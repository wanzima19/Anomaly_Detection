from flask import Flask, request, jsonify

import joblib
import json
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
import tensorflow as tf
#from tensorflow.keras.models import load_model
import keras
from keras.models import load_model



app = Flask(__name__)

# Load artifacts (expects files in output/ produced by the notebook)
BASE_OUTPUT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")

class FeatureEngineer:
    def __init__(self):
        from sklearn.preprocessing import RobustScaler, StandardScaler
        self.scaler_amount = RobustScaler()
        self.scaler_time = StandardScaler()

    def create_features(self, df, fit=True):
        import numpy as np
        df = df.copy()
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Day'] = (df['Time'] // (3600 * 24))
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        if fit:
            df['Amount_scaled'] = self.scaler_amount.fit_transform(df[['Amount']])
        else:
            df['Amount_scaled'] = self.scaler_amount.transform(df[['Amount']])
        df['Amount_log'] = np.log1p(df['Amount'])
        df['Amount_deviation'] = np.abs(df['Amount'] - df['Amount'].median()) / df['Amount'].std()
        df['V_amount_interaction'] = df.get('V1', 0) * df['Amount_scaled']
        pca_cols = [f'V{i}' for i in range(1, 29) if f'V{i}' in df.columns]
        if len(pca_cols) > 0:
            df['PCA_magnitude'] = np.sqrt(np.sum(df[pca_cols]**2, axis=1))
        else:
            df['PCA_magnitude'] = 0
        return df

    def prepare_features(self, df, fit=True):
        df = self.create_features(df, fit)
        feature_cols = [col for col in df.columns if col not in ['Time', 'Amount', 'Class', 'Hour', 'Day']]
        X = df[feature_cols]
        y = df['Class'] if 'Class' in df.columns else None
        return X, y, feature_cols

def _load_artifacts():
    cfg_path = os.path.join(BASE_OUTPUT, "model_config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path, "r") as f:
            cfg = json.load(f)

    # Load preprocessing
    fe = None
    scaler = None
    fe_path = os.path.join(BASE_OUTPUT, "feature_engineer.pkl")
    scaler_path = os.path.join(BASE_OUTPUT, "scaler.pkl")
    if os.path.exists(fe_path):
        fe = joblib.load(fe_path)
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    # Load classical models
    classical_models = {}
    for name in ["random_forest", "xgboost", "logistic"]:
        p = os.path.join(BASE_OUTPUT, f"model_{name}.pkl")
        if os.path.exists(p):
            classical_models[name] = joblib.load(p)

    # Load deep learning model
    dl_path = os.path.join(BASE_OUTPUT, "model_deep_learning.keras")
    deep_model = None
    if os.path.exists(dl_path):
        deep_model = load_model(dl_path, compile=False)

    return cfg, fe, scaler, classical_models, deep_model


CFG, FE, SCALER, CLASSICAL_MODELS, DEEP_MODEL = _load_artifacts()


class RealTimeFraudDetector:
    def __init__(self, cfg, feature_engineer, scaler, classical_models, deep_model, threshold=0.5):
        self.cfg = cfg or {}
        self.fe = feature_engineer
        self.scaler = scaler
        self.classical_models = classical_models
        self.deep_model = deep_model
        self.threshold = self.cfg.get("optimal_threshold", threshold)
        self.weights = self.cfg.get("model_weights", None)

    def preprocess_transaction(self, transaction_dict):
        tx_df = pd.DataFrame([transaction_dict])
        X, _, _ = self.fe.prepare_features(tx_df, fit=False) if self.fe is not None else (tx_df, None, list(tx_df.columns))
        X_scaled = self.scaler.transform(X) if self.scaler is not None else X.values
        return X, X_scaled

    def predict(self, transaction_dict, return_details=False):
        X, X_scaled = self.preprocess_transaction(transaction_dict)

        predictions = {}
        # Classical models expect the same features used during training (unscaled in notebook), try on X
        for name, model in self.classical_models.items():
            try:
                if hasattr(model, 'predict_proba'):
                    predictions[name] = model.predict_proba(X)[:, 1]
                else:
                    predictions[name] = model.predict(X)
            except Exception:
                # fallback to scaled
                try:
                    if hasattr(model, 'predict_proba'):
                        predictions[name] = model.predict_proba(X_scaled)[:, 1]
                    else:
                        predictions[name] = model.predict(X_scaled)
                except Exception:
                    predictions[name] = np.array([0.0])

        # Deep model uses scaled features
        if self.deep_model is not None:
            try:
                pred_dl, _ = self.deep_model.predict(X_scaled, verbose=0)
                predictions['deep_learning'] = pred_dl.flatten()
            except Exception:
                # try older keras predict shape
                try:
                    pred_dl = self.deep_model.predict(X_scaled, verbose=0)
                    if isinstance(pred_dl, list):
                        predictions['deep_learning'] = pred_dl[0].flatten()
                    else:
                        predictions['deep_learning'] = np.asarray(pred_dl).flatten()
                except Exception:
                    predictions['deep_learning'] = np.array([0.0])

        # Combine with weights if available
        if self.weights:
            total = np.zeros_like(next(iter(predictions.values())))
            for k, v in predictions.items():
                w = self.weights.get(k, 1.0 / max(1, len(predictions)))
                total = total + v * w
            final = total
        else:
            # simple average
            final = np.mean(list(predictions.values()), axis=0)

        risk_score = float(final[0])
        is_fraud = risk_score > self.threshold

        # Determine action
        if risk_score < 0.3:
            risk_level = "LOW"
            action = "Approve"
        elif risk_score < 0.7:
            risk_level = "MEDIUM"
            action = "Review"
        else:
            risk_level = "HIGH"
            action = "Block"

        result = {
            'transaction_id': transaction_dict.get('transaction_id', 'unknown'),
            'risk_score': risk_score,
            'risk_level': risk_level,
            'is_fraud': bool(is_fraud),
            'recommended_action': action,
            'threshold_used': self.threshold,
            'model_contributions': {k: float(v[0]) for k, v in predictions.items()}
        }

        return result if return_details else risk_score


detector = RealTimeFraudDetector(CFG, FE, SCALER, CLASSICAL_MODELS, DEEP_MODEL, threshold=CFG.get('optimal_threshold', 0.5))

@app.route('/')
def index():
   return"welcome to the fraud detection API. System"


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})



@app.route('/predict', methods=['POST'])
def predict_endpoint():
    payload = request.get_json()
    if not payload:
        return jsonify({'error': 'invalid json payload'}), 400

    result = detector.predict(payload, return_details=True)
    return jsonify(result)


if __name__ == '__main__':
    app.run(debug=True)
