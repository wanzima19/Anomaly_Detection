
from flask import Flask, request, jsonify
import joblib
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import keras
from sklearn.preprocessing import RobustScaler, StandardScaler


# Add at top, after imports
from sklearn.ensemble import IsolationForest

# In the loading section, add anomaly detector (train/save it in anomaly.ipynb if needed)
anomaly_detector = IsolationForest(contamination=0.1, random_state=42)  # Pre-trained on training data
# For simplicity, fit on a sample; in production, fit on X_train and save/load like other models
sample_data = np.random.randn(1000, 28)  # Placeholder; replace with actual training features
anomaly_detector.fit(sample_data)



app = Flask(__name__)

script_dir = os.path.dirname(os.path.abspath(__file__))
# Load models at startup
print("Loading fraud detection models...")

#-------------------------------------------
# ...existing code...

class FeatureEngineer:
    def __init__(self):
        self.scaler_amount = RobustScaler()
        self.scaler_time = StandardScaler()
        
    def create_features(self, df, fit=True):
        """Create advanced features for fraud detection"""
        df = df.copy()
        
        # 1. Temporal Features
        df['Hour'] = (df['Time'] // 3600) % 24
        df['Day'] = (df['Time'] // (3600 * 24))
        
        # Cyclical encoding for hour (captures circular nature of time)
        df['Hour_sin'] = np.sin(2 * np.pi * df['Hour'] / 24)
        df['Hour_cos'] = np.cos(2 * np.pi * df['Hour'] / 24)
        
        # 2. Amount-based features
        if fit:
            df['Amount_scaled'] = self.scaler_amount.fit_transform(df[['Amount']])
        else:
            df['Amount_scaled'] = self.scaler_amount.transform(df[['Amount']])
            
        df['Amount_log'] = np.log1p(df['Amount'])
        
        # 3. Statistical aggregations (simulating user history)
        # In production, these would be calculated from historical data
        amount_std = df['Amount'].std()
        if amount_std == 0 or np.isnan(amount_std):
            df['Amount_deviation'] = 0  # No deviation if std is 0
        else:
            df['Amount_deviation'] = np.abs(df['Amount'] - df['Amount'].median()) / amount_std
        
        # 4. Interaction features (important for capturing complex patterns)
        df['V_amount_interaction'] = df['V1'] * df['Amount_scaled']
        
        # 5. Risk scores based on PCA components
        # Components with high variance often indicate anomalies
        pca_cols = [f'V{i}' for i in range(1, 29)]
        df['PCA_magnitude'] = np.sqrt(np.sum(df[pca_cols]**2, axis=1))
        
        # 6. Handle any remaining NaN values
        df = df.fillna(0)
        
        # 7. Replace inf values with 0
        df = df.replace([np.inf, -np.inf], 0)
        
        return df
    
    def prepare_features(self, df, fit=True):
        df = self.create_features(df, fit)
        
        # Select features for modeling (exclude raw Time and original Amount, plus metadata like transaction_id)
        feature_cols = [col for col in df.columns if col not in ['Time', 'Amount', 'Class', 'Hour', 'Day', 'transaction_id']]
        
        X = df[feature_cols]
        y = df['Class'] if 'Class' in df.columns else None
        
        # Final safety check: fill any remaining NaN/inf values
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        return X, y, feature_cols

# ...existing code...


# Get the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# Load models at startup
print("Loading fraud detection models...")
models = {
    'random_forest': joblib.load(os.path.join(script_dir, 'model_random_forest.pkl')),
    'xgboost': joblib.load(os.path.join(script_dir, 'model_xgboost.pkl')),
    'logistic': joblib.load(os.path.join(script_dir, 'model_logistic.pkl')),
    'deep_learning': keras.models.load_model(os.path.join(script_dir, 'model_deep_learning.keras'))
}
fe = joblib.load(os.path.join(script_dir, 'feature_engineer.pkl'))
scaler = joblib.load(os.path.join(script_dir, 'scaler.pkl'))

with open(os.path.join(script_dir, 'model_config.json')) as f:
    config = json.load(f)

THRESHOLD = config['optimal_threshold']


def _normalize_prediction_output(pred_output):
    """Return model predictions as a flat 1D numpy array.

    Handles Keras models that may return a single array or a list/tuple
    of arrays for multi-output architectures.
    """
    if isinstance(pred_output, (list, tuple)):
        if len(pred_output) == 0:
            return np.array([], dtype=float)
        pred_output = pred_output[0]
    return np.asarray(pred_output, dtype=float).reshape(-1)


#-------------------------------------------


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'threshold': THRESHOLD,
        'timestamp': datetime.now().isoformat()
    })

# ...existing code...




# ...existing code...

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['transaction_id', 'Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400
        
        # Preprocess and detect anomalies
        tx_df = pd.DataFrame([data])
        X, _, _ = fe.prepare_features(tx_df, fit=False)
        X_scaled = scaler.transform(X)
        
        # Anomaly detection: use only raw PCA components V1..V28 (exactly 28 features).
        v_cols = [f'V{i}' for i in range(1, 29)]
        v_idx = [X.columns.get_loc(col) for col in v_cols]
        v_features = X_scaled[:, v_idx]
        anomaly_score = anomaly_detector.decision_function(v_features)[0]  # -1 to 1
        
        # Flag and preprocess if anomalous
        is_anomalous = anomaly_score < -0.5  # Threshold for anomaly
        if is_anomalous:
            # Corrections: Clip extreme values, fill masked (e.g., if many V=0)
            for col in X.columns:
                if col.startswith('V'):
                    X[col] = np.clip(X[col], -3, 3)  # Clip outliers
                elif col == 'Amount_scaled':
                    X[col] = np.clip(X[col], -2, 2)
            X_scaled = scaler.transform(X)  # Re-scale after correction
            anomaly_flag = "Input corrected for anomalies (potential corruption/OOD)"
        else:
            anomaly_flag = "Input appears normal"
        
        # Proceed with prediction
        predictions = {}
        for name, model in models.items():
            if name == 'deep_learning':
                pred = _normalize_prediction_output(model.predict(X_scaled, verbose=0))
                predictions[name] = float(pred[0])
            else:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                predictions[name] = float(pred[0])
        
        weights = config['model_weights']
        risk_score = sum(predictions[name] * weights[name] for name in predictions.keys())
        
        # Boost risk if anomalous
        if is_anomalous:
            risk_score = min(1.0, risk_score + 0.2)  # Increase risk for suspicious inputs
        
        # Determine action
        if risk_score < 0.3:
            action, level = "approve", "low"
        elif risk_score < 0.7:
            action, level = "review", "medium"
        else:
            action, level = "block", "high"
        
        return jsonify({
            'transaction_id': data['transaction_id'],
            'risk_score': float(risk_score),
            'risk_level': level,
            'recommended_action': action,
            'is_fraud': bool(risk_score > THRESHOLD),
            'anomaly_flag': anomaly_flag,  # New: Explain preprocessing
            'model_breakdown': {k: float(v) for k, v in predictions.items()},
            'threshold': float(THRESHOLD),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

# ...existing code...

# ...existing code...

@app.route('/batch_predict', methods=['POST', 'GET'])
def batch_predict():
    try:
        data = request.get_json()
        transactions = data.get('transactions', []) if data else []

        if not isinstance(transactions, list) or len(transactions) == 0:
            return jsonify({'error': 'Payload must include non-empty "transactions" list'}), 400

        tx_df = pd.DataFrame(transactions)
        required_fields = ['transaction_id', 'Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing = [f for f in required_fields if f not in tx_df.columns]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400

        X, _, _ = fe.prepare_features(tx_df, fit=False)
        X_scaled = scaler.transform(X)

        weights = config['model_weights']
        weighted_sum = np.zeros(len(tx_df), dtype=float)

        for name, model in models.items():
            if name == 'deep_learning':
                dl_pred = _normalize_prediction_output(model.predict(X_scaled, verbose=0))
                weighted_sum += dl_pred * weights[name]
            else:
                if hasattr(model, 'predict_proba'):
                    p = model.predict_proba(X)[:, 1]
                else:
                    p = model.predict(X)
                weighted_sum += np.asarray(p).reshape(-1) * weights[name]

        results = []
        for idx, tx in enumerate(transactions):
            risk_score = float(weighted_sum[idx])
            results.append({
                'transaction_id': tx.get('transaction_id', f'tx_{idx}'),
                'risk_score': risk_score,
                'is_fraud': bool(risk_score > THRESHOLD)
            })
        
        return jsonify({
            'results': results,
            'processed_count': len(results),
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=False)
