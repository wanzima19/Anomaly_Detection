"""
Flask API for Real-Time Fraud Detection
Save this as 'fraud_api.py' and run with: python fraud_api.py
"""

flask_code = '''
from flask import Flask, request, jsonify
import joblib
import json
import numpy as np
import pandas as pd
from datetime import datetime

app = Flask(__name__)

# Load models at startup
print("Loading fraud detection models...")
models = {
    'random_forest': joblib.load('model_random_forest.pkl'),
    'xgboost': joblib.load('model_xgboost.pkl'),
    'logistic': joblib.load('model_logistic.pkl'),
    'deep_learning': keras.models.load_model('model_deep_learning.keras')
}
fe = joblib.load('feature_engineer.pkl')
scaler = joblib.load('scaler.pkl')

with open('model_config.json') as f:
    config = json.load(f)

THRESHOLD = config['optimal_threshold']

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'models_loaded': list(models.keys()),
        'threshold': THRESHOLD,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        # Validate input
        required_fields = ['transaction_id', 'Time', 'Amount'] + [f'V{i}' for i in range(1, 29)]
        missing = [f for f in required_fields if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400
        
        # Preprocess
        tx_df = pd.DataFrame([data])
        X, _, _ = fe.prepare_features(tx_df, fit=False)
        X_scaled = scaler.transform(X)
        
        # Get predictions from all models
        predictions = {}
        for name, model in models.items():
            if name == 'deep_learning':
                pred, _ = model.predict(X_scaled, verbose=0)
                predictions[name] = pred.flatten()[0]
            else:
                if hasattr(model, 'predict_proba'):
                    pred = model.predict_proba(X)[:, 1]
                else:
                    pred = model.predict(X)
                predictions[name] = pred[0]
        
        # Weighted ensemble
        weights = config['model_weights']
        risk_score = sum(predictions[name] * weights[name] for name in predictions.keys())
        
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
            'is_fraud': risk_score > THRESHOLD,
            'model_breakdown': {k: float(v) for k, v in predictions.items()},
            'threshold': THRESHOLD,
            'timestamp': datetime.now().isoformat()
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        data = request.get_json()
        transactions = data.get('transactions', [])
        
        results = []
        for tx in transactions:
            # Process each transaction (simplified)
            tx_df = pd.DataFrame([tx])
            X, _, _ = fe.prepare_features(tx_df, fit=False)
            X_scaled = scaler.transform(X)
            
            # Quick ensemble prediction
            preds = []
            for name, model in models.items():
                if name == 'deep_learning':
                    p, _ = model.predict(X_scaled, verbose=0)
                    preds.append(p.flatten()[0] * config['model_weights'][name])
                else:
                    if hasattr(model, 'predict_proba'):
                        p = model.predict_proba(X)[:, 1][0]
                    else:
                        p = model.predict(X)[0]
                    preds.append(p * config['model_weights'][name])
            
            risk_score = sum(preds)
            results.append({
                'transaction_id': tx['transaction_id'],
                'risk_score': float(risk_score),
                'is_fraud': risk_score > THRESHOLD
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
'''

# Save Flask API code
with open('output/fraud_api.py', 'w') as f:
    f.write(flask_code)

print("✓ Saved Flask API code to fraud_api.py")
print("\nTo deploy:")
print("  1. pip install flask scikit-learn xgboost tensorflow joblib pandas numpy")
print("  2. python fraud_api.py")
print("  3. Test with: curl -X POST http://localhost:5000/predict -H 'Content-Type: application/json' -d '{...}'")


