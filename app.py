from flask import Flask, request, jsonify
import joblib
import numpy as np

# Define AEFD class and predict_aefd function 
def predict_aefd(features, lr, rf, ae):
    lr_pred = lr.predict_proba(features)[:, 1]
    rf_pred = rf.predict_proba(features)[:, 1]
    recon_error = np.mean((features - ae.predict(features))**2, axis=1)
    anomaly_score = (recon_error - np.min(recon_error)) / (np.max(recon_error) - np.min(recon_error))
    ensemble_pred = 0.4 * lr_pred + 0.4 * rf_pred + 0.2 * anomaly_score
    return (ensemble_pred > 0.5).astype(int)

class AEFD:
    def __init__(self, lr, rf, ae):
        self.lr = lr
        self.rf = rf
        self.ae = ae
    def predict(self, X):
        return predict_aefd(X, self.lr, self.rf, self.ae)

app = Flask(__name__)

# Load updated AEFD model
try:
    model = joblib.load('models/aefraud_detector.pkl')
    print("AEFD model loaded successfully.")
except FileNotFoundError:
    print("Error: 'aefraud_detector.pkl' not found. Run model.py first.")
    model = None

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'AEFD Fraud Detection API. Use POST /predict with {"features": [list of 23 features]}'})

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({'error': 'Model not loaded'}), 500
    try:
        data = request.get_json()
        if 'features' not in data:
            return jsonify({'error': 'Missing "features" key'}), 400
        features = np.array(data['features']).reshape(1, -1)
        if features.shape[1] != 23:  
            return jsonify({'error': 'Features must be a list of 23 values'}), 400
        prediction = model.predict(features)
        return jsonify({'fraud': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
