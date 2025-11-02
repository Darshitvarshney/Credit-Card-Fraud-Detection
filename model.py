import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from imblearn.over_sampling import SMOTE
import tensorflow as tf
from tensorflow.keras import layers
import joblib
import time

# Load processed data
try:
    X = pd.read_csv('data/X_processed.csv').values
    y = pd.read_csv('data/y_processed.csv').values.ravel()
    print("Processed data loaded successfully. Shape:", X.shape)
except FileNotFoundError:
    print("Error: Run preprocessing.py first to generate X_processed.csv and y_processed.csv.")
    exit()

# Split data 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Autoencoder for anomaly detection 
def build_autoencoder(input_dim):
    input_layer = layers.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(input_layer)
    decoded = layers.Dense(input_dim, activation='sigmoid')(encoded)
    autoencoder = tf.keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')
    return autoencoder

autoencoder = build_autoencoder(X_train.shape[1])
start_time = time.time()
autoencoder.fit(X_train, X_train, epochs=50, batch_size=32, validation_data=(X_test, X_test), verbose=0)
ae_train_time = time.time() - start_time
print(f"Autoencoder training time: {ae_train_time:.2f} seconds")

# Base models
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)

lr.fit(X_train, y_train)
rf.fit(X_train, y_train)

# Cross-validation for robustness 
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_cv_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring='f1')
rf_cv_scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='f1')
print(f"Logistic Regression CV F1: {lr_cv_scores.mean():.3f} (+/- {lr_cv_scores.std():.3f})")
print(f"Random Forest CV F1: {rf_cv_scores.mean():.3f} (+/- {rf_cv_scores.std():.3f})")

# AEFD Ensemble: Weighted voting with anomaly scores
def predict_aefd(features):
    lr_pred = lr.predict_proba(features)[:, 1]
    rf_pred = rf.predict_proba(features)[:, 1]
    recon_error = np.mean((features - autoencoder.predict(features))**2, axis=1)
    anomaly_score = (recon_error - np.min(recon_error)) / (np.max(recon_error) - np.min(recon_error))
    ensemble_pred = 0.4 * lr_pred + 0.4 * rf_pred + 0.2 * anomaly_score
    return (ensemble_pred > 0.5).astype(int)

# Evaluate on test set
y_pred = predict_aefd(X_test)
print("AEFD Test Results:")
print(classification_report(y_test, y_pred))
print(f"AEFD F1-Score: {f1_score(y_test, y_pred):.3f}")

# Compare with base models 
lr_pred_base = lr.predict(X_test)
rf_pred_base = rf.predict(X_test)
print(f"Logistic Regression F1: {f1_score(y_test, lr_pred_base):.3f}")
print(f"Random Forest F1: {f1_score(y_test, rf_pred_base):.3f}")


print("Space complexity: AEFD model size is compact (ensemble + small autoencoder).")

# Save model 
class AEFD:
    def __init__(self, lr, rf, ae):
        self.lr = lr
        self.rf = rf
        self.ae = ae
    def predict(self, X):
        return predict_aefd(X)

aefraud_detector = AEFD(lr, rf, autoencoder)
joblib.dump(aefraud_detector, 'aefraud_detector.pkl')
print("AEFD model saved as 'aefraud_detector.pkl'")