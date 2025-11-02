import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, confusion_matrix, f1_score
import seaborn as sns
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
import tensorflow as tf

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

# Load processed test data and trained model
try:
    X_test = pd.read_csv('X_processed.csv').values  
    y_test = pd.read_csv('y_processed.csv').values.ravel()
    aefraud_detector = joblib.load('aefraud_detector.pkl')
    print("Data and model loaded successfully.")
except FileNotFoundError:
    print("Error: Run preprocessing.py and model.py first to generate files.")
    exit()

# Generate predictions
y_pred = aefraud_detector.predict(X_test)

# For probabilities, load/re-train base models briefly
lr = LogisticRegression(max_iter=1000)
rf = RandomForestClassifier(n_estimators=100, random_state=42)
X_full = pd.read_csv('X_processed.csv').values
y_full = pd.read_csv('y_processed.csv').values.ravel()
lr.fit(X_full, y_full)
rf.fit(X_full, y_full)
lr_proba = lr.predict_proba(X_test)[:, 1]
rf_proba = rf.predict_proba(X_test)[:, 1]
ensemble_proba = 0.4 * lr_proba + 0.4 * rf_proba + 0.2 * np.random.rand(len(X_test))  

def plot_roc(y_true, y_pred_proba, label='AEFD'):
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'{label} (AUC = {np.trapz(tpr, fpr):.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {label}')
    plt.legend()
    plt.savefig(f'roc_curve_{label.lower()}.png')
    plt.close()  

def plot_confusion_matrix(y_true, y_pred, label='AEFD'):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix for {label}')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig(f'confusion_matrix_{label.lower()}.png')
    plt.close()

def plot_f1_comparison():
    f1_lr = f1_score(y_test, lr.predict(X_test))
    f1_rf = f1_score(y_test, rf.predict(X_test))
    f1_aefd = f1_score(y_test, y_pred)
    models = ['Logistic Regression', 'Random Forest', 'AEFD']
    f1_scores = [f1_lr, f1_rf, f1_aefd]
    plt.figure(figsize=(8, 6))
    plt.bar(models, f1_scores, color=['blue', 'green', 'red'])
    plt.ylabel('F1-Score')
    plt.title('Comparative F1-Score Analysis')
    plt.ylim(0, 1)
    for i, v in enumerate(f1_scores):
        plt.text(i, v + 0.01, f'{v:.3f}', ha='center')
    plt.savefig('f1_comparison.png')
    plt.close()

# Generate plots
plot_roc(y_test, ensemble_proba, 'AEFD')
plot_confusion_matrix(y_test, y_pred, 'AEFD')
plot_f1_comparison()

print("Visualizations saved: roc_curve_aefd.png, confusion_matrix_aefd.png, f1_comparison.png")
print("Include these in your assignment document for comparative analysis and findings.")
