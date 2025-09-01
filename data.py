import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# Load dataset
df = pd.read_csv(r"C:\Users\darsh\Python\Credit Card Fraud Detection\default of credit card clients.csv",skiprows=1)

# print(df.info())
# print(df.describe())


if "ID" in df.columns:
    df.drop(columns=["ID"],inplace=True)

# Rename target to a friendly name
if "default payment next month" in df.columns:
    df.rename(columns={"default payment next month": "default_next_month"}, inplace=True)

# Fix categorical label issues
if "EDUCATION" in df.columns:
    df["EDUCATION"] = df["EDUCATION"].replace({0:4, 5:4, 6:4})
if "MARRIAGE" in df.columns:
    df["MARRIAGE"] = df["MARRIAGE"].replace({0:3})

# Ensure PAY_* columns are ints
pay_cols = [c for c in df.columns if c.startswith("PAY_")]
for c in pay_cols:
    df[c] = df[c].astype(int)


# Save target distribution plot
sns.countplot(x="default_next_month", data=df)
plt.title("Target Variable Distribution (Default vs Non-default)")
plt.savefig(r"C:\Users\darsh\Python\Credit Card Fraud Detection\eda_results/target_distribution.png")
plt.close()

# Save categorical plots
for col in ["SEX", "EDUCATION", "MARRIAGE"]:
    sns.countplot(x=col, hue="default_next_month", data=df)
    plt.title(f"{col} Distribution vs Default")
    plt.savefig(rf"C:\Users\darsh\Python\Credit Card Fraud Detection\eda_results/{col}_vs_default.png")
    plt.close()

# Repayment history
for col in pay_cols:
    sns.countplot(x=col, hue="default_next_month", data=df)
    plt.title(f"{col} vs Default")
    plt.savefig(rf"C:\Users\darsh\Python\Credit Card Fraud Detection\eda_results/{col}_vs_default.png")
    plt.close()

# Correlation heatmap
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), cmap="coolwarm", center=0)
plt.title("Correlation Heatmap")
plt.savefig(r"C:\Users\darsh\Python\Credit Card Fraud Detection\eda_results/correlation_heatmap.png")
plt.close()



with open("eda_results/eda_summary.txt", "w") as f:
    f.write("=== Credit Card Default Dataset: EDA Summary ===\n\n")
    f.write("1. Target variable is imbalanced (more non-defaults).\n")
    f.write("2. Repayment history (PAY_1 … PAY_6) shows strong link with default.\n")
    f.write("3. EDUCATION and MARRIAGE categories had invalid values; fixed.\n")
    f.write("4. Bill amounts and payment amounts are correlated with default risk.\n")
    f.write("5. Demographics (SEX, AGE) show weaker influence.\n")


# -----------------------------
# Train/test split


X = df.drop(columns=["default_next_month"])
y = df["default_next_month"].astype(int)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print("Train/test split done.")


# -----------------------------
# Preprocessing and model pipelines
# -----------------------------
categorical = ["SEX", "EDUCATION", "MARRIAGE"]
numeric = [c for c in df.columns if c not in categorical + pay_cols + ["default_next_month"]]

preprocess = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric),
    ("ord", "passthrough", pay_cols),
    ("cat", "passthrough", categorical)
], remainder='drop')

# Import extra models

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# Ratio for imbalance (non-default / default)
ratio = y_train.value_counts()[0] / y_train.value_counts()[1]

# Use ImbPipeline to insert SMOTE before the classifier
models = {
    "LogisticRegression": ImbPipeline([
        ("prep", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("clf", LogisticRegression(class_weight='balanced', max_iter=1000))
    ]),
    "RandomForest": ImbPipeline([
        ("prep", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("clf", RandomForestClassifier(class_weight="balanced", n_estimators=200, random_state=42))
    ]),
    "GradientBoosting": ImbPipeline([
        ("prep", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("clf", GradientBoostingClassifier())
    ]),
    "XGBoost": ImbPipeline([
        ("prep", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("clf", XGBClassifier(scale_pos_weight=ratio, eval_metric="logloss", random_state=42))
    ]),
    "LightGBM": ImbPipeline([
        ("prep", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("clf", LGBMClassifier(class_weight="balanced", random_state=42))
    ]),
    "CatBoost": ImbPipeline([
        ("prep", preprocess),
        ("smote", SMOTE(random_state=42)),
        ("clf", CatBoostClassifier(class_weights=[1, ratio], verbose=0, random_state=42))
    ])
}

# Train, evaluate, and save models
# -----------------------------
metrics = []
for name, pipeline in models.items():
    print(f"Training {name} ...")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    y_prob = pipeline.predict_proba(X_test)[:,1] if hasattr(pipeline, "predict_proba") else pipeline.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    auc = roc_auc_score(y_test, y_prob)

    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall: {rec:.4f}")
    print(f"F1-score: {f1:.4f}")
    print(f"ROC-AUC: {auc:.4f}\n")

    metrics.append({
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "roc_auc": auc
    })

# Convert results to DataFrame
metrics_df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
print("\n=== Model Performance Summary ===")
print(metrics_df)

gb_model = models["GradientBoosting"]
gb_model.fit(X_train, y_train)   # make sure it's trained
joblib.dump(gb_model, "gradient_boosting_pipeline.pkl")

print("GradientBoosting model saved, as its returned best overall performance.")


cat_model = models["GradientBoosting"]
cat_model.fit(X_train, y_train)   # make sure it's trained
joblib.dump(cat_model, "CatBoost_pipeline.pkl")

print("CatBoost model saved, as its Best for Detecting Defaults")