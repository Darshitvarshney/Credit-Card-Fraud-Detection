import joblib
import pandas as pd


# Load the trained GradientBoosting pipeline
gradient_model = joblib.load("gradient_boosting_pipeline.pkl")
cat_model = joblib.load("CatBoost_pipeline.pkl")

# Define expected feature columns (except target)
feature_cols = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6",
    "BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6",
    "PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"
]

# Instructions for categorical inputs
print("\n=== Please provide input values ===")
print("SEX: 1 = Male, 2 = Female")
print("EDUCATION: 1 = Graduate School, 2 = University, 3 = High School, 4 = Others")
print("MARRIAGE: 1 = Married, 2 = Single, 3 = Others")
print("PAY_X (Payment Status): -1 = Paid duly, 0 = No delay, 1 = 1 month delay, 2 = 2 months delay ... up to 8 = 8 months delay")

# Collect input from user
data = {}

data["LIMIT_BAL"] = float(input("Enter LIMIT_BAL (Amount of credit given, e.g., 20000): "))
data["SEX"] = int(input("Enter SEX (1=Male, 2=Female): "))
data["EDUCATION"] = int(input("Enter EDUCATION (1=Graduate School, 2=University, 3=High School, 4=Others): "))
data["MARRIAGE"] = int(input("Enter MARRIAGE (1=Married, 2=Single, 3=Others): "))
data["AGE"] = int(input("Enter AGE (e.g., 35): "))

# Payment history
for col in ["PAY_0","PAY_2","PAY_3","PAY_4","PAY_5","PAY_6"]:
    data[col] = int(input(f"Enter {col} (payment status, -1=paid duly, 0=no delay, 1=1-month delay ...): "))

# Bill statement amounts
for col in ["BILL_AMT1","BILL_AMT2","BILL_AMT3","BILL_AMT4","BILL_AMT5","BILL_AMT6"]:
    data[col] = float(input(f"Enter {col} (bill statement amount, e.g., 5000): "))

# Payment amounts
for col in ["PAY_AMT1","PAY_AMT2","PAY_AMT3","PAY_AMT4","PAY_AMT5","PAY_AMT6"]:
    data[col] = float(input(f"Enter {col} (amount paid, e.g., 2000): "))

# Convert to DataFrame
input_df = pd.DataFrame([data])

# Predict
print("\n--- Making Prediction ---")

# Gradient Boosting
gb_pred = gradient_model.predict(input_df)[0]
gb_prob = gradient_model.predict_proba(input_df)[0][1]

print("\n[GradientBoosting Result]")
print("Default Next Month:", "YES (1)" if gb_pred == 1 else "NO (0)")
print(f"Probability of Default: {gb_prob:.2f}")

# CatBoost
cat_pred = cat_model.predict(input_df)[0]
cat_prob = cat_model.predict_proba(input_df)[0][1]

print("\n[CatBoost Result]")
print("Default Next Month:", "YES (1)" if cat_pred == 1 else "NO (0)")
print(f"Probability of Default: {cat_prob:.2f}")