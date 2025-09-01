
# Credit Card Fraud Detection

This project predicts **credit card default risk** using machine learning models, including Gradient Boosting and CatBoost. It is based on the **Default of Credit Card Clients Dataset**, and includes data preprocessing, exploratory data analysis (EDA), model training with handling class imbalance, and a user-friendly prediction script.

---

## ⚙️ Features

* **Data Preprocessing**:

  * Handles invalid values in categorical features (`EDUCATION`, `MARRIAGE`).
  * Normalizes numeric features using `StandardScaler`.
  * Ensures proper data types for repayment history (`PAY_*`).

* **Exploratory Data Analysis (EDA)**:

  * Distribution plots of target variable.
  * Relationship plots (`SEX`, `EDUCATION`, `MARRIAGE` vs default).
  * Repayment history analysis.
  * Correlation heatmap.
  * Summary report (`eda_summary.txt`).

* **Modeling**:

  * Handles class imbalance using **SMOTE**.
  * Trains multiple models: Logistic Regression, Random Forest, Gradient Boosting, XGBoost, LightGBM, and CatBoost.
  * Evaluates using **Accuracy, Precision, Recall, F1-score, ROC-AUC**.
  * Saves best-performing models (`.pkl`).

* **Prediction Script (`project.py`)**:

  * CLI-based user input for credit card features.
  * Predicts **default next month (Yes/No)** with probabilities using both **GradientBoosting** and **CatBoost** models.

---

## 🗂 Dataset

* **Source**: [UCI Machine Learning Repository – Default of Credit Card Clients Dataset](https://archive.ics.uci.edu/ml/datasets/default+of+credit+card+clients)
* **Description**:

  * Records: 30,000 clients
  * Features: 23 (Demographics, credit amount, repayment history, bill statements, payments)
  * Target: `default_next_month` (1 = default, 0 = non-default)

---

## 📊 Example Results

Model Performance Summary:
| Model              | Accuracy  | Precision | Recall    | F1        | ROC-AUC   |
| ------------------ | --------- | --------- | --------- | --------- | --------- |
| GradientBoosting   | **0.813** | 0.608     | 0.436     | **0.508** | 0.771     |
| LightGBM           | 0.813     | **0.613** | 0.414     | 0.494     | **0.773** |
| CatBoost           | 0.739     | 0.438     | **0.635** | **0.518** | 0.769     |
| XGBoost            | 0.740     | 0.437     | 0.607     | 0.508     | 0.756     |
| RandomForest       | 0.805     | 0.585     | 0.411     | 0.483     | 0.755     |
| LogisticRegression | 0.685     | 0.372     | 0.619     | 0.465     | 0.705     |

---

## 🚀 Getting Started

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/Credit-Card-Fraud-Detection.git
cd Credit-Card-Fraud-Detection
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Train Models & Generate EDA

```bash
python data.py
```

This will:

* Preprocess the dataset
* Generate EDA plots (saved in `eda_results/`)
* Train and evaluate models
* Save best pipelines (`gradient_boosting_pipeline.pkl`, `CatBoost_pipeline.pkl`)

### 4️⃣ Run Prediction Script

```bash
python project.py
```

You will be prompted to enter details such as **credit limit, age, education, marriage status, repayment history, bill amounts, and payments**.
The script will output predictions from both **GradientBoosting** and **CatBoost**.

---

## 📦 Requirements

See `requirements.txt`:

* pandas, scikit-learn, imbalanced-learn
* xgboost, lightgbm, catboost
* matplotlib, seaborn
* joblib

Install with:

```bash
pip install -r requirements.txt
```
---

## 👨‍💻 Author

* **Darshit Varshney**
* B.Tech 2 Year Engineering Student At AKGEC 

---
