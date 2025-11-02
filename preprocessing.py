import pandas as pd
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

def preprocess_data(file_path=r"C:\Users\darsh\Python\ccf\default of credit card clients.csv", target_col='Y'):
    try:
        data = pd.read_csv(file_path, encoding='utf-8') 
        print("Dataset loaded successfully.")
        print("Columns in dataset:", list(data.columns))
        print("Data types:\n", data.dtypes)  
        
        # Check if target column exists
        if target_col not in data.columns:
            raise KeyError(f"'{target_col}' column not found. Available columns: {list(data.columns)}. "
                           f"Update target_col to the correct name (e.g., 'Class' for Fraud dataset).")
        
        # Drop unnecessary columns
        X = data.drop([target_col, 'Unnamed: 0'], axis=1, errors='ignore')
        y = data[target_col]
        
        #Convert object columns to numeric
        for col in X.columns:
            if X[col].dtype == 'object':
                try:
                    X[col] = pd.to_numeric(X[col].str.strip(), errors='coerce')  # Strip spaces, convert
                    print(f"Converted column '{col}' to numeric.")
                except:
                    print(f"Dropping non-convertible column '{col}'.")
                    X = X.drop(col, axis=1)
        
        # Drop rows with NaN after conversion
        X = X.dropna()
        y = y[X.index]  
        
        # Proceed with preprocessing
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        smote = SMOTE()
        X_res, y_res = smote.fit_resample(X_scaled, y)
        print("Preprocessing completed. Shape after SMOTE:", X_res.shape)
        print("Class distribution after SMOTE:", pd.Series(y_res).value_counts())
        return X_res, y_res
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
    except Exception as e:
        print(f"Unexpected error: {e}")
        return None, None

# Usage
if __name__ == "__main__":
    X, y = preprocess_data()
    if X is not None and y is not None:
        pd.DataFrame(X).to_csv('X_processed.csv', index=False)
        pd.DataFrame(y).to_csv('y_processed.csv', index=False)
        print("Processed data saved.")