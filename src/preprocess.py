import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def load_data(file_path):
    """
    Loads the raw CSV dataset from the specified file path.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No dataset found at: {file_path}")
    return pd.read_csv(file_path)

def clean_data(df):
    """
    Performs initial data cleaning:
    1. Removes 'customerID' as it is a unique identifier with no predictive power.
    2. Converts 'TotalCharges' to numeric (handles spaces/empty strings).
    3. Imputes missing values generated during conversion using the column median.
    4. Maps the target variable 'Churn' to binary format (1 for Yes, 0 for No).
    """
    df_clean = df.copy()
    
    # Drop customerID column
    if 'customerID' in df_clean.columns:
        df_clean = df_clean.drop(columns=['customerID'])
        
    # Convert TotalCharges to numeric, turning empty spaces or errors into NaN
    df_clean['TotalCharges'] = pd.to_numeric(df_clean['TotalCharges'], errors='coerce')
    
    # Impute missing values in TotalCharges using the median
    # (Median is preferred over mean to avoid outlier distortion)
    median_val = df_clean['TotalCharges'].median()
    df_clean['TotalCharges'] = df_clean['TotalCharges'].fillna(median_val)
    
    # Map target variable 'Churn' to binary values (Yes -> 1, No -> 0)
    if 'Churn' in df_clean.columns:
        df_clean['Churn'] = df_clean['Churn'].map({'Yes': 1, 'No': 0})
        
    return df_clean

def split_and_encode(df_clean):
    """
    Splits the cleaned dataframe into Train and Test sets BEFORE scaling/encoding
    to prevent Data Leakage. Then applies One-Hot Encoding and StandardScaler.
    """
    # Separate features (X) and target (y)
    X = df_clean.drop(columns=['Churn'])
    y = df_clean['Churn']
    
    # Split into 80% Train and 20% Test
    # stratify=y ensures identical churn/non-churn distribution in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Identify numerical and categorical columns
    num_cols = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
    cat_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    
    # Encode categorical variables using One-Hot Encoding
    X_train_encoded = pd.get_dummies(X_train, columns=cat_cols, drop_first=True)
    X_test_encoded = pd.get_dummies(X_test, columns=cat_cols, drop_first=True)
    
    # Reindex test columns to match train columns (fills newly introduced columns with 0)
    X_test_encoded = X_test_encoded.reindex(columns=X_train_encoded.columns, fill_value=0)
    
    # Scale numerical features using StandardScaler (mean=0, std=1)
    scaler = StandardScaler()
    
    # Crucial: fit parameters ONLY on Train set to prevent Data Leakage
    X_train_encoded[num_cols] = scaler.fit_transform(X_train_encoded[num_cols])
    X_test_encoded[num_cols] = scaler.transform(X_test_encoded[num_cols])
    
    return X_train_encoded, X_test_encoded, y_train, y_test

if __name__ == "__main__":
    # Self-run block to verify the pipeline functionality
    print("--- Starting Data Preprocessing ---")
    try:
        raw_df = load_data("data/raw_data.csv")
        print(f"1. Raw data successfully loaded. Total rows: {raw_df.shape[0]}")
        
        cleaned_df = clean_data(raw_df)
        print("2. Data cleaning and missing value imputation completed.")
        
        X_train, X_test, y_train, y_test = split_and_encode(cleaned_df)
        print("3. Train-Test split and feature encoding completed!")
        print(f"   Train set shape: {X_train.shape}")
        print(f"   Test set shape:  {X_test.shape}")
        print("--- Preprocessing pipeline completed successfully! ---")
    except Exception as e:
        print(f"Error occurred during preprocessing: {e}")