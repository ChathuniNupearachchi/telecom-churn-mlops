import sys
import pandas as pd
import sklearn
import xgboost
import mlflow
import fastapi

print("Python version:", sys.version)
print("Pandas:", pd.__version__)
print("Scikit-learn:", sklearn.__version__)
print("XGBoost:", xgboost.__version__)
print("MLflow:", mlflow.__version__)
print("FastAPI:", fastapi.__version__)

# Check if dataset exists
import os
if os.path.exists('data/raw/telecom_churn.csv'):
    df = pd.read_csv('data/raw/telecom_churn.csv')
    print(f"\n Dataset loaded successfully!")
    print(f" Shape: {df.shape}")
    print(f" Columns: {len(df.columns)}")
else:
    print(" Dataset not found! Please download it.")