"""
Churn Prediction Module
=======================
Production-ready prediction functions for customer churn.
"""

import joblib
import pickle
import pandas as pd
import numpy as np
import os

# Get the directory where this module is located
MODULE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODULE_DIR))
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')


def load_artifacts():
    """Load all model artifacts"""
    model = joblib.load(os.path.join(MODELS_DIR, 'best_model_rf_smote.pkl'))
    scaler = joblib.load(os.path.join(MODELS_DIR, 'scaler.pkl'))
    
    with open(os.path.join(MODELS_DIR, 'feature_names.pkl'), 'rb') as f:
        feature_names = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, 'encoding_info.pkl'), 'rb') as f:
        encoding_info = pickle.load(f)
    
    with open(os.path.join(MODELS_DIR, 'model_metadata.pkl'), 'rb') as f:
        metadata = pickle.load(f)
    
    return model, scaler, feature_names, encoding_info, metadata

# Load artifacts once at module import
MODEL, SCALER, FEATURE_NAMES, ENCODING_INFO, METADATA = load_artifacts()

def preprocess_customer_data(customer_data):
    """Preprocess raw customer data for prediction"""
    
    # Convert to DataFrame if dict
    if isinstance(customer_data, dict):
        df = pd.DataFrame([customer_data])
    else:
        df = customer_data.copy()
    
    # Binary encoding
    for col in ENCODING_INFO['binary_columns']:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # Gender encoding
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(ENCODING_INFO['gender_encoding'])
    
    # One-hot encoding
    df_encoded = pd.get_dummies(df, columns=ENCODING_INFO['categorical_columns'], drop_first=True)
    
    # Ensure all features exist
    for feature in FEATURE_NAMES:
        if feature not in df_encoded.columns:
            df_encoded[feature] = 0
    
    # Reorder columns
    df_encoded = df_encoded[FEATURE_NAMES]
    
    # Scale numerical features
    cols_to_scale = ENCODING_INFO['numerical_columns']
    df_encoded[cols_to_scale] = SCALER.transform(df_encoded[cols_to_scale])
    
    return df_encoded

def predict_churn(customer_data):
    """
    Predict customer churn
    
    Parameters:
    -----------
    customer_data : dict or pd.DataFrame
        Customer features in original format
    
    Returns:
    --------
    dict : Prediction results
    """
    
    # Preprocess
    X = preprocess_customer_data(customer_data)
    
    # Predict
    prediction = MODEL.predict(X)[0]
    probability = MODEL.predict_proba(X)[0, 1]
    
    # Determine risk level
    if probability > 0.7:
        risk_level = 'High'
        recommendation = 'Immediate retention action needed'
    elif probability > 0.4:
        risk_level = 'Medium'
        recommendation = 'Monitor closely and engage proactively'
    else:
        risk_level = 'Low'
        recommendation = 'Standard customer service'
    
    return {
        'will_churn': bool(prediction),
        'churn_probability': float(probability),
        'risk_level': risk_level,
        'recommendation': recommendation,
        'model_version': METADATA['training_date']
    }

def predict_batch(customers_df):
    """
    Predict churn for multiple customers
    
    Parameters:
    -----------
    customers_df : pd.DataFrame
        DataFrame with multiple customers
    
    Returns:
    --------
    pd.DataFrame : Original data with predictions
    """
    
    # Preprocess
    X = preprocess_customer_data(customers_df)
    
    # Predict
    predictions = MODEL.predict(X)
    probabilities = MODEL.predict_proba(X)[:, 1]
    
    # Add to original dataframe
    result = customers_df.copy()
    result['churn_prediction'] = predictions
    result['churn_probability'] = probabilities
    result['risk_level'] = ['High' if p > 0.7 else 'Medium' if p > 0.4 else 'Low' 
                           for p in probabilities]
    
    return result

def get_model_info():
    """Get model metadata"""
    return METADATA