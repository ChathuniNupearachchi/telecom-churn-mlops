"""
FastAPI Application for Churn Prediction
=========================================
REST API for customer churn prediction using ML model.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import Optional
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.predict import predict_churn, predict_batch, get_model_info
import pandas as pd

# Initialize FastAPI app
app = FastAPI(
    title="Telecom Churn Prediction API",
    description="Predict customer churn probability using machine learning",
    version="1.0.0"
)

# Define request schema
class CustomerData(BaseModel):
    """Customer data schema for prediction"""
    gender: str = Field(..., description="Customer gender: Male or Female")
    SeniorCitizen: int = Field(..., description="Senior citizen: 0 or 1")
    Partner: str = Field(..., description="Has partner: Yes or No")
    Dependents: str = Field(..., description="Has dependents: Yes or No")
    tenure: int = Field(..., ge=0, description="Months with company")
    PhoneService: str = Field(..., description="Has phone service: Yes or No")
    MultipleLines: str = Field(..., description="Has multiple lines: Yes, No, or No phone service")
    InternetService: str = Field(..., description="Internet service type: DSL, Fiber optic, or No")
    OnlineSecurity: str = Field(..., description="Has online security: Yes, No, or No internet service")
    OnlineBackup: str = Field(..., description="Has online backup: Yes, No, or No internet service")
    DeviceProtection: str = Field(..., description="Has device protection: Yes, No, or No internet service")
    TechSupport: str = Field(..., description="Has tech support: Yes, No, or No internet service")
    StreamingTV: str = Field(..., description="Has streaming TV: Yes, No, or No internet service")
    StreamingMovies: str = Field(..., description="Has streaming movies: Yes, No, or No internet service")
    Contract: str = Field(..., description="Contract type: Month-to-month, One year, or Two year")
    PaperlessBilling: str = Field(..., description="Has paperless billing: Yes or No")
    PaymentMethod: str = Field(..., description="Payment method: Electronic check, Mailed check, Bank transfer (automatic), or Credit card (automatic)")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charges in dollars")
    
    class Config:
        schema_extra = {
            "example": {
                "gender": "Female",
                "SeniorCitizen": 0,
                "Partner": "No",
                "Dependents": "No",
                "tenure": 3,
                "PhoneService": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "No",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaperlessBilling": "Yes",
                "PaymentMethod": "Electronic check",
                "MonthlyCharges": 95.50
            }
        }

# Define response schema
class PredictionResponse(BaseModel):
    """Prediction response schema"""
    will_churn: bool
    churn_probability: float
    risk_level: str
    recommendation: str
    model_version: str

class ModelInfo(BaseModel):
    """Model information schema"""
    model_name: str
    training_date: str
    metrics: dict

# Root endpoint
@app.get("/")
def root():
    """Root endpoint with API information"""
    return {
        "message": "Telecom Churn Prediction API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict churn for a single customer",
            "POST /predict/batch": "Predict churn for multiple customers",
            "GET /model/info": "Get model information",
            "GET /health": "Health check",
            "GET /docs": "Interactive API documentation"
        }
    }

# Health check endpoint
@app.get("/health")
def health_check():
    """Check if API is running"""
    return {"status": "healthy", "message": "API is running"}

# Model info endpoint
@app.get("/model/info", response_model=ModelInfo)
def model_info():
    """Get information about the trained model"""
    try:
        info = get_model_info()
        return {
            "model_name": info['model_name'],
            "training_date": info['training_date'],
            "metrics": info['metrics']
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error retrieving model info: {str(e)}")

# Single prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(customer: CustomerData):
    """
    Predict churn for a single customer
    
    Returns prediction with probability and risk level
    """
    try:
        # Convert to dict
        customer_dict = customer.dict()
        
        # Make prediction
        result = predict_churn(customer_dict)
        
        return result
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# Batch prediction endpoint
@app.post("/predict/batch")
def predict_batch_endpoint(customers: list[CustomerData]):
    """
    Predict churn for multiple customers
    
    Returns predictions for all customers
    """
    try:
        # Convert to DataFrame
        customers_data = [customer.dict() for customer in customers]
        df = pd.DataFrame(customers_data)
        
        # Make predictions
        results = predict_batch(df)
        
        # Convert to list of dicts
        return {
            "count": len(results),
            "predictions": results.to_dict(orient='records')
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction error: {str(e)}")
