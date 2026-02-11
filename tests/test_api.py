"""
API Testing Script
==================
Test the churn prediction API endpoints.
"""

import requests
import json

# API base URL
BASE_URL = "http://localhost:8000"

def test_root():
    """Test root endpoint"""
    print("\nTesting Root Endpoint...")
    response = requests.get(f"{BASE_URL}/")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_health():
    """Test health check"""
    print("\nTesting Health Check...")
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_model_info():
    """Test model info endpoint"""
    print("\nTesting Model Info...")
    response = requests.get(f"{BASE_URL}/model/info")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_prediction():
    """Test single prediction"""
    print("\nTesting Single Prediction...")
    
    # High-risk customer
    customer = {
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
    
    response = requests.post(f"{BASE_URL}/predict", json=customer)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")

def test_batch_prediction():
    """Test batch prediction"""
    print("\nTesting Batch Prediction...")
    
    customers = [
        {
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
        },
        {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "Yes",
            "tenure": 48,
            "PhoneService": "Yes",
            "MultipleLines": "Yes",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "Two year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Bank transfer (automatic)",
            "MonthlyCharges": 65.00
        }
    ]
    
    response = requests.post(f"{BASE_URL}/predict/batch", json=customers)
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)[:500]}...")  # Truncate for readability

if __name__ == "__main__":
    print("="*60)
    print("TESTING CHURN PREDICTION API")
    print("="*60)
    
    try:
        test_root()
        test_health()
        test_model_info()
        test_prediction()
        test_batch_prediction()
        
        print("\n" + "="*60)
        print("ALL TESTS PASSED!")
        print("="*60)
        
    except requests.exceptions.ConnectionError:
        print("\nERROR: Could not connect to API")
        print("Make sure the API is running with:")
        print("uvicorn src.api.app:app --reload")
    except Exception as e:
        print(f"\nERROR: {str(e)}")