"""
Tests for the FastAPI application.
"""

import sys
import os
import pytest
from fastapi.testclient import TestClient
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from unittest.mock import patch, MagicMock

# Add the src directory to the path so we can import the app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

# Import the app
from app import app, CustomerData

client = TestClient(app)

# Mock data for tests
mock_customer_data = {
    "CreditScore": 650,
    "Geography": "France",
    "Gender": "Male",
    "Age": 35,
    "Tenure": 5,
    "Balance": 75000.0,
    "NumOfProducts": 2,
    "HasCrCard": 1,
    "IsActiveMember": 1,
    "EstimatedSalary": 50000.0
}

def test_home_endpoint():
    """Test the home endpoint."""
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()
    assert "Welcome" in response.json()["message"]

def test_health_endpoint():
    """Test the health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert "status" in response.json()
    assert response.json()["status"] == "healthy"

@patch('app.model')
@patch('app.transformer')
def test_predict_endpoint(mock_model, mock_transformer):
    """Test the predict endpoint with mocked model and transformer."""
    # Setup mocks
    mock_transformer.transform.return_value = np.array([[1, 0, 0, 1, 0, 650, 35, 5, 75000, 2, 1, 1, 50000]])
    mock_transformer.get_feature_names_out.return_value = [
        'standardscaler__CreditScore', 'onehotencoder__Geography_France', 
        'onehotencoder__Geography_Germany', 'onehotencoder__Gender_Male', 
        'standardscaler__Age', 'standardscaler__Tenure', 
        'standardscaler__Balance', 'standardscaler__NumOfProducts', 
        'standardscaler__HasCrCard', 'standardscaler__IsActiveMember', 
        'standardscaler__EstimatedSalary'
    ]
    
    # Mock prediction
    mock_model.predict.return_value = np.array([0])
    mock_model.predict_proba.return_value = np.array([[0.8, 0.2]])
    
    # Make request
    response = client.post("/predict", json=mock_customer_data)
    
    # Assertions
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert "prediction_label" in response.json()
    assert response.json()["prediction"] == 0
    assert response.json()["prediction_label"] == "Not Churned"
    assert "probability" in response.json()
    
    # Verify mocks were called
    mock_transformer.transform.assert_called_once()
    mock_model.predict.assert_called_once()

def test_predict_endpoint_invalid_data():
    """Test the predict endpoint with invalid data."""
    # Missing required fields
    invalid_data = {"CreditScore": 650}
    response = client.post("/predict", json=invalid_data)
    assert response.status_code == 422  # Validation error

def test_customer_data_model():
    """Test the CustomerData pydantic model."""
    # Valid data
    customer = CustomerData(**mock_customer_data)
    assert customer.CreditScore == 650
    assert customer.Geography == "France"
    
    # Invalid data should raise ValidationError
    with pytest.raises(Exception):
        CustomerData(CreditScore="invalid")