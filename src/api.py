"""
FastAPI application for bank customer churn prediction.
"""

import os
import logging
import pandas as pd
import mlflow
import mlflow.sklearn
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import Optional
import uvicorn
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Bank Customer Churn Prediction API",
    description="API for predicting bank customer churn",
    version="1.0.0"
)

# Load the model and transformer
try:
    logger.info("Loading model and transformer...")
    model = mlflow.sklearn.load_model("C:/Users/algon/mlruns/0/latest/artifacts/model")
    transformer = mlflow.sklearn.load_model("C:/Users/algon/mlruns/0/latest/artifacts/model")
    logger.info("Model and transformer loaded successfully")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define input data model
class CustomerData(BaseModel):
    CreditScore: int = Field(..., example=650)
    Geography: str = Field(..., example="France")
    Gender: str = Field(..., example="Male")
    Age: int = Field(..., example=35)
    Tenure: int = Field(..., example=5)
    Balance: float = Field(..., example=75000.0)
    NumOfProducts: int = Field(..., example=2)
    HasCrCard: int = Field(..., example=1)
    IsActiveMember: int = Field(..., example=1)
    EstimatedSalary: float = Field(..., example=50000.0)

# Define response model
class PredictionResponse(BaseModel):
    prediction: int
    probability: Optional[float] = None
    prediction_label: str

# Middleware for request timing
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    logger.info(f"Request to {request.url.path} took {process_time:.4f} seconds")
    return response

@app.get("/", tags=["General"])
async def home():
    """Home endpoint that returns a welcome message."""
    logger.info("Home endpoint accessed")
    return {"message": "Welcome to the Bank Customer Churn Prediction API"}

@app.get("/health", tags=["General"])
async def health():
    """Health check endpoint."""
    logger.info("Health check endpoint accessed")
    return {"status": "healthy", "model_loaded": model is not None, "transformer_loaded": transformer is not None}

@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
async def predict(data: CustomerData):
    """
    Predict customer churn based on input features.
    
    Returns:
        prediction: 0 (not churned) or 1 (churned)
        prediction_label: "Not Churned" or "Churned"
    """
    try:
        logger.info("Received prediction request")
        
        # Convert input data to DataFrame
        input_df = pd.DataFrame([data.dict()])
        logger.debug(f"Input data: {input_df}")
        
        # Transform the input data
        transformed_data = transformer.transform(input_df)
        transformed_df = pd.DataFrame(transformed_data, columns=transformer.get_feature_names_out())
        logger.debug(f"Transformed data shape: {transformed_df.shape}")
        
        # Make prediction
        prediction = int(model.predict(transformed_df)[0])
        logger.info(f"Prediction result: {prediction}")
        
        # Get prediction label
        prediction_label = "Churned" if prediction == 1 else "Not Churned"
        
        # Try to get probability if the model supports it
        probability = None
        try:
            if hasattr(model, "predict_proba"):
                probability = float(model.predict_proba(transformed_df)[0][1])
                logger.debug(f"Prediction probability: {probability}")
        except Exception as e:
            logger.warning(f"Could not get probability: {str(e)}")
        
        return {
            "prediction": prediction,
            "probability": probability,
            "prediction_label": prediction_label
        }
    except Exception as e:
        logger.error(f"Error making prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred", "detail": str(exc)}
    )

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
