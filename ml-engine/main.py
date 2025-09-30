from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import torch
import torch.nn as nn
import numpy as np
import os
import joblib
import pandas as pd
from typing import List, Optional
import logging
from datetime import datetime

from models import LSTMForecastModel, GRUForecastModel
from data_fetcher import get_realtime_weather_forecast, process_weather_data

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
SAVED_MODELS_DIR = os.path.join(os.path.dirname(__file__), 'saved_models')
SEQUENCE_LENGTH = 24
INPUT_SIZE_SOLAR = 4
INPUT_SIZE_WIND = 3

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    location_lat: float = Field(..., ge=-90, le=90, description="Latitude")
    location_lng: float = Field(..., ge=-180, le=180, description="Longitude")
    hours: int = Field(24, ge=1, le=168, description="Hours to predict (1-168)")

class PredictionResponse(BaseModel):
    predicted_kwh: List[float]
    model: str
    timestamp: float
    location: dict
    confidence_score: Optional[float] = None

class HealthResponse(BaseModel):
    status: str
    models_loaded: dict
    timestamp: float

# --- FastAPI App Setup ---
app = FastAPI(
    title="SolarSync ML Prediction Service",
    description="Machine learning API for renewable energy generation predictions",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for loaded assets
solar_model: Optional[nn.Module] = None
wind_model: Optional[nn.Module] = None
solar_scaler: Optional[object] = None
wind_scaler: Optional[object] = None

# --- Helper Functions ---

def load_assets(model_name: str, input_size: int, model_type: str = 'lstm'):
    """Loads a saved PyTorch model and its corresponding scaler"""
    model_path = os.path.join(SAVED_MODELS_DIR, f'{model_name}_model.pth')
    scaler_path = os.path.join(SAVED_MODELS_DIR, f'{model_name}_scaler.pkl')
    
    if not os.path.exists(model_path):
        logger.warning(f"Model not found: {model_path}")
        return None, None
    
    if not os.path.exists(scaler_path):
        logger.warning(f"Scaler not found: {scaler_path}")
        return None, None

    try:
        # Load appropriate model architecture
        if model_type == 'lstm':
            model = LSTMForecastModel(
                input_size=input_size, 
                hidden_size=128, 
                num_layers=3, 
                output_size=1
            )
        else:  # GRU
            model = GRUForecastModel(
                input_size=input_size, 
                hidden_size=128, 
                num_layers=2, 
                output_size=1
            )
        
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
        
        # Load Scaler
        scaler = joblib.load(scaler_path)
        
        logger.info(f"Successfully loaded {model_name} model and scaler")
        return model, scaler
        
    except Exception as e:
        logger.error(f"Error loading {model_name} assets: {e}")
        return None, None


def run_autoregressive_prediction(
    model: nn.Module, 
    scaler: object, 
    future_features: List[np.ndarray],
    sequence_length: int = 24
) -> List[float]:
    """
    Performs multi-step forecasting using auto-regression
    
    Args:
        model: The loaded PyTorch model
        scaler: The loaded MinMaxScaler
        future_features: List of normalized feature vectors for forecast period
        sequence_length: Length of input sequence
        
    Returns:
        List of denormalized KWh predictions
    """
    predictions = []
    
    # Initialize with random data (in production, use actual historical data from DB)
    # This should be replaced with Redis/PostgreSQL lookup
    initial_sequence = np.random.rand(
        sequence_length, 
        future_features[0].shape[-1]
    ).astype(np.float32) * 0.5  # Scaled initial values
    
    current_sequence = torch.from_numpy(initial_sequence).float().unsqueeze(0)

    with torch.no_grad():
        for i, future_step_features in enumerate(future_features):
            # Predict the next step
            prediction_scaled_tensor = model(current_sequence)
            prediction_scaled = prediction_scaled_tensor.item()
            
            # Denormalize the prediction
            target_index = scaler.n_features_in_ - 1
            
            # Create inverse transform input
            inverse_input = np.zeros((1, scaler.n_features_in_))
            inverse_input[0, target_index] = prediction_scaled
            
            # Get actual KWh value
            predicted_kwh = scaler.inverse_transform(inverse_input)[0, target_index]
            
            # Ensure non-negative predictions
            predicted_kwh = max(0, predicted_kwh)
            predictions.append(float(predicted_kwh))
            
            # Auto-regression: prepare next input
            next_sequence = current_sequence.squeeze(0)[1:]
            
            # Create new input vector with predicted value
            if i < len(future_features) - 1:
                new_input_vector = np.concatenate([
                    future_step_features[0][:-1], 
                    [prediction_scaled]
                ]).astype(np.float32)
                
                next_sequence = torch.cat([
                    next_sequence, 
                    torch.from_numpy(new_input_vector).float().unsqueeze(0)
                ], dim=0)
                
                current_sequence = next_sequence.unsqueeze(0)

    return predictions


# --- API Endpoints ---

@app.on_event("startup")
async def load_startup_models():
    """Load models and scalers when the API starts"""
    global solar_model, wind_model, solar_scaler, wind_scaler
    
    logger.info("Loading ML models...")
    
    try:
        solar_model, solar_scaler = load_assets('solar', INPUT_SIZE_SOLAR, 'lstm')
        wind_model, wind_scaler = load_assets('wind', INPUT_SIZE_WIND, 'gru')
        
        if solar_model and solar_scaler:
            logger.info("✅ Solar model loaded successfully")
        else:
            logger.warning("⚠️ Solar model not available")
            
        if wind_model and wind_scaler:
            logger.info("✅ Wind model loaded successfully")
        else:
            logger.warning("⚠️ Wind model not available")
            
    except Exception as e:
        logger.error(f"FATAL: Could not load ML assets. Error: {e}")


@app.get("/", response_model=dict)
async def root():
    """Root endpoint"""
    return {
        "service": "SolarSync ML Prediction Service",
        "version": "1.0.0",
        "status": "operational",
        "endpoints": {
            "solar_prediction": "/api/v1/predict/solar",
            "wind_prediction": "/api/v1/predict/wind",
            "health": "/health"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "solar": solar_model is not None,
            "wind": wind_model is not None
        },
        timestamp=datetime.now().timestamp()
    )


@app.get("/api/v1/predict/solar", response_model=PredictionResponse)
async def predict_solar(
    location_lat: float = Query(..., ge=-90, le=90),
    location_lng: float = Query(..., ge=-180, le=180),
    hours: int = Query(24, ge=1, le=168)
):
    """
    Predicts solar energy generation for the next N hours
    
    Chainlink Oracle Target: Use 'predicted_kwh[0]' for first hour prediction
    """
    if not solar_model or not solar_scaler:
        raise HTTPException(
            status_code=503, 
            detail="Solar model not loaded. Please train the model first."
        )
    
    try:
        # Fetch weather forecast
        raw_weather = get_realtime_weather_forecast(location_lat, location_lng)
        
        if not raw_weather:
            raise HTTPException(
                status_code=503, 
                detail="Failed to fetch weather data from external API"
            )
        
        # Process weather data into model features
        future_features = process_weather_data(raw_weather, 'solar', hours)
        
        if not future_features:
            raise HTTPException(
                status_code=500, 
                detail="Failed to process weather data into model features"
            )
        
        # Run prediction
        predictions = run_autoregressive_prediction(
            solar_model, 
            solar_scaler, 
            future_features,
            SEQUENCE_LENGTH
        )
        
        # Calculate confidence score (simplified)
        confidence = min(0.95, max(0.6, 1.0 - (len(predictions) / 100)))
        
        return PredictionResponse(
            predicted_kwh=predictions,
            model="LSTM",
            timestamp=pd.Timestamp.now().timestamp(),
            location={"lat": location_lat, "lng": location_lng},
            confidence_score=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Solar prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.get("/api/v1/predict/wind", response_model=PredictionResponse)
async def predict_wind(
    location_lat: float = Query(..., ge=-90, le=90),
    location_lng: float = Query(..., ge=-180, le=180),
    hours: int = Query(24, ge=1, le=168)
):
    """
    Predicts wind energy generation for the next N hours
    """
    if not wind_model or not wind_scaler:
        raise HTTPException(
            status_code=503, 
            detail="Wind model not loaded. Please train the model first."
        )
    
    try:
        raw_weather = get_realtime_weather_forecast(location_lat, location_lng)
        
        if not raw_weather:
            raise HTTPException(
                status_code=503, 
                detail="Failed to fetch weather data"
            )
        
        future_features = process_weather_data(raw_weather, 'wind', hours)
        
        if not future_features:
            raise HTTPException(
                status_code=500, 
                detail="Failed to process weather data"
            )
        
        predictions = run_autoregressive_prediction(
            wind_model, 
            wind_scaler, 
            future_features,
            18  # Wind model uses 18-hour sequence
        )
        
        confidence = min(0.92, max(0.65, 1.0 - (len(predictions) / 120)))
        
        return PredictionResponse(
            predicted_kwh=predictions,
            model="GRU+Attention",
            timestamp=pd.Timestamp.now().timestamp(),
            location={"lat": location_lat, "lng": location_lng},
            confidence_score=confidence
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Wind prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)