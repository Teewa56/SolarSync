from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
import os
import joblib # Required for loading the MinMaxScaler
import pandas as pd
from typing import List

from models import LSTMForecastModel 
from data_fetcher import get_realtime_weather_forecast, process_weather_data
# Assuming GRUForecastModel is also defined in models.py

# --- Configuration ---
SAVED_MODELS_DIR = 'ml-engine/saved_models'
SEQUENCE_LENGTH = 24
INPUT_SIZE_SOLAR = 4
INPUT_SIZE_WIND = 3 

# --- Pydantic Models ---
class PredictionRequest(BaseModel):
    location_lat: float
    location_lng: float
    hours: int = 24

class PredictionResponse(BaseModel):
    predicted_kwh: List[float] # Changed to list of floats for the time series prediction
    model: str
    timestamp: float

# --- FastAPI App Setup ---
app = FastAPI(title="SolarSync ML Prediction Service")

# Global variables for loaded assets
solar_model: nn.Module = None
wind_model: nn.Module = None
solar_scaler: joblib.MinMaxScaler = None
wind_scaler: joblib.MinMaxScaler = None

# --- Helper Functions ---

def load_assets(model_name: str, input_size: int):
    """Loads a saved PyTorch model and its corresponding scaler."""
    model_path = os.path.join(SAVED_MODELS_DIR, f'{model_name}_model.pth')
    scaler_path = os.path.join(SAVED_MODELS_DIR, f'{model_name}_scaler.pkl')
    
    if not os.path.exists(model_path) or not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Missing model/scaler for {model_name}. Run train_models.py.")

    # Load Model (using LSTM for both as a template, replace with GRU/Transformer if needed)
    model = LSTMForecastModel(input_size=input_size, hidden_size=128, num_layers=3, output_size=1)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()

    # Load Scaler (Critical for denormalization)
    scaler = joblib.load(scaler_path)
    
    return model, scaler


def run_autoregressive_prediction(model: nn.Module, scaler: joblib.MinMaxScaler, future_features: List[np.ndarray]) -> List[float]:
    """
    Performs multi-step forecasting using auto-regression.
    
    Args:
        model: The loaded PyTorch model.
        scaler: The loaded MinMaxScaler.
        future_features: List of normalized feature vectors for the forecast period.
        
    Returns:
        List of denormalized KWh predictions.
    """
    predictions = []
    
    # 1. Get initial sequence: Last (SEQUENCE_LENGTH) historical steps (normalized)
    # **Placeholder: In a real system, fetch from Redis/PostgreSQL**
    initial_sequence = np.random.rand(SEQUENCE_LENGTH, future_features[0].shape[-1]).astype(np.float32)
    current_sequence = torch.from_numpy(initial_sequence).float().unsqueeze(0) # (1, seq_len, features)

    with torch.no_grad():
        for i, future_step_features in enumerate(future_features):
            # 2. Predict the next step
            prediction_scaled_tensor = model(current_sequence)
            prediction_scaled = prediction_scaled_tensor.item()
            
            # 3. Denormalize the prediction (CRITICAL)
            # Find the index of the target feature ('energy_output') in the scaler's features list
            target_index = scaler.n_features_in_ - 1
            
            # Create an array of zeros and insert the scaled prediction at the target index
            inverse_input = np.zeros((1, scaler.n_features_in_))
            inverse_input[0, target_index] = prediction_scaled
            
            # Inverse transform to get the KWh value
            predicted_kwh = scaler.inverse_transform(inverse_input)[0, target_index]
            predictions.append(predicted_kwh)
            
            # 4. Auto-regression: Create the next input sequence
            
            # a) Shift the sequence: Remove the oldest step
            next_sequence = current_sequence.squeeze(0)[1:] 
            
            # b) Create the input vector for the predicted time step
            # Replace the old 'energy_output' feature with the new prediction
            
            # The new input vector uses the *actual* weather forecast for time 'i+1' 
            # and the *predicted* energy output from the previous step.
            new_input_vector = np.concatenate((future_step_features[0][:-1], [prediction_scaled])).astype(np.float32)
            
            # c) Append the new step
            next_sequence = torch.cat((next_sequence, torch.from_numpy(new_input_vector).float().unsqueeze(0)), dim=0)
            current_sequence = next_sequence.unsqueeze(0)


    return predictions


# --- API Endpoints ---

@app.on_event("startup")
def load_startup_models():
    """Load models and scalers when the API starts."""
    global solar_model, wind_model, solar_scaler, wind_scaler
    try:
        solar_model, solar_scaler = load_assets('solar', INPUT_SIZE_SOLAR)
        wind_model, wind_scaler = load_assets('wind', INPUT_SIZE_WIND) 
        print("ML Models and Scalers loaded successfully.")
    except Exception as e:
        print(f"FATAL: Could not load ML assets. Error: {e}")


@app.get("/api/v1/predict/solar", response_model=PredictionResponse)
async def predict_solar(location_lat: float, location_lng: float, hours: int = 24):
    """
    Predicts solar energy generation.
    
    Chainlink Oracle Target: The oracle will target the 'predicted_kwh[0]' path for the first hour's prediction.
    """
    if not solar_model:
        raise HTTPException(status_code=500, detail="Solar model not loaded.")
        
    raw_weather = get_realtime_weather_forecast(location_lat, location_lng)
    if not raw_weather:
         raise HTTPException(status_code=503, detail="Failed to fetch weather data for prediction.")
         
    # Get the normalized features for the forecast period (N hours)
    future_features = process_weather_data(raw_weather, 'solar', hours)
    
    if not future_features:
        raise HTTPException(status_code=500, detail="Failed to process weather data into features.")

    # Run the autoregressive prediction loop
    predictions = run_autoregressive_prediction(solar_model, solar_scaler, future_features)
    
    # Return the full list of N predictions
    return PredictionResponse(
        predicted_kwh=predictions,
        model="LSTM",
        timestamp=pd.Timestamp.now().timestamp()
    )

# ... (wind prediction endpoint is similar)

@app.get("/api/v1/predict/wind", response_model=PredictionResponse)
async def predict_wind(location_lat: float, location_lng: float, hours: int = 24):
    """Predicts wind energy generation for the next N hours."""
    # (Implementation is similar to predict_solar, using wind_model and wind_scaler)
    if not wind_model:
        raise HTTPException(status_code=500, detail="Wind model not loaded.")
        
    raw_weather = get_realtime_weather_forecast(location_lat, location_lng)
    
    future_features = process_weather_data(raw_weather, 'wind', hours)
    predictions = run_autoregressive_prediction(wind_model, wind_scaler, future_features)
    
    return PredictionResponse(
        predicted_kwh=predictions,
        model="GRU+Attention",
        timestamp=pd.Timestamp.now().timestamp()
    )