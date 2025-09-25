import requests
import pandas as pd
import numpy as np
from typing import List, Dict
import os

# Placeholder for feature names (must match the order used in model training)
SOLAR_FEATURES_API = ['temp', 'humidity', 'cloud_cover', 'solar_irradiance'] 
WIND_FEATURES_API = ['wind_speed', 'wind_direction', 'pressure'] # Note: 'energy_output' is the target, not an input feature here

def get_realtime_weather_forecast(lat: float, lon: float) -> Dict:
    """
    Fetches the 48-hour hourly weather forecast from OpenWeatherMap.
    
    Args:
        lat: Latitude of the producer.
        lon: Longitude of the producer.
        
    Returns:
        Dictionary containing hourly forecast data.
    """
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    if not OPENWEATHER_API_KEY:
        print("Error: OPENWEATHER_API_KEY not set.")
        return {}

    # Using the One Call API 3.0 for hourly forecast
    # Exclude unnecessary blocks for efficiency
    url = (
        f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}"
        f"&exclude=current,minutely,daily,alerts"
        f"&appid={OPENWEATHER_API_KEY}&units=metric"
    )
    
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except requests.RequestException as e:
        print(f"Weather API Error: {e}")
        return {}

def process_weather_data(raw_data: Dict, model_type: str, hours: int) -> List[np.ndarray]:
    """
    Extracts, normalizes, and sequences the necessary features from the raw forecast.
    
    NOTE: This is the most complex step in the ML pipeline. It requires:
    1. Historical context (last 24 hours) from the DB.
    2. Merging historical and forecast data.
    3. Applying the saved MinMaxScaler from training.
    """
    if not raw_data or 'hourly' not in raw_data:
        return []

    # Simplified extraction of the next N hours of forecast data
    forecast_list = raw_data['hourly'][:hours]
    
    # Placeholder for feature extraction (must align with model's expectations)
    processed_features = []
    
    if model_type == 'solar':
        # Example: extract features relevant to solar prediction
        for hour_data in forecast_list:
            # Note: OpenWeatherMap does not provide direct solar irradiance, 
            # so we estimate or use a proxy like cloudiness/UV.
            irradiance_proxy = 1000 * (1 - hour_data.get('clouds', 0) / 100) # Simple proxy
            
            features_vector = [
                hour_data.get('temp', 0),
                hour_data.get('humidity', 0),
                hour_data.get('clouds', 0),
                irradiance_proxy
            ]
            processed_features.append(features_vector)
            
    # Convert to NumPy array
    processed_array = np.array(processed_features)
    
    # 
    # !!! Critical Step: Normalization and Sequence Creation !!!
    # 
    # 1. Load the saved SCALER object (e.g., joblib.load('solar_scaler.pkl'))
    # 2. Load the last SEQUENCE_LENGTH-1 of actual historical data from PostgreSQL/Redis.
    # 3. Concatenate (Historical_Data, Forecast_Data)
    # 4. Normalize the full dataset using the loaded scaler.
    # 5. Create the sequence array for the prediction loop.
    
    # **Simplified Placeholder:** Just returning the raw (unnormalized) forecast data as the list of future features
    return [np.array([features]) for features in processed_array]