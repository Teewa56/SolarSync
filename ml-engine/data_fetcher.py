import requests
import numpy as np
from typing import List, Dict, Optional
import os
from dotenv import load_dotenv
import logging
from datetime import datetime

load_dotenv()
logger = logging.getLogger(__name__)

# Feature names (must match training)
SOLAR_FEATURES = ['temp', 'humidity', 'cloud_cover', 'solar_irradiance']
WIND_FEATURES = ['wind_speed', 'wind_direction', 'pressure']

def get_realtime_weather_forecast(lat: float, lon: float) -> Optional[Dict]:
    """
    Fetches 48-hour hourly weather forecast from OpenWeatherMap
    
    Args:
        lat: Latitude of the location
        lon: Longitude of the location
        
    Returns:
        Dictionary containing hourly forecast data or None if failed
    """
    OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")
    
    if not OPENWEATHER_API_KEY:
        logger.error("OPENWEATHER_API_KEY not set in environment")
        return None

    # OpenWeatherMap One Call API 3.0
    url = (
        f"https://api.openweathermap.org/data/3.0/onecall?"
        f"lat={lat}&lon={lon}"
        f"&exclude=current,minutely,daily,alerts"
        f"&appid={OPENWEATHER_API_KEY}"
        f"&units=metric"
    )
    
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        
        logger.info(f"Successfully fetched weather data for ({lat}, {lon})")
        return data
        
    except requests.exceptions.Timeout:
        logger.error("Weather API request timed out")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Weather API Error: {e}")
        return None
    except ValueError as e:
        logger.error(f"Failed to parse weather API response: {e}")
        return None


def process_weather_data(
    raw_data: Dict, 
    model_type: str, 
    hours: int
) -> List[np.ndarray]:
    """
    Extracts and processes weather features from raw forecast data
    
    Args:
        raw_data: Raw weather API response
        model_type: 'solar' or 'wind'
        hours: Number of hours to process
        
    Returns:
        List of numpy arrays containing processed features
    """
    if not raw_data or 'hourly' not in raw_data:
        logger.error("Invalid weather data format")
        return []

    try:
        forecast_list = raw_data['hourly'][:hours]
        processed_features = []
        
        if model_type == 'solar':
            processed_features = _process_solar_features(forecast_list)
        elif model_type == 'wind':
            processed_features = _process_wind_features(forecast_list)
        else:
            logger.error(f"Unknown model type: {model_type}")
            return []
        
        logger.info(f"Processed {len(processed_features)} hours of {model_type} features")
        return processed_features
        
    except Exception as e:
        logger.error(f"Error processing weather data: {e}")
        return []


def _process_solar_features(forecast_list: List[Dict]) -> List[np.ndarray]:
    """
    Extract solar-relevant features from weather forecast
    
    Features:
    - Temperature (°C)
    - Humidity (%)
    - Cloud cover (%)
    - Solar irradiance (W/m²) - estimated from cloud cover
    """
    processed = []
    
    for hour_data in forecast_list:
        try:
            temp = hour_data.get('temp', 20.0)
            humidity = hour_data.get('humidity', 50.0)
            clouds = hour_data.get('clouds', 0.0)
            
            # Estimate solar irradiance from cloud cover
            # Clear sky: ~1000 W/m², fully cloudy: ~200 W/m²
            base_irradiance = 1000.0
            cloud_factor = 1.0 - (clouds / 100.0) * 0.8
            irradiance = base_irradiance * cloud_factor
            
            # Additional factors
            if 'dt' in hour_data:
                hour_of_day = datetime.fromtimestamp(hour_data['dt']).hour
                # Reduce irradiance at night (simple day/night cycle)
                if hour_of_day < 6 or hour_of_day > 20:
                    irradiance *= 0.1
                elif hour_of_day < 9 or hour_of_day > 17:
                    irradiance *= 0.6
            
            # UV index can also help refine irradiance if available
            if 'uvi' in hour_data:
                uvi = hour_data.get('uvi', 0)
                irradiance = max(irradiance, uvi * 100)  # Rough correlation
            
            features_vector = [
                float(temp),
                float(humidity),
                float(clouds),
                float(irradiance)
            ]
            
            processed.append(np.array([features_vector]))
            
        except Exception as e:
            logger.warning(f"Error processing solar feature: {e}")
            # Use default values if processing fails
            processed.append(np.array([[20.0, 50.0, 50.0, 500.0]]))
    
    return processed


def _process_wind_features(forecast_list: List[Dict]) -> List[np.ndarray]:
    """
    Extract wind-relevant features from weather forecast
    
    Features:
    - Wind speed (m/s)
    - Wind direction (degrees)
    - Pressure (hPa)
    """
    processed = []
    
    for hour_data in forecast_list:
        try:
            wind_speed = hour_data.get('wind_speed', 0.0)
            wind_deg = hour_data.get('wind_deg', 0.0)
            pressure = hour_data.get('pressure', 1013.0)
            
            # Wind gust can indicate higher energy potential
            if 'wind_gust' in hour_data:
                wind_gust = hour_data.get('wind_gust', wind_speed)
                wind_speed = max(wind_speed, wind_gust * 0.8)
            
            features_vector = [
                float(wind_speed),
                float(wind_deg),
                float(pressure)
            ]
            
            processed.append(np.array([features_vector]))
            
        except Exception as e:
            logger.warning(f"Error processing wind feature: {e}")
            processed.append(np.array([[5.0, 180.0, 1013.0]]))
    
    return processed


def normalize_features(
    features: List[np.ndarray], 
    scaler: object
) -> List[np.ndarray]:
    """
    Normalize features using a fitted scaler
    
    Args:
        features: List of feature arrays
        scaler: Fitted MinMaxScaler or StandardScaler
        
    Returns:
        List of normalized feature arrays
    """
    try:
        # Stack all features
        stacked = np.vstack([f[0] for f in features])
        
        # Normalize
        normalized = scaler.transform(stacked)
        
        # Reshape back to list format
        return [np.array([normalized[i]]) for i in range(len(normalized))]
        
    except Exception as e:
        logger.error(f"Error normalizing features: {e}")
        return features


def get_historical_context(
    location_id: str, 
    sequence_length: int = 24
) -> Optional[np.ndarray]:
    """
    Fetch historical data from database/cache for context
    
    In production, this would query Redis or PostgreSQL for the last
    N hours of actual weather and generation data
    
    Args:
        location_id: Identifier for the location
        sequence_length: Number of historical hours needed
        
    Returns:
        Array of historical features or None
    """
    # TODO: Implement Redis/PostgreSQL lookup
    # For now, return None to indicate no historical data available
    
    logger.warning("Historical context not implemented - using mock data")
    return None


def validate_weather_data(data: Dict) -> bool:
    """
    Validate weather API response structure
    
    Args:
        data: Weather API response
        
    Returns:
        True if valid, False otherwise
    """
    required_fields = ['hourly']
    
    if not all(field in data for field in required_fields):
        return False
    
    if not isinstance(data['hourly'], list) or len(data['hourly']) == 0:
        return False
    
    # Check first hourly entry has required fields
    first_hour = data['hourly'][0]
    required_hour_fields = ['temp', 'humidity', 'clouds']
    
    return all(field in first_hour for field in required_hour_fields)