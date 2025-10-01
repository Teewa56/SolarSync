# openmeteo_historical.py
import requests
import pandas as pd
from datetime import datetime, timedelta

def get_openmeteo_historical(lat, lon, start_date, end_date):
    """
    Get historical weather data from Open-Meteo API
    
    Args:
        lat (float): Latitude
        lon (float): Longitude
        start_date (str): Start date in YYYY-MM-DD format  
        end_date (str): End date in YYYY-MM-DD format
    
    Returns:
        pandas.DataFrame: Historical weather data
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'temperature_2m,relative_humidity_2m,pressure_msl,wind_speed_10m,wind_direction_10m,shortwave_radiation',
        'timezone': 'auto'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return parse_openmeteo_data(data)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching Open-Meteo data: {e}")
        return None

def parse_openmeteo_data(meteo_json):
    """Parse Open-Meteo JSON response into DataFrame"""
    hourly_data = meteo_json['hourly']
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly_data['time']),
        'temperature': hourly_data['temperature_2m'],
        'humidity': hourly_data['relative_humidity_2m'],
        'pressure': hourly_data['pressure_msl'],
        'wind_speed': hourly_data['wind_speed_10m'],
        'wind_direction': hourly_data['wind_direction_10m'],
        'solar_radiation': hourly_data['shortwave_radiation']
    })
    
    return df

# Example usage  
if __name__ == "__main__":
    # Get last 30 days of historical data
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=30)).strftime('%Y-%m-%d')
    
    meteo_data = get_openmeteo_historical(40.7128, -74.0060, start_date, end_date)
    
    if meteo_data is not None:
        meteo_data.to_csv('data/raw/wind/weather_wind_data/openmeteo_historical.csv', index=False)
        print(f"Saved {len(meteo_data)} hours of Open-Meteo historical data")