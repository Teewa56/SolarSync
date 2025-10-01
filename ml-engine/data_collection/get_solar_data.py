# nasa_solar_data.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_nasa_solar_data(lat, lon, start_date, end_date):
    """
    Get solar data from NASA POWER API
    
    Args:
        lat (float): Latitude
        lon (float): Longitude  
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format
    
    Returns:
        pandas.DataFrame: Solar data with timestamp and weather parameters
    """
    url = "https://power.larc.nasa.gov/api/temporal/daily/point"
    
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M,PS',
        'community': 'RE',
        'longitude': lon,
        'latitude': lat,
        'start': start_date,
        'end': end_date,
        'format': 'JSON'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        
        data = response.json()
        return parse_nasa_solar_data(data)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching NASA data: {e}")
        return None

def parse_nasa_solar_data(nasa_json):
    """Parse NASA JSON response into DataFrame"""
    parameters = nasa_json['properties']['parameter']
    
    # Extract dates from any parameter (all should have same dates)
    dates = list(parameters['ALLSKY_SFC_SW_DWN'].keys())
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(dates),
        'solar_radiation': [parameters['ALLSKY_SFC_SW_DWN'][date] for date in dates],
        'temperature': [parameters['T2M'][date] for date in dates],
        'humidity': [parameters['RH2M'][date] for date in dates],
        'wind_speed': [parameters['WS10M'][date] for date in dates],
        'pressure': [parameters['PS'][date] for date in dates]
    })
    
    return df

# Example usage
if __name__ == "__main__":
    # Get last 3 years of solar data for New York
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=3*365)).strftime('%Y%m%d')
    
    solar_data = get_nasa_solar_data(40.7128, -74.0060, start_date, end_date)
    
    if solar_data is not None:
        solar_data.to_csv('data/raw/solar/weather_solar_data/nasa_solar_data.csv', index=False)
        print(f"Saved {len(solar_data)} days of NASA solar data")