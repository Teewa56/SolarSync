# nasa_solar_data.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

def get_nasa_solar_data_hourly(lat, lon, start_date, end_date):
    """
    Get HOURLY solar data from NASA POWER API
    
    Args:
        lat (float): Latitude
        lon (float): Longitude  
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format
    
    Returns:
        pandas.DataFrame: Hourly solar data with timestamp and weather parameters
    """
    # Change to hourly endpoint
    url = "https://power.larc.nasa.gov/api/temporal/hourly/point"
    
    params = {
        'parameters': 'ALLSKY_SFC_SW_DWN,T2M,RH2M,WS10M,PS',
        'community': 'RE',
        'longitude': lon,
        'latitude': lat,
        'start': start_date,
        'end': end_date,
        'format': 'JSON',
        'time-standard': 'UTC'
    }
    
    try:
        print(f"Fetching hourly data from NASA POWER API...")
        print(f"Location: ({lat}, {lon})")
        print(f"Date range: {start_date} to {end_date}")
        
        response = requests.get(url, params=params, timeout=60)
        response.raise_for_status()
        
        data = response.json()
        return parse_nasa_hourly_data(data)
        
    except requests.exceptions.RequestException as e:
        print(f"Error fetching NASA data: {e}")
        return None

def parse_nasa_hourly_data(nasa_json):
    """Parse NASA JSON response for HOURLY data into DataFrame"""
    parameters = nasa_json['properties']['parameter']
    
    # Extract timestamps - format is YYYYMMDDHH
    timestamps = list(parameters['ALLSKY_SFC_SW_DWN'].keys())
    
    # Parse timestamps: '2020051500' -> '2020-05-15 00:00:00'
    parsed_timestamps = []
    for ts in timestamps:
        year = ts[:4]
        month = ts[4:6]
        day = ts[6:8]
        hour = ts[8:10]
        parsed_timestamps.append(f"{year}-{month}-{day} {hour}:00:00")
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(parsed_timestamps),
        'solar_radiation': [parameters['ALLSKY_SFC_SW_DWN'][ts] for ts in timestamps],
        'temperature': [parameters['T2M'][ts] for ts in timestamps],
        'humidity': [parameters['RH2M'][ts] for ts in timestamps],
        'wind_speed': [parameters['WS10M'][ts] for ts in timestamps],
        'pressure': [parameters['PS'][ts] for ts in timestamps]
    })
    
    # Replace -999 (missing values) with NaN
    df = df.replace(-999, pd.NA)
    
    return df

# Example usage
if __name__ == "__main__":
    # CONFIGURATION - UPDATE THESE
    LAT = 28.6139   # Your solar plant latitude
    LON = 77.2090   # Your solar plant longitude
    
    # Date range matching your generation data (May 2020)
    START_DATE = '20200515'  # May 15, 2020
    END_DATE = '20200531'    # May 31, 2020
    
    OUTPUT_PATH = 'data/solar/weather_solar_data/nasa_solar_hourly_2020.csv'
    
    print("="*60)
    print("NASA POWER Hourly Data Fetcher")
    print("="*60)
    
    solar_data = get_nasa_solar_data_hourly(LAT, LON, START_DATE, END_DATE)
    
    if solar_data is not None:
        print(f"\nSuccessfully fetched {len(solar_data)} hourly records")
        print(f"Date range: {solar_data['timestamp'].min()} to {solar_data['timestamp'].max()}")
        print(f"\nData preview:")
        print(solar_data.head())
        print(f"\nData statistics:")
        print(solar_data.describe())
        
        # Save to CSV
        import os
        os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
        solar_data.to_csv(OUTPUT_PATH, index=False)
        print(f"\n✓ Saved to: {OUTPUT_PATH}")
        
        print("\nNext steps:")
        print("1. Update data_merger.py with:")
        print(f"   WEATHER_DATA_PATH = '{OUTPUT_PATH}'")
        print("2. Run: python data_merger.py")
    else:
        print("\n✗ Failed to fetch data")