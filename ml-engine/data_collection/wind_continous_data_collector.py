# scheduled_wind_data_db.py
import requests
import pandas as pd
from datetime import datetime, timedelta
import schedule
import time
import psycopg2
from sqlalchemy import create_engine

def get_openmeteo_wind_data(lat, lon, start_date, end_date):
    """Get wind data from Open-Meteo API"""
    url = "https://archive-api.open-meteo.com/v1/archive"
    
    params = {
        'latitude': lat,
        'longitude': lon,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': 'wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,pressure_msl',
        'timezone': 'auto'
    }
    
    try:
        response = requests.get(url, params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        return parse_openmeteo_wind_data(data)
    except Exception as e:
        print(f"Error fetching Open-Meteo wind data: {e}")
        return None

def parse_openmeteo_wind_data(meteo_json):
    """Parse Open-Meteo JSON response into DataFrame"""
    hourly_data = meteo_json['hourly']
    
    df = pd.DataFrame({
        'timestamp': pd.to_datetime(hourly_data['time']),
        'wind_speed': hourly_data['wind_speed_10m'],
        'wind_direction': hourly_data['wind_direction_10m'],
        'wind_gust': hourly_data['wind_gusts_10m'],
        'temperature': hourly_data['temperature_2m'],
        'pressure': hourly_data['pressure_msl']
    })
    
    return df

def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host="localhost",
        database="solarsync",
        user="your_username",
        password="your_password",
        port="5432"
    )

def create_wind_table():
    """Create wind data table if not exists"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS wind_data (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP UNIQUE,
        wind_speed FLOAT,
        wind_direction FLOAT,
        wind_gust FLOAT,
        temperature FLOAT,
        pressure FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_wind_timestamp ON wind_data(timestamp);
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    conn.close()

def store_wind_data_db(wind_df):
    """Store wind data in PostgreSQL"""
    try:
        engine = create_engine('postgresql://your_username:your_password@localhost:5432/solarsync')
        wind_df.to_sql('wind_data', engine, if_exists='append', index=False, method='multi')
        print(f"Stored {len(wind_df)} wind data records in database")
        return True
    except Exception as e:
        print(f"Error storing wind data in DB: {e}")
        return False

def scheduled_wind_data_db():
    """Scheduled task to collect and store wind data in PostgreSQL"""
    print(f"Running wind data collection at {datetime.now()}")
    
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
    
    wind_data = get_openmeteo_wind_data(40.7128, -74.0060, start_date, end_date)
    
    if wind_data is not None:
        success = store_wind_data_db(wind_data)
        if success:
            print("Wind data successfully stored in database")
        else:
            print("Failed to store wind data in database")
    else:
        print("Failed to fetch wind data")

create_wind_table()

schedule.every().day.at("06:00").do(scheduled_wind_data_db)
schedule.every().day.at("18:00").do(scheduled_wind_data_db)

if __name__ == "__main__":
    print("Wind data DB scheduler started...")
    scheduled_wind_data_db()
    
    while True:
        schedule.run_pending()
        time.sleep(60)