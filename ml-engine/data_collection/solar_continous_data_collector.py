import requests
import pandas as pd
from datetime import datetime, timedelta
import schedule
import time
import psycopg2
from sqlalchemy import create_engine

def get_nasa_solar_data(lat, lon, start_date, end_date):
    """Get solar data from NASA POWER API"""
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
    except Exception as e:
        print(f"Error fetching NASA solar data: {e}")
        return None

def parse_nasa_solar_data(nasa_json):
    """Parse NASA JSON response into DataFrame"""
    parameters = nasa_json['properties']['parameter']
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

def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host="localhost",
        database="solarsync",
        user="your_username",
        password="your_password",
        port="5432"
    )

def create_solar_table():
    """Create solar data table if not exists"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS solar_data (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP UNIQUE,
        solar_radiation FLOAT,
        temperature FLOAT,
        humidity FLOAT,
        wind_speed FLOAT,
        pressure FLOAT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_solar_timestamp ON solar_data(timestamp);
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    conn.close()

def store_solar_data_db(solar_df):
    """Store solar data in PostgreSQL"""
    try:
        engine = create_engine('postgresql://your_username:your_password@localhost:5432/solarsync')
        solar_df.to_sql('solar_data', engine, if_exists='append', index=False, method='multi')
        print(f"Stored {len(solar_df)} solar data records in database")
        return True
    except Exception as e:
        print(f"Error storing solar data in DB: {e}")
        return False

def scheduled_solar_data_db():
    """Scheduled task to collect and store solar data in PostgreSQL"""
    print(f"Running solar data collection at {datetime.now()}")
    
    end_date = datetime.now().strftime('%Y%m%d')
    start_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
    
    solar_data = get_nasa_solar_data(40.7128, -74.0060, start_date, end_date)
    
    if solar_data is not None:
        success = store_solar_data_db(solar_data)
        if success:
            print("Solar data successfully stored in database")
        else:
            print("Failed to store solar data in database")
    else:
        print("Failed to fetch solar data")

create_solar_table()

schedule.every().day.at("06:00").do(scheduled_solar_data_db)
schedule.every().day.at("18:00").do(scheduled_solar_data_db)

if __name__ == "__main__":
    print("Solar data DB scheduler started...")
    scheduled_solar_data_db()
    
    while True:
        schedule.run_pending()
        time.sleep(60)