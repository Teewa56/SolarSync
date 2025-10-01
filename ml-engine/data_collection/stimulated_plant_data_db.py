import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import schedule
import time
import psycopg2
from sqlalchemy import create_engine

def get_simulated_plant_data():
    """Generate realistic plant data based on weather patterns"""
    # Get current weather to make realistic simulations
    current_weather = get_current_weather()
    
    if current_weather:
        # Generate realistic plant output based on weather
        solar_data = simulate_solar_generation(current_weather)
        wind_data = simulate_wind_generation(current_weather)
        
        # Combine both plant types
        plant_data = pd.concat([solar_data, wind_data], ignore_index=True)
        return plant_data
    else:
        return generate_fallback_plant_data()

def get_current_weather():
    """Get current weather for realistic simulation"""
    try:
        return {
            'solar_radiation': np.random.uniform(200, 900),
            'temperature': np.random.uniform(15, 35),
            'wind_speed': np.random.uniform(2, 12),
            'timestamp': datetime.now()
        }
    except:
        return None

def simulate_solar_generation(weather):
    """Simulate solar plant generation based on weather"""
    current_time = datetime.now()
    
    # Solar generation follows daily pattern + weather effects
    hour = current_time.hour
    solar_factor = max(0, np.sin((hour - 6) * np.pi / 12))  # 0 at night, peak at noon
    
    base_output = weather['solar_radiation'] * 0.2  # Assume 20% efficiency
    temperature_effect = max(0, 1 - (weather['temperature'] - 25) * 0.005)  # Temp derating
    
    actual_output = base_output * solar_factor * temperature_effect
    
    return pd.DataFrame({
        'timestamp': [current_time],
        'plant_id': ['solar_farm_001'],
        'plant_type': ['solar'],
        'energy_output_kwh': [actual_output],
        'capacity_factor': [actual_output / 1000],  # Assume 1kW system
        'efficiency': [0.18 + np.random.normal(0, 0.01)],
        'status': 'operational'
    })

def simulate_wind_generation(weather):
    """Simulate wind turbine generation based on weather"""
    current_time = datetime.now()
    
    # Wind power curve simulation
    wind_speed = weather['wind_speed']
    if wind_speed < 3:  # Cut-in speed
        power_output = 0
    elif wind_speed > 25:  # Cut-out speed  
        power_output = 0
    else:
        # Simplified power curve
        power_output = min(2000, 0.5 * 1.225 * 0.5 * (wind_speed ** 3))  # Basic wind power formula
    
    return pd.DataFrame({
        'timestamp': [current_time],
        'plant_id': ['wind_farm_001'],
        'plant_type': ['wind'],
        'energy_output_kwh': [power_output / 1000],  # Convert to kWh
        'capacity_factor': [power_output / 2000],  # Assume 2MW turbine
        'efficiency': [0.45 + np.random.normal(0, 0.02)],
        'status': 'operational'
    })

def generate_fallback_plant_data():
    """Fallback data if weather fetch fails"""
    current_time = datetime.now()
    
    # Simple time-based pattern
    hour = current_time.hour
    solar_output = max(0, 500 * np.sin((hour - 6) * np.pi / 12))
    wind_output = np.random.uniform(100, 800)
    
    solar_data = pd.DataFrame({
        'timestamp': [current_time],
        'plant_id': ['solar_farm_001'],
        'plant_type': ['solar'], 
        'energy_output_kwh': [solar_output],
        'capacity_factor': [solar_output / 1000],
        'efficiency': [0.18],
        'status': 'operational'
    })
    
    wind_data = pd.DataFrame({
        'timestamp': [current_time],
        'plant_id': ['wind_farm_001'],
        'plant_type': ['wind'],
        'energy_output_kwh': [wind_output],
        'capacity_factor': [wind_output / 2000],
        'efficiency': [0.45],
        'status': 'operational'
    })
    
    return pd.concat([solar_data, wind_data], ignore_index=True)

def create_plant_table():
    """Create plant generation data table"""
    conn = get_db_connection()
    cursor = conn.cursor()
    
    create_table_query = """
    CREATE TABLE IF NOT EXISTS plant_generation_data (
        id SERIAL PRIMARY KEY,
        timestamp TIMESTAMP,
        plant_id VARCHAR(50),
        plant_type VARCHAR(20),
        energy_output_kwh FLOAT,
        capacity_factor FLOAT,
        efficiency FLOAT,
        status VARCHAR(20),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    );
    CREATE INDEX IF NOT EXISTS idx_plant_timestamp ON plant_generation_data(timestamp);
    CREATE INDEX IF NOT EXISTS idx_plant_type ON plant_generation_data(plant_type);
    """
    
    cursor.execute(create_table_query)
    conn.commit()
    cursor.close()
    conn.close()

def store_plant_data_db(plant_df):
    """Store plant data in PostgreSQL"""
    try:
        engine = create_engine('postgresql://your_username:your_password@localhost:5432/solarsync')
        plant_df.to_sql('plant_generation_data', engine, if_exists='append', index=False)
        print(f"Stored {len(plant_df)} plant data records")
        return True
    except Exception as e:
        print(f"Error storing plant data: {e}")
        return False

def scheduled_plant_data_db():
    """Collect and store simulated plant data"""
    print(f"Running plant data simulation at {datetime.now()}")
    
    plant_data = get_simulated_plant_data()
    
    if plant_data is not None:
        success = store_plant_data_db(plant_data)
        if success:
            print("Simulated plant data stored successfully")
        else:
            print("Failed to store plant data")
    else:
        print("Failed to generate plant data")

def get_db_connection():
    """Create database connection"""
    return psycopg2.connect(
        host="localhost",
        database="solarsync",
        user="your_username",
        password="your_password",
        port="5432"
    )

# Initialize
create_plant_table()

# Schedule every 15 minutes for realistic data flow
schedule.every(60).minutes.do(scheduled_plant_data_db)

if __name__ == "__main__":
    print("Simulated plant data scheduler started...")
    scheduled_plant_data_db()
    
    while True:
        schedule.run_pending()
        time.sleep(60)