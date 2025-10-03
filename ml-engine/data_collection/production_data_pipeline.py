import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import logging
import time
from typing import Dict, List, Optional
import redis
import requests
from db_config import get_db_connection, get_engine
from data_validator import DataValidator
import schedule

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionDataPipeline:
    """
    Production-grade data collection and storage pipeline
    with error handling, monitoring, and retry logic
    """
    
    def __init__(self):
        self.db_engine = get_engine()
        
        # Redis for caching (optional but recommended)
        try:
            self.redis_client = redis.Redis(
                host=os.getenv('REDIS_HOST', 'localhost'),
                port=int(os.getenv('REDIS_PORT', 6379)),
                db=0,
                decode_responses=True
            )
            self.redis_available = True
        except:
            logger.warning("Redis not available, running without cache")
            self.redis_available = False
        
        self.validators = {
            'solar': DataValidator('solar'),
            'wind': DataValidator('wind')
        }
        
        # Monitoring metrics
        self.metrics = {
            'solar_fetches': 0,
            'wind_fetches': 0,
            'solar_errors': 0,
            'wind_errors': 0,
            'last_successful_fetch': None
        }
    
    def fetch_nasa_solar_data(self, lat: float, lon: float, 
                              start_date: str, end_date: str,
                              max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch solar data with retry logic and error handling
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
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                df = self._parse_nasa_response(data)
                
                # Validate data
                validator = self.validators['solar']
                clean_df, report = validator.validate_dataframe(df)
                
                if report['data_quality_score'] < 0.7:
                    logger.warning(f"Low data quality score: {report['data_quality_score']:.2%}")
                
                self.metrics['solar_fetches'] += 1
                self.metrics['last_successful_fetch'] = datetime.now()
                
                logger.info(f"✅ Fetched {len(clean_df)} solar data points")
                return clean_df
                
            except requests.exceptions.Timeout:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: NASA API timeout")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    
            except requests.exceptions.RequestException as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: NASA API error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    
            except Exception as e:
                logger.error(f"Unexpected error fetching solar data: {e}")
                break
        
        self.metrics['solar_errors'] += 1
        return None
    
    def fetch_openmeteo_wind_data(self, lat: float, lon: float,
                                   start_date: str, end_date: str,
                                   max_retries: int = 3) -> Optional[pd.DataFrame]:
        """
        Fetch wind data with retry logic and error handling
        """
        url = "https://archive-api.open-meteo.com/v1/archive"
        
        params = {
            'latitude': lat,
            'longitude': lon,
            'start_date': start_date,
            'end_date': end_date,
            'hourly': 'wind_speed_10m,wind_direction_10m,wind_gusts_10m,temperature_2m,pressure_msl',
            'timezone': 'auto'
        }
        
        for attempt in range(max_retries):
            try:
                response = requests.get(url, params=params, timeout=30)
                response.raise_for_status()
                
                data = response.json()
                df = self._parse_openmeteo_response(data)
                
                # Validate data
                validator = self.validators['wind']
                clean_df, report = validator.validate_dataframe(df)
                
                if report['data_quality_score'] < 0.7:
                    logger.warning(f"Low data quality score: {report['data_quality_score']:.2%}")
                
                self.metrics['wind_fetches'] += 1
                self.metrics['last_successful_fetch'] = datetime.now()
                
                logger.info(f"✅ Fetched {len(clean_df)} wind data points")
                return clean_df
                
            except Exception as e:
                logger.error(f"Attempt {attempt + 1}/{max_retries}: Wind data error: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
        
        self.metrics['wind_errors'] += 1
        return None
    
    def _parse_nasa_response(self, nasa_json: Dict) -> pd.DataFrame:
        """Parse NASA API response"""
        parameters = nasa_json['properties']['parameter']
        dates = list(parameters['ALLSKY_SFC_SW_DWN'].keys())
        
        return pd.DataFrame({
            'timestamp': pd.to_datetime(dates),
            'solar_radiation': [parameters['ALLSKY_SFC_SW_DWN'][date] for date in dates],
            'temperature': [parameters['T2M'][date] for date in dates],
            'humidity': [parameters['RH2M'][date] for date in dates],
            'wind_speed': [parameters['WS10M'][date] for date in dates],
            'pressure': [parameters['PS'][date] for date in dates]
        })
    
    def _parse_openmeteo_response(self, meteo_json: Dict) -> pd.DataFrame:
        """Parse Open-Meteo API response"""
        hourly_data = meteo_json['hourly']
        
        return pd.DataFrame({
            'timestamp': pd.to_datetime(hourly_data['time']),
            'wind_speed': hourly_data['wind_speed_10m'],
            'wind_direction': hourly_data['wind_direction_10m'],
            'wind_gust': hourly_data['wind_gusts_10m'],
            'temperature': hourly_data['temperature_2m'],
            'pressure': hourly_data['pressure_msl']
        })
    
    def store_to_database(self, df: pd.DataFrame, table_name: str) -> bool:
        """
        Store data to PostgreSQL with duplicate handling
        """
        if df is None or len(df) == 0:
            logger.warning(f"No data to store in {table_name}")
            return False
        
        try:
            # Use INSERT ... ON CONFLICT DO NOTHING for duplicate handling
            df.to_sql(
                table_name,
                self.db_engine,
                if_exists='append',
                index=False,
                method='multi',
                chunksize=1000
            )
            
            logger.info(f"✅ Stored {len(df)} records to {table_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to store data to {table_name}: {e}")
            return False
    
    def cache_to_redis(self, key: str, data: pd.DataFrame, expiry: int = 3600):
        """Cache data to Redis (optional)"""
        if not self.redis_available:
            return
        
        try:
            # Store as JSON
            self.redis_client.setex(
                key,
                expiry,
                data.to_json(orient='records', date_format='iso')
            )
            logger.debug(f"Cached {len(data)} records to Redis key: {key}")
        except Exception as e:
            logger.warning(f"Redis cache failed: {e}")
    
    def scheduled_solar_collection(self):
        """
        Scheduled task for solar data collection
        Runs every 6 hours
        """
        logger.info("="*60)
        logger.info(f"SOLAR DATA COLLECTION - {datetime.now()}")
        logger.info("="*60)
        
        # Fetch yesterday's data
        end_date = datetime.now().strftime('%Y%m%d')
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y%m%d')
        
        # Default location (should be configurable per plant)
        lat, lon = 40.7128, -74.0060
        
        df = self.fetch_nasa_solar_data(lat, lon, start_date, end_date)
        
        if df is not None:
            # Store to database
            success = self.store_to_database(df, 'solar_data')
            
            # Cache recent data
            if success:
                self.cache_to_redis('latest_solar_data', df, expiry=21600)  # 6 hours
        
        self._log_metrics()
    
    def scheduled_wind_collection(self):
        """
        Scheduled task for wind data collection
        Runs every 6 hours
        """
        logger.info("="*60)
        logger.info(f"WIND DATA COLLECTION - {datetime.now()}")
        logger.info("="*60)
        
        # Fetch yesterday's data
        end_date = datetime.now().strftime('%Y-%m-%d')
        start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        
        # Default location
        lat, lon = 40.7128, -74.0060
        
        df = self.fetch_openmeteo_wind_data(lat, lon, start_date, end_date)
        
        if df is not None:
            # Store to database
            success = self.store_to_database(df, 'wind_data')
            
            # Cache recent data
            if success:
                self.cache_to_redis('latest_wind_data', df, expiry=21600)
        
        self._log_metrics()
    
    def _log_metrics(self):
        """Log pipeline metrics"""
        logger.info(f"\nPipeline Metrics:")
        logger.info(f"  Solar fetches: {self.metrics['solar_fetches']}")
        logger.info(f"  Solar errors: {self.metrics['solar_errors']}")
        logger.info(f"  Wind fetches: {self.metrics['wind_fetches']}")
        logger.info(f"  Wind errors: {self.metrics['wind_errors']}")
        logger.info(f"  Last success: {self.metrics['last_successful_fetch']}\n")
    
    def run_scheduler(self):
        """
        Run the data collection scheduler
        """
        logger.info("="*60)
        logger.info("PRODUCTION DATA PIPELINE STARTED")
        logger.info("="*60)
        
        # Schedule tasks
        schedule.every(6).hours.do(self.scheduled_solar_collection)
        schedule.every(6).hours.do(self.scheduled_wind_collection)
        
        # Run immediately on startup
        self.scheduled_solar_collection()
        self.scheduled_wind_collection()
        
        # Keep running
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


import os

if __name__ == "__main__":
    pipeline = ProductionDataPipeline()
    pipeline.run_scheduler()