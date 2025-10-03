import pandas as pd
import numpy as np
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

class FeatureEngineer:
    """Advanced feature engineering for time series forecasting"""
    
    def __init__(self, data_type: str = 'solar'):
        self.data_type = data_type
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from timestamp
        
        Features:
        - hour, day_of_week, month, season
        - is_weekend, is_holiday
        - cyclical encoding (sin/cos)
        """
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            logger.warning("No timestamp column found")
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Basic temporal features
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.dayofweek
        df['day_of_month'] = df['timestamp'].dt.day
        df['month'] = df['timestamp'].dt.month
        df['quarter'] = df['timestamp'].dt.quarter
        df['day_of_year'] = df['timestamp'].dt.dayofyear
        
        # Weekend indicator
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        
        # Season (Northern Hemisphere)
        df['season'] = df['month'].apply(lambda x: 
            0 if x in [12, 1, 2] else      # Winter
            1 if x in [3, 4, 5] else        # Spring
            2 if x in [6, 7, 8] else        # Summer
            3                               # Fall
        )
        
        # Cyclical encoding for periodic features
        # Hour (24-hour cycle)
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
        
        # Day of week (7-day cycle)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Month (12-month cycle)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        
        # Day of year (365-day cycle)
        df['doy_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['doy_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        
        logger.info(f"Created {17} temporal features")
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str, 
                           lags: List[int] = None) -> pd.DataFrame:
        """
        Create lagged features for target variable
        
        Args:
            df: DataFrame with time series data
            target_col: Column to create lags for
            lags: List of lag periods (default: [1, 2, 3, 6, 12, 24])
        """
        if lags is None:
            lags = [1, 2, 3, 6, 12, 24]  # 1h, 2h, 3h, 6h, 12h, 24h
        
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        logger.info(f"Created {len(lags)} lag features for {target_col}")
        
        return df
    
    def create_rolling_features(self, df: pd.DataFrame, target_col: str,
                               windows: List[int] = None) -> pd.DataFrame:
        """
        Create rolling window statistics
        
        Args:
            df: DataFrame with time series data
            target_col: Column to calculate rolling stats for
            windows: List of window sizes (default: [3, 6, 12, 24])
        """
        if windows is None:
            windows = [3, 6, 12, 24]  # 3h, 6h, 12h, 24h
        
        df = df.copy()
        
        for window in windows:
            # Rolling mean
            df[f'{target_col}_rolling_mean_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).mean()
            )
            
            # Rolling std
            df[f'{target_col}_rolling_std_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).std()
            )
            
            # Rolling min/max
            df[f'{target_col}_rolling_min_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).min()
            )
            df[f'{target_col}_rolling_max_{window}'] = (
                df[target_col].rolling(window=window, min_periods=1).max()
            )
        
        logger.info(f"Created {len(windows) * 4} rolling features for {target_col}")
        
        return df
    
    def create_solar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create solar-specific features
        """
        df = df.copy()
        
        if 'timestamp' not in df.columns:
            return df
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Solar elevation angle (simplified)
        lat = 40.0  # Default latitude, should be parameterized
        df['solar_elevation'] = self._calculate_solar_elevation(
            df['timestamp'], lat
        )
        
        # Daylight hours indicator
        df['is_daylight'] = (
            (df['hour'] >= 6) & (df['hour'] <= 20)
        ).astype(int)
        
        # Clear sky index (if solar radiation available)
        if 'solar_radiation' in df.columns:
            max_radiation = 1000  # W/m²
            df['clear_sky_index'] = (
                df['solar_radiation'] / max_radiation
            ).clip(0, 1)
        
        # Cloud impact factor
        if 'cloud_cover' in df.columns:
            df['cloud_impact'] = 1 - (df['cloud_cover'] / 100)
        
        # Temperature effect on efficiency
        if 'temperature' in df.columns:
            optimal_temp = 25  # °C
            df['temp_efficiency'] = 1 - (
                abs(df['temperature'] - optimal_temp) * 0.005
            )
            df['temp_efficiency'] = df['temp_efficiency'].clip(0, 1)
        
        logger.info("Created solar-specific features")
        
        return df
    
    def create_wind_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create wind-specific features
        """
        df = df.copy()
        
        if 'wind_speed' in df.columns:
            # Wind power (cubic relationship)
            df['wind_power_potential'] = df['wind_speed'] ** 3
            
            # Wind speed categories
            df['wind_speed_category'] = pd.cut(
                df['wind_speed'],
                bins=[0, 3, 7, 12, 25, 100],
                labels=['calm', 'light', 'moderate', 'strong', 'extreme']
            ).astype(str)
            
            # Cut-in and cut-out indicators
            df['below_cutin'] = (df['wind_speed'] < 3).astype(int)
            df['above_cutout'] = (df['wind_speed'] > 25).astype(int)
            df['optimal_range'] = (
                (df['wind_speed'] >= 7) & (df['wind_speed'] <= 15)
            ).astype(int)
        
        if 'wind_direction' in df.columns:
            # Wind direction components (N-S, E-W)
            df['wind_north_south'] = np.cos(np.radians(df['wind_direction']))
            df['wind_east_west'] = np.sin(np.radians(df['wind_direction']))
            
            # Prevailing wind indicator
            prevailing_direction = 270  # West (default)
            df['aligned_with_prevailing'] = (
                np.cos(np.radians(df['wind_direction'] - prevailing_direction))
            )
        
        if 'wind_gust' in df.columns and 'wind_speed' in df.columns:
            # Gust factor
            df['gust_factor'] = df['wind_gust'] / (df['wind_speed'] + 0.1)
        
        logger.info("Created wind-specific features")
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables
        """
        df = df.copy()
        
        if self.data_type == 'solar':
            # Solar-specific interactions
            if all(col in df.columns for col in ['solar_radiation', 'cloud_cover']):
                df['radiation_cloud_interaction'] = (
                    df['solar_radiation'] * (1 - df['cloud_cover'] / 100)
                )
            
            if all(col in df.columns for col in ['temperature', 'humidity']):
                df['temp_humidity_interaction'] = (
                    df['temperature'] * df['humidity'] / 100
                )
        
        elif self.data_type == 'wind':
            # Wind-specific interactions
            if all(col in df.columns for col in ['wind_speed', 'pressure']):
                df['wind_pressure_interaction'] = (
                    df['wind_speed'] * (df['pressure'] / 1013)
                )
            
            if all(col in df.columns for col in ['wind_speed', 'temperature']):
                df['wind_temp_interaction'] = (
                    df['wind_speed'] * df['temperature']
                )
        
        logger.info("Created interaction features")
        
        return df
    
    def _calculate_solar_elevation(self, timestamps: pd.Series, lat: float) -> np.ndarray:
        """
        Calculate solar elevation angle (simplified)
        
        Note: This is a simplified calculation. For production,
        use a library like pvlib or ephem for accurate calculations.
        """
        hour = timestamps.dt.hour + timestamps.dt.minute / 60
        day_of_year = timestamps.dt.dayofyear
        
        # Solar declination (simplified)
        declination = 23.45 * np.sin(np.radians(360 * (284 + day_of_year) / 365))
        
        # Hour angle
        hour_angle = 15 * (hour - 12)
        
        # Solar elevation
        elevation = np.arcsin(
            np.sin(np.radians(lat)) * np.sin(np.radians(declination)) +
            np.cos(np.radians(lat)) * np.cos(np.radians(declination)) * 
            np.cos(np.radians(hour_angle))
        )
        
        return np.degrees(elevation)
    
    def engineer_all_features(self, df: pd.DataFrame, 
                            target_col: str = 'energy_output') -> pd.DataFrame:
        """
        Apply all feature engineering steps
        
        Args:
            df: Raw DataFrame with timestamp and features
            target_col: Target variable column name
        
        Returns:
            DataFrame with all engineered features
        """
        logger.info("Starting comprehensive feature engineering...")
        
        # 1. Temporal features
        df = self.create_temporal_features(df)
        
        # 2. Type-specific features
        if self.data_type == 'solar':
            df = self.create_solar_features(df)
        elif self.data_type == 'wind':
            df = self.create_wind_features(df)
        
        # 3. Lag features (if target exists)
        if target_col in df.columns:
            df = self.create_lag_features(df, target_col)
            df = self.create_rolling_features(df, target_col)
        
        # 4. Interaction features
        df = self.create_interaction_features(df)
        
        # 5. Drop rows with NaN from lag/rolling features
        initial_rows = len(df)
        df = df.dropna()
        logger.info(f"Dropped {initial_rows - len(df)} rows with NaN values")
        
        logger.info(f"Feature engineering complete. Final shape: {df.shape}")
        
        return df


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample solar data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=200, freq='H'),
        'solar_radiation': np.random.uniform(200, 1000, 200),
        'temperature': np.random.uniform(15, 35, 200),
        'humidity': np.random.uniform(30, 80, 200),
        'cloud_cover': np.random.uniform(0, 100, 200),
        'energy_output': np.random.uniform(50, 500, 200)
    })
    
    engineer = FeatureEngineer('solar')
    enriched_data = engineer.engineer_all_features(sample_data)
    
    print(f"\nOriginal features: {sample_data.shape[1]}")
    print(f"Engineered features: {enriched_data.shape[1]}")
    print(f"\nNew columns: {enriched_data.columns.tolist()}")