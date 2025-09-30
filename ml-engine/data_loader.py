import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, List
import logging
import os

logger = logging.getLogger(__name__)

def load_and_preprocess_data(
    file_path: str, 
    features: List[str]
) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Loads historical weather and energy data and preprocesses it
    
    Args:
        file_path: Path to the CSV file (e.g., 'data/solar/solar_history.csv')
        features: List of features to use (target should be last)
        
    Returns:
        A tuple of (scaled_data, scaler_object)
    """
    if not os.path.exists(file_path):
        logger.error(f"Data file not found at {file_path}")
        logger.info(f"Please create {file_path} with columns: {', '.join(features)}")
        return np.array([]), MinMaxScaler()

    try:
        # Load data
        df = pd.read_csv(file_path, parse_dates=['timestamp'] if 'timestamp' in pd.read_csv(file_path, nrows=0).columns else None)
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        
        # Check if all features exist
        missing_features = [f for f in features if f not in df.columns]
        if missing_features:
            logger.error(f"Missing features in data: {missing_features}")
            logger.info(f"Available columns: {list(df.columns)}")
            return np.array([]), MinMaxScaler()
        
        # Select and extract features
        data = df[features].values.astype(np.float32)
        
        # Check for NaN values
        if np.isnan(data).any():
            logger.warning(f"Found {np.isnan(data).sum()} NaN values, filling with forward fill")
            df[features] = df[features].fillna(method='ffill').fillna(method='bfill')
            data = df[features].values.astype(np.float32)
        
        # Normalize the data
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(data)
        
        logger.info(f"Data preprocessed successfully: shape {scaled_data.shape}")
        logger.info(f"Data range: [{scaled_data.min():.4f}, {scaled_data.max():.4f}]")
        
        return scaled_data, scaler
        
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return np.array([]), MinMaxScaler()


def create_sequences(
    data: np.ndarray, 
    seq_length: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms time series data into input (X) and target (Y) sequences
    
    Args:
        data: The 2D numpy array of scaled features [samples, features]
        seq_length: The number of previous time steps to use as input (look-back window)
        
    Returns:
        A tuple of (X_sequences, Y_targets)
        - X_sequences: shape [num_sequences, seq_length, num_features]
        - Y_targets: shape [num_sequences] - next timestep's energy output
    """
    if data.size == 0:
        logger.error("Empty data array provided to create_sequences")
        return np.array([]), np.array([])
    
    if len(data) <= seq_length:
        logger.error(f"Data length ({len(data)}) must be > sequence length ({seq_length})")
        return np.array([]), np.array([])
    
    X, Y = [], []
    
    for i in range(len(data) - seq_length):
        # Look-back window as input (X)
        X.append(data[i:(i + seq_length), :])
        
        # The target (Y) is the next time step's energy output (last column)
        Y.append(data[i + seq_length, -1])
    
    X = np.array(X)
    Y = np.array(Y)
    
    logger.info(f"Created sequences: X={X.shape}, Y={Y.shape}")
    
    return X, Y


def generate_sample_data(
    output_path: str,
    num_samples: int = 8760,  # 1 year of hourly data
    data_type: str = 'solar'
) -> None:
    """
    Generate sample training data for testing
    
    Args:
        output_path: Where to save the CSV
        num_samples: Number of hourly samples to generate
        data_type: 'solar' or 'wind'
    """
    logger.info(f"Generating {num_samples} samples of {data_type} data")
    
    timestamps = pd.date_range(start='2023-01-01', periods=num_samples, freq='H')
    
    if data_type == 'solar':
        # Simulate solar data with daily patterns
        hours = np.array([t.hour for t in timestamps])
        
        # Solar irradiance (higher during day)
        base_irradiance = 500 + 400 * np.sin((hours - 6) * np.pi / 12)
        base_irradiance = np.clip(base_irradiance, 0, 1000)
        solar_irradiance = base_irradiance + np.random.normal(0, 50, num_samples)
        solar_irradiance = np.clip(solar_irradiance, 0, 1000)
        
        # Temperature (correlated with irradiance)
        temperature = 15 + 10 * (solar_irradiance / 1000) + np.random.normal(0, 3, num_samples)
        
        # Cloud cover (inversely correlated with irradiance)
        cloud_cover = 100 - (solar_irradiance / 10) + np.random.normal(0, 15, num_samples)
        cloud_cover = np.clip(cloud_cover, 0, 100)
        
        # Energy output (function of irradiance, temp, clouds)
        energy_output = (
            solar_irradiance * 0.15  # Base conversion
            * (1 - cloud_cover / 200)  # Cloud penalty
            * (1 + (temperature - 25) / 100)  # Temperature effect
            + np.random.normal(0, 10, num_samples)
        )
        energy_output = np.clip(energy_output, 0, 200)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'solar_irradiance': solar_irradiance,
            'temperature': temperature,
            'cloud_cover': cloud_cover,
            'energy_output': energy_output
        })
        
    else:  # wind
        # Wind speed (varies more randomly)
        wind_speed = 5 + 3 * np.sin(np.arange(num_samples) * 2 * np.pi / 168)  # Weekly pattern
        wind_speed += np.random.normal(0, 2, num_samples)
        wind_speed = np.clip(wind_speed, 0, 25)
        
        # Wind direction
        wind_direction = np.random.uniform(0, 360, num_samples)
        
        # Pressure
        pressure = 1013 + np.random.normal(0, 10, num_samples)
        
        # Energy output (cubic relationship with wind speed)
        energy_output = (
            wind_speed ** 3 * 0.02  # Cubic power curve
            * (1 + (pressure - 1013) / 5000)  # Pressure effect
            + np.random.normal(0, 15, num_samples)
        )
        energy_output = np.clip(energy_output, 0, 300)
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'wind_speed': wind_speed,
            'wind_direction': wind_direction,
            'pressure': pressure,
            'energy_output': energy_output
        })
    
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    logger.info(f"Sample data saved to {output_path}")
    logger.info(f"Columns: {list(df.columns)}")
    logger.info(f"Shape: {df.shape}")


def split_train_val_test(
    X: np.ndarray,
    Y: np.ndarray,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train, validation, and test sets
    
    Args:
        X: Input sequences
        Y: Target values
        train_ratio: Proportion of data for training
        val_ratio: Proportion of data for validation
        test_ratio: Proportion of data for testing
        
    Returns:
        X_train, X_val, X_test, Y_train, Y_val, Y_test
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-5, "Ratios must sum to 1"
    
    n_samples = len(X)
    train_end = int(n_samples * train_ratio)
    val_end = int(n_samples * (train_ratio + val_ratio))
    
    X_train = X[:train_end]
    Y_train = Y[:train_end]
    
    X_val = X[train_end:val_end]
    Y_val = Y[train_end:val_end]
    
    X_test = X[val_end:]
    Y_test = Y[val_end:]
    
    logger.info(f"Split data - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    
    return X_train, X_val, X_test, Y_train, Y_val, Y_test


# Example usage for generating sample data
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    
    # Generate sample solar data
    generate_sample_data(
        'data/solar/solar_history.csv',
        num_samples=8760,
        data_type='solar'
    )
    
    # Generate sample wind data
    generate_sample_data(
        'data/wind/wind_history.csv',
        num_samples=8760,
        data_type='wind'
    )
    
    logger.info("Sample data generation complete!")