import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from typing import Tuple, Dict

def load_and_preprocess_data(file_path: str, features: list) -> Tuple[np.ndarray, MinMaxScaler]:
    """
    Loads historical weather and energy data and preprocesses it.
    
    Args:
        file_path: Path to the CSV file (e.g., 'data/solar_history.csv').
        features: List of features to use (e.g., ['irradiance', 'temp', 'humidity', 'energy_output']).
        
    Returns:
        A tuple of (scaled_data, scaler_object).
    """
    try:
        df = pd.read_csv(file_path, parse_dates=['timestamp'])
    except FileNotFoundError:
        print(f"Error: Data file not found at {file_path}")
        return np.array([]), MinMaxScaler()

    # Ensure the target feature ('energy_output' in a real dataset) is included and is the last feature
    data = df[features].values.astype(np.float32)
    
    # Normalize the data
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    
    return scaled_data, scaler

def create_sequences(data: np.ndarray, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transforms time series data into input (X) and target (Y) sequences.
    
    Args:
        data: The 2D numpy array of scaled features.
        seq_length: The number of previous time steps to use as input (look-back window).
        
    Returns:
        A tuple of (X_sequences, Y_targets).
    """
    X, Y = [], []
    for i in range(len(data) - seq_length):
        # Look-back window as input (X)
        X.append(data[i:(i + seq_length), :])
        # The target (Y) is the next time step's energy output (last column)
        Y.append(data[i + seq_length, -1]) 
        
    return np.array(X), np.array(Y)

# Example Usage (placeholder data needed for a full run)
# data_path = './data/solar_history.csv'
# features = ['solar_irradiance', 'temperature', 'cloud_cover', 'energy_output']
# scaled_data, scaler = load_and_preprocess_data(data_path, features)
# X, Y = create_sequences(scaled_data, seq_length=24)