# ml-engine/tests/test_models.py
import pytest
import numpy as np
import torch
import os
import joblib

# NOTE: Need to ensure Python path is configured to find modules
from ml_engine.models import LSTMForecastModel 
from ml_engine.data_loader import load_and_preprocess_data, create_sequences

# --- Fixtures and Setup ---

# Create a mock CSV file for testing data loading
@pytest.fixture(scope="module", autouse=True)
def mock_data_file():
    # Ensure data directory exists
    os.makedirs('data/mock', exist_ok=True)
    # 50 hours of data, 4 features + target
    data = {
        'timestamp': pd.to_datetime(pd.date_range(start='2024-01-01', periods=50, freq='H')),
        'feature_1': np.random.rand(50),
        'feature_2': np.random.rand(50),
        'target_output': np.random.rand(50) * 100 
    }
    df = pd.DataFrame(data)
    test_path = 'data/mock/mock_history.csv'
    df.to_csv(test_path, index=False)
    yield test_path
    # Cleanup (optional)
    os.remove(test_path)

# --- Tests ---

def test_data_loading_and_scaling(mock_data_file):
    """Test if data loads, selects correct features, and scales correctly."""
    features = ['feature_1', 'feature_2', 'target_output']
    scaled_data, scaler = load_and_preprocess_data(mock_data_file, features)
    
    assert scaled_data.shape[1] == 3, "Should have 3 features (2 inputs + 1 target)"
    assert scaled_data.max() <= 1.0 and scaled_data.min() >= 0.0, "Data should be normalized [0, 1]"
    assert isinstance(scaler, joblib.MinMaxScaler), "Scaler object should be returned"

def test_sequence_creation():
    """Test conversion of 2D time series data into 3D sequences (X) and 1D targets (Y)."""
    # 10 data points, 3 features
    data = np.array([[i, i * 2, i * 3] for i in range(10)]) 
    seq_length = 3
    X, Y = create_sequences(data, seq_length)
    
    assert X.ndim == 3, "Input X should be 3D (samples, sequence_length, features)"
    assert X.shape[0] == 7, "Should create 10 - 3 = 7 samples"
    assert X.shape[1] == 3, "Sequence length must be 3"
    assert X.shape[2] == 3, "Feature size must be 3"
    assert Y.shape[0] == 7, "Target Y should have 7 samples"
    # Check sequence content (e.g., first sequence)
    assert np.allclose(X[0, 0, :], data[0]), "First step of first sequence is incorrect"
    assert Y[0] == data[3, 2], "First target value is incorrect (target is next step's last feature)"

def test_lstm_model_architecture():
    """Test model instantiation and tensor flow through the LSTM model."""
    input_size = 4
    hidden_size = 128
    num_layers = 3
    output_size = 1
    model = LSTMForecastModel(input_size, hidden_size, num_layers, output_size)
    
    # Mock input: Batch size 1, sequence length 24, 4 features
    mock_input = torch.randn(1, 24, 4) 
    
    output = model(mock_input)
    
    assert output.ndim == 2, "Output tensor should be 2D"
    assert output.shape == (1, output_size), "Output shape should be (batch_size, output_size)"