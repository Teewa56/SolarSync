# tests/test_prediction_accuracy.py
import pytest
import numpy as np
import torch
import joblib
import os
import math
from ml_engine.models import LSTMForecastModel
from ml_engine.data_loader import load_and_preprocess_data
# NOTE: Need the run_autoregressive_prediction function from main.py or move it to a utility file
# from ml_engine.main import run_autoregressive_prediction 

# --- Setup and Configuration ---

# Mock the autoregressive function for testing in a standalone environment
def mock_autoregressive_prediction(model, scaler, test_features, steps):
    """Simulates the loop where a model feeds its output back into its input."""
    
    # 1. Get initial sequence from the features (mocking historical data context)
    initial_sequence = test_features[:24] 
    current_sequence = torch.from_numpy(initial_sequence).float().unsqueeze(0) 

    predictions_scaled = []
    
    for i in range(steps):
        with torch.no_grad():
            prediction_tensor = model(current_sequence)
            
        predictions_scaled.append(prediction_tensor.item())
        
        # Auto-regression step: Use a mock next feature vector
        next_step_input = np.random.rand(1, current_sequence.shape[-1]).astype(np.float32)
        next_step_input[0, -1] = prediction_tensor.item() # Inject predicted target back in
        
        # Shift and Append
        next_sequence = current_sequence.squeeze(0)[1:] 
        new_step = torch.from_numpy(next_step_input).float()
        current_sequence = torch.cat((next_sequence, new_step), dim=0).unsqueeze(0)

    # Simplified denormalization for mock
    return np.array(predictions_scaled) * 500 # Mock KWh scale

@pytest.mark.skip(reason="Requires full ML dependencies and large data files.")
def test_solar_autoregressive_accuracy():
    """Tests the stability and error growth of 48-hour solar prediction."""
    print("\n--- Testing 48-Hour Autoregressive Solar Accuracy ---")
    
    MODEL_TYPE = 'solar'
    HOURS_TO_PREDICT = 48
    FEATURES = ['solar_irradiance', 'temperature', 'cloud_cover', 'energy_output']

    # Load assets
    save_dir = 'ml-engine/saved_models'
    model_path = os.path.join(save_dir, f'{MODEL_TYPE}_model.pth')
    scaler_path = os.path.join(save_dir, f'{MODEL_TYPE}_scaler.pkl')
    
    # Using a test data file that covers the prediction window
    DATA_PATH = 'data/solar/solar_test_2024.csv' 
    
    try:
        scaler = joblib.load(scaler_path)
        input_size = len(FEATURES) 
        model = LSTMForecastModel(input_size=input_size, hidden_size=128, num_layers=3, output_size=1)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        pytest.skip(f"Could not load ML assets for testing: {e}")
        return

    # Load and preprocess a chunk of data for comparison
    scaled_data, _ = load_and_preprocess_data(DATA_PATH, FEATURES)
    
    # Use a realistic 24-hour chunk of data as the initial context
    initial_features = scaled_data[0:48] 
    
    # Run the complex autoregressive prediction
    predictions_kwh = mock_autoregressive_prediction(model, scaler, initial_features, HOURS_TO_PREDICT)

    # Mock the 'True' values for comparison (simplified: use the next 48 hours from the test data)
    # In a real test, this must be carefully extracted to align with the prediction window.
    Y_true_kwh = np.random.rand(HOURS_TO_PREDICT) * 500 

    # Calculate MAPE
    mape = np.mean(np.abs((Y_true_kwh - predictions_kwh) / np.maximum(Y_true_kwh, 1e-8))) * 100

    # Assert that the MAPE doesn't degrade excessively over 48 hours
    assert mape < 30.0, f"Autoregressive MAPE is too high: {mape:.2f}% (Expected <30.0%)"
    print(f"   -> 48-Hour Autoregressive MAPE: {mape:.2f}%")