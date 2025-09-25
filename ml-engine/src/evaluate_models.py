# ml-engine/evaluate_models.py
import torch
import numpy as np
import pandas as pd
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error
from data_loader import load_and_preprocess_data, create_sequences
from models import LSTMForecastModel # Re-use the model definition
import argparse
import os
import math

# --- Metrics Function ---

def mean_absolute_percentage_error(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error (MAPE)."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    # Avoid division by zero by setting a floor for y_true
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-8))) * 100

# --- Evaluation Logic ---

def evaluate_model_performance(model_type: str, data_path: str, features: list):
    
    print(f"\n--- Evaluating {model_type.upper()} Model ---")
    
    # 1. Load Assets
    save_dir = 'ml-engine/saved_models'
    model_path = os.path.join(save_dir, f'{model_type}_model.pth')
    scaler_path = os.path.join(save_dir, f'{model_type}_scaler.pkl')

    try:
        scaler = joblib.load(scaler_path)
        # Determine input size based on features (target is the last feature)
        input_size = len(features) 
        model = LSTMForecastModel(input_size=input_size, hidden_size=128, num_layers=3, output_size=1)
        model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()
    except Exception as e:
        print(f"Error loading assets: {e}. Ensure training has run successfully.")
        return

    # 2. Data Preparation for Testing
    scaled_data, _ = load_and_preprocess_data(data_path, features)
    if scaled_data.size == 0: return

    # Assuming a hardcoded sequence length from training
    SEQ_LENGTH = 24 if model_type == 'solar' else 18
    X_test, Y_test_scaled = create_sequences(scaled_data, SEQ_LENGTH)
    
    X_tensor = torch.from_numpy(X_test).float()

    # 3. Prediction
    with torch.no_grad():
        predictions_scaled = model(X_tensor).numpy()

    # 4. Denormalization (Crucial step to get back to KWh)
    target_index = scaler.n_features_in_ - 1
    
    # Denormalize targets (true values)
    inverse_input_true = np.zeros((Y_test_scaled.shape[0], scaler.n_features_in_))
    inverse_input_true[:, target_index] = Y_test_scaled.flatten()
    Y_test_kwh = scaler.inverse_transform(inverse_input_true)[:, target_index]
    
    # Denormalize predictions
    inverse_input_pred = np.zeros((predictions_scaled.shape[0], scaler.n_features_in_))
    inverse_input_pred[:, target_index] = predictions_scaled.flatten()
    Y_pred_kwh = scaler.inverse_transform(inverse_input_pred)[:, target_index]

    # 5. Calculate Metrics
    mape = mean_absolute_percentage_error(Y_test_kwh, Y_pred_kwh)
    rmse = math.sqrt(mean_squared_error(Y_test_kwh, Y_pred_kwh))
    mae = mean_absolute_error(Y_test_kwh, Y_pred_kwh)
    
    print(f"✅ Evaluation Results for {model_type.upper()}:")
    print(f"   - MAPE: {mape:.2f}% (Target: 80-85%)")
    print(f"   - RMSE: {rmse:.2f} KWh")
    print(f"   - MAE: {mae:.2f} KWh")
    print(f"   - Total Test Samples: {len(Y_test_kwh)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate SolarSync ML Models.')
    parser.add_argument('--model_type', type=str, required=True, choices=['solar', 'wind'], help='Model to evaluate (solar or wind).')
    parser.add_argument('--test_period', type=str, default='2024', help='Year/period for test data path (e.g., 2024).')
    args = parser.parse_args()
    
    # Define features based on model type (must match training features)
    if args.model_type == 'solar':
        FEATURES = ['solar_irradiance', 'temperature', 'cloud_cover', 'energy_output']
        DATA_PATH = f'data/solar/solar_test_{args.test_period}.csv'
    elif args.model_type == 'wind':
        FEATURES = ['wind_speed', 'wind_direction', 'pressure', 'energy_output']
        DATA_PATH = f'data/wind/wind_test_{args.test_period}.csv'
    else:
        FEATURES = []
        DATA_PATH = ''

    if FEATURES:
        evaluate_model_performance(args.model_type, DATA_PATH, FEATURES)