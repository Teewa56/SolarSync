# ml-engine/train_solar_model.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from data_loader import load_and_preprocess_data, create_sequences
from models import LSTMForecastModel 
import os
import joblib 
import argparse # Keep argparse for standalone execution

# --- SOLAR MODEL CONFIGURATION ---
SEQ_LENGTH = 24             
HIDDEN_SIZE = 128           
NUM_LAYERS = 3              
OUTPUT_SIZE = 1             
LR = 0.001
SOLAR_FEATURES = ['solar_irradiance', 'temperature', 'cloud_cover', 'energy_output']
MODEL_NAME = 'solar'
DEFAULT_EPOCHS = 100
DEFAULT_DATA_PATH = 'data/solar/solar_history.csv'

class SimpleMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
    def forward(self, input, target):
        return self.mse(input, target)

def run_solar_training(data_path: str = DEFAULT_DATA_PATH, epochs: int = DEFAULT_EPOCHS):
    """Orchestrates the training process for the Solar LSTM model."""
    print(f"\n--- Starting Training for {MODEL_NAME.upper()} Model (LSTM) ---")
    
    # 1. Data Preparation
    scaled_data, scaler = load_and_preprocess_data(data_path, SOLAR_FEATURES)
    if scaled_data.size == 0: return
        
    X_sequences, Y_targets = create_sequences(scaled_data, SEQ_LENGTH)
    
    X_tensor = torch.from_numpy(X_sequences).float()
    Y_tensor = torch.from_numpy(Y_targets).float().unsqueeze(1)
    
    train_size = int(0.8 * len(X_tensor))
    X_train, Y_train = X_tensor[:train_size], Y_tensor[:train_size]
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=64, shuffle=True)

    # 2. Model, Loss, and Optimizer
    input_size = X_train.shape[2]
    model = LSTMForecastModel(input_size, HIDDEN_SIZE, NUM_LAYERS, OUTPUT_SIZE)
    criterion = SimpleMSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    
    # 3. Training Loop (Placeholder for actual loop)
    model.train()
    for epoch in range(epochs):
        # ... Training steps ...
        if (epoch + 1) % (epochs // 5) == 0:
             print(f'  [Solar] Epoch {epoch+1}/{epochs} simulated.') 
    
    print(f"Training completed for {MODEL_NAME}.")

    # 4. Save Model and Scaler
    save_dir = 'ml-engine/saved_models'
    os.makedirs(save_dir, exist_ok=True)
    
    torch.save(model.state_dict(), os.path.join(save_dir, f'{MODEL_NAME}_model.pth'))
    joblib.dump(scaler, os.path.join(save_dir, f'{MODEL_NAME}_scaler.pkl'))
    
    print(f"Solar Model and Scaler saved to {save_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Solar Energy Forecasting Model.')
    parser.add_argument('--data_path', type=str, default=DEFAULT_DATA_PATH, help='Path to the historical solar data CSV.')
    parser.add_argument('--epochs', type=int, default=DEFAULT_EPOCHS, help='Number of training epochs.')
    args = parser.parse_args()
    
    run_solar_training(args.data_path, args.epochs)