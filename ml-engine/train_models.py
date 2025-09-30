import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import os
import sys
import joblib
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Setup paths
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models import LSTMForecastModel, GRUForecastModel
from data_loader import load_and_preprocess_data, create_sequences

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# --- Configuration ---
class ModelConfig:
    """Configuration for model training"""
    def __init__(self, model_type: str):
        self.model_type = model_type
        
        if model_type == 'solar':
            self.seq_length = 24
            self.hidden_size = 128
            self.num_layers = 3
            self.features = ['solar_irradiance', 'temperature', 'cloud_cover', 'energy_output']
            self.data_path = 'data/solar/solar_history.csv'
            self.epochs = 100
            self.learning_rate = 0.001
            self.batch_size = 64
            self.architecture = 'lstm'
            
        elif model_type == 'wind':
            self.seq_length = 18
            self.hidden_size = 128
            self.num_layers = 2
            self.features = ['wind_speed', 'wind_direction', 'pressure', 'energy_output']
            self.data_path = 'data/wind/wind_history.csv'
            self.epochs = 80
            self.learning_rate = 0.002
            self.batch_size = 64
            self.architecture = 'gru'
        
        self.output_size = 1
        self.dropout = 0.2
        self.weight_decay = 1e-5
        self.early_stopping_patience = 15
        self.save_dir = 'saved_models'


def train_model(config: ModelConfig):
    """
    Main training function for solar/wind models
    
    Args:
        config: ModelConfig object with training parameters
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting {config.model_type.upper()} Model Training")
    logger.info(f"{'='*60}\n")
    
    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)
    
    # 1. Load and preprocess data
    logger.info("Loading and preprocessing data...")
    scaled_data, scaler = load_and_preprocess_data(config.data_path, config.features)
    
    if scaled_data.size == 0:
        logger.error(f"No data loaded from {config.data_path}")
        logger.info("Please ensure the data file exists and contains the required columns")
        return None
    
    logger.info(f"Data shape: {scaled_data.shape}")
    
    # 2. Create sequences
    logger.info(f"Creating sequences (length: {config.seq_length})...")
    X, Y = create_sequences(scaled_data, config.seq_length)
    
    logger.info(f"Sequences created: X={X.shape}, Y={Y.shape}")
    
    # 3. Convert to PyTorch tensors
    X_tensor = torch.from_numpy(X).float()
    Y_tensor = torch.from_numpy(Y).float().unsqueeze(1)
    
    # 4. Split data (80% train, 10% validation, 10% test)
    dataset = TensorDataset(X_tensor, Y_tensor)
    train_size = int(0.8 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, 
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False)
    
    logger.info(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    
    # 5. Initialize model
    input_size = X.shape[2]
    
    if config.architecture == 'lstm':
        model = LSTMForecastModel(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            dropout=config.dropout
        )
    else:  # GRU
        model = GRUForecastModel(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            output_size=config.output_size,
            dropout=config.dropout
        )
    
    logger.info(f"Model architecture: {config.architecture.upper()}")
    logger.info(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # 7. Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    logger.info(f"\nStarting training for {config.epochs} epochs...")
    logger.info(f"{'='*60}\n")
    
    for epoch in range(config.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        # Logging
        if (epoch + 1) % 10 == 0 or epoch == 0:
            logger.info(
                f"Epoch [{epoch+1}/{config.epochs}] "
                f"Train Loss: {avg_train_loss:.6f} | "
                f"Val Loss: {avg_val_loss:.6f}"
            )
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            
            # Save best model
            best_model_path = os.path.join(
                config.save_dir, 
                f'{config.model_type}_model_best.pth'
            )
            torch.save(model.state_dict(), best_model_path)
            
        else:
            patience_counter += 1
            
            if patience_counter >= config.early_stopping_patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch+1}")
                break
    
    # 8. Load best model and evaluate on test set
    logger.info("\nEvaluating best model on test set...")
    model.load_state_dict(torch.load(best_model_path))
    model.eval()
    
    test_loss = 0.0
    with torch.no_grad():
        for batch_X, batch_Y in test_loader:
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            test_loss += loss.item()
    
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Test Loss: {avg_test_loss:.6f}")
    
    # 9. Save final model and scaler
    final_model_path = os.path.join(config.save_dir, f'{config.model_type}_model.pth')
    scaler_path = os.path.join(config.save_dir, f'{config.model_type}_scaler.pkl')
    
    torch.save(model.state_dict(), final_model_path)
    joblib.dump(scaler, scaler_path)
    
    logger.info(f"\n{'='*60}")
    logger.info(f"Training completed for {config.model_type.upper()}")
    logger.info(f"Model saved to: {final_model_path}")
    logger.info(f"Scaler saved to: {scaler_path}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    logger.info(f"Final test loss: {avg_test_loss:.6f}")
    logger.info(f"{'='*60}\n")
    
    return model, scaler, {'train': train_losses, 'val': val_losses, 'test': avg_test_loss}


def main():
    """Main entry point for training"""
    parser = argparse.ArgumentParser(description='Train SolarSync ML Models')
    parser.add_argument(
        '--model',
        type=str,
        choices=['solar', 'wind', 'both'],
        default='both',
        help='Which model to train'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        help='Number of training epochs (overrides default)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='data',
        help='Directory containing training data'
    )
    
    args = parser.parse_args()
    
    logger.info("\n" + "="*60)
    logger.info("SolarSync ML Training Pipeline")
    logger.info("="*60 + "\n")
    
    # Train solar model
    if args.model in ['solar', 'both']:
        solar_config = ModelConfig('solar')
        if args.epochs:
            solar_config.epochs = args.epochs
        train_model(solar_config)
    
    # Train wind model
    if args.model in ['wind', 'both']:
        wind_config = ModelConfig('wind')
        if args.epochs:
            wind_config.epochs = args.epochs
        train_model(wind_config)
    
    logger.info("\n" + "="*60)
    logger.info("All training completed successfully!")
    logger.info("="*60 + "\n")


if __name__ == '__main__':
    main()