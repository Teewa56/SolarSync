import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import os
import joblib
import logging
from datetime import datetime
from pathlib import Path
import json
from typing import Dict, Tuple, Optional
import optuna 

from models import LSTMForecastModel, GRUForecastModel
from data_loader import load_and_preprocess_data, create_sequences
from data_validator import DataValidator
from feature_engineering import FeatureEngineer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelTrainer:
    """Enhanced model training with experiment tracking and versioning"""
    
    def __init__(self, config: dict):
        self.config = config
        self.model_type = config['model_type']
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Experiment tracking
        self.experiment_id = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.experiment_dir = Path(f"experiments/{self.model_type}/{self.experiment_id}")
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Initialized trainer for {self.model_type}")
        logger.info(f"Using device: {self.device}")
        logger.info(f"Experiment directory: {self.experiment_dir}")
    
    def prepare_data(self) -> Tuple[DataLoader, DataLoader, DataLoader, object]:
        """
        Comprehensive data preparation pipeline
        
        Returns:
            train_loader, val_loader, test_loader, scaler
        """
        logger.info("="*60)
        logger.info("STEP 1: DATA PREPARATION")
        logger.info("="*60)
        
        # 1. Load raw data
        logger.info(f"Loading data from {self.config['data_path']}...")
        raw_data = pd.read_csv(self.config['data_path'])
        logger.info(f"Loaded {len(raw_data)} rows")
        
        # 2. Validate and clean data
        logger.info("Validating data quality...")
        validator = DataValidator(self.model_type)
        clean_data, validation_report = validator.validate_dataframe(raw_data)
        
        # Save validation report
        def convert_to_serializable(obj):
            """Convert numpy types to native Python types"""
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        with open(self.experiment_dir / 'validation_report.json', 'w') as f:
            json.dump(convert_to_serializable(validation_report), f, indent=2)
                
        logger.info(f"Data quality score: {validation_report['data_quality_score']:.2%}")
        
        # 3. Feature engineering
        logger.info("Engineering features...")
        engineer = FeatureEngineer(self.model_type)
        enriched_data = engineer.engineer_all_features(
            clean_data, 
            target_col=self.config['target_col']
        )
        
        logger.info(f"Features created: {enriched_data.shape[1]} columns")
        
        # Save feature list
        feature_list = enriched_data.columns.tolist()
        with open(self.experiment_dir / 'features.json', 'w') as f:
            json.dump(feature_list, f, indent=2)
        
        # 4. Select features for training
        # Use all numeric columns except timestamp and target
        feature_cols = [col for col in enriched_data.columns 
                       if col not in ['timestamp', self.config['target_col']]
                       and enriched_data[col].dtype in [np.float32, np.float64, np.int32, np.int64]]
        
        # Add target column at the end
        feature_cols.append(self.config['target_col'])
        
        logger.info(f"Training features: {len(feature_cols)-1}, Target: {self.config['target_col']}")
        
        # 5. Scale data
        from sklearn.preprocessing import MinMaxScaler
        scaler = MinMaxScaler(feature_range=(0, 1))
        
        data_array = enriched_data[feature_cols].values.astype(np.float32)
        scaled_data = scaler.fit_transform(data_array)
        
        logger.info(f"Data scaled to range [0, 1]")
        
        # 6. Create sequences
        logger.info(f"Creating sequences (length: {self.config['seq_length']})...")
        X, Y = create_sequences(scaled_data, self.config['seq_length'])
        
        logger.info(f"Sequences: X={X.shape}, Y={Y.shape}")
        
        # 7. Convert to tensors
        X_tensor = torch.from_numpy(X).float()
        Y_tensor = torch.from_numpy(Y).float().unsqueeze(1)
        
        # 8. Split data
        dataset = TensorDataset(X_tensor, Y_tensor)
        
        train_size = int(0.7 * len(dataset))
        val_size = int(0.15 * len(dataset))
        test_size = len(dataset) - train_size - val_size
        
        from torch.utils.data import random_split
        train_dataset, val_dataset, test_dataset = random_split(
            dataset, 
            [train_size, val_size, test_size],
            generator=torch.Generator().manual_seed(42)
        )
        
        # 9. Create data loaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=True,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config['batch_size'], 
            shuffle=False,
            num_workers=2,
            pin_memory=True if torch.cuda.is_available() else False
        )
        
        logger.info(f"Data split: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
        
        return train_loader, val_loader, test_loader, scaler, len(feature_cols)-1
    
    def build_model(self, input_size: int) -> nn.Module:
        """Build model based on configuration"""
        if self.config['architecture'] == 'lstm':
            model = LSTMForecastModel(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=1,
                dropout=self.config['dropout']
            )
        elif self.config['architecture'] == 'gru':
            model = GRUForecastModel(
                input_size=input_size,
                hidden_size=self.config['hidden_size'],
                num_layers=self.config['num_layers'],
                output_size=1,
                dropout=self.config['dropout']
            )
        else:
            raise ValueError(f"Unknown architecture: {self.config['architecture']}")
        
        model = model.to(self.device)
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        logger.info(f"Model: {self.config['architecture'].upper()}")
        logger.info(f"Total parameters: {total_params:,}")
        logger.info(f"Trainable parameters: {trainable_params:,}")
        
        return model
    
    def train_epoch(self, model, train_loader, criterion, optimizer) -> float:
        """Train for one epoch"""
        model.train()
        total_loss = 0.0
        
        for batch_X, batch_Y in train_loader:
            batch_X = batch_X.to(self.device)
            batch_Y = batch_Y.to(self.device)
            
            optimizer.zero_grad()
            outputs = model(batch_X)
            loss = criterion(outputs, batch_Y)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate_epoch(self, model, val_loader, criterion) -> float:
        """Validate for one epoch"""
        model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_Y in val_loader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def train(self, train_loader, val_loader, input_size) -> Tuple[nn.Module, Dict]:
        """
        Full training loop with early stopping and checkpointing
        """
        logger.info("="*60)
        logger.info("STEP 2: MODEL TRAINING")
        logger.info("="*60)
        
        # Build model
        model = self.build_model(input_size)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 1e-5)
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            verbose=True
        )
        
        # Training history
        history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': []
        }
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        logger.info(f"Training for {self.config['epochs']} epochs...")
        logger.info("="*60)
        
        for epoch in range(self.config['epochs']):
            # Train
            train_loss = self.train_epoch(model, train_loader, criterion, optimizer)
            
            # Validate
            val_loss = self.validate_epoch(model, val_loader, criterion)
            
            # Update scheduler
            scheduler.step(val_loss)
            
            # Record history
            history['train_loss'].append(train_loss)
            history['val_loss'].append(val_loss)
            history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                logger.info(
                    f"Epoch [{epoch+1:3d}/{self.config['epochs']}] "
                    f"Train Loss: {train_loss:.6f} | Val Loss: {val_loss:.6f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                
                # Save checkpoint
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                    'config': self.config
                }, self.experiment_dir / 'best_model.pth')
                
            else:
                patience_counter += 1
                
                if patience_counter >= self.config.get('early_stopping_patience', 15):
                    logger.info(f"\nEarly stopping at epoch {epoch+1}")
                    break
        
        logger.info("="*60)
        logger.info(f"Training completed!")
        logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch}")
        logger.info("="*60)
        
        # Load best model
        checkpoint = torch.load(self.experiment_dir / 'best_model.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model, history
    
    def evaluate(self, model, test_loader, scaler) -> Dict:
        """Evaluate model on test set"""
        logger.info("="*60)
        logger.info("STEP 3: MODEL EVALUATION")
        logger.info("="*60)
        
        model.eval()
        criterion = nn.MSELoss()
        
        all_predictions = []
        all_targets = []
        test_loss = 0.0
        
        with torch.no_grad():
            for batch_X, batch_Y in test_loader:
                batch_X = batch_X.to(self.device)
                batch_Y = batch_Y.to(self.device)
                
                outputs = model(batch_X)
                loss = criterion(outputs, batch_Y)
                test_loss += loss.item()
                
                all_predictions.extend(outputs.cpu().numpy())
                all_targets.extend(batch_Y.cpu().numpy())
        
        avg_test_loss = test_loss / len(test_loader)
        
        # Denormalize predictions
        predictions = np.array(all_predictions)
        targets = np.array(all_targets)
        
        # Calculate metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        
        mae = mean_absolute_error(targets, predictions)
        mse = mean_squared_error(targets, predictions)
        rmse = np.sqrt(mse)
        r2 = r2_score(targets, predictions)
        mape = np.mean(np.abs((targets - predictions) / (targets + 1e-8))) * 100
        
        metrics = {
            'test_loss': avg_test_loss,
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mape': float(mape)
        }
        
        logger.info("Test Metrics:")
        logger.info(f"  MSE Loss: {avg_test_loss:.6f}")
        logger.info(f"  MAE: {mae:.4f}")
        logger.info(f"  RMSE: {rmse:.4f}")
        logger.info(f"  R²: {r2:.4f}")
        logger.info(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def save_artifacts(self, model, scaler, metrics, history):
        """Save all training artifacts"""
        logger.info("="*60)
        logger.info("STEP 4: SAVING ARTIFACTS")
        logger.info("="*60)
        
        # Save model for production
        production_dir = Path(f"saved_models/{self.model_type}")
        production_dir.mkdir(parents=True, exist_ok=True)
        
        torch.save(
            model.state_dict(),
            production_dir / f'{self.model_type}_model.pth'
        )
        
        joblib.dump(scaler, production_dir / f'{self.model_type}_scaler.pkl')
        
        # Save to experiment directory
        joblib.dump(scaler, self.experiment_dir / 'scaler.pkl')
        
        # Save config
        with open(self.experiment_dir / 'config.json', 'w') as f:
            json.dump(self.config, f, indent=2)
        
        # Save metrics
        with open(self.experiment_dir / 'metrics.json', 'w') as f:
            json.dump(metrics, f, indent=2)
        
        # Save history
        with open(self.experiment_dir / 'history.json', 'w') as f:
            json.dump(history, f, indent=2)
        
        logger.info(f"✅ Model saved to: {production_dir}")
        logger.info(f"✅ Experiment artifacts saved to: {self.experiment_dir}")


import pandas as pd

def main():
    """Main training pipeline"""
    
    # Solar model configuration
    solar_config = {
        'model_type': 'solar',
        'architecture': 'lstm',
        'data_path': 'data/solar/merged_solar_data.csv',
        'target_col': 'energy_output',
        'seq_length': 24,
        'hidden_size': 128,
        'num_layers': 3,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'epochs': 100,
        'early_stopping_patience': 15
    }
    
    # Wind model configuration
    wind_config = {
        'model_type': 'wind',
        'architecture': 'gru',
        'data_path': 'data/wind/merged_wind_data.csv',
        'target_col': 'energy_output',
        'seq_length': 18,
        'hidden_size': 128,
        'num_layers': 2,
        'dropout': 0.2,
        'learning_rate': 0.002,
        'weight_decay': 1e-5,
        'batch_size': 64,
        'epochs': 80,
        'early_stopping_patience': 15
    }
    
    # Train both models
    for config in [solar_config, wind_config]:
        logger.info("\n" + "="*80)
        logger.info(f"TRAINING {config['model_type'].upper()} MODEL")
        logger.info("="*80 + "\n")
        
        trainer = ModelTrainer(config)
        
        # Prepare data
        train_loader, val_loader, test_loader, scaler, input_size = trainer.prepare_data()
        
        # Train model
        model, history = trainer.train(train_loader, val_loader, input_size)
        
        # Evaluate
        metrics = trainer.evaluate(model, test_loader, scaler)
        
        # Save artifacts
        trainer.save_artifacts(model, scaler, metrics, history)
        
        logger.info(f"\n✅ {config['model_type'].upper()} model training complete!\n")


if __name__ == '__main__':
    main()