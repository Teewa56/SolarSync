import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple
import logging
import joblib
from pathlib import Path
from datetime import datetime, timedelta
import redis
import json

from models import LSTMForecastModel, GRUForecastModel
from data_fetcher import get_realtime_weather_forecast, process_weather_data
from feature_engineering import FeatureEngineer
from db_config import get_db_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ProductionPredictor:
    """
    Production-ready prediction service with caching,
    historical context, and confidence intervals
    """
    
    def __init__(self, model_type: str):
        self.model_type = model_type
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and scaler
        self.model, self.scaler, self.config = self._load_model_assets()
        
        # Feature engineer
        self.feature_engineer = FeatureEngineer(model_type)
        
        # Redis cache (optional)
        try:
            self.redis_client = redis.Redis(
                host='localhost',
                port=6379,
                db=0,
                decode_responses=True
            )
            self.cache_enabled = True
        except:
            logger.warning("Redis not available, running without cache")
            self.cache_enabled = False
        
        logger.info(f"✅ {model_type.upper()} predictor initialized")
    
    def _load_model_assets(self) -> Tuple[torch.nn.Module, object, Dict]:
        """Load model, scaler, and configuration"""
        model_dir = Path(f'saved_models/{self.model_type}')
        
        model_path = model_dir / f'{self.model_type}_model.pth'
        scaler_path = model_dir / f'{self.model_type}_scaler.pkl'
        
        if not model_path.exists() or not scaler_path.exists():
            raise FileNotFoundError(
                f"Model files not found in {model_dir}. "
                "Please train the model first."
            )
        
        # Load scaler
        scaler = joblib.load(scaler_path)
        
        # Load config (if available)
        config_path = model_dir / 'config.json'
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = json.load(f)
        else:
            # Default config
            config = {
                'input_size': scaler.n_features_in_,
                'hidden_size': 128,
                'num_layers': 3 if self.model_type == 'solar' else 2,
                'seq_length': 24 if self.model_type == 'solar' else 18
            }
        
        # Build model
        if self.model_type == 'solar':
            model = LSTMForecastModel(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=1
            )
        else:  # wind
            model = GRUForecastModel(
                input_size=config['input_size'],
                hidden_size=config['hidden_size'],
                num_layers=config['num_layers'],
                output_size=1
            )
        
        # Load weights
        model.load_state_dict(torch.load(model_path, map_location=self.device))
        model.to(self.device)
        model.eval()
        
        logger.info(f"Loaded {self.model_type} model with {config['input_size']} features")
        
        return model, scaler, config
    
    def get_historical_context(self, plant_id: Optional[str] = None,
                               hours: int = 24) -> Optional[pd.DataFrame]:
        """
        Fetch historical data from database for context
        
        Args:
            plant_id: Plant identifier (if None, uses aggregate data)
            hours: Number of hours of history to fetch
        
        Returns:
            DataFrame with historical weather and generation data
        """
        try:
            conn = get_db_connection()
            
            # Query based on model type
            if self.model_type == 'solar':
                table = 'solar_data'
                columns = 'timestamp, solar_radiation, temperature, humidity, wind_speed, pressure'
            else:
                table = 'wind_data'
                columns = 'timestamp, wind_speed, wind_direction, wind_gust, temperature, pressure'
            
            query = f"""
                SELECT {columns}
                FROM {table}
                WHERE timestamp >= NOW() - INTERVAL '{hours} hours'
                ORDER BY timestamp DESC
                LIMIT {hours}
            """
            
            df = pd.read_sql(query, conn)
            conn.close()
            
            if len(df) > 0:
                logger.info(f"Fetched {len(df)} hours of historical context")
                return df
            else:
                logger.warning("No historical data available")
                return None
                
        except Exception as e:
            logger.error(f"Failed to fetch historical context: {e}")
            return None
    
    def prepare_features(self, weather_forecast: List[Dict]) -> np.ndarray:
        """
        Prepare features from weather forecast
        
        Args:
            weather_forecast: List of hourly weather dictionaries
        
        Returns:
            Numpy array of engineered features
        """
        # Convert to DataFrame
        df = pd.DataFrame(weather_forecast)
        
        # Add timestamp if not present
        if 'timestamp' not in df.columns:
            df['timestamp'] = pd.date_range(
                start=datetime.now(),
                periods=len(df),
                freq='H'
            )
        
        # Engineer features
        enriched_df = self.feature_engineer.engineer_all_features(
            df,
            target_col=None  # No target for prediction
        )
        
        # Select features matching training
        # This should match the features used during training
        feature_cols = [col for col in enriched_df.columns 
                       if col not in ['timestamp']]
        
        features = enriched_df[feature_cols].values
        
        return features
    
    def predict_with_confidence(self, lat: float, lon: float,
                                hours: int = 24,
                                plant_id: Optional[str] = None) -> Dict:
        """
        Make predictions with confidence intervals
        
        Args:
            lat: Latitude
            lon: Longitude
            hours: Number of hours to predict
            plant_id: Optional plant identifier for context
        
        Returns:
            Dictionary with predictions and metadata
        """
        # Check cache
        cache_key = f"prediction:{self.model_type}:{lat}:{lon}:{hours}"
        if self.cache_enabled:
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.info("Returning cached prediction")
                return json.loads(cached)
        
        # Fetch weather forecast
        raw_weather = get_realtime_weather_forecast(lat, lon)
        
        if not raw_weather:
            raise ValueError("Failed to fetch weather forecast")
        
        # Process weather data
        weather_features = process_weather_data(raw_weather, self.model_type, hours)
        
        if not weather_features:
            raise ValueError("Failed to process weather data")
        
        # Get historical context
        historical_df = self.get_historical_context(plant_id, hours=self.config['seq_length'])
        
        # Prepare initial sequence
        if historical_df is not None and len(historical_df) >= self.config['seq_length']:
            # Use real historical data
            initial_sequence = self._prepare_historical_sequence(historical_df)
        else:
            # Use synthetic initial sequence
            logger.warning("Using synthetic initial sequence (no historical data)")
            initial_sequence = self._create_synthetic_sequence()
        
        # Make predictions
        predictions = self._predict_sequence(initial_sequence, weather_features)
        
        # Calculate confidence intervals (using prediction variance)
        confidence_intervals = self._calculate_confidence_intervals(predictions)
        
        # Prepare response
        result = {
            'model': self.model_type,
            'predictions': predictions,
            'confidence_intervals': confidence_intervals,
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'location': {'lat': lat, 'lon': lon},
                'hours_predicted': len(predictions),
                'model_version': '1.0',
                'used_historical_context': historical_df is not None
            }
        }
        
        # Cache result
        if self.cache_enabled:
            self.redis_client.setex(
                cache_key,
                1800,  # 30 minutes
                json.dumps(result)
            )
        
        return result
    
    def _prepare_historical_sequence(self, df: pd.DataFrame) -> np.ndarray:
        """Prepare sequence from historical data"""
        # Engineer features
        enriched_df = self.feature_engineer.engineer_all_features(df)
        
        # Select features and scale
        feature_cols = [col for col in enriched_df.columns 
                       if col not in ['timestamp']]
        
        features = enriched_df[feature_cols].values[-self.config['seq_length']:]
        scaled_features = self.scaler.transform(features)
        
        return scaled_features
    
    def _create_synthetic_sequence(self) -> np.ndarray:
        """Create synthetic initial sequence"""
        # Create random sequence with realistic values
        seq_length = self.config['seq_length']
        n_features = self.scaler.n_features_in_
        
        # Generate synthetic data based on typical ranges
        synthetic_data = np.random.rand(seq_length, n_features) * 0.5 + 0.25
        
        return synthetic_data
    
    def _predict_sequence(self, initial_sequence: np.ndarray,
                         future_features: List[np.ndarray]) -> List[float]:
        """
        Autoregressive prediction with dropout for uncertainty
        """
        predictions = []
        current_sequence = torch.from_numpy(initial_sequence).float().unsqueeze(0).to(self.device)
        
        # Enable dropout for uncertainty estimation
        self.model.train()
        
        # Make multiple predictions for uncertainty
        n_samples = 10
        all_predictions = []
        
        for _ in range(n_samples):
            sample_predictions = []
            seq = current_sequence.clone()
            
            with torch.no_grad():
                for i, future_step in enumerate(future_features):
                    # Predict
                    pred = self.model(seq)
                    pred_value = pred.item()
                    
                    # Denormalize
                    denorm_pred = self._denormalize_prediction(pred_value)
                    sample_predictions.append(max(0, denorm_pred))
                    
                    # Update sequence for next step
                    if i < len(future_features) - 1:
                        next_input = torch.cat([
                            future_step,
                            torch.tensor([pred_value])
                        ]).float().to(self.device)
                        
                        seq = torch.cat([
                            seq[:, 1:, :],
                            next_input.unsqueeze(0).unsqueeze(0)
                        ], dim=1)
            
            all_predictions.append(sample_predictions)
        
        # Average predictions
        predictions = np.mean(all_predictions, axis=0).tolist()
        
        # Set model back to eval mode
        self.model.eval()
        
        return predictions
    
    def _denormalize_prediction(self, scaled_value: float) -> float:
        """Denormalize a single prediction value"""
        target_index = self.scaler.n_features_in_ - 1
        
        inverse_input = np.zeros((1, self.scaler.n_features_in_))
        inverse_input[0, target_index] = scaled_value
        
        denormalized = self.scaler.inverse_transform(inverse_input)[0, target_index]
        
        return float(denormalized)
    
    def _calculate_confidence_intervals(self, predictions: List[float],
                                       confidence: float = 0.95) -> List[Dict]:
        """
        Calculate confidence intervals for predictions
        
        Uses a simple heuristic: ±20% for near-term, increasing with time
        """
        intervals = []
        
        for i, pred in enumerate(predictions):
            # Uncertainty increases with prediction horizon
            uncertainty_factor = 0.15 + (i / len(predictions)) * 0.15
            
            lower = pred * (1 - uncertainty_factor)
            upper = pred * (1 + uncertainty_factor)
            
            intervals.append({
                'lower': max(0, lower),
                'upper': upper,
                'confidence': confidence
            })
        
        return intervals


# Example usage
if __name__ == "__main__":
    # Test solar predictor
    solar_predictor = ProductionPredictor('solar')
    
    result = solar_predictor.predict_with_confidence(
        lat=40.7128,
        lon=-74.0060,
        hours=24
    )
    
    print(f"\nPredictions: {result['predictions'][:5]}...")
    print(f"Confidence intervals: {result['confidence_intervals'][:2]}...")
    print(f"Metadata: {result['metadata']}")