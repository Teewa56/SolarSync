import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
import logging
from scipy import stats

logger = logging.getLogger(__name__)

class DataValidator:
    """Comprehensive data validation and cleaning"""
    
    def __init__(self, data_type: str = 'solar'):
        self.data_type = data_type
        
        # Define valid ranges for each feature
        self.valid_ranges = {
            'solar': {
                'solar_radiation': (0, 1500),  # W/m²
                'temperature': (-50, 60),       # °C
                'humidity': (0, 100),           # %
                'wind_speed': (0, 50),          # m/s
                'pressure': (900, 1100),        # hPa
                'energy_output': (0, 10000)     # kWh (adjust based on capacity)
            },
            'wind': {
                'wind_speed': (0, 50),          # m/s
                'wind_direction': (0, 360),     # degrees
                'wind_gust': (0, 70),           # m/s
                'temperature': (-50, 60),       # °C
                'pressure': (900, 1100),        # hPa
                'energy_output': (0, 5000)      # kWh
            }
        }
    
    def validate_dataframe(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """
        Comprehensive data validation and cleaning
        
        Returns:
            Tuple of (cleaned_df, validation_report)
        """
        report = {
            'original_rows': len(df),
            'issues_found': [],
            'rows_removed': 0,
            'values_imputed': 0,
            'outliers_detected': 0
        }
        
        df_clean = df.copy()
        
        # 1. Check for missing timestamps
        if 'timestamp' in df_clean.columns:
            missing_timestamps = df_clean['timestamp'].isnull().sum()
            if missing_timestamps > 0:
                report['issues_found'].append(f"Missing timestamps: {missing_timestamps}")
                df_clean = df_clean.dropna(subset=['timestamp'])
                report['rows_removed'] += missing_timestamps
        
        # 2. Check for duplicate timestamps
        duplicates = df_clean.duplicated(subset=['timestamp'], keep='first').sum()
        if duplicates > 0:
            report['issues_found'].append(f"Duplicate timestamps: {duplicates}")
            df_clean = df_clean.drop_duplicates(subset=['timestamp'], keep='first')
            report['rows_removed'] += duplicates
        
        # 3. Validate feature ranges
        ranges = self.valid_ranges.get(self.data_type, {})
        for feature, (min_val, max_val) in ranges.items():
            if feature in df_clean.columns:
                # Detect out-of-range values
                out_of_range = (
                    (df_clean[feature] < min_val) | 
                    (df_clean[feature] > max_val)
                ).sum()
                
                if out_of_range > 0:
                    report['issues_found'].append(
                        f"{feature}: {out_of_range} values out of range [{min_val}, {max_val}]"
                    )
                    # Clip values to valid range
                    df_clean[feature] = df_clean[feature].clip(min_val, max_val)
        
        # 4. Detect and handle outliers using IQR method
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col in ranges:
                outliers = self._detect_outliers_iqr(df_clean[col])
                if outliers.sum() > 0:
                    report['outliers_detected'] += outliers.sum()
                    # Replace outliers with median
                    median_val = df_clean.loc[~outliers, col].median()
                    df_clean.loc[outliers, col] = median_val
                    report['values_imputed'] += outliers.sum()
        
        # 5. Handle missing values
        for col in numeric_cols:
            missing = df_clean[col].isnull().sum()
            if missing > 0:
                report['issues_found'].append(f"{col}: {missing} missing values")
                
                # Forward fill then backward fill
                df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                
                # If still missing, use median
                if df_clean[col].isnull().sum() > 0:
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                
                report['values_imputed'] += missing
        
        # 6. Check for monotonic timestamp
        if 'timestamp' in df_clean.columns:
            df_clean = df_clean.sort_values('timestamp')
            non_monotonic = (df_clean['timestamp'].diff().dt.total_seconds() <= 0).sum()
            if non_monotonic > 0:
                report['issues_found'].append(f"Non-monotonic timestamps: {non_monotonic}")
        
        # 7. Check data frequency consistency
        if 'timestamp' in df_clean.columns and len(df_clean) > 1:
            time_diffs = df_clean['timestamp'].diff().dt.total_seconds()
            expected_freq = time_diffs.mode()[0] if len(time_diffs.mode()) > 0 else 3600
            irregular = (time_diffs != expected_freq).sum() - 1  # -1 for first NaN
            if irregular > 0:
                report['issues_found'].append(
                    f"Irregular time intervals: {irregular} gaps (expected {expected_freq}s)"
                )
        
        report['final_rows'] = len(df_clean)
        report['data_quality_score'] = self._calculate_quality_score(report)
        
        logger.info(f"Data validation complete. Quality score: {report['data_quality_score']:.2%}")
        
        return df_clean, report
    
    def _detect_outliers_iqr(self, series: pd.Series, multiplier: float = 1.5) -> pd.Series:
        """Detect outliers using Interquartile Range method"""
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - multiplier * IQR
        upper_bound = Q3 + multiplier * IQR
        
        return (series < lower_bound) | (series > upper_bound)
    
    def _calculate_quality_score(self, report: Dict) -> float:
        """Calculate overall data quality score (0-1)"""
        if report['original_rows'] == 0:
            return 0.0
        
        # Penalize for issues
        retention_rate = report['final_rows'] / report['original_rows']
        imputation_rate = 1 - (report['values_imputed'] / (report['original_rows'] * 10))  # Assume ~10 features
        outlier_rate = 1 - (report['outliers_detected'] / (report['original_rows'] * 10))
        
        quality_score = (retention_rate * 0.4) + (imputation_rate * 0.3) + (outlier_rate * 0.3)
        return max(0.0, min(1.0, quality_score))
    
    def validate_realtime_data(self, data: Dict) -> Tuple[bool, str]:
        """
        Validate real-time weather data from API
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = {
            'solar': ['temp', 'humidity', 'clouds'],
            'wind': ['wind_speed', 'wind_deg', 'pressure']
        }
        
        fields = required_fields.get(self.data_type, [])
        
        for field in fields:
            if field not in data:
                return False, f"Missing required field: {field}"
        
        # Validate ranges
        ranges = self.valid_ranges.get(self.data_type, {})
        mapping = {
            'temp': 'temperature',
            'clouds': 'cloud_cover',
            'wind_deg': 'wind_direction'
        }
        
        for field in fields:
            range_key = mapping.get(field, field)
            if range_key in ranges:
                min_val, max_val = ranges[range_key]
                value = data[field]
                if not (min_val <= value <= max_val):
                    return False, f"{field} value {value} out of range [{min_val}, {max_val}]"
        
        return True, "Valid"


# Example usage
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Test with sample data
    sample_data = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=100, freq='H'),
        'solar_radiation': np.random.uniform(200, 1000, 100),
        'temperature': np.random.uniform(15, 35, 100),
        'humidity': np.random.uniform(30, 80, 100),
        'energy_output': np.random.uniform(50, 500, 100)
    })
    
    # Add some issues for testing
    sample_data.loc[10, 'temperature'] = 150  # Outlier
    sample_data.loc[20:22, 'humidity'] = np.nan  # Missing values
    
    validator = DataValidator('solar')
    clean_data, report = validator.validate_dataframe(sample_data)
    
    print("\nValidation Report:")
    for key, value in report.items():
        print(f"{key}: {value}")