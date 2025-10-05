import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataMerger:
    """Merge weather data with energy generation data for solar"""
    
    def __init__(self, weather_path: str, generation_path: str, output_path: str):
        self.weather_path = weather_path
        self.generation_path = generation_path
        self.output_path = output_path
    
    def load_and_inspect(self):
        """Load both datasets and show their structure"""
        logger.info("Loading datasets...")
        
        # Load weather data
        self.weather_df = pd.read_csv(self.weather_path)
        logger.info(f"\n{'='*60}")
        logger.info("WEATHER DATA")
        logger.info(f"{'='*60}")
        logger.info(f"Shape: {self.weather_df.shape}")
        logger.info(f"Columns: {self.weather_df.columns.tolist()}")
        logger.info(f"\nFirst few rows:")
        logger.info(f"\n{self.weather_df.head()}")
        
        # Load generation data
        self.generation_df = pd.read_csv(self.generation_path)
        logger.info(f"\n{'='*60}")
        logger.info("GENERATION DATA")
        logger.info(f"{'='*60}")
        logger.info(f"Shape: {self.generation_df.shape}")
        logger.info(f"Columns: {self.generation_df.columns.tolist()}")
        logger.info(f"\nFirst few rows:")
        logger.info(f"\n{self.generation_df.head()}")
    
    def merge_datasets(self, 
                       weather_time_col: str = 'timestamp',
                       generation_time_col: str = 'timestamp',
                       generation_value_col: str = 'energy_output',
                       merge_tolerance='1D'):
        """
        Merge weather and generation data on timestamp
        
        Parameters:
        -----------
        weather_time_col : str
            Name of timestamp column in weather data
        generation_time_col : str
            Name of timestamp column in generation data
        generation_value_col : str
            Name of energy output column in generation data
        merge_tolerance : str
            Time tolerance for merging (e.g., '1H' = 1 hour, '30min' = 30 minutes)
        """
        logger.info(f"\n{'='*60}")
        logger.info("MERGING DATASETS")
        logger.info(f"{'='*60}")
        
        # 1. Parse timestamps
        logger.info("Parsing timestamps...")
        self.weather_df[weather_time_col] = pd.to_datetime(
            self.weather_df[weather_time_col], 
            errors='coerce'
        )
        self.generation_df[generation_time_col] = pd.to_datetime(
            self.generation_df[generation_time_col], 
            errors='coerce'
        )
        
        # 2. Sort by timestamp
        self.weather_df = self.weather_df.sort_values(weather_time_col)
        self.generation_df = self.generation_df.sort_values(generation_time_col)
        
        # 3. Check date ranges
        logger.info(f"\nWeather data range: {self.weather_df[weather_time_col].min()} to {self.weather_df[weather_time_col].max()}")
        logger.info(f"Generation data range: {self.generation_df[generation_time_col].min()} to {self.generation_df[generation_time_col].max()}")
        
        # 4. Find overlapping period
        overlap_start = max(
            self.weather_df[weather_time_col].min(),
            self.generation_df[generation_time_col].min()
        )
        overlap_end = min(
            self.weather_df[weather_time_col].max(),
            self.generation_df[generation_time_col].max()
        )
        
        logger.info(f"\nOverlapping period: {overlap_start} to {overlap_end}")
        
        # 5. Filter to overlapping period
        weather_filtered = self.weather_df[
            (self.weather_df[weather_time_col] >= overlap_start) &
            (self.weather_df[weather_time_col] <= overlap_end)
        ].copy()
        
        generation_filtered = self.generation_df[
            (self.generation_df[generation_time_col] >= overlap_start) &
            (self.generation_df[generation_time_col] <= overlap_end)
        ].copy()
        
        logger.info(f"\nFiltered weather data: {len(weather_filtered)} rows")
        logger.info(f"Filtered generation data: {len(generation_filtered)} rows")
        
        # 6. Prepare generation data for merge (keep only timestamp and energy)
        generation_for_merge = generation_filtered[[generation_time_col, generation_value_col]].copy()
        generation_for_merge = generation_for_merge.rename(
            columns={generation_value_col: 'energy_output'}
        )
        
        # 7. Merge using pd.merge_asof for nearest timestamp matching
        logger.info(f"\nMerging with tolerance: {merge_tolerance}...")
        
        merged_df = pd.merge_asof(
            weather_filtered,
            generation_for_merge,
            left_on=weather_time_col,
            right_on=generation_time_col,
            direction='nearest',
            tolerance=pd.Timedelta(merge_tolerance)
        )
        
        # 8. Remove rows where merge failed (no matching generation data)
        before_dropna = len(merged_df)
        merged_df = merged_df.dropna(subset=['energy_output'])
        after_dropna = len(merged_df)
        
        logger.info(f"\nMerge complete!")
        logger.info(f"Rows before removing NaN: {before_dropna}")
        logger.info(f"Rows after removing NaN: {after_dropna}")
        logger.info(f"Successfully merged: {after_dropna} rows")
        logger.info(f"Merge success rate: {(after_dropna/before_dropna)*100:.2f}%")
        
        # 9. Rename timestamp column to standard 'timestamp'
        if weather_time_col != 'timestamp':
            merged_df = merged_df.rename(columns={weather_time_col: 'timestamp'})
        
        self.merged_df = merged_df
        
        return merged_df
    
    def validate_merged_data(self):
        """Validate the merged dataset"""
        logger.info(f"\n{'='*60}")
        logger.info("VALIDATING MERGED DATA")
        logger.info(f"{'='*60}")
        
        # Check for missing values
        missing = self.merged_df.isnull().sum()
        if missing.sum() > 0:
            logger.warning(f"\nMissing values found:")
            logger.warning(f"\n{missing[missing > 0]}")
        else:
            logger.info("✓ No missing values")
        
        # Check energy output range
        energy_stats = self.merged_df['energy_output'].describe()
        logger.info(f"\nEnergy Output Statistics:")
        logger.info(f"\n{energy_stats}")
        
        # Check for negative or zero energy during daytime
        if 'solar_radiation' in self.merged_df.columns or 'ALLSKY_SFC_SW_DWN' in self.merged_df.columns:
            solar_col = 'solar_radiation' if 'solar_radiation' in self.merged_df.columns else 'ALLSKY_SFC_SW_DWN'
            daytime = self.merged_df[self.merged_df[solar_col] > 100]
            zero_energy_daytime = (daytime['energy_output'] <= 0).sum()
            if zero_energy_daytime > 0:
                logger.warning(f"⚠ Found {zero_energy_daytime} rows with zero energy during daytime (solar > 100)")
        
        # Final shape
        logger.info(f"\nFinal merged dataset shape: {self.merged_df.shape}")
        logger.info(f"Columns: {self.merged_df.columns.tolist()}")
    
    def save_merged_data(self):
        """Save the merged dataset"""
        output_path = Path(self.output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.merged_df.to_csv(output_path, index=False)
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ Merged data saved to: {output_path}")
        logger.info(f"{'='*60}\n")
    
    def run_full_pipeline(self, 
                          weather_time_col='timestamp',
                          generation_time_col='timestamp', 
                          generation_value_col='energy_output',
                          merge_tolerance='1H'):
        """Run the complete merge pipeline"""
        
        # Step 1: Load and inspect
        self.load_and_inspect()
        
        # Step 2: Merge
        merged_df = self.merge_datasets(
            weather_time_col=weather_time_col,
            generation_time_col=generation_time_col,
            generation_value_col=generation_value_col,
            merge_tolerance=merge_tolerance
        )
        
        # Step 3: Validate
        self.validate_merged_data()
        
        # Step 4: Save
        self.save_merged_data()
        
        return merged_df


# Example usage
if __name__ == "__main__":
    
    # CONFIGURATION - UPDATE THESE PATHS
    WEATHER_DATA_PATH = 'data/wind/weather_wind_data/openmeteo_historical.csv'
    GENERATION_DATA_PATH = 'data/wind/plant_generation_data/T1.csv'  # UPDATE THIS
    OUTPUT_DATA_PATH = 'data/wind/merged_wind_data.csv'
    
    # Column name mapping - UPDATE THESE IF YOUR COLUMNS HAVE DIFFERENT NAMES
    WEATHER_TIME_COL = 'timestamp'  # or 'date', 'datetime', etc.
    GENERATION_TIME_COL = 'timestamp'  # or 'date', 'datetime', etc.
    GENERATION_VALUE_COL = 'power'  # or 'generation', 'output', 'AC_POWER', etc.
    
    # Time tolerance for matching (how close timestamps need to be)
    MERGE_TOLERANCE = '1D'  # 1 day
    
    # Create merger and run
    merger = DataMerger(
        weather_path=WEATHER_DATA_PATH,
        generation_path=GENERATION_DATA_PATH,
        output_path=OUTPUT_DATA_PATH
    )
    
    merged_data = merger.run_full_pipeline(
        weather_time_col=WEATHER_TIME_COL,
        generation_time_col=GENERATION_TIME_COL,
        generation_value_col=GENERATION_VALUE_COL,
        merge_tolerance=MERGE_TOLERANCE
    )
    
    print("\n✓ Merge complete! Use the merged data for training:")
    print(f"   {OUTPUT_DATA_PATH}")