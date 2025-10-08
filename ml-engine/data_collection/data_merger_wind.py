import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataMerger:
    """Merge weather data with energy generation data for wind"""
    
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
    
    def merge_datasets(self, weather_df, gen_df, weather_time_col, gen_time_col, merge_tolerance="1h"):
    
        logger.info("\n============================================================")
        logger.info("MERGING DATASETS (MATCHING BY MONTH, DAY, TIME — IGNORING YEAR)")
        logger.info("============================================================")

        # Parse timestamps
        weather_df[weather_time_col] = pd.to_datetime(weather_df[weather_time_col], errors="coerce")
        gen_df[gen_time_col] = pd.to_datetime(gen_df[gen_time_col], errors="coerce")

        # Drop any invalid timestamps
        weather_df = weather_df.dropna(subset=[weather_time_col])
        gen_df = gen_df.dropna(subset=[gen_time_col])

        # Extract comparable time components
        for df, name in [(weather_df, "Weather"), (gen_df, "Generation")]:
            time_col = weather_time_col if name == "Weather" else gen_time_col
            df["month"] = df[time_col].dt.month
            df["day"] = df[time_col].dt.day
            df["hour"] = df[time_col].dt.hour
            df["minute"] = df[time_col].dt.minute

        logger.info(f"Weather data range (month/day): "
                    f"{weather_df['month'].min()}/{weather_df['day'].min()} "
                    f"to {weather_df['month'].max()}/{weather_df['day'].max()}")
        logger.info(f"Generation data range (month/day): "
                    f"{gen_df['month'].min()}/{gen_df['day'].min()} "
                    f"to {gen_df['month'].max()}/{gen_df['day'].max()}")

        # ✅ Create proper datetime merge keys ignoring the year (fixes MergeError)
        for df in [weather_df, gen_df]:
            df["merge_key"] = pd.to_datetime({
                "year": 2000,  # constant dummy year
                "month": df["month"],
                "day": df["day"],
                "hour": df["hour"],
                "minute": df["minute"]
            }, errors="coerce")

        # Sort before merging (required for merge_asof)
        weather_df = weather_df.sort_values("merge_key")
        gen_df = gen_df.sort_values("merge_key")

        # Merge datasets regardless of year
        logger.info(f"Merging with tolerance: {merge_tolerance} (ignoring year)...")
        merged_df = pd.merge_asof(
            gen_df,
            weather_df,
            on="merge_key",
            direction="nearest",
            tolerance=pd.Timedelta(merge_tolerance.lower())
        )

        # Drop rows with missing merged values
        before_dropna = len(merged_df)
        merged_df = merged_df.dropna()
        after_dropna = len(merged_df)

        logger.info(f"Merge complete! Rows before dropna: {before_dropna}, after dropna: {after_dropna}")
        logger.info(f"Successfully merged {(after_dropna / before_dropna * 100) if before_dropna else 0:.2f}% of records.")

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
        energy_stats = self.merged_df['LV-ActivePower(kW)'].describe()
        logger.info(f"\nEnergy Output Statistics:")
        logger.info(f"\n{energy_stats}")
        
        # Check for negative or zero energy during daytime
        if 'solar_radiation' in self.merged_df.columns or 'ALLSKY_SFC_SW_DWN' in self.merged_df.columns:
            solar_col = 'solar_radiation' if 'solar_radiation' in self.merged_df.columns else 'ALLSKY_SFC_SW_DWN'
            daytime = self.merged_df[self.merged_df[solar_col] > 100]
            zero_energy_daytime = (daytime['LV-ActivePower(kW)'] <= 0).sum()
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
                          generation_time_col='Date/Time', 
                          generation_value_col='LV-ActivePower(kW)',
                          merge_tolerance='1H'):
        """Run the complete merge pipeline"""
        
        # Step 1: Load and inspect
        self.load_and_inspect()
        
        # Step 2: Merge
        merged_df = self.merge_datasets(
            weather_df=self.weather_df,
            gen_df=self.generation_df,
            weather_time_col=weather_time_col,
            gen_time_col=generation_time_col,
            merge_tolerance=merge_tolerance
        )
        
        # Step 3: Validate
        self.merged_df = merged_df
        self.validate_merged_data()
        
        # Step 4: Save
        self.save_merged_data()
        
        return merged_df


# Example usage
if __name__ == "__main__":
    
    # CONFIGURATION - UPDATE THESE PATHS
    WEATHER_DATA_PATH = 'data/wind/weather_wind_data/openmeteo_historical_2018.csv'
    GENERATION_DATA_PATH = 'data/wind/plant_generation_data/T1.csv'
    OUTPUT_DATA_PATH = 'data/wind/merged_wind_data.csv'
    
    # Column name mapping - UPDATE THESE IF YOUR COLUMNS HAVE DIFFERENT NAMES
    WEATHER_TIME_COL = 'timestamp'
    GENERATION_TIME_COL = 'Date/Time'
    GENERATION_VALUE_COL = 'LV-ActivePower(kW)'
    
    # Time tolerance for matching
    MERGE_TOLERANCE = '1H'
    
    # Run pipeline
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