# ml-engine/train_models.py
import argparse
import os
import sys

# Ensure the parent directory (ml-engine) is in the path to allow imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import the main training functions from the dedicated files
try:
    from train_solar_model import run_solar_training
    from train_wind_model import run_wind_training
except ImportError as e:
    print(f"Error importing training modules: {e}")
    print("Please ensure train_solar_model.py and train_wind_model.py are in the ml-engine directory.")
    sys.exit(1)


def main():
    """
    Main function to run both solar and wind model training sequentially.
    """
    print("================================================")
    print("☀️⚡ SolarSync ML Engine Training Orchestrator ⚡🌬️")
    print("================================================")
    
    parser = argparse.ArgumentParser(description='Run SolarSync ML Model Training.')
    parser.add_argument('--solar_epochs', type=int, default=100, help='Epochs for solar model.')
    parser.add_argument('--wind_epochs', type=int, default=80, help='Epochs for wind model.')
    # You can add arguments here to override data paths if needed
    args = parser.parse_args()

    # 1. Run Solar Model Training
    try:
        run_solar_training(epochs=args.solar_epochs)
        print("\nSolar Model Training Finished.")
    except Exception as e:
        print(f"\n❌ ERROR: Solar Model Training Failed: {e}")
        # Continue to the next training run even if one fails
        
    # 2. Run Wind Model Training
    try:
        run_wind_training(epochs=args.wind_epochs)
        print("\nWind Model Training Finished.")
    except Exception as e:
        print(f"\n❌ ERROR: Wind Model Training Failed: {e}")

    print("\n================================================")
    print("✅ All Training Processes Attempted. Check logs for completion.")
    print("================================================")


if __name__ == '__main__':
    # Running this script will execute:
    # python ml-engine/train_models.py
    main()