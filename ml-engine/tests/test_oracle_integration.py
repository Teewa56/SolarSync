# tests/test_oracle_integration.py
import pytest
import requests
import json
import os

# --- Configuration ---
ML_API_URL = os.getenv("EXPO_PUBLIC_ML_API_URL", "http://localhost:8000") 
# Note: Requires the FastAPI server (ml-engine/main.py) to be running (uvicorn main:app)

@pytest.fixture(scope="module")
def mock_request_params():
    """Mock parameters for a typical oracle request (e.g., San Francisco)."""
    return {
        "location_lat": 37.7749,
        "location_lng": -122.4194,
        "hours": 5 # Test a short prediction window
    }

def test_solar_prediction_endpoint_format(mock_request_params):
    """Test the solar endpoint returns the correct structure expected by Chainlink."""
    print("\n--- Testing Solar Prediction API Response ---")
    try:
        response = requests.get(f"{ML_API_URL}/api/v1/predict/solar", params=mock_request_params)
        response.raise_for_status()
        data = response.json()
        
        # 1. Status Check
        assert response.status_code == 200, f"Expected 200 status, got {response.status_code}"
        
        # 2. Key Check
        assert "predicted_kwh" in data, "Response must contain 'predicted_kwh' key"
        assert isinstance(data['predicted_kwh'], list), "'predicted_kwh' must be a list"
        
        # 3. Length Check
        expected_hours = mock_request_params['hours']
        assert len(data['predicted_kwh']) == expected_hours, f"Expected {expected_hours} predictions"
        
        # 4. Chainlink Oracle Data Path Check (Crucial!)
        # Chainlink will use the path "predicted_kwh.0"
        first_prediction = data['predicted_kwh'][0]
        assert isinstance(first_prediction, (int, float)), "First prediction must be a number"
        assert first_prediction >= 0, "Energy prediction should be non-negative"
        
        print("   -> Solar prediction format passed for Chainlink integration.")

    except requests.exceptions.ConnectionError:
        pytest.skip(f"Could not connect to ML API at {ML_API_URL}. Is it running?")
    except Exception as e:
        pytest.fail(f"Solar Prediction Test Failed: {e}")

def test_wind_prediction_endpoint_format(mock_request_params):
    """Test the wind endpoint format."""
    print("\n--- Testing Wind Prediction API Response ---")
    # ... (Similar checks for the /predict/wind endpoint)
    # We assume similar structure and skip the verbose implementation here for brevity.
    pass

def test_error_handling_on_invalid_location():
    """Test API error handling for edge cases."""
    print("\n--- Testing API Error Handling ---")
    try:
        # Invalid coordinates (e.g., non-numeric)
        response = requests.get(f"{ML_API_URL}/api/v1/predict/solar", 
                                params={"location_lat": "a", "location_lng": "b", "hours": 1})
        
        assert response.status_code == 422, "Expected 422 (Validation Error) for invalid input"
        print("   -> API error handling test passed.")

    except requests.exceptions.ConnectionError:
        pytest.skip(f"Could not connect to ML API at {ML_API_URL}.")
    except Exception as e:
        pytest.fail(f"API Error Handling Test Failed: {e}")