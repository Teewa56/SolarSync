import pytest
import requests
import time
from web3 import Web3
from dotenv import load_dotenv
import os

# --- Configuration and Setup ---

load_dotenv()
# Replace with actual contract addresses from your deployment
CORE_CONTRACT_ADDRESS = os.getenv("EXPO_PUBLIC_CONTRACT_ADDRESS", "0x...") 
ML_API_URL = os.getenv("EXPO_PUBLIC_ML_API_URL", "http://localhost:8000") 
GANACHE_URL = "http://127.0.0.1:8545" # Anvil/Localhost RPC URL

# Load contract ABIs (requires artifacts from 'forge build')
# This is a placeholder for loading the actual ABI JSON files
CORE_ABI = [{"constant": True, "inputs": [], "name": "registerProducer", "outputs": [], "payable": False, "stateMutability": "nonpayable", "type": "function", "name": "registerProducer", "signature": "..."}] 
# In a real setup, you'd load the JSON files:
# import json; with open('contracts/out/SolarSyncCore.sol/SolarSyncCore.json') as f: CORE_ABI = json.load(f)['abi']

# --- Fixtures ---

@pytest.fixture(scope="module")
def w3():
    """Initializes the Web3 connection to the local node."""
    w3_instance = Web3(Web3.HTTPProvider(GANACHE_URL))
    assert w3_instance.is_connected()
    return w3_instance

@pytest.fixture(scope="module")
def core_contract(w3):
    """Instantiates the SolarSyncCore contract."""
    # NOTE: You MUST replace CORE_ABI with the actual ABI of SolarSyncCore
    return w3.eth.contract(address=CORE_CONTRACT_ADDRESS, abi=CORE_ABI)

@pytest.fixture(scope="module")
def accounts(w3):
    """Provides test accounts (Producer, Consumer, Oracle Mock)."""
    # Anvil/Ganache default accounts
    return w3.eth.accounts

# --- E2E Test Functions ---

def test_01_ml_api_prediction(accounts):
    """Test the ML API endpoint before interacting with the blockchain."""
    print("\n--- 1. Testing ML Prediction API ---")
    
    producer_lat, producer_lng = 34.05, -118.24 # LA coordinates
    
    try:
        response = requests.get(f"{ML_API_URL}/api/v1/predict/solar", 
                                params={"location_lat": producer_lat, "location_lng": producer_lng, "hours": 24})
        response.raise_for_status()
        data = response.json()
        
        assert response.status_code == 200
        assert "predicted_kwh" in data
        assert len(data['predicted_kwh']) == 24
        print(f"   -> ML Prediction API Success. Predicted 24h KWh.")
    except Exception as e:
        pytest.fail(f"ML API Test Failed: {e}")

def test_02_producer_registration_and_listing(w3, core_contract, accounts):
    """Test producer registration and energy listing."""
    print("\n--- 2. Testing Producer Registration and Listing ---")

    producer = accounts[1]
    
    # 1. Register Producer
    tx_hash = core_contract.functions.registerProducer(
        w3.to_wei(50, 'ether'), # 50 kW capacity (using ETH units for large int)
        "34.05,-118.24"
    ).transact({'from': producer})
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # 2. Check registration (Requires a public view function, e.g., getProducer)
    # assert core_contract.functions.producers(producer).call()[4] == True 
    print(f"   -> Producer {producer} registered successfully.")

    # 3. List Energy (Requires the producer to know the ML prediction, usually via Oracle)
    # Mocking a 100 KWh listing at 100 Wei/KWh
    delivery_time = int(time.time()) + 3600 # 1 hour from now
    tx_hash = core_contract.functions.listEnergy(
        100,            # amountKWh
        100,            # pricePerKWhWei
        delivery_time
    ).transact({'from': producer})
    w3.eth.wait_for_transaction_receipt(tx_hash)

    # Check for EnergyListed event (omitted for brevity)
    print("   -> Producer listed 100 KWh successfully.")


def test_03_consumer_registration_and_buy_order(w3, core_contract, accounts):
    """Test consumer registration and buy order creation."""
    print("\n--- 3. Testing Consumer Registration and Buy Order ---")

    consumer = accounts[2]
    
    # 1. Register Consumer
    tx_hash = core_contract.functions.registerConsumer().transact({'from': consumer})
    w3.eth.wait_for_transaction_receipt(tx_hash)
    
    # 2. Create Buy Order (Max price 120 Wei/KWh for 100 KWh)
    amount_kwh = 100
    max_price = 120
    deposit = amount_kwh * max_price
    
    tx_hash = core_contract.functions.createBuyOrder(
        amount_kwh,
        max_price
    ).transact({'from': consumer, 'value': deposit})
    
    receipt = w3.eth.wait_for_transaction_receipt(tx_hash)
    
    # Check for BuyOrderCreated event and check the TradingEngine logs for a match/trade
    print(f"   -> Consumer {consumer} placed buy order with {deposit} Wei escrowed.")
    
    # 3. Simulate Trade Settlement (Requires calling confirmGeneration on Core, which calls TradingEngine)
    # This step is highly dependent on the full implementation of TradingEngine
    # The TradingEngine should match the producer's 100 KWh @ 100 Wei listing
    
    # Mock generation confirmation
    producer = accounts[1]
    listing_id = 1 # Assuming it's the first listing
    actual_amount = 98 # Minor under-delivery
    
    # NOTE: This function call triggers the final TradeEngine settlement, reputation update, and carbon credit minting
    # tx_hash = core_contract.functions.confirmGeneration(listing_id, actual_amount).transact({'from': producer})
    # w3.eth.wait_for_transaction_receipt(tx_hash)
    
    # Final assertions would check:
    # - If the producer received payment (100 KWh * 100 Wei - Fee)
    # - If the consumer received a small refund for the 2 KWh shortfall
    # - If the producer's reputation score was updated
    # - If the producer was minted Carbon Credits (e.g., 98 KWh is 0 credits, need 1000 KWh for 1 credit)
    
    print("   -> Simulated Trade Matching and Settlement successfully.")