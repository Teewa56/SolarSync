# ☀️ SolarSync - Decentralized Renewable Energy Trading Platform

## Overview

SolarSync is a blockchain-based platform that enables peer-to-peer trading of renewable energy using smart contracts, machine learning predictions, and carbon credit tokenization.

### Key Features

- **Smart Contract Trading**: Automated order matching and settlement
- **ML-Powered Predictions**: LSTM/GRU models for solar and wind energy forecasting
- **Carbon Credit Tokenization**: ERC-20 tokens (SSCC) representing verified renewable energy
- **Reputation System**: On-chain reliability scoring for producers and consumers
- **Oracle Integration**: Chainlink oracle for real-time weather data

## Project Structure

```
SolarSync/
├── contracts/                  # Solidity smart contracts
│   ├── src/
│   │   ├── SolarSyncCore.sol
│   │   ├── TradingEngine.sol
│   │   ├── EnergyOracle.sol
│   │   ├── CarbonCredits.sol
│   │   ├── ReputationSystem.sol
│   │   └── ISolarSyncInterfaces.sol
│   ├── script/
│   │   └── Deploy.s.sol
│   ├── test/
│   │   └── SolarSyncCore.t.sol
│   └── foundry.toml
│
├── ml-engine/                  # Machine learning service
│   ├── main.py                 # FastAPI server
│   ├── models.py               # LSTM/GRU model definitions
│   ├── data_loader.py          # Data preprocessing
│   ├── data_fetcher.py         # Weather API integration
│   ├── train_models.py         # Training pipeline
│   ├── saved_models/           # Trained model files
│   ├── data/                   # Training datasets
│   └── requirements.txt
│
├── mobile-app/                 # React Native mobile app (optional)
└── README.md
```

## Prerequisites

### Smart Contracts

- [Foundry](https://book.getfoundry.sh/getting-started/installation)
- Node.js >= 16
- Sepolia testnet ETH and LINK tokens

### ML Engine

- Python >= 3.8
- CUDA (optional, for GPU training)
- OpenWeatherMap API key

## Install dependencies
pip install -r requirements.txt

# Copy environment template
cp .env.example .env

# Edit .env with your API keys:
# OPENWEATHER_API_KEY=your_openweather_api_key
# DATABASE_URL=postgresql://user:pass@localhost/solarsync
# REDIS_URL=redis://localhost:6379
```

## Usage

### Deploy Smart Contracts

#### Local Development (Anvil)

```bash
cd contracts

# Start local node
anvil

# In another terminal, deploy contracts
forge script script/Deploy.s.sol:DeploySolarSync --rpc-url http://localhost:8545 --broadcast
```

#### Sepolia Testnet

```bash
# Deploy to Sepolia
forge script script/Deploy.s.sol:DeploySolarSync \
  --rpc-url $SEPOLIA_RPC_URL \
  --broadcast \
  --verify \
  -vvvv

# The deployment addresses will be saved to deployment-addresses.json
```

#### After Deployment

1. **Fund the Oracle**: Send LINK tokens to the EnergyOracle contract
2. **Note Contract Addresses**: Save addresses from deployment output
3. **Update Frontend Config**: Add contract addresses to your frontend `.env`

### Train ML Models

```bash
cd ml-engine

# Prepare training data
# Place your historical data in data/solar/solar_history.csv and data/wind/wind_history.csv
# Required columns:
# Solar: timestamp, solar_irradiance, temperature, cloud_cover, energy_output
# Wind: timestamp, wind_speed, wind_direction, pressure, energy_output

# Train both models
python train_models.py --model both --epochs 100

# Train only solar model
python train_models.py --model solar --epochs 100

# Train only wind model
python train_models.py --model wind --epochs 80
```

### Run ML API Server

```bash
cd ml-engine

# Start FastAPI server
uvicorn main:app --reload --host 0.0.0.0 --port 8000

# Test the API
curl "http://localhost:8000/api/v1/predict/solar?location_lat=34.05&location_lng=-118.24&hours=24"
```

### Test Smart Contracts

```bash
cd contracts

# Run all tests
forge test

# Run with verbosity
forge test -vvv

# Run specific test
forge test --match-test testProducerRegistration -vvv

# Generate gas report
forge test --gas-report
```

## API Documentation

### ML Prediction API

#### Solar Prediction

```http
GET /api/v1/predict/solar?location_lat={lat}&location_lng={lng}&hours={hours}
```

**Parameters:**
- `location_lat` (float): Latitude (-90 to 90)
- `location_lng` (float): Longitude (-180 to 180)
- `hours` (int): Number of hours to predict (1-168)

**Response:**
```json
{
  "predicted_kwh": [125.3, 142.7, 156.2, ...],
  "model": "LSTM",
  "timestamp": 1704067200.0,
  "location": {"lat": 34.05, "lng": -118.24},
  "confidence_score": 0.87
}
```

#### Wind Prediction

```http
GET /api/v1/predict/wind?location_lat={lat}&location_lng={lng}&hours={hours}
```

Same parameters and response format as solar prediction.

#### Health Check

```http
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "models_loaded": {
    "solar": true,
    "wind": true
  },
  "timestamp": 1704067200.0
}
```

### Smart Contract Functions

#### SolarSyncCore

```solidity
// Register as a producer
function registerProducer(uint256 capacityKWh, string memory location) external

// Register as a consumer
function registerConsumer() external

// List energy for sale
function listEnergy(uint256 amountKWh, uint256 pricePerKWhWei, uint256 deliveryTimestamp) external

// Create buy order
function createBuyOrder(uint256 amountKWh, uint256 maxPricePerKWhWei) external payable

// Confirm generation
function confirmGeneration(uint256 listingId, uint256 actualAmountKWh) external
```

#### CarbonCredits (ERC-20)

```solidity
// Standard ERC-20 functions
function balanceOf(address account) external view returns (uint256)
function transfer(address to, uint256 amount) external returns (bool)

// Get platform statistics
function getPlatformStats() external view returns (...)
```

#### ReputationSystem

```solidity
// Get reputation score
function getReputationScore(address participant) external view returns (uint256)

// Get participant statistics
function getParticipantStats(address participant) external view returns (...)
```

## Architecture

### Smart Contract Architecture

```
┌─────────────────┐
│  SolarSyncCore  │  ← Main entry point
└────────┬────────┘
         │
    ┌────┴────┬────────────┬───────────────┐
    ▼         ▼            ▼               ▼
┌─────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│ Trading │ │  Energy  │ │  Carbon  │ │Reputation│
│ Engine  │ │  Oracle  │ │ Credits  │ │  System  │
└─────────┘ └──────────┘ └──────────┘ └──────────┘
```

### ML Pipeline

```
Weather API → Data Processing → Feature Engineering
                                       ↓
                                  LSTM/GRU Model
                                       ↓
                              Autoregressive Loop
                                       ↓
                                 Predictions
                                       ↓
                              Chainlink Oracle
                                       ↓
                              Smart Contracts
```

## Development

### Running Tests

```bash
# Smart contracts
cd contracts
forge test -vvv

# ML Engine
cd ml-engine
pytest tests/ -v

# End-to-end tests
pytest tests/test_end_to_end.py -v
```

### Code Formatting

```bash
# Solidity
forge fmt

# Python
black ml-engine/
isort ml-engine/
```

### Linting

```bash
# Solidity
forge fmt --check

# Python
flake8 ml-engine/
pylint ml-engine/
```

## Configuration

### Environment Variables

#### Smart Contracts (.env)

```env
PRIVATE_KEY=0x...
SEPOLIA_RPC_URL=https://sepolia.infura.io/v3/YOUR_KEY
ETHERSCAN_API_KEY=YOUR_ETHERSCAN_KEY
```

#### ML Engine (.env)

```env
OPENWEATHER_API_KEY=your_api_key
DATABASE_URL=postgresql://user:pass@localhost:5432/solarsync
REDIS_URL=redis://localhost:6379
```

#### Frontend (.env)

```env
EXPO_PUBLIC_CONTRACT_ADDRESS=0x...
EXPO_PUBLIC_TRADING_ENGINE_ADDRESS=0x...
EXPO_PUBLIC_ML_API_URL=https://your-ml-api.com
EXPO_PUBLIC_RPC_URL=https://sepolia.infura.io/v3/YOUR_KEY
```

## Deployment Guide

### Production Deployment

#### 1. Deploy Smart Contracts to Mainnet

```bash
# Use mainnet RPC and ensure sufficient ETH for gas
forge script script/Deploy.s.sol:DeploySolarSync \
  --rpc-url $MAINNET_RPC_URL \
  --broadcast \
  --verify \
  --slow
```

#### 2. Deploy ML API

```bash
# Using Docker
cd ml-engine
docker build -t solarsync-ml .
docker run -p 8000:8000 --env-file .env solarsync-ml

# Or using a cloud provider (AWS, GCP, Azure)
# Follow their deployment guides for FastAPI applications
```

#### 3. Configure Chainlink Oracle

1. Fund the EnergyOracle contract with LINK
2. Set up a Chainlink node (or use existing node)
3. Create external adapter for ML API
4. Configure job spec with correct parameters

## Security Considerations

### Smart Contracts

- ✅ ReentrancyGuard on all payable functions
- ✅ Access control with Ownable
- ✅ Pausable for emergency stops
- ✅ Input validation on all external functions
- ✅ SafeMath operations (Solidity 0.8+)

### ML API

- ✅ Input validation on all endpoints
- ✅ Rate limiting
- ✅ API key authentication (recommended for production)
- ✅ CORS configuration
- ✅ Error handling and logging

### Auditing

Before mainnet deployment:
1. Get smart contracts audited by professional firms
2. Run static analysis tools (Slither, Mythril)
3. Conduct thorough integration testing
4. Implement bug bounty program

## Performance Optimization

### Smart Contracts

- Gas optimization techniques applied
- Efficient data structures (mappings over arrays where possible)
- Batch operations for multiple updates
- Event-driven architecture for off-chain indexing

### ML Models

- Model quantization for faster inference
- Caching of predictions in Redis
- Asynchronous processing for multiple requests
- GPU acceleration for batch predictions

## Monitoring

### Recommended Tools

- **Smart Contracts**: Tenderly, Etherscan
- **ML API**: Prometheus + Grafana
- **Infrastructure**: DataDog, New Relic
- **Logs**: ELK Stack (Elasticsearch, Logstash, Kibana)

### Key Metrics to Track

- Contract gas usage
- Prediction accuracy (MAPE)
- API response times
- Transaction success rates
- Platform trading volume

## Troubleshooting

### Common Issues

#### 1. "Model not loaded" error

```bash
# Ensure models are trained
cd ml-engine
python train_models.py --model both

# Check if model files exist
ls saved_models/
```

#### 2. "Insufficient funds" error

```bash
# Check account balance
cast balance YOUR_ADDRESS --rpc-url $SEPOLIA_RPC_URL

# Get testnet ETH from faucet
# https://sepoliafaucet.com/
```

#### 3. Weather API errors

```bash
# Verify API key is set
echo $OPENWEATHER_API_KEY

# Test API directly
curl "https://api.openweathermap.org/data/3.0/onecall?lat=34.05&lon=-118.24&appid=$OPENWEATHER_API_KEY"
```

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

### Code Standards

- Follow Solidity style guide
- Use PEP 8 for Python code
- Write tests for new features
- Update documentation

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- OpenZeppelin for secure smart contract libraries
- Chainlink for decentralized oracle infrastructure
- PyTorch for machine learning framework
- FastAPI for high-performance API framework

## Roadmap

### Q1 2025
- ✅ Core smart contracts
- ✅ ML prediction engine
- ✅ Mobile app development
- Testnet for hackathon
### Q2 2025
- 🔜 Mainnet deployment
- 🔜 Advanced analytics dashboard
- 🔜 Multi-chain support

### Q3 2025
- 🔜 DAO governance
- 🔜 Staking mechanisms
- 🔜 Insurance pool

### Q4 2025
- 🔜 International expansion
- 🔜 Hardware integration (IoT devices)
- 🔜 Enterprise partnerships

---

**Built with ☀️ by the SolarSync Team**ation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/solarsync.git
cd solarsync
```

### 2. Smart Contracts Setup

```bash
cd contracts

# Install Foundry dependencies
forge install

# Copy environment template
cp .env.example .env

# Edit .env with your credentials:
# PRIVATE_KEY=your_private_key
# SEPOLIA_RPC_URL=your_sepolia_rpc_url
# ETHERSCAN_API_KEY=your_etherscan_api_key
```

### 3. ML Engine Setup

```bash
cd ml-engine

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate