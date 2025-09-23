# SolarSync 🌞⚡

**Decentralized Energy Trading Platform with AI-Powered Renewable Energy Forecasting**

SolarSync is a blockchain-based peer-to-peer energy trading platform that uses machine learning to predict renewable energy generation and optimize decentralized energy markets. Built with PyTorch for ML forecasting and Solidity for smart contract automation.

## 🚀 Overview

SolarSync enables renewable energy producers (solar panel owners, wind farm operators) to trade excess energy directly with consumers through an intelligent marketplace. Our ML models predict energy generation patterns using weather data, while smart contracts automate trading, pricing, and settlements.

### Key Features

- **AI-Powered Forecasting**: Time series models predict solar/wind energy generation 24-48 hours ahead
- **Automated Trading**: Smart contracts handle order matching, execution, and settlement
- **Dynamic Pricing**: ML-driven pricing optimization based on supply/demand predictions
- **Carbon Credit Integration**: Automatic carbon credit generation for renewable energy trades
- **Reputation System**: Producer/consumer reliability scoring based on trading history
- **Grid Balancing**: Incentivize energy storage and demand shifting for grid stability

## 🛠 Tech Stack

### Blockchain Layer
- **Solidity**: Smart contracts for trading logic, escrow, and settlements
- **Base**: Blockchain network for contract deployment
- **Ethers.js**: Frontend blockchain interactions
- **Chainlink Oracles**: Weather data and price feeds integration

### Machine Learning
- **PyTorch**: Deep learning framework for time series prediction
- **Python**: ML model development and API backend
- **LSTM/GRU Networks**: For energy generation forecasting
- **Transformer Models**: Advanced sequence prediction for complex weather patterns
- **Scikit-learn**: Data preprocessing and traditional ML algorithms

### Data & APIs
- **OpenWeatherMap API**: Real-time weather data
- **NOAA Climate Data**: Historical weather patterns
- **FastAPI**: ML model serving and API endpoints
- **PostgreSQL**: Time series data storage
- **Redis**: Real-time data caching

### Frontend
- **React.js**: User interface
- **Chart.js/D3.js**: Data visualization and analytics
- **MetaMask Integration**: Wallet connectivity
- **Material-UI**: Component library

## 📋 Prerequisites

- Node.js (v16+)
- Python (3.8+)
- MetaMask wallet
- Base testnet ETH (for development)

## 🚧 Installation

### 1. Clone Repository
```bash
git clone https://github.com/teewa56/solarsync.git
cd solarsync
```

### 2. Smart Contract Setup
```bash
cd contracts
npm install
forge build

# Deploy to Sepolia
forge create --rpc-url $SEPOLIA_RPC_URL --private-key $PRIVATE_KEY src/EnerCastCore.sol:EnerCastCore


### 3. ML Model Setup
```bash
cd ml-engine
pip install -r requirements.txt
python train_models.py
uvicorn main:app --reload --port 8000
```

### 4. Frontend Setup
```bash
cd frontend
npm install
npm start
```

### 5. Environment Configuration
Create `.env` files in each directory:

**contracts/.env**
```
PRIVATE_KEY=your_private_key
SEPOLIA_RPC_URL=your_sepolia_rpc_url
ETHERSCAN_API_KEY=your_etherscan_key
```

**ml-engine/.env**
```
OPENWEATHER_API_KEY=your_openweather_key
DATABASE_URL=postgresql://user:pass@localhost/energycast
REDIS_URL=redis://localhost:6379
```

**frontend/.env**
```
REACT_APP_CONTRACT_ADDRESS=deployed_contract_address
REACT_APP_ML_API_URL=http://localhost:8000
```

## 🏗 Architecture

### Smart Contract Architecture

```
SolarSyncCore.sol           # Main trading logic
├── EnergyOracle.sol       # Weather data and ML predictions
├── TradingEngine.sol      # Order matching and execution
├── ReputationSystem.sol   # Producer/consumer scoring
└── CarbonCredits.sol      # Environmental impact tracking
```

### ML Model Pipeline

```
Data Ingestion → Feature Engineering → Model Training → Prediction API
     ↓                    ↓                ↓              ↓
Weather APIs         Time Series        PyTorch        FastAPI
NOAA Data           Processing         LSTM/GRU       Endpoints
```

### System Workflow

1. **Data Collection**: Weather APIs provide real-time meteorological data
2. **ML Prediction**: PyTorch models forecast energy generation for registered producers
3. **Oracle Updates**: Chainlink oracles feed predictions to smart contracts
4. **Market Making**: Smart contracts automatically create buy/sell orders
5. **Order Matching**: Trading engine matches buyers with sellers based on price/preferences
6. **Settlement**: Automated payment and energy delivery confirmation
7. **Reputation Update**: System updates participant reliability scores

## 📊 ML Models

### Energy Generation Forecasting

**Solar Energy Prediction**
- **Input Features**: Solar irradiance, temperature, humidity, cloud cover, time of day/year
- **Architecture**: LSTM with 3 layers, 128 hidden units
- **Prediction Window**: 24-48 hours ahead
- **Accuracy**: ~85% MAPE on historical data

**Wind Energy Prediction**
- **Input Features**: Wind speed, direction, pressure, temperature gradients
- **Architecture**: GRU with attention mechanism
- **Prediction Window**: 12-36 hours ahead
- **Accuracy**: ~80% MAPE on historical data

**Demand Forecasting**
- **Input Features**: Historical consumption, weather, time patterns, economic indicators
- **Architecture**: Transformer-based sequence model
- **Prediction Window**: 1-7 days ahead

### Model Training
```bash
# Train solar prediction model
python ml-engine/train_solar_model.py --data_path ./data/solar/ --epochs 100

# Train wind prediction model
python ml-engine/train_wind_model.py --data_path ./data/wind/ --epochs 80

# Evaluate model performance
python ml-engine/evaluate_models.py --model_type solar --test_period 2023-01-01:2023-12-31
```

## 🔗 Smart Contracts

### Core Functions

**For Energy Producers**
```solidity
function registerProducer(uint256 capacity, string location) external
function listEnergy(uint256 amount, uint256 price, uint256 deliveryTime) external
function confirmGeneration(uint256 actualAmount) external
```

**For Energy Consumers**
```solidity
function registerConsumer() external
function createBuyOrder(uint256 amount, uint256 maxPrice) external
function confirmConsumption(uint256 amount) external
```

**Trading Engine**
```solidity
function matchOrders() external
function executeTradesettlement(uint256 tradeId) external
function calculateCarbonCredits(uint256 energyAmount) external pure returns (uint256)
```

### Contract Deployment
```bash

# Or using a deploy script (recommended)
forge script script/Deploy.s.sol --rpc-url $SEPOLIA_RPC_URL --private-key $PRIVATE_KEY --broadcast

```

## 🌐 API Endpoints

### ML Prediction Service

```http
GET /api/v1/predict/solar?location={lat,lng}&hours=24
GET /api/v1/predict/wind?location={lat,lng}&hours=24
GET /api/v1/predict/demand?user_id={id}&days=7
POST /api/v1/models/retrain
```

### Market Data Service

```http
GET /api/v1/market/prices/current
GET /api/v1/market/volume/24h
GET /api/v1/producers?location={lat,lng}&radius=50km
GET /api/v1/carbon-credits/calculate?energy_amount={kwh}
```

## 📱 User Interfaces

### Producer Dashboard
- Real-time generation monitoring and forecasts
- Revenue analytics and trading history
- Market price trends and optimization recommendations
- System performance and maintenance alerts

### Consumer Dashboard
- Energy consumption patterns and cost analysis
- Available energy sources with carbon footprint data
- Automated purchasing preferences and budget controls
- Savings compared to traditional grid pricing

### Marketplace
- Live energy listings with ML-predicted availability
- Producer profiles with reliability and sustainability scores
- Real-time price charts and market depth
- Community trading activity and leaderboards

## 🧪 Testing

### Smart Contract Tests
```bash
cd contracts
npx forge test
npx forge coverage
```

### ML Model Tests
```bash
cd ml-engine
pytest tests/test_models.py -v
python tests/test_prediction_accuracy.py
```

### Integration Tests
```bash
cd tests
python test_end_to_end.py
python test_oracle_integration.py
```

## 🚀 Deployment

### Production Deployment

1. **Smart Contracts**: Deploy to base mainnet
2. **ML Services**: Deploy to AWS/GCP with auto-scaling
3. **Frontend**: Deploy to Vercel/Netlify with CDN
4. **Database**: Managed PostgreSQL with replication
5. **Monitoring**: Grafana dashboards for system metrics

### Environment Setup
```bash
# Production environment
export NODE_ENV=production
export NETWORK=mainnet
export ML_API_URL=https://api.energycast.io
```

## 🌱 Environmental Impact

- **Carbon Footprint Tracking**: Real-time monitoring of environmental benefits
- **Renewable Energy Incentives**: Preferential pricing for clean energy sources
- **Grid Decarbonization**: Accelerate adoption of renewable energy
- **Community Impact**: Local energy independence and resilience


### Development Workflow
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- OpenWeatherMap for weather data APIs
- Chainlink for reliable oracle services
- The Ethereum and PyTorch communities for excellent documentation
- Renewable energy pioneers making clean energy accessible



**Built with ❤️ for a sustainable energy future**