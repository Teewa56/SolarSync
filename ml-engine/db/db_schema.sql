-- Main database schema for SolarSync

-- Producers and their facilities
CREATE TABLE producers (
    id SERIAL PRIMARY KEY,
    wallet_address VARCHAR(42) UNIQUE NOT NULL,
    producer_type VARCHAR(20) NOT NULL CHECK (producer_type IN ('solar', 'wind', 'hydro', 'other')),
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'inactive', 'suspended')),
    reputation_score DECIMAL(5,4) DEFAULT 1.0,
    total_energy_sold_kwh DECIMAL(15,2) DEFAULT 0,
    total_earnings DECIMAL(15,6) DEFAULT 0
);

-- Producer plants/facilities
CREATE TABLE producer_plants (
    id SERIAL PRIMARY KEY,
    producer_id INTEGER REFERENCES producers(id) ON DELETE CASCADE,
    plant_name VARCHAR(100) NOT NULL,
    location_lat DECIMAL(10,6),
    location_lon DECIMAL(10,6),
    installed_capacity_kw DECIMAL(10,2) NOT NULL,
    technology_type VARCHAR(50),
    installation_date DATE,
    efficiency_rating DECIMAL(5,4),
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'maintenance', 'inactive')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Production data from plants
CREATE TABLE production_data (
    id SERIAL PRIMARY KEY,
    plant_id INTEGER REFERENCES producer_plants(id) ON DELETE CASCADE,
    timestamp TIMESTAMP NOT NULL,
    energy_produced_kwh DECIMAL(10,2) NOT NULL,
    submitted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    verified BOOLEAN DEFAULT FALSE,
    source VARCHAR(20) DEFAULT 'manual' CHECK (source IN ('manual', 'api', 'oracle'))
);

-- Weather data for predictions
CREATE TABLE weather_data (
    id SERIAL PRIMARY KEY,
    timestamp TIMESTAMP NOT NULL,
    location_lat DECIMAL(10,6),
    location_lon DECIMAL(10,6),
    solar_radiation DECIMAL(8,2),
    temperature DECIMAL(5,2),
    humidity DECIMAL(5,2),
    wind_speed DECIMAL(6,2),
    wind_direction DECIMAL(5,2),
    pressure DECIMAL(7,2),
    cloud_cover DECIMAL(4,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Energy consumers
CREATE TABLE consumers (
    id SERIAL PRIMARY KEY,
    wallet_address VARCHAR(42) UNIQUE NOT NULL,
    consumer_type VARCHAR(20) DEFAULT 'residential' CHECK (consumer_type IN ('residential', 'commercial', 'industrial')),
    registration_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    location_lat DECIMAL(10,6),
    location_lon DECIMAL(10,6),
    status VARCHAR(20) DEFAULT 'active',
    reputation_score DECIMAL(5,4) DEFAULT 1.0,
    total_energy_bought_kwh DECIMAL(15,2) DEFAULT 0
);

-- Energy trading orders
CREATE TABLE energy_orders (
    id SERIAL PRIMARY KEY,
    producer_id INTEGER REFERENCES producers(id),
    plant_id INTEGER REFERENCES producer_plants(id),
    order_type VARCHAR(10) NOT NULL CHECK (order_type IN ('sell', 'buy')),
    energy_amount_kwh DECIMAL(10,2) NOT NULL,
    price_per_kwh DECIMAL(10,6) NOT NULL,
    status VARCHAR(20) DEFAULT 'open' CHECK (status IN ('open', 'matched', 'executed', 'cancelled')),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP,
    location_lat DECIMAL(10,6),
    location_lon DECIMAL(10,6)
);

-- Executed trades
CREATE TABLE energy_trades (
    id SERIAL PRIMARY KEY,
    sell_order_id INTEGER REFERENCES energy_orders(id),
    buy_order_id INTEGER REFERENCES energy_orders(id),
    producer_id INTEGER REFERENCES producers(id),
    consumer_id INTEGER REFERENCES consumers(id),
    energy_amount_kwh DECIMAL(10,2) NOT NULL,
    price_per_kwh DECIMAL(10,6) NOT NULL,
    total_value DECIMAL(15,6) NOT NULL,
    trade_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'disputed')),
    transaction_hash VARCHAR(66),
    carbon_credits_issued DECIMAL(10,2) DEFAULT 0
);

-- Carbon credits tracking
CREATE TABLE carbon_credits (
    id SERIAL PRIMARY KEY,
    producer_id INTEGER REFERENCES producers(id),
    trade_id INTEGER REFERENCES energy_trades(id),
    credits_amount DECIMAL(10,2) NOT NULL,
    credit_type VARCHAR(20) DEFAULT 'renewable',
    issued_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20) DEFAULT 'active' CHECK (status IN ('active', 'retired', 'traded')),
    transaction_hash VARCHAR(66)
);

-- ML model predictions
CREATE TABLE energy_predictions (
    id SERIAL PRIMARY KEY,
    plant_id INTEGER REFERENCES producer_plants(id),
    prediction_type VARCHAR(20) NOT NULL CHECK (prediction_type IN ('solar', 'wind', 'demand')),
    timestamp TIMESTAMP NOT NULL,
    predicted_energy_kwh DECIMAL(10,2) NOT NULL,
    confidence_interval DECIMAL(5,4),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    actual_energy_kwh DECIMAL(10,2) -- For model validation
);

-- Reputation system
CREATE TABLE reputation_events (
    id SERIAL PRIMARY KEY,
    user_id INTEGER, -- Can be producer or consumer
    user_type VARCHAR(10) CHECK (user_type IN ('producer', 'consumer')),
    event_type VARCHAR(50) NOT NULL,
    score_change DECIMAL(4,2) NOT NULL,
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System oracles for external data
CREATE TABLE oracle_data (
    id SERIAL PRIMARY KEY,
    oracle_type VARCHAR(50) NOT NULL,
    data_key VARCHAR(100) NOT NULL,
    data_value TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source VARCHAR(100),
    verified BOOLEAN DEFAULT FALSE
);

-- Indexes for performance
CREATE INDEX idx_production_data_timestamp ON production_data(timestamp);
CREATE INDEX idx_weather_data_timestamp ON weather_data(timestamp);
CREATE INDEX idx_energy_orders_status ON energy_orders(status);
CREATE INDEX idx_energy_orders_type ON energy_orders(order_type);
CREATE INDEX idx_energy_trades_date ON energy_trades(trade_date);
CREATE INDEX idx_producers_wallet ON producers(wallet_address);
CREATE INDEX idx_consumers_wallet ON consumers(wallet_address);
CREATE INDEX idx_predictions_plant ON energy_predictions(plant_id);