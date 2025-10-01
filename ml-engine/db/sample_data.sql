-- Sample data for development and testing

-- Insert sample producers
INSERT INTO producers (wallet_address, producer_type, reputation_score) VALUES
('0x742d35Cc6634C0532925a3b8D4B5A3B8D4B5A3B8', 'solar', 0.95),
('0x843d46Dc8765A3b8D4B5A3B8D4B5A3B8D4B5A3B9', 'wind', 0.88),
('0x944e57Cc6634C0532925a3b8D4B5A3B8D4B5A3B0', 'solar', 0.92);

-- Insert sample plants
INSERT INTO producer_plants (producer_id, plant_name, location_lat, location_lon, installed_capacity_kw, technology_type) VALUES
(1, 'Sunny Valley Solar Farm', 40.7128, -74.0060, 5000.00, 'monocrystalline'),
(2, 'Windy Ridge Turbines', 34.0522, -118.2437, 2000.00, 'onshore_wind'),
(3, 'Eco Home Solar', 41.8781, -87.6298, 10.50, 'polycrystalline');

-- Insert sample consumers
INSERT INTO consumers (wallet_address, consumer_type, location_lat, location_lon) VALUES
('0xA43d46Dc8765A3b8D4B5A3B8D4B5A3B8D4B5A3B1', 'residential', 40.7128, -74.0060),
('0xB44e57Cc6634C0532925a3b8D4B5A3B8D4B5A3B2', 'commercial', 34.0522, -118.2437),
('0xC45f68Dc8765A3b8D4B5A3B8D4B5A3B8D4B5A3B3', 'industrial', 41.8781, -87.6298);

-- Insert sample weather data
INSERT INTO weather_data (timestamp, location_lat, location_lon, solar_radiation, temperature, humidity, wind_speed) VALUES
(NOW() - INTERVAL '1 hour', 40.7128, -74.0060, 650.25, 22.5, 65.0, 3.2),
(NOW() - INTERVAL '2 hours', 40.7128, -74.0060, 720.50, 23.1, 63.5, 3.8),
(NOW() - INTERVAL '3 hours', 40.7128, -74.0060, 580.75, 21.8, 68.2, 2.9);

-- Insert sample production data
INSERT INTO production_data (plant_id, timestamp, energy_produced_kwh, verified) VALUES
(1, NOW() - INTERVAL '1 hour', 1250.50, true),
(1, NOW() - INTERVAL '2 hours', 1180.25, true),
(2, NOW() - INTERVAL '1 hour', 850.75, true);

-- Insert sample energy orders
INSERT INTO energy_orders (producer_id, plant_id, order_type, energy_amount_kwh, price_per_kwh, status) VALUES
(1, 1, 'sell', 100.00, 0.15, 'open'),
(2, 2, 'sell', 50.00, 0.12, 'open'),
(3, 3, 'sell', 5.50, 0.18, 'open');