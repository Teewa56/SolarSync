-- Useful views and functions

-- View for active energy orders
CREATE VIEW active_energy_orders AS
SELECT 
    eo.*,
    p.wallet_address as producer_wallet,
    pp.plant_name,
    pp.installed_capacity_kw
FROM energy_orders eo
JOIN producers p ON eo.producer_id = p.id
JOIN producer_plants pp ON eo.plant_id = pp.id
WHERE eo.status = 'open' AND eo.expires_at > NOW();

-- View for producer performance
CREATE VIEW producer_performance AS
SELECT 
    p.id,
    p.wallet_address,
    p.producer_type,
    COUNT(DISTINCT pp.id) as plant_count,
    SUM(pp.installed_capacity_kw) as total_capacity,
    AVG(pd.energy_produced_kwh) as avg_daily_production,
    p.reputation_score,
    p.total_energy_sold_kwh,
    p.total_earnings
FROM producers p
LEFT JOIN producer_plants pp ON p.id = pp.producer_id
LEFT JOIN production_data pd ON pp.id = pd.plant_id
GROUP BY p.id, p.wallet_address, p.producer_type, p.reputation_score, p.total_energy_sold_kwh, p.total_earnings;

-- Function to calculate carbon credits for a trade
CREATE OR REPLACE FUNCTION calculate_carbon_credits(energy_kwh DECIMAL)
RETURNS DECIMAL AS $$
BEGIN
    -- 1 kWh of renewable energy = 0.5 kg CO2 equivalent saved
    RETURN energy_kwh * 0.5;
END;
$$ LANGUAGE plpgsql;

-- Function to update reputation score
CREATE OR REPLACE FUNCTION update_reputation_score(user_id INTEGER, user_type VARCHAR, change_amount DECIMAL)
RETURNS VOID AS $$
BEGIN
    IF user_type = 'producer' THEN
        UPDATE producers 
        SET reputation_score = GREATEST(0, LEAST(1, reputation_score + change_amount))
        WHERE id = user_id;
    ELSIF user_type = 'consumer' THEN
        UPDATE consumers 
        SET reputation_score = GREATEST(0, LEAST(1, reputation_score + change_amount))
        WHERE id = user_id;
    END IF;
    
    -- Log the reputation event
    INSERT INTO reputation_events (user_id, user_type, event_type, score_change)
    VALUES (user_id, user_type, 'manual_adjustment', change_amount);
END;
$$ LANGUAGE plpgsql;