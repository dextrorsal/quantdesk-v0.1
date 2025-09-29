-- System Mode Management Tables
-- Tables for managing demo/live mode switching

-- System configuration table
CREATE TABLE IF NOT EXISTS system_config (
    id SERIAL PRIMARY KEY,
    key VARCHAR(255) UNIQUE NOT NULL,
    value TEXT NOT NULL,
    mode VARCHAR(50) NOT NULL DEFAULT 'demo',
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Risk limits per mode
CREATE TABLE IF NOT EXISTS risk_limits (
    id SERIAL PRIMARY KEY,
    mode VARCHAR(50) UNIQUE NOT NULL,
    max_position_size DECIMAL(20, 8) NOT NULL,
    max_leverage INTEGER NOT NULL,
    max_daily_loss DECIMAL(20, 8) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- API endpoints per mode
CREATE TABLE IF NOT EXISTS api_endpoints (
    id SERIAL PRIMARY KEY,
    mode VARCHAR(50) UNIQUE NOT NULL,
    endpoints JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Mode change history
CREATE TABLE IF NOT EXISTS mode_change_history (
    id SERIAL PRIMARY KEY,
    old_mode VARCHAR(50) NOT NULL,
    new_mode VARCHAR(50) NOT NULL,
    timestamp TIMESTAMP NOT NULL,
    user_id VARCHAR(255),
    reason TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- System health logs
CREATE TABLE IF NOT EXISTS system_health_logs (
    id SERIAL PRIMARY KEY,
    component VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    message TEXT,
    details JSONB,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Admin actions log
CREATE TABLE IF NOT EXISTS admin_actions (
    id SERIAL PRIMARY KEY,
    user_id VARCHAR(255) NOT NULL,
    action VARCHAR(100) NOT NULL,
    target_type VARCHAR(50),
    target_id VARCHAR(255),
    details JSONB,
    ip_address INET,
    user_agent TEXT,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert default system configuration
INSERT INTO system_config (key, value, mode, description) VALUES
('current_mode', 'demo', 'demo', 'Current system mode (demo/live)'),
('trading_enabled', 'true', 'demo', 'Whether trading is enabled'),
('maintenance_mode', 'false', 'demo', 'Whether system is in maintenance mode'),
('max_concurrent_users', '1000', 'demo', 'Maximum concurrent users allowed'),
('api_rate_limit', '1000', 'demo', 'API rate limit per minute')
ON CONFLICT (key) DO NOTHING;

-- Insert default risk limits
INSERT INTO risk_limits (mode, max_position_size, max_leverage, max_daily_loss) VALUES
('demo', 10000.00, 100, 1000.00),
('live', 100000.00, 50, 10000.00)
ON CONFLICT (mode) DO NOTHING;

-- Insert default API endpoints
INSERT INTO api_endpoints (mode, endpoints) VALUES
('demo', '["https://api-devnet.solana.com", "https://devnet.helius-rpc.com"]'),
('live', '["https://api.mainnet-beta.solana.com", "https://mainnet.helius-rpc.com"]')
ON CONFLICT (mode) DO NOTHING;

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_system_config_key ON system_config(key);
CREATE INDEX IF NOT EXISTS idx_risk_limits_mode ON risk_limits(mode);
CREATE INDEX IF NOT EXISTS idx_api_endpoints_mode ON api_endpoints(mode);
CREATE INDEX IF NOT EXISTS idx_mode_change_history_timestamp ON mode_change_history(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_system_health_logs_timestamp ON system_health_logs(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_admin_actions_timestamp ON admin_actions(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_admin_actions_user_id ON admin_actions(user_id);

-- Create updated_at trigger function
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers for updated_at
CREATE TRIGGER update_system_config_updated_at BEFORE UPDATE ON system_config FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_risk_limits_updated_at BEFORE UPDATE ON risk_limits FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_api_endpoints_updated_at BEFORE UPDATE ON api_endpoints FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
