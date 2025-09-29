-- Create advanced_orders table
CREATE TABLE IF NOT EXISTS advanced_orders (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id) ON DELETE CASCADE,
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN (
        'market', 'limit', 'stop_loss', 'take_profit', 'trailing_stop',
        'post_only', 'ioc', 'fok', 'iceberg', 'twap', 'stop_limit', 'bracket'
    )),
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    size DECIMAL(20, 8) NOT NULL CHECK (size > 0),
    price DECIMAL(20, 8) NOT NULL CHECK (price >= 0),
    stop_price DECIMAL(20, 8) CHECK (stop_price > 0),
    trailing_distance DECIMAL(20, 8) CHECK (trailing_distance > 0),
    leverage INTEGER NOT NULL CHECK (leverage > 0),
    status VARCHAR(20) NOT NULL DEFAULT 'pending' CHECK (status IN (
        'pending', 'filled', 'cancelled', 'expired', 'partially_filled', 'rejected'
    )),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    filled_size DECIMAL(20, 8) DEFAULT 0 CHECK (filled_size >= 0),
    
    -- Advanced order fields
    hidden_size DECIMAL(20, 8) CHECK (hidden_size > 0),
    display_size DECIMAL(20, 8) CHECK (display_size > 0),
    time_in_force VARCHAR(10) NOT NULL DEFAULT 'gtc' CHECK (time_in_force IN ('gtc', 'ioc', 'fok', 'gtd')),
    target_price DECIMAL(20, 8) CHECK (target_price > 0),
    parent_order_id UUID REFERENCES advanced_orders(id) ON DELETE SET NULL,
    twap_duration INTEGER CHECK (twap_duration > 0), -- in seconds
    twap_interval INTEGER CHECK (twap_interval > 0), -- in seconds
    
    -- Execution fields
    current_price DECIMAL(20, 8),
    unrealized_pnl DECIMAL(20, 8),
    execution_price DECIMAL(20, 8),
    execution_time TIMESTAMP WITH TIME ZONE,
    
    -- Metadata
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_advanced_orders_user_id ON advanced_orders(user_id);
CREATE INDEX IF NOT EXISTS idx_advanced_orders_market_id ON advanced_orders(market_id);
CREATE INDEX IF NOT EXISTS idx_advanced_orders_status ON advanced_orders(status);
CREATE INDEX IF NOT EXISTS idx_advanced_orders_order_type ON advanced_orders(order_type);
CREATE INDEX IF NOT EXISTS idx_advanced_orders_created_at ON advanced_orders(created_at);
CREATE INDEX IF NOT EXISTS idx_advanced_orders_expires_at ON advanced_orders(expires_at);

-- Index for conditional orders (stop-loss, take-profit, trailing-stop)
CREATE INDEX IF NOT EXISTS idx_advanced_orders_conditional 
ON advanced_orders(market_id, status, order_type) 
WHERE order_type IN ('stop_loss', 'take_profit', 'trailing_stop') AND status = 'pending';

-- Index for TWAP orders
CREATE INDEX IF NOT EXISTS idx_advanced_orders_twap 
ON advanced_orders(order_type, status, created_at) 
WHERE order_type = 'twap' AND status = 'pending';

-- Index for bracket orders
CREATE INDEX IF NOT EXISTS idx_advanced_orders_bracket 
ON advanced_orders(parent_order_id) 
WHERE parent_order_id IS NOT NULL;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_advanced_orders_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_advanced_orders_updated_at
    BEFORE UPDATE ON advanced_orders
    FOR EACH ROW
    EXECUTE FUNCTION update_advanced_orders_updated_at();

-- Add constraints for advanced order types
ALTER TABLE advanced_orders ADD CONSTRAINT chk_stop_loss_stop_price 
CHECK (order_type != 'stop_loss' OR stop_price IS NOT NULL);

ALTER TABLE advanced_orders ADD CONSTRAINT chk_take_profit_stop_price 
CHECK (order_type != 'take_profit' OR stop_price IS NOT NULL);

ALTER TABLE advanced_orders ADD CONSTRAINT chk_trailing_stop_distance 
CHECK (order_type != 'trailing_stop' OR trailing_distance IS NOT NULL);

ALTER TABLE advanced_orders ADD CONSTRAINT chk_iceberg_sizes 
CHECK (order_type != 'iceberg' OR (hidden_size IS NOT NULL AND display_size IS NOT NULL));

ALTER TABLE advanced_orders ADD CONSTRAINT chk_twap_duration 
CHECK (order_type != 'twap' OR (twap_duration IS NOT NULL AND twap_interval IS NOT NULL));

ALTER TABLE advanced_orders ADD CONSTRAINT chk_bracket_prices 
CHECK (order_type != 'bracket' OR (target_price IS NOT NULL AND stop_price IS NOT NULL));

-- Add constraint for TWAP interval <= duration
ALTER TABLE advanced_orders ADD CONSTRAINT chk_twap_interval_valid 
CHECK (twap_interval IS NULL OR twap_duration IS NULL OR twap_interval <= twap_duration);

-- Add constraint for iceberg order sizes
ALTER TABLE advanced_orders ADD CONSTRAINT chk_iceberg_size_sum 
CHECK (hidden_size IS NULL OR display_size IS NULL OR (hidden_size + display_size = size));

-- Create view for order statistics
CREATE OR REPLACE VIEW advanced_order_stats AS
SELECT 
    order_type,
    status,
    COUNT(*) as count,
    SUM(size) as total_size,
    AVG(size) as avg_size,
    SUM(filled_size) as total_filled_size,
    COUNT(CASE WHEN status = 'filled' THEN 1 END) as filled_count,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_count,
    COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_count
FROM advanced_orders
GROUP BY order_type, status;

-- Create view for user order summary
CREATE OR REPLACE VIEW user_order_summary AS
SELECT 
    user_id,
    COUNT(*) as total_orders,
    COUNT(CASE WHEN status = 'filled' THEN 1 END) as filled_orders,
    COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_orders,
    COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_orders,
    SUM(size) as total_size,
    SUM(filled_size) as total_filled_size,
    AVG(size) as avg_order_size,
    MAX(created_at) as last_order_time
FROM advanced_orders
GROUP BY user_id;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON advanced_orders TO your_app_user;
-- GRANT SELECT ON advanced_order_stats TO your_app_user;
-- GRANT SELECT ON user_order_summary TO your_app_user;
