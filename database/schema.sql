-- QuantDesk Production Database Schema
-- PostgreSQL with Supabase extensions

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb";

-- Create custom types
CREATE TYPE position_side AS ENUM ('long', 'short');
CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop_loss', 'take_profit', 'trailing_stop', 'post_only', 'ioc', 'fok');
CREATE TYPE order_status AS ENUM ('pending', 'filled', 'cancelled', 'expired', 'partially_filled');
CREATE TYPE trade_side AS ENUM ('buy', 'sell');
CREATE TYPE liquidation_type AS ENUM ('market', 'backstop');

-- JIT auction enums
DO $$ BEGIN
  IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'auction_side') THEN
    CREATE TYPE auction_side AS ENUM ('buy', 'sell');
  END IF;
END $$;

-- Users table (wallet-based authentication)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wallet_address TEXT UNIQUE NOT NULL,
    username TEXT,
    email TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    kyc_status TEXT DEFAULT 'pending', -- pending, verified, rejected
    risk_level TEXT DEFAULT 'medium', -- low, medium, high
    total_volume DECIMAL(20,8) DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    metadata JSONB DEFAULT '{}'
);

-- Markets table
CREATE TABLE markets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT UNIQUE NOT NULL, -- e.g., "BTC-PERP"
    base_asset TEXT NOT NULL, -- e.g., "BTC"
    quote_asset TEXT NOT NULL, -- e.g., "USDT"
    program_id TEXT NOT NULL, -- Solana program ID
    market_account TEXT NOT NULL, -- Solana market account
    oracle_account TEXT NOT NULL, -- Pyth oracle account
    is_active BOOLEAN DEFAULT true,
    max_leverage INTEGER NOT NULL DEFAULT 100,
    initial_margin_ratio INTEGER NOT NULL DEFAULT 500, -- 5% in basis points
    maintenance_margin_ratio INTEGER NOT NULL DEFAULT 300, -- 3% in basis points
    tick_size DECIMAL(20,8) NOT NULL DEFAULT 0.01,
    step_size DECIMAL(20,8) NOT NULL DEFAULT 0.001,
    min_order_size DECIMAL(20,8) NOT NULL DEFAULT 0.001,
    max_order_size DECIMAL(20,8) NOT NULL DEFAULT 1000000,
    funding_interval INTEGER NOT NULL DEFAULT 3600, -- seconds
    last_funding_time TIMESTAMP WITH TIME ZONE,
    current_funding_rate DECIMAL(10,6) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- User balances (collateral)
CREATE TABLE user_balances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    asset TEXT NOT NULL, -- e.g., "USDC", "SOL"
    balance DECIMAL(20,8) NOT NULL DEFAULT 0,
    locked_balance DECIMAL(20,8) NOT NULL DEFAULT 0, -- locked in positions
    available_balance DECIMAL(20,8) GENERATED ALWAYS AS (balance - locked_balance) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, asset)
);

-- Positions table
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    position_account TEXT NOT NULL, -- Solana position account
    side position_side NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    margin DECIMAL(20,8) NOT NULL,
    leverage INTEGER NOT NULL,
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    funding_fees DECIMAL(20,8) DEFAULT 0,
    is_liquidated BOOLEAN DEFAULT false,
    liquidation_price DECIMAL(20,8),
    health_factor DECIMAL(10,6), -- Current health factor
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE
);

-- Orders table
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    order_account TEXT NOT NULL, -- Solana order account
    order_type order_type NOT NULL,
    side position_side NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8),
    stop_price DECIMAL(20,8),
    trailing_distance DECIMAL(20,8),
    leverage INTEGER NOT NULL,
    status order_status NOT NULL DEFAULT 'pending',
    filled_size DECIMAL(20,8) DEFAULT 0,
    remaining_size DECIMAL(20,8) GENERATED ALWAYS AS (size - filled_size) STORED,
    average_fill_price DECIMAL(20,8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE
);

-- Trades table (for trade history)
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    position_id UUID REFERENCES positions(id),
    order_id UUID REFERENCES orders(id),
    trade_account TEXT NOT NULL, -- Solana trade account
    side trade_side NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    value DECIMAL(20,8) GENERATED ALWAYS AS (size * price) STORED,
    fees DECIMAL(20,8) NOT NULL DEFAULT 0,
    pnl DECIMAL(20,8), -- Realized P&L for this trade
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Funding rates history
CREATE TABLE funding_rates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID NOT NULL REFERENCES markets(id),
    funding_rate DECIMAL(10,6) NOT NULL,
    premium_index DECIMAL(10,6) NOT NULL,
    oracle_price DECIMAL(20,8) NOT NULL,
    mark_price DECIMAL(20,8) NOT NULL,
    total_funding DECIMAL(20,8) NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Liquidations table
CREATE TABLE liquidations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    position_id UUID NOT NULL REFERENCES positions(id),
    liquidator_address TEXT NOT NULL,
    liquidation_type liquidation_type NOT NULL,
    liquidated_size DECIMAL(20,8) NOT NULL,
    liquidation_price DECIMAL(20,8) NOT NULL,
    liquidation_fee DECIMAL(20,8) NOT NULL,
    remaining_margin DECIMAL(20,8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Oracle prices (for historical tracking)
CREATE TABLE oracle_prices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID NOT NULL REFERENCES markets(id),
    price DECIMAL(20,8) NOT NULL,
    confidence DECIMAL(20,8) NOT NULL,
    exponent INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market statistics (for analytics)
CREATE TABLE market_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID NOT NULL REFERENCES markets(id),
    date DATE NOT NULL,
    open_price DECIMAL(20,8),
    high_price DECIMAL(20,8),
    low_price DECIMAL(20,8),
    close_price DECIMAL(20,8),
    volume DECIMAL(20,8) DEFAULT 0,
    volume_usd DECIMAL(20,8) DEFAULT 0,
    trades_count INTEGER DEFAULT 0,
    open_interest DECIMAL(20,8) DEFAULT 0,
    funding_rate DECIMAL(10,6) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(market_id, date)
);

-- User statistics (for analytics)
CREATE TABLE user_stats (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    date DATE NOT NULL,
    total_volume DECIMAL(20,8) DEFAULT 0,
    total_volume_usd DECIMAL(20,8) DEFAULT 0,
    trades_count INTEGER DEFAULT 0,
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    fees_paid DECIMAL(20,8) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, date)
);

-- System events (for monitoring and debugging)
CREATE TABLE system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type TEXT NOT NULL, -- 'liquidation', 'funding', 'oracle_update', 'error'
    event_data JSONB NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info', -- 'info', 'warning', 'error', 'critical'
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_users_wallet_address ON users(wallet_address);
CREATE INDEX idx_users_created_at ON users(created_at);

CREATE INDEX idx_markets_symbol ON markets(symbol);
CREATE INDEX idx_markets_is_active ON markets(is_active);

CREATE INDEX idx_user_balances_user_id ON user_balances(user_id);
CREATE INDEX idx_user_balances_asset ON user_balances(asset);

CREATE INDEX idx_positions_user_id ON positions(user_id);
CREATE INDEX idx_positions_market_id ON positions(market_id);
CREATE INDEX idx_positions_created_at ON positions(created_at);
CREATE INDEX idx_positions_is_liquidated ON positions(is_liquidated);
CREATE INDEX idx_positions_health_factor ON positions(health_factor);

CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_market_id ON orders(market_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_orders_expires_at ON orders(expires_at);

CREATE INDEX idx_trades_user_id ON trades(user_id);
CREATE INDEX idx_trades_market_id ON trades(market_id);
CREATE INDEX idx_trades_position_id ON trades(position_id);
CREATE INDEX idx_trades_order_id ON trades(order_id);
CREATE INDEX idx_trades_created_at ON trades(created_at);

CREATE INDEX idx_funding_rates_market_id ON funding_rates(market_id);
CREATE INDEX idx_funding_rates_created_at ON funding_rates(created_at);

CREATE INDEX idx_liquidations_user_id ON liquidations(user_id);
CREATE INDEX idx_liquidations_market_id ON liquidations(market_id);
CREATE INDEX idx_liquidations_position_id ON liquidations(position_id);
CREATE INDEX idx_liquidations_created_at ON liquidations(created_at);

CREATE INDEX idx_oracle_prices_market_id ON oracle_prices(market_id);
CREATE INDEX idx_oracle_prices_created_at ON oracle_prices(created_at);

CREATE INDEX idx_market_stats_market_id ON market_stats(market_id);
CREATE INDEX idx_market_stats_date ON market_stats(date);

CREATE INDEX idx_user_stats_user_id ON user_stats(user_id);
CREATE INDEX idx_user_stats_date ON user_stats(date);

CREATE INDEX idx_system_events_event_type ON system_events(event_type);
CREATE INDEX idx_system_events_severity ON system_events(severity);
CREATE INDEX idx_system_events_created_at ON system_events(created_at);

-- Create hypertables for time-series data (TimescaleDB)
SELECT create_hypertable('oracle_prices', 'created_at');
SELECT create_hypertable('funding_rates', 'created_at');
SELECT create_hypertable('trades', 'created_at');
SELECT create_hypertable('system_events', 'created_at');

-- JIT liquidity tables
CREATE TABLE IF NOT EXISTS auctions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT NOT NULL,
    side auction_side NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    reference_price DECIMAL(20,8) NOT NULL,
    max_slippage_bps INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    settled BOOLEAN DEFAULT false
);

CREATE TABLE IF NOT EXISTS auction_quotes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    auction_id UUID NOT NULL REFERENCES auctions(id) ON DELETE CASCADE,
    maker_id TEXT NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS auction_settlements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    auction_id UUID NOT NULL REFERENCES auctions(id) ON DELETE CASCADE,
    maker_id TEXT,
    fill_price DECIMAL(20,8),
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_auctions_symbol ON auctions(symbol);
CREATE INDEX IF NOT EXISTS idx_auctions_expires_at ON auctions(expires_at);
CREATE INDEX IF NOT EXISTS idx_auction_quotes_auction_id ON auction_quotes(auction_id);
CREATE INDEX IF NOT EXISTS idx_auction_settlements_auction_id ON auction_settlements(auction_id);

-- Row Level Security (RLS) policies
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_balances ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_stats ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY "Users can view own data" ON users
    FOR ALL USING (auth.jwt() ->> 'wallet_address' = wallet_address);

CREATE POLICY "Users can view own balances" ON user_balances
    FOR ALL USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

CREATE POLICY "Users can view own positions" ON positions
    FOR ALL USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

CREATE POLICY "Users can view own orders" ON orders
    FOR ALL USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

CREATE POLICY "Users can view own trades" ON trades
    FOR ALL USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

CREATE POLICY "Users can view own stats" ON user_stats
    FOR ALL USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

-- Functions for common operations
CREATE OR REPLACE FUNCTION update_user_balance(
    p_user_id UUID,
    p_asset TEXT,
    p_amount DECIMAL(20,8)
) RETURNS VOID AS $$
BEGIN
    INSERT INTO user_balances (user_id, asset, balance)
    VALUES (p_user_id, p_asset, p_amount)
    ON CONFLICT (user_id, asset)
    DO UPDATE SET 
        balance = user_balances.balance + p_amount,
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

CREATE OR REPLACE FUNCTION calculate_position_health(
    p_position_id UUID
) RETURNS DECIMAL(10,6) AS $$
DECLARE
    position_record RECORD;
    current_price DECIMAL(20,8);
    equity DECIMAL(20,8);
    health_factor DECIMAL(10,6);
BEGIN
    SELECT * INTO position_record FROM positions WHERE id = p_position_id;
    
    -- Get current price from oracle
    SELECT price INTO current_price 
    FROM oracle_prices 
    WHERE market_id = position_record.market_id 
    ORDER BY created_at DESC 
    LIMIT 1;
    
    -- Calculate unrealized P&L
    IF position_record.side = 'long' THEN
        equity := position_record.margin + ((current_price - position_record.entry_price) * position_record.size);
    ELSE
        equity := position_record.margin + ((position_record.entry_price - current_price) * position_record.size);
    END IF;
    
    -- Calculate health factor
    health_factor := (equity / (position_record.size * current_price)) * 100;
    
    RETURN health_factor;
END;
$$ LANGUAGE plpgsql;

-- Triggers for automatic updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_markets_updated_at BEFORE UPDATE ON markets
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_balances_updated_at BEFORE UPDATE ON user_balances
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON positions
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_orders_updated_at BEFORE UPDATE ON orders
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Views for common queries
CREATE VIEW active_positions AS
SELECT 
    p.*,
    u.wallet_address,
    m.symbol,
    m.base_asset,
    m.quote_asset
FROM positions p
JOIN users u ON p.user_id = u.id
JOIN markets m ON p.market_id = m.id
WHERE p.size > 0 AND NOT p.is_liquidated;

CREATE VIEW pending_orders AS
SELECT 
    o.*,
    u.wallet_address,
    m.symbol,
    m.base_asset,
    m.quote_asset
FROM orders o
JOIN users u ON o.user_id = u.id
JOIN markets m ON o.market_id = m.id
WHERE o.status = 'pending' AND o.expires_at > NOW();

CREATE VIEW market_summary AS
SELECT 
    m.*,
    COALESCE(SUM(p.size), 0) as total_open_interest,
    COALESCE(COUNT(DISTINCT p.user_id), 0) as active_traders,
    COALESCE(AVG(p.leverage), 0) as avg_leverage,
    COALESCE(SUM(CASE WHEN p.side = 'long' THEN p.size ELSE 0 END), 0) as long_interest,
    COALESCE(SUM(CASE WHEN p.side = 'short' THEN p.size ELSE 0 END), 0) as short_interest
FROM markets m
LEFT JOIN positions p ON m.id = p.market_id AND p.size > 0 AND NOT p.is_liquidated
GROUP BY m.id;

-- Insert sample data for development
INSERT INTO markets (symbol, base_asset, quote_asset, program_id, market_account, oracle_account, max_leverage, initial_margin_ratio, maintenance_margin_ratio) VALUES
('BTC-PERP', 'BTC', 'USDT', 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J', 'BTC_MARKET_ACCOUNT', 'BTC_ORACLE_ACCOUNT', 100, 500, 300),
('ETH-PERP', 'ETH', 'USDT', 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J', 'ETH_MARKET_ACCOUNT', 'ETH_ORACLE_ACCOUNT', 100, 500, 300),
('SOL-PERP', 'SOL', 'USDT', 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J', 'SOL_MARKET_ACCOUNT', 'SOL_ORACLE_ACCOUNT', 100, 500, 300);

-- Grant permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;
