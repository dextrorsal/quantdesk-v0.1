-- QuantDesk Production Database Schema for Solana Perpetual DEX
-- PostgreSQL with Supabase extensions - Production Ready
-- This schema implements all features needed for a professional perpetual trading DEX

-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- Create custom types for better type safety
CREATE TYPE position_side AS ENUM ('long', 'short');
CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop_loss', 'take_profit', 'trailing_stop', 'post_only', 'ioc', 'fok');
CREATE TYPE order_status AS ENUM ('pending', 'filled', 'cancelled', 'expired', 'partially_filled', 'rejected');
CREATE TYPE trade_side AS ENUM ('buy', 'sell');
CREATE TYPE liquidation_type AS ENUM ('market', 'backstop', 'insurance_fund');
CREATE TYPE funding_type AS ENUM ('automatic', 'manual', 'emergency');
CREATE TYPE risk_level AS ENUM ('low', 'medium', 'high', 'critical');
CREATE TYPE kyc_status AS ENUM ('pending', 'verified', 'rejected', 'suspended');
CREATE TYPE auction_side AS ENUM ('buy', 'sell');
CREATE TYPE settlement_type AS ENUM ('auction', 'insurance_fund', 'socialized_loss');

-- =============================================
-- CORE TRADING TABLES
-- =============================================

-- Users table (wallet-based authentication)
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wallet_address TEXT UNIQUE NOT NULL,
    username TEXT UNIQUE,
    email TEXT UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true,
    kyc_status kyc_status DEFAULT 'pending',
    risk_level risk_level DEFAULT 'medium',
    total_volume DECIMAL(20,8) DEFAULT 0,
    total_volume_usd DECIMAL(20,8) DEFAULT 0,
    total_trades INTEGER DEFAULT 0,
    total_pnl DECIMAL(20,8) DEFAULT 0,
    total_fees_paid DECIMAL(20,8) DEFAULT 0,
    referral_code TEXT UNIQUE,
    referred_by TEXT,
    metadata JSONB DEFAULT '{}',
    -- Solana specific fields
    program_account TEXT, -- Main program account for user
    authority_pubkey TEXT, -- User's authority public key
    sub_accounts JSONB DEFAULT '[]' -- Array of sub-account addresses
);

-- Markets table (perpetual contracts)
CREATE TABLE markets (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT UNIQUE NOT NULL, -- e.g., "BTC-PERP"
    base_asset TEXT NOT NULL, -- e.g., "BTC"
    quote_asset TEXT NOT NULL, -- e.g., "USDT"
    -- Solana program integration
    program_id TEXT NOT NULL, -- Solana program ID
    market_account TEXT NOT NULL, -- Solana market account
    oracle_account TEXT NOT NULL, -- Pyth oracle account
    -- Market configuration
    is_active BOOLEAN DEFAULT true,
    max_leverage INTEGER NOT NULL DEFAULT 100, -- Max leverage in basis points (100 = 1x)
    initial_margin_ratio INTEGER NOT NULL DEFAULT 500, -- 5% in basis points
    maintenance_margin_ratio INTEGER NOT NULL DEFAULT 300, -- 3% in basis points
    liquidation_fee_ratio INTEGER NOT NULL DEFAULT 200, -- 2% in basis points
    -- Order book configuration
    tick_size DECIMAL(20,8) NOT NULL DEFAULT 0.01,
    step_size DECIMAL(20,8) NOT NULL DEFAULT 0.001,
    min_order_size DECIMAL(20,8) NOT NULL DEFAULT 0.001,
    max_order_size DECIMAL(20,8) NOT NULL DEFAULT 1000000,
    -- Funding configuration
    funding_interval INTEGER NOT NULL DEFAULT 3600, -- seconds
    last_funding_time TIMESTAMP WITH TIME ZONE,
    current_funding_rate DECIMAL(10,6) DEFAULT 0,
    max_funding_rate DECIMAL(10,6) DEFAULT 1000, -- 10% max funding rate
    -- Risk management
    max_open_interest DECIMAL(20,8) DEFAULT 10000000, -- Max total open interest
    insurance_fund_ratio DECIMAL(10,6) DEFAULT 100, -- 1% insurance fund contribution
    -- Price limits
    max_price_change_ratio DECIMAL(10,6) DEFAULT 1000, -- 10% max price change per update
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- User balances (collateral management)
CREATE TABLE user_balances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    asset TEXT NOT NULL, -- e.g., "USDC", "SOL"
    balance DECIMAL(20,8) NOT NULL DEFAULT 0,
    locked_balance DECIMAL(20,8) NOT NULL DEFAULT 0, -- locked in positions
    available_balance DECIMAL(20,8) GENERATED ALWAYS AS (balance - locked_balance) STORED,
    -- Solana account info
    token_account TEXT, -- SPL token account address
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, asset)
);

-- Positions table (active trading positions)
CREATE TABLE positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    -- Solana account info
    position_account TEXT NOT NULL, -- Solana position account
    -- Position details
    side position_side NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    entry_price DECIMAL(20,8) NOT NULL,
    current_price DECIMAL(20,8),
    margin DECIMAL(20,8) NOT NULL,
    leverage INTEGER NOT NULL,
    -- P&L tracking
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    funding_fees DECIMAL(20,8) DEFAULT 0,
    trading_fees DECIMAL(20,8) DEFAULT 0,
    -- Risk management
    is_liquidated BOOLEAN DEFAULT false,
    liquidation_price DECIMAL(20,8),
    health_factor DECIMAL(10,6), -- Current health factor
    margin_ratio DECIMAL(10,6), -- Current margin ratio
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Orders table (order management)
CREATE TABLE orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    -- Solana account info
    order_account TEXT NOT NULL, -- Solana order account
    -- Order details
    order_type order_type NOT NULL,
    side position_side NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8),
    stop_price DECIMAL(20,8),
    trailing_distance DECIMAL(20,8),
    leverage INTEGER NOT NULL,
    -- Order status
    status order_status NOT NULL DEFAULT 'pending',
    filled_size DECIMAL(20,8) DEFAULT 0,
    remaining_size DECIMAL(20,8) GENERATED ALWAYS AS (size - filled_size) STORED,
    average_fill_price DECIMAL(20,8),
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE,
    filled_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Trades table (trade execution history)
CREATE TABLE trades (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    position_id UUID REFERENCES positions(id),
    order_id UUID REFERENCES orders(id),
    -- Solana account info
    trade_account TEXT NOT NULL, -- Solana trade account
    transaction_signature TEXT, -- Solana transaction signature
    -- Trade details
    side trade_side NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    value DECIMAL(20,8) GENERATED ALWAYS AS (size * price) STORED,
    fees DECIMAL(20,8) NOT NULL DEFAULT 0,
    pnl DECIMAL(20,8), -- Realized P&L for this trade
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- =============================================
-- FUNDING & RISK MANAGEMENT
-- =============================================

-- Funding rates history
CREATE TABLE funding_rates (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID NOT NULL REFERENCES markets(id),
    funding_rate DECIMAL(10,6) NOT NULL,
    premium_index DECIMAL(10,6) NOT NULL,
    oracle_price DECIMAL(20,8) NOT NULL,
    mark_price DECIMAL(20,8) NOT NULL,
    total_funding DECIMAL(20,8) NOT NULL DEFAULT 0,
    funding_type funding_type DEFAULT 'automatic',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Liquidations table
CREATE TABLE liquidations (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    position_id UUID NOT NULL REFERENCES positions(id),
    -- Liquidation details
    liquidator_address TEXT NOT NULL,
    liquidation_type liquidation_type NOT NULL,
    liquidated_size DECIMAL(20,8) NOT NULL,
    liquidation_price DECIMAL(20,8) NOT NULL,
    liquidation_fee DECIMAL(20,8) NOT NULL,
    remaining_margin DECIMAL(20,8),
    -- Solana transaction
    transaction_signature TEXT,
    -- Timestamps
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    -- Metadata
    metadata JSONB DEFAULT '{}'
);

-- Insurance fund
CREATE TABLE insurance_fund (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID NOT NULL REFERENCES markets(id),
    balance DECIMAL(20,8) NOT NULL DEFAULT 0,
    total_contributions DECIMAL(20,8) NOT NULL DEFAULT 0,
    total_payouts DECIMAL(20,8) NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- ORACLE & PRICE DATA
-- =============================================

-- Oracle prices (for historical tracking)
CREATE TABLE oracle_prices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID NOT NULL REFERENCES markets(id),
    price DECIMAL(20,8) NOT NULL,
    confidence DECIMAL(20,8) NOT NULL,
    exponent INTEGER NOT NULL,
    slot BIGINT, -- Solana slot number
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Mark prices (calculated from oracle + funding)
CREATE TABLE mark_prices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID NOT NULL REFERENCES markets(id),
    mark_price DECIMAL(20,8) NOT NULL,
    oracle_price DECIMAL(20,8) NOT NULL,
    funding_rate DECIMAL(10,6) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- ANALYTICS & STATISTICS
-- =============================================

-- Market statistics (daily rollups)
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

-- User statistics (daily rollups)
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

-- =============================================
-- JIT LIQUIDITY & AUCTIONS
-- =============================================

-- JIT liquidity auctions
CREATE TABLE auctions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT NOT NULL,
    side auction_side NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    reference_price DECIMAL(20,8) NOT NULL,
    max_slippage_bps INTEGER NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    settled BOOLEAN DEFAULT false,
    settlement_type settlement_type DEFAULT 'auction'
);

-- Auction quotes from market makers
CREATE TABLE auction_quotes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    auction_id UUID NOT NULL REFERENCES auctions(id) ON DELETE CASCADE,
    maker_id TEXT NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Auction settlements
CREATE TABLE auction_settlements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    auction_id UUID NOT NULL REFERENCES auctions(id) ON DELETE CASCADE,
    maker_id TEXT,
    fill_price DECIMAL(20,8),
    fill_size DECIMAL(20,8),
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- SYSTEM MONITORING
-- =============================================

-- System events (for monitoring and debugging)
CREATE TABLE system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type TEXT NOT NULL, -- 'liquidation', 'funding', 'oracle_update', 'error', 'risk_alert'
    event_data JSONB NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info', -- 'info', 'warning', 'error', 'critical'
    market_id UUID REFERENCES markets(id),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk alerts
CREATE TABLE risk_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    position_id UUID REFERENCES positions(id),
    alert_type TEXT NOT NULL, -- 'low_margin', 'high_leverage', 'large_position', 'unusual_activity'
    severity risk_level NOT NULL,
    message TEXT NOT NULL,
    is_resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- =============================================
-- ADMIN & AUDIT
-- =============================================

-- Admin users (for admin panel)
CREATE TABLE admin_users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin', -- 'admin', 'super_admin', 'risk_manager'
    permissions JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES admin_users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    -- OAuth fields
    google_id TEXT UNIQUE,
    github_id TEXT UNIQUE,
    avatar_url TEXT,
    oauth_provider TEXT -- 'google' or 'github'
);

-- Admin audit logs
CREATE TABLE admin_audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    admin_user_id UUID NOT NULL REFERENCES admin_users(id),
    action TEXT NOT NULL,
    resource TEXT NOT NULL,
    resource_id UUID,
    details JSONB DEFAULT '{}',
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- =============================================
-- PERFORMANCE INDEXES
-- =============================================

-- Users indexes
CREATE INDEX idx_users_wallet_address ON users(wallet_address);
CREATE INDEX idx_users_created_at ON users(created_at);
CREATE INDEX idx_users_kyc_status ON users(kyc_status);
CREATE INDEX idx_users_risk_level ON users(risk_level);

-- Markets indexes
CREATE INDEX idx_markets_symbol ON markets(symbol);
CREATE INDEX idx_markets_is_active ON markets(is_active);
CREATE INDEX idx_markets_program_id ON markets(program_id);

-- User balances indexes
CREATE INDEX idx_user_balances_user_id ON user_balances(user_id);
CREATE INDEX idx_user_balances_asset ON user_balances(asset);
CREATE INDEX idx_user_balances_user_asset ON user_balances(user_id, asset);

-- Positions indexes (critical for performance)
CREATE INDEX idx_positions_user_id ON positions(user_id);
CREATE INDEX idx_positions_market_id ON positions(market_id);
CREATE INDEX idx_positions_created_at ON positions(created_at);
CREATE INDEX idx_positions_is_liquidated ON positions(is_liquidated);
CREATE INDEX idx_positions_health_factor ON positions(health_factor);
CREATE INDEX idx_positions_margin_ratio ON positions(margin_ratio);
CREATE INDEX idx_positions_user_market ON positions(user_id, market_id);

-- Orders indexes
CREATE INDEX idx_orders_user_id ON orders(user_id);
CREATE INDEX idx_orders_market_id ON orders(market_id);
CREATE INDEX idx_orders_status ON orders(status);
CREATE INDEX idx_orders_created_at ON orders(created_at);
CREATE INDEX idx_orders_expires_at ON orders(expires_at);
CREATE INDEX idx_orders_user_market_status ON orders(user_id, market_id, status);

-- Trades indexes
CREATE INDEX idx_trades_user_id ON trades(user_id);
CREATE INDEX idx_trades_market_id ON trades(market_id);
CREATE INDEX idx_trades_position_id ON trades(position_id);
CREATE INDEX idx_trades_order_id ON trades(order_id);
CREATE INDEX idx_trades_created_at ON trades(created_at);
CREATE INDEX idx_trades_transaction_signature ON trades(transaction_signature);

-- Funding rates indexes
CREATE INDEX idx_funding_rates_market_id ON funding_rates(market_id);
CREATE INDEX idx_funding_rates_created_at ON funding_rates(created_at);

-- Liquidations indexes
CREATE INDEX idx_liquidations_user_id ON liquidations(user_id);
CREATE INDEX idx_liquidations_market_id ON liquidations(market_id);
CREATE INDEX idx_liquidations_position_id ON liquidations(position_id);
CREATE INDEX idx_liquidations_created_at ON liquidations(created_at);

-- Oracle prices indexes (critical for performance)
CREATE INDEX idx_oracle_prices_market_id ON oracle_prices(market_id);
CREATE INDEX idx_oracle_prices_created_at ON oracle_prices(created_at);
CREATE INDEX idx_oracle_prices_market_time ON oracle_prices(market_id, created_at DESC);

-- Mark prices indexes
CREATE INDEX idx_mark_prices_market_id ON mark_prices(market_id);
CREATE INDEX idx_mark_prices_created_at ON mark_prices(created_at);

-- Market stats indexes
CREATE INDEX idx_market_stats_market_id ON market_stats(market_id);
CREATE INDEX idx_market_stats_date ON market_stats(date);

-- User stats indexes
CREATE INDEX idx_user_stats_user_id ON user_stats(user_id);
CREATE INDEX idx_user_stats_date ON user_stats(date);

-- System events indexes
CREATE INDEX idx_system_events_event_type ON system_events(event_type);
CREATE INDEX idx_system_events_severity ON system_events(severity);
CREATE INDEX idx_system_events_created_at ON system_events(created_at);
CREATE INDEX idx_system_events_market_id ON system_events(market_id);

-- Risk alerts indexes
CREATE INDEX idx_risk_alerts_user_id ON risk_alerts(user_id);
CREATE INDEX idx_risk_alerts_market_id ON risk_alerts(market_id);
CREATE INDEX idx_risk_alerts_severity ON risk_alerts(severity);
CREATE INDEX idx_risk_alerts_is_resolved ON risk_alerts(is_resolved);

-- Auction indexes
CREATE INDEX idx_auctions_symbol ON auctions(symbol);
CREATE INDEX idx_auctions_expires_at ON auctions(expires_at);
CREATE INDEX idx_auctions_settled ON auctions(settled);
CREATE INDEX idx_auction_quotes_auction_id ON auction_quotes(auction_id);
CREATE INDEX idx_auction_settlements_auction_id ON auction_settlements(auction_id);

-- =============================================
-- TIMESCALEDB HYPERTABLES
-- =============================================

-- Convert time-series tables to hypertables for better performance
SELECT create_hypertable('oracle_prices', 'created_at');
SELECT create_hypertable('mark_prices', 'created_at');
SELECT create_hypertable('funding_rates', 'created_at');
SELECT create_hypertable('trades', 'created_at');
SELECT create_hypertable('system_events', 'created_at');

-- =============================================
-- ROW LEVEL SECURITY (RLS)
-- =============================================

-- Enable RLS on all user-related tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_balances ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_alerts ENABLE ROW LEVEL SECURITY;

-- Users can only see their own data
CREATE POLICY "Users can view own profile" ON users
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

CREATE POLICY "Users can view own risk alerts" ON risk_alerts
    FOR ALL USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

-- Markets are public read-only
ALTER TABLE markets ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read markets" ON markets
    FOR SELECT USING (true);

-- Oracle prices are public read-only
ALTER TABLE oracle_prices ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read oracle prices" ON oracle_prices
    FOR SELECT USING (true);

-- Mark prices are public read-only
ALTER TABLE mark_prices ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read mark prices" ON mark_prices
    FOR SELECT USING (true);

-- Market stats are public read-only
ALTER TABLE market_stats ENABLE ROW LEVEL SECURITY;
CREATE POLICY "Anyone can read market stats" ON market_stats
    FOR SELECT USING (true);

-- =============================================
-- DATABASE FUNCTIONS
-- =============================================

-- Function to update user balance
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

-- Function to calculate position health factor
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

-- Function to calculate funding rate
CREATE OR REPLACE FUNCTION calculate_funding_rate(
    p_market_id UUID
) RETURNS DECIMAL(10,6) AS $$
DECLARE
    market_record RECORD;
    oracle_price DECIMAL(20,8);
    mark_price DECIMAL(20,8);
    premium_index DECIMAL(10,6);
    funding_rate DECIMAL(10,6);
BEGIN
    SELECT * INTO market_record FROM markets WHERE id = p_market_id;
    
    -- Get latest oracle price
    SELECT price INTO oracle_price 
    FROM oracle_prices 
    WHERE market_id = p_market_id 
    ORDER BY created_at DESC 
    LIMIT 1;
    
    -- Get latest mark price
    SELECT mark_price INTO mark_price 
    FROM mark_prices 
    WHERE market_id = p_market_id 
    ORDER BY created_at DESC 
    LIMIT 1;
    
    -- Calculate premium index
    premium_index := ((mark_price - oracle_price) / oracle_price) * 10000; -- in basis points
    
    -- Calculate funding rate (simplified formula)
    funding_rate := premium_index * 0.1; -- 0.1% per hour max
    
    -- Cap funding rate
    IF funding_rate > market_record.max_funding_rate THEN
        funding_rate := market_record.max_funding_rate;
    ELSIF funding_rate < -market_record.max_funding_rate THEN
        funding_rate := -market_record.max_funding_rate;
    END IF;
    
    RETURN funding_rate;
END;
$$ LANGUAGE plpgsql;

-- Function to check liquidation conditions
CREATE OR REPLACE FUNCTION check_liquidation(
    p_position_id UUID
) RETURNS BOOLEAN AS $$
DECLARE
    position_record RECORD;
    market_record RECORD;
    current_price DECIMAL(20,8);
    equity DECIMAL(20,8);
    margin_ratio DECIMAL(10,6);
BEGIN
    SELECT * INTO position_record FROM positions WHERE id = p_position_id;
    SELECT * INTO market_record FROM markets WHERE id = position_record.market_id;
    
    -- Get current price
    SELECT price INTO current_price 
    FROM oracle_prices 
    WHERE market_id = position_record.market_id 
    ORDER BY created_at DESC 
    LIMIT 1;
    
    -- Calculate equity
    IF position_record.side = 'long' THEN
        equity := position_record.margin + ((current_price - position_record.entry_price) * position_record.size);
    ELSE
        equity := position_record.margin + ((position_record.entry_price - current_price) * position_record.size);
    END IF;
    
    -- Calculate margin ratio
    margin_ratio := (equity / (position_record.size * current_price)) * 100;
    
    -- Check if position should be liquidated
    RETURN margin_ratio <= market_record.maintenance_margin_ratio;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- TRIGGERS
-- =============================================

-- Trigger function for updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Apply updated_at triggers
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

CREATE TRIGGER update_insurance_fund_updated_at BEFORE UPDATE ON insurance_fund
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_admin_users_updated_at BEFORE UPDATE ON admin_users
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create indexes for OAuth fields
CREATE INDEX IF NOT EXISTS idx_admin_users_google_id ON admin_users(google_id);
CREATE INDEX IF NOT EXISTS idx_admin_users_github_id ON admin_users(github_id);

-- =============================================
-- VIEWS
-- =============================================

-- Active positions view
CREATE VIEW active_positions AS
SELECT 
    p.*,
    u.wallet_address,
    m.symbol,
    m.base_asset,
    m.quote_asset,
    m.maintenance_margin_ratio,
    m.liquidation_fee_ratio
FROM positions p
JOIN users u ON p.user_id = u.id
JOIN markets m ON p.market_id = m.id
WHERE p.size > 0 AND NOT p.is_liquidated;

-- Pending orders view
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

-- Market summary view
CREATE VIEW market_summary AS
SELECT 
    m.*,
    COALESCE(SUM(p.size), 0) as total_open_interest,
    COALESCE(COUNT(DISTINCT p.user_id), 0) as active_traders,
    COALESCE(AVG(p.leverage), 0) as avg_leverage,
    COALESCE(SUM(CASE WHEN p.side = 'long' THEN p.size ELSE 0 END), 0) as long_interest,
    COALESCE(SUM(CASE WHEN p.side = 'short' THEN p.size ELSE 0 END), 0) as short_interest,
    COALESCE(SUM(p.unrealized_pnl), 0) as total_unrealized_pnl
FROM markets m
LEFT JOIN positions p ON m.id = p.market_id AND p.size > 0 AND NOT p.is_liquidated
GROUP BY m.id;

-- User portfolio view
CREATE VIEW user_portfolio AS
SELECT 
    u.id,
    u.wallet_address,
    u.total_volume,
    u.total_pnl,
    u.total_fees_paid,
    COALESCE(SUM(ub.balance), 0) as total_balance,
    COALESCE(SUM(ub.locked_balance), 0) as total_locked,
    COALESCE(COUNT(p.id), 0) as active_positions,
    COALESCE(SUM(p.unrealized_pnl), 0) as total_unrealized_pnl
FROM users u
LEFT JOIN user_balances ub ON u.id = ub.user_id
LEFT JOIN positions p ON u.id = p.user_id AND p.size > 0 AND NOT p.is_liquidated
GROUP BY u.id, u.wallet_address, u.total_volume, u.total_pnl, u.total_fees_paid;

-- Risk dashboard view
CREATE VIEW risk_dashboard AS
SELECT 
    p.id,
    p.user_id,
    u.wallet_address,
    m.symbol,
    p.side,
    p.size,
    p.leverage,
    p.health_factor,
    p.margin_ratio,
    p.unrealized_pnl,
    CASE 
        WHEN p.health_factor <= 150 THEN 'critical'
        WHEN p.health_factor <= 200 THEN 'high'
        WHEN p.health_factor <= 300 THEN 'medium'
        ELSE 'low'
    END as risk_level
FROM positions p
JOIN users u ON p.user_id = u.id
JOIN markets m ON p.market_id = m.id
WHERE p.size > 0 AND NOT p.is_liquidated
ORDER BY p.health_factor ASC;

-- =============================================
-- SAMPLE DATA
-- =============================================

-- Insert sample markets
INSERT INTO markets (symbol, base_asset, quote_asset, program_id, market_account, oracle_account, max_leverage, initial_margin_ratio, maintenance_margin_ratio) VALUES
('BTC-PERP', 'BTC', 'USDT', 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J', 'BTC_MARKET_ACCOUNT', 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J', 100, 500, 300),
('ETH-PERP', 'ETH', 'USDT', 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J', 'ETH_MARKET_ACCOUNT', 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB', 100, 500, 300),
('SOL-PERP', 'SOL', 'USDT', 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J', 'SOL_MARKET_ACCOUNT', 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG', 100, 500, 300),
('AVAX-PERP', 'AVAX', 'USDT', 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J', 'AVAX_MARKET_ACCOUNT', 'AVAX_ORACLE_ACCOUNT', 50, 1000, 500),
('MATIC-PERP', 'MATIC', 'USDT', 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J', 'MATIC_MARKET_ACCOUNT', 'MATIC_ORACLE_ACCOUNT', 50, 1000, 500);

-- Initialize insurance fund for each market
INSERT INTO insurance_fund (market_id, balance, total_contributions)
SELECT id, 0, 0 FROM markets;

-- =============================================
-- PERMISSIONS
-- =============================================

-- Grant permissions to Supabase roles
GRANT USAGE ON SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL TABLES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO anon, authenticated;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO anon, authenticated;

-- Grant permissions to service role
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO service_role;

-- =============================================
-- COMMENTS
-- =============================================

COMMENT ON TABLE users IS 'User accounts with wallet-based authentication';
COMMENT ON TABLE markets IS 'Perpetual trading markets configuration';
COMMENT ON TABLE user_balances IS 'User collateral balances for trading';
COMMENT ON TABLE positions IS 'Active trading positions';
COMMENT ON TABLE orders IS 'Trading orders (pending, filled, cancelled)';
COMMENT ON TABLE trades IS 'Executed trades history';
COMMENT ON TABLE funding_rates IS 'Funding rate history for each market';
COMMENT ON TABLE liquidations IS 'Liquidation events and details';
COMMENT ON TABLE oracle_prices IS 'Oracle price feeds from Pyth';
COMMENT ON TABLE mark_prices IS 'Mark prices calculated from oracle + funding';
COMMENT ON TABLE insurance_fund IS 'Insurance fund balances per market';
COMMENT ON TABLE auctions IS 'JIT liquidity auctions';
COMMENT ON TABLE system_events IS 'System monitoring and debugging events';
COMMENT ON TABLE risk_alerts IS 'Risk management alerts for users';

COMMENT ON COLUMN users.wallet_address IS 'Primary identifier - Solana wallet address';
COMMENT ON COLUMN users.program_account IS 'Main program account for user on Solana';
COMMENT ON COLUMN users.authority_pubkey IS 'User authority public key for signing transactions';
COMMENT ON COLUMN markets.max_leverage IS 'Maximum leverage in basis points (100 = 1x)';
COMMENT ON COLUMN markets.initial_margin_ratio IS 'Initial margin requirement in basis points (500 = 5%)';
COMMENT ON COLUMN markets.maintenance_margin_ratio IS 'Maintenance margin requirement in basis points (300 = 3%)';
COMMENT ON COLUMN positions.health_factor IS 'Current health factor (margin / position_value * 100)';
COMMENT ON COLUMN positions.liquidation_price IS 'Price at which position will be liquidated';
COMMENT ON COLUMN trades.transaction_signature IS 'Solana transaction signature for trade execution';
COMMENT ON COLUMN oracle_prices.slot IS 'Solana slot number for price update';
COMMENT ON COLUMN auctions.max_slippage_bps IS 'Maximum slippage allowed in basis points';
