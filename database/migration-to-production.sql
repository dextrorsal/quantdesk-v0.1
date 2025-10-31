-- QuantDesk Database Migration Script
-- Migrates existing database to production-ready perpetual DEX schema
-- Run this script to upgrade your current database

-- =============================================
-- BACKUP EXISTING DATA (IMPORTANT!)
-- =============================================

-- Before running this migration, backup your existing data:
-- pg_dump -h your-host -U your-user -d your-database > backup_before_migration.sql

-- =============================================
-- STEP 1: ADD MISSING EXTENSIONS
-- =============================================

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "timescaledb";
CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";

-- =============================================
-- STEP 2: CREATE MISSING CUSTOM TYPES
-- =============================================

-- Create custom types if they don't exist
DO $$ BEGIN
    CREATE TYPE position_side AS ENUM ('long', 'short');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE order_type AS ENUM ('market', 'limit', 'stop_loss', 'take_profit', 'trailing_stop', 'post_only', 'ioc', 'fok');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE order_status AS ENUM ('pending', 'filled', 'cancelled', 'expired', 'partially_filled', 'rejected');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE trade_side AS ENUM ('buy', 'sell');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE liquidation_type AS ENUM ('market', 'backstop', 'insurance_fund');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE funding_type AS ENUM ('automatic', 'manual', 'emergency');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE risk_level AS ENUM ('low', 'medium', 'high', 'critical');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE kyc_status AS ENUM ('pending', 'verified', 'rejected', 'suspended');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE auction_side AS ENUM ('buy', 'sell');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

DO $$ BEGIN
    CREATE TYPE settlement_type AS ENUM ('auction', 'insurance_fund', 'socialized_loss');
EXCEPTION
    WHEN duplicate_object THEN null;
END $$;

-- =============================================
-- STEP 3: UPDATE EXISTING TABLES
-- =============================================

-- Update users table with new columns
ALTER TABLE users 
ADD COLUMN IF NOT EXISTS kyc_status kyc_status DEFAULT 'pending',
ADD COLUMN IF NOT EXISTS risk_level risk_level DEFAULT 'medium',
ADD COLUMN IF NOT EXISTS total_volume_usd DECIMAL(20,8) DEFAULT 0,
ADD COLUMN IF NOT EXISTS total_pnl DECIMAL(20,8) DEFAULT 0,
ADD COLUMN IF NOT EXISTS total_fees_paid DECIMAL(20,8) DEFAULT 0,
ADD COLUMN IF NOT EXISTS referral_code TEXT UNIQUE,
ADD COLUMN IF NOT EXISTS referred_by TEXT,
ADD COLUMN IF NOT EXISTS program_account TEXT,
ADD COLUMN IF NOT EXISTS authority_pubkey TEXT,
ADD COLUMN IF NOT EXISTS sub_accounts JSONB DEFAULT '[]';

-- Update markets table with new columns
ALTER TABLE markets 
ADD COLUMN IF NOT EXISTS program_id TEXT DEFAULT 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J',
ADD COLUMN IF NOT EXISTS market_account TEXT DEFAULT 'MARKET_ACCOUNT',
ADD COLUMN IF NOT EXISTS oracle_account TEXT DEFAULT 'ORACLE_ACCOUNT',
ADD COLUMN IF NOT EXISTS max_leverage INTEGER DEFAULT 100,
ADD COLUMN IF NOT EXISTS initial_margin_ratio INTEGER DEFAULT 500,
ADD COLUMN IF NOT EXISTS maintenance_margin_ratio INTEGER DEFAULT 300,
ADD COLUMN IF NOT EXISTS liquidation_fee_ratio INTEGER DEFAULT 200,
ADD COLUMN IF NOT EXISTS tick_size DECIMAL(20,8) DEFAULT 0.01,
ADD COLUMN IF NOT EXISTS step_size DECIMAL(20,8) DEFAULT 0.001,
ADD COLUMN IF NOT EXISTS min_order_size DECIMAL(20,8) DEFAULT 0.001,
ADD COLUMN IF NOT EXISTS max_order_size DECIMAL(20,8) DEFAULT 1000000,
ADD COLUMN IF NOT EXISTS funding_interval INTEGER DEFAULT 3600,
ADD COLUMN IF NOT EXISTS last_funding_time TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS current_funding_rate DECIMAL(10,6) DEFAULT 0,
ADD COLUMN IF NOT EXISTS max_funding_rate DECIMAL(10,6) DEFAULT 1000,
ADD COLUMN IF NOT EXISTS max_open_interest DECIMAL(20,8) DEFAULT 10000000,
ADD COLUMN IF NOT EXISTS insurance_fund_ratio DECIMAL(10,6) DEFAULT 100,
ADD COLUMN IF NOT EXISTS max_price_change_ratio DECIMAL(10,6) DEFAULT 1000;

-- Update user_balances table
ALTER TABLE user_balances 
ADD COLUMN IF NOT EXISTS token_account TEXT;

-- Update positions table with new columns
ALTER TABLE positions 
ADD COLUMN IF NOT EXISTS position_account TEXT DEFAULT 'POSITION_ACCOUNT',
ADD COLUMN IF NOT EXISTS current_price DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS funding_fees DECIMAL(20,8) DEFAULT 0,
ADD COLUMN IF NOT EXISTS trading_fees DECIMAL(20,8) DEFAULT 0,
ADD COLUMN IF NOT EXISTS liquidation_price DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS health_factor DECIMAL(10,6),
ADD COLUMN IF NOT EXISTS margin_ratio DECIMAL(10,6),
ADD COLUMN IF NOT EXISTS closed_at TIMESTAMP WITH TIME ZONE;

-- Update orders table with new columns
ALTER TABLE orders 
ADD COLUMN IF NOT EXISTS order_account TEXT DEFAULT 'ORDER_ACCOUNT',
ADD COLUMN IF NOT EXISTS stop_price DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS trailing_distance DECIMAL(20,8),
ADD COLUMN IF NOT EXISTS leverage INTEGER DEFAULT 1,
ADD COLUMN IF NOT EXISTS remaining_size DECIMAL(20,8) GENERATED ALWAYS AS (size - filled_size) STORED,
ADD COLUMN IF NOT EXISTS expires_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS filled_at TIMESTAMP WITH TIME ZONE,
ADD COLUMN IF NOT EXISTS cancelled_at TIMESTAMP WITH TIME ZONE;

-- Update trades table with new columns
ALTER TABLE trades 
ADD COLUMN IF NOT EXISTS trade_account TEXT DEFAULT 'TRADE_ACCOUNT',
ADD COLUMN IF NOT EXISTS transaction_signature TEXT,
ADD COLUMN IF NOT EXISTS pnl DECIMAL(20,8);

-- =============================================
-- STEP 4: CREATE NEW TABLES
-- =============================================

-- Funding rates history
CREATE TABLE IF NOT EXISTS funding_rates (
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
CREATE TABLE IF NOT EXISTS liquidations (
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
    transaction_signature TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Insurance fund
CREATE TABLE IF NOT EXISTS insurance_fund (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID NOT NULL REFERENCES markets(id),
    balance DECIMAL(20,8) NOT NULL DEFAULT 0,
    total_contributions DECIMAL(20,8) NOT NULL DEFAULT 0,
    total_payouts DECIMAL(20,8) NOT NULL DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Oracle prices (rename existing price_feeds if it exists)
DO $$ 
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'price_feeds') THEN
        ALTER TABLE price_feeds RENAME TO oracle_prices;
        ALTER TABLE oracle_prices ADD COLUMN IF NOT EXISTS slot BIGINT;
    ELSE
        CREATE TABLE oracle_prices (
            id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
            market_id UUID NOT NULL REFERENCES markets(id),
            price DECIMAL(20,8) NOT NULL,
            confidence DECIMAL(20,8) NOT NULL,
            exponent INTEGER NOT NULL,
            slot BIGINT,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
        );
    END IF;
END $$;

-- Mark prices
CREATE TABLE IF NOT EXISTS mark_prices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    market_id UUID NOT NULL REFERENCES markets(id),
    mark_price DECIMAL(20,8) NOT NULL,
    oracle_price DECIMAL(20,8) NOT NULL,
    funding_rate DECIMAL(10,6) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Market statistics (daily rollups)
CREATE TABLE IF NOT EXISTS market_stats (
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
CREATE TABLE IF NOT EXISTS user_stats (
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

-- JIT liquidity auctions
CREATE TABLE IF NOT EXISTS auctions (
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
CREATE TABLE IF NOT EXISTS auction_quotes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    auction_id UUID NOT NULL REFERENCES auctions(id) ON DELETE CASCADE,
    maker_id TEXT NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    size DECIMAL(20,8) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Auction settlements
CREATE TABLE IF NOT EXISTS auction_settlements (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    auction_id UUID NOT NULL REFERENCES auctions(id) ON DELETE CASCADE,
    maker_id TEXT,
    fill_price DECIMAL(20,8),
    fill_size DECIMAL(20,8),
    reason TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- System events (for monitoring and debugging)
CREATE TABLE IF NOT EXISTS system_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    severity TEXT NOT NULL DEFAULT 'info',
    market_id UUID REFERENCES markets(id),
    user_id UUID REFERENCES users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Risk alerts
CREATE TABLE IF NOT EXISTS risk_alerts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_id UUID NOT NULL REFERENCES markets(id),
    position_id UUID REFERENCES positions(id),
    alert_type TEXT NOT NULL,
    severity risk_level NOT NULL,
    message TEXT NOT NULL,
    is_resolved BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    resolved_at TIMESTAMP WITH TIME ZONE
);

-- Admin users (for admin panel)
CREATE TABLE IF NOT EXISTS admin_users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    username TEXT UNIQUE NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password_hash TEXT NOT NULL,
    role TEXT NOT NULL DEFAULT 'admin',
    permissions JSONB DEFAULT '{}',
    is_active BOOLEAN DEFAULT true,
    created_by UUID REFERENCES admin_users(id),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE
);

-- Admin audit logs
CREATE TABLE IF NOT EXISTS admin_audit_logs (
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
-- STEP 5: ADD MISSING INDEXES
-- =============================================

-- Users indexes
CREATE INDEX IF NOT EXISTS idx_users_kyc_status ON users(kyc_status);
CREATE INDEX IF NOT EXISTS idx_users_risk_level ON users(risk_level);

-- Markets indexes
CREATE INDEX IF NOT EXISTS idx_markets_program_id ON markets(program_id);

-- User balances indexes
CREATE INDEX IF NOT EXISTS idx_user_balances_user_asset ON user_balances(user_id, asset);

-- Positions indexes (critical for performance)
CREATE INDEX IF NOT EXISTS idx_positions_health_factor ON positions(health_factor);
CREATE INDEX IF NOT EXISTS idx_positions_margin_ratio ON positions(margin_ratio);
CREATE INDEX IF NOT EXISTS idx_positions_user_market ON positions(user_id, market_id);

-- Orders indexes
CREATE INDEX IF NOT EXISTS idx_orders_expires_at ON orders(expires_at);
CREATE INDEX IF NOT EXISTS idx_orders_user_market_status ON orders(user_id, market_id, status);

-- Trades indexes
CREATE INDEX IF NOT EXISTS idx_trades_transaction_signature ON trades(transaction_signature);

-- Funding rates indexes
CREATE INDEX IF NOT EXISTS idx_funding_rates_market_id ON funding_rates(market_id);
CREATE INDEX IF NOT EXISTS idx_funding_rates_created_at ON funding_rates(created_at);

-- Liquidations indexes
CREATE INDEX IF NOT EXISTS idx_liquidations_user_id ON liquidations(user_id);
CREATE INDEX IF NOT EXISTS idx_liquidations_market_id ON liquidations(market_id);
CREATE INDEX IF NOT EXISTS idx_liquidations_position_id ON liquidations(position_id);
CREATE INDEX IF NOT EXISTS idx_liquidations_created_at ON liquidations(created_at);

-- Oracle prices indexes (critical for performance)
CREATE INDEX IF NOT EXISTS idx_oracle_prices_market_time ON oracle_prices(market_id, created_at DESC);

-- Mark prices indexes
CREATE INDEX IF NOT EXISTS idx_mark_prices_market_id ON mark_prices(market_id);
CREATE INDEX IF NOT EXISTS idx_mark_prices_created_at ON mark_prices(created_at);

-- Market stats indexes
CREATE INDEX IF NOT EXISTS idx_market_stats_market_id ON market_stats(market_id);
CREATE INDEX IF NOT EXISTS idx_market_stats_date ON market_stats(date);

-- User stats indexes
CREATE INDEX IF NOT EXISTS idx_user_stats_user_id ON user_stats(user_id);
CREATE INDEX IF NOT EXISTS idx_user_stats_date ON user_stats(date);

-- System events indexes
CREATE INDEX IF NOT EXISTS idx_system_events_event_type ON system_events(event_type);
CREATE INDEX IF NOT EXISTS idx_system_events_severity ON system_events(severity);
CREATE INDEX IF NOT EXISTS idx_system_events_created_at ON system_events(created_at);
CREATE INDEX IF NOT EXISTS idx_system_events_market_id ON system_events(market_id);

-- Risk alerts indexes
CREATE INDEX IF NOT EXISTS idx_risk_alerts_user_id ON risk_alerts(user_id);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_market_id ON risk_alerts(market_id);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_severity ON risk_alerts(severity);
CREATE INDEX IF NOT EXISTS idx_risk_alerts_is_resolved ON risk_alerts(is_resolved);

-- Auction indexes
CREATE INDEX IF NOT EXISTS idx_auctions_symbol ON auctions(symbol);
CREATE INDEX IF NOT EXISTS idx_auctions_expires_at ON auctions(expires_at);
CREATE INDEX IF NOT EXISTS idx_auctions_settled ON auctions(settled);
CREATE INDEX IF NOT EXISTS idx_auction_quotes_auction_id ON auction_quotes(auction_id);
CREATE INDEX IF NOT EXISTS idx_auction_settlements_auction_id ON auction_settlements(auction_id);

-- =============================================
-- STEP 6: CONVERT TO HYPERTABLES (TimescaleDB)
-- =============================================

-- Convert time-series tables to hypertables for better performance
DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'oracle_prices') THEN
        PERFORM create_hypertable('oracle_prices', 'created_at');
    END IF;
END $$;

DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'mark_prices') THEN
        PERFORM create_hypertable('mark_prices', 'created_at');
    END IF;
END $$;

DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'funding_rates') THEN
        PERFORM create_hypertable('funding_rates', 'created_at');
    END IF;
END $$;

DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'trades') THEN
        PERFORM create_hypertable('trades', 'created_at');
    END IF;
END $$;

DO $$ 
BEGIN
    IF NOT EXISTS (SELECT 1 FROM timescaledb_information.hypertables WHERE hypertable_name = 'system_events') THEN
        PERFORM create_hypertable('system_events', 'created_at');
    END IF;
END $$;

-- =============================================
-- STEP 7: CREATE DATABASE FUNCTIONS
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
-- STEP 8: ENABLE ROW LEVEL SECURITY
-- =============================================

-- Enable RLS on all user-related tables
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_balances ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE user_stats ENABLE ROW LEVEL SECURITY;
ALTER TABLE risk_alerts ENABLE ROW LEVEL SECURITY;

-- Drop existing policies if they exist
DROP POLICY IF EXISTS "Users can view own profile" ON users;
DROP POLICY IF EXISTS "Users can view own balances" ON user_balances;
DROP POLICY IF EXISTS "Users can view own positions" ON positions;
DROP POLICY IF EXISTS "Users can view own orders" ON orders;
DROP POLICY IF EXISTS "Users can view own trades" ON trades;
DROP POLICY IF EXISTS "Users can view own stats" ON user_stats;
DROP POLICY IF EXISTS "Users can view own risk alerts" ON risk_alerts;

-- Create new RLS policies
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
DROP POLICY IF EXISTS "Anyone can read markets" ON markets;
CREATE POLICY "Anyone can read markets" ON markets
    FOR SELECT USING (true);

-- Oracle prices are public read-only
ALTER TABLE oracle_prices ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Anyone can read oracle prices" ON oracle_prices;
CREATE POLICY "Anyone can read oracle prices" ON oracle_prices
    FOR SELECT USING (true);

-- Mark prices are public read-only
ALTER TABLE mark_prices ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Anyone can read mark prices" ON mark_prices;
CREATE POLICY "Anyone can read mark prices" ON mark_prices
    FOR SELECT USING (true);

-- Market stats are public read-only
ALTER TABLE market_stats ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "Anyone can read market stats" ON market_stats;
CREATE POLICY "Anyone can read market stats" ON market_stats
    FOR SELECT USING (true);

-- =============================================
-- STEP 9: CREATE UPDATED VIEWS
-- =============================================

-- Drop existing views
DROP VIEW IF EXISTS active_positions;
DROP VIEW IF EXISTS pending_orders;
DROP VIEW IF EXISTS order_history;

-- Create updated views
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
    COALESCE(SUM(CASE WHEN p.side = 'short' THEN p.size ELSE 0 END), 0) as short_interest,
    COALESCE(SUM(p.unrealized_pnl), 0) as total_unrealized_pnl
FROM markets m
LEFT JOIN positions p ON m.id = p.market_id AND p.size > 0 AND NOT p.is_liquidated
GROUP BY m.id;

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
-- STEP 10: INITIALIZE DATA
-- =============================================

-- Initialize insurance fund for each market
INSERT INTO insurance_fund (market_id, balance, total_contributions)
SELECT id, 0, 0 FROM markets
WHERE id NOT IN (SELECT market_id FROM insurance_fund);

-- Update existing markets with proper oracle accounts
UPDATE markets SET 
    oracle_account = CASE 
        WHEN symbol = 'BTC/USDT' THEN 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J'
        WHEN symbol = 'ETH/USDT' THEN 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB'
        WHEN symbol = 'SOL/USDT' THEN 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG'
        ELSE oracle_account
    END
WHERE oracle_account = 'ORACLE_ACCOUNT';

-- =============================================
-- STEP 11: GRANT PERMISSIONS
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
-- MIGRATION COMPLETE
-- =============================================

-- Log migration completion
INSERT INTO system_events (event_type, event_data, severity)
VALUES ('migration', '{"version": "2.0", "description": "Perpetual DEX schema migration completed"}', 'info');

-- Display completion message
DO $$
BEGIN
    RAISE NOTICE 'Migration completed successfully!';
    RAISE NOTICE 'Your database is now ready for production perpetual trading.';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Update your application code to use new schema';
    RAISE NOTICE '2. Test all trading functions';
    RAISE NOTICE '3. Set up monitoring for system_events table';
    RAISE NOTICE '4. Configure proper environment variables';
END $$;
