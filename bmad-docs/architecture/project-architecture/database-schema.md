# Database Schema

```sql
-- Enable required extensions
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pgcrypto";

-- Users table
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    wallet_address VARCHAR(44) UNIQUE NOT NULL,
    email VARCHAR(255) UNIQUE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    last_login TIMESTAMP WITH TIME ZONE,
    trading_preferences JSONB DEFAULT '{}',
    kyc_status VARCHAR(20) DEFAULT 'pending' CHECK (kyc_status IN ('pending', 'verified', 'rejected')),
    risk_level VARCHAR(20) DEFAULT 'moderate' CHECK (risk_level IN ('conservative', 'moderate', 'aggressive')),
    is_active BOOLEAN DEFAULT true,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trading positions table
CREATE TABLE trading_positions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    market_symbol VARCHAR(20) NOT NULL,
    side VARCHAR(10) NOT NULL CHECK (side IN ('long', 'short')),
    size DECIMAL(20,8) NOT NULL CHECK (size > 0),
    entry_price DECIMAL(20,8) NOT NULL CHECK (entry_price > 0),
    current_price DECIMAL(20,8) NOT NULL CHECK (current_price > 0),
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    margin_used DECIMAL(20,8) NOT NULL CHECK (margin_used >= 0),
    leverage DECIMAL(5,2) NOT NULL CHECK (leverage > 0 AND leverage <= 100),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    closed_at TIMESTAMP WITH TIME ZONE,
    is_active BOOLEAN DEFAULT true
);

-- Trading orders table
CREATE TABLE trading_orders (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    position_id UUID REFERENCES trading_positions(id) ON DELETE SET NULL,
    order_type VARCHAR(20) NOT NULL CHECK (order_type IN ('market', 'limit', 'stop', 'stop_limit')),
    side VARCHAR(10) NOT NULL CHECK (side IN ('buy', 'sell')),
    market_symbol VARCHAR(20) NOT NULL,
    quantity DECIMAL(20,8) NOT NULL CHECK (quantity > 0),
    price DECIMAL(20,8) CHECK (price > 0),
    stop_price DECIMAL(20,8) CHECK (stop_price > 0),
    status VARCHAR(20) DEFAULT 'pending' CHECK (status IN ('pending', 'filled', 'cancelled', 'rejected')),
    filled_quantity DECIMAL(20,8) DEFAULT 0 CHECK (filled_quantity >= 0),
    average_fill_price DECIMAL(20,8),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    executed_at TIMESTAMP WITH TIME ZONE,
    cancelled_at TIMESTAMP WITH TIME ZONE,
    transaction_hash VARCHAR(88),
    error_message TEXT
);

-- Market data table
CREATE TABLE market_data (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol VARCHAR(20) NOT NULL,
    price DECIMAL(20,8) NOT NULL CHECK (price > 0),
    timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
    confidence DECIMAL(10,4) CHECK (confidence >= 0),
    exponent INTEGER DEFAULT -8,
    source VARCHAR(50) DEFAULT 'pyth',
    staleness_threshold INTEGER DEFAULT 30,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Portfolio snapshots table
CREATE TABLE portfolio_snapshots (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    total_balance DECIMAL(20,8) NOT NULL CHECK (total_balance >= 0),
    available_balance DECIMAL(20,8) NOT NULL CHECK (available_balance >= 0),
    margin_used DECIMAL(20,8) NOT NULL CHECK (margin_used >= 0),
    unrealized_pnl DECIMAL(20,8) DEFAULT 0,
    realized_pnl DECIMAL(20,8) DEFAULT 0,
    snapshot_time TIMESTAMP WITH TIME ZONE NOT NULL,
    trading_fees_paid DECIMAL(20,8) DEFAULT 0 CHECK (trading_fees_paid >= 0),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- AI analysis table
CREATE TABLE ai_analysis (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    analysis_type VARCHAR(30) NOT NULL CHECK (analysis_type IN ('market_analysis', 'trading_signal', 'risk_assessment')),
    market_symbol VARCHAR(20) NOT NULL,
    confidence_score DECIMAL(3,2) NOT NULL CHECK (confidence_score >= 0 AND confidence_score <= 1),
    recommendation VARCHAR(10) CHECK (recommendation IN ('buy', 'sell', 'hold', 'close')),
    reasoning TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL,
    is_active BOOLEAN DEFAULT true
);

-- Position history table (for audit trail)
CREATE TABLE position_history (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    position_id UUID NOT NULL REFERENCES trading_positions(id) ON DELETE CASCADE,
    action VARCHAR(20) NOT NULL CHECK (action IN ('opened', 'updated', 'closed', 'liquidated')),
    size_change DECIMAL(20,8) DEFAULT 0,
    price_change DECIMAL(20,8) DEFAULT 0,
    pnl_change DECIMAL(20,8) DEFAULT 0,
    margin_change DECIMAL(20,8) DEFAULT 0,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    transaction_hash VARCHAR(88),
    notes TEXT
);

-- Indexes for performance optimization
CREATE INDEX idx_users_wallet_address ON users(wallet_address);
CREATE INDEX idx_users_email ON users(email);
CREATE INDEX idx_users_created_at ON users(created_at);

CREATE INDEX idx_positions_user_id ON trading_positions(user_id);
CREATE INDEX idx_positions_symbol ON trading_positions(market_symbol);
CREATE INDEX idx_positions_active ON trading_positions(is_active);
CREATE INDEX idx_positions_created_at ON trading_positions(created_at);

CREATE INDEX idx_orders_user_id ON trading_orders(user_id);
CREATE INDEX idx_orders_position_id ON trading_orders(position_id);
CREATE INDEX idx_orders_status ON trading_orders(status);
CREATE INDEX idx_orders_symbol ON trading_orders(market_symbol);
CREATE INDEX idx_orders_created_at ON trading_orders(created_at);

CREATE INDEX idx_market_data_symbol ON market_data(symbol);
CREATE INDEX idx_market_data_timestamp ON market_data(timestamp);
CREATE INDEX idx_market_data_symbol_timestamp ON market_data(symbol, timestamp DESC);

CREATE INDEX idx_portfolio_user_id ON portfolio_snapshots(user_id);
CREATE INDEX idx_portfolio_snapshot_time ON portfolio_snapshots(snapshot_time);

CREATE INDEX idx_ai_analysis_user_id ON ai_analysis(user_id);
CREATE INDEX idx_ai_analysis_symbol ON ai_analysis(market_symbol);
CREATE INDEX idx_ai_analysis_created_at ON ai_analysis(created_at);
CREATE INDEX idx_ai_analysis_active ON ai_analysis(is_active);

CREATE INDEX idx_position_history_position_id ON position_history(position_id);
CREATE INDEX idx_position_history_created_at ON position_history(created_at);

-- Row Level Security (RLS) policies
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading_positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE trading_orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE portfolio_snapshots ENABLE ROW LEVEL SECURITY;
ALTER TABLE ai_analysis ENABLE ROW LEVEL SECURITY;
ALTER TABLE position_history ENABLE ROW LEVEL SECURITY;

-- RLS policies for user data isolation
CREATE POLICY "Users can view own data" ON users FOR SELECT USING (auth.uid() = id);
CREATE POLICY "Users can update own data" ON users FOR UPDATE USING (auth.uid() = id);

CREATE POLICY "Users can view own positions" ON trading_positions FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own positions" ON trading_positions FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own positions" ON trading_positions FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can view own orders" ON trading_orders FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own orders" ON trading_orders FOR INSERT WITH CHECK (auth.uid() = user_id);
CREATE POLICY "Users can update own orders" ON trading_orders FOR UPDATE USING (auth.uid() = user_id);

CREATE POLICY "Users can view own portfolio" ON portfolio_snapshots FOR SELECT USING (auth.uid() = user_id);
CREATE POLICY "Users can insert own portfolio" ON portfolio_snapshots FOR INSERT WITH CHECK (auth.uid() = user_id);

CREATE POLICY "Users can view own AI analysis" ON ai_analysis FOR SELECT USING (auth.uid() = user_id OR user_id IS NULL);
CREATE POLICY "Users can insert own AI analysis" ON ai_analysis FOR INSERT WITH CHECK (auth.uid() = user_id OR user_id IS NULL);

CREATE POLICY "Users can view own position history" ON position_history FOR SELECT USING (
    EXISTS (SELECT 1 FROM trading_positions WHERE id = position_id AND user_id = auth.uid())
);

-- Functions for automatic timestamp updates
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_users_updated_at BEFORE UPDATE ON users FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();
CREATE TRIGGER update_positions_updated_at BEFORE UPDATE ON trading_positions FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Function for calculating unrealized P&L
CREATE OR REPLACE FUNCTION calculate_unrealized_pnl(
    p_size DECIMAL(20,8),
    p_entry_price DECIMAL(20,8),
    p_current_price DECIMAL(20,8),
    p_side VARCHAR(10)
) RETURNS DECIMAL(20,8) AS $$
BEGIN
    IF p_side = 'long' THEN
        RETURN p_size * (p_current_price - p_entry_price);
    ELSE
        RETURN p_size * (p_entry_price - p_current_price);
    END IF;
END;
$$ LANGUAGE plpgsql;

-- Trigger to automatically update unrealized P&L
CREATE OR REPLACE FUNCTION update_position_pnl()
RETURNS TRIGGER AS $$
BEGIN
    NEW.unrealized_pnl = calculate_unrealized_pnl(NEW.size, NEW.entry_price, NEW.current_price, NEW.side);
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_position_pnl_trigger 
    BEFORE UPDATE ON trading_positions 
    FOR EACH ROW 
    EXECUTE FUNCTION update_position_pnl();
```
