-- Deposit and Withdrawal Tables
-- Run this migration to add deposit/withdrawal functionality

-- Create deposits table
CREATE TABLE IF NOT EXISTS deposits (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    trading_account_id UUID REFERENCES trading_accounts(id) ON DELETE SET NULL,
    asset VARCHAR(10) NOT NULL,
    amount DECIMAL(18,8) NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- pending, completed, failed
    wallet_address TEXT NOT NULL,
    transaction_signature TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confirmed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create withdrawals table
CREATE TABLE IF NOT EXISTS withdrawals (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    trading_account_id UUID REFERENCES trading_accounts(id) ON DELETE SET NULL,
    asset VARCHAR(10) NOT NULL,
    amount DECIMAL(18,8) NOT NULL,
    destination_address TEXT NOT NULL,
    status VARCHAR(20) DEFAULT 'pending', -- pending, processing, completed, failed
    transaction_signature TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    confirmed_at TIMESTAMP WITH TIME ZONE,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create user_balances table if it doesn't exist
CREATE TABLE IF NOT EXISTS user_balances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    trading_account_id UUID REFERENCES trading_accounts(id) ON DELETE SET NULL,
    asset VARCHAR(10) NOT NULL,
    balance DECIMAL(18,8) DEFAULT 0,
    locked_balance DECIMAL(18,8) DEFAULT 0,
    available_balance DECIMAL(18,8) GENERATED ALWAYS AS (balance - locked_balance) STORED,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(user_id, COALESCE(trading_account_id, '00000000-0000-0000-0000-000000000000'::uuid), asset)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_deposits_user_id ON deposits(user_id);
CREATE INDEX IF NOT EXISTS idx_deposits_status ON deposits(status);
CREATE INDEX IF NOT EXISTS idx_deposits_created_at ON deposits(created_at);
CREATE INDEX IF NOT EXISTS idx_deposits_trading_account ON deposits(trading_account_id);

CREATE INDEX IF NOT EXISTS idx_withdrawals_user_id ON withdrawals(user_id);
CREATE INDEX IF NOT EXISTS idx_withdrawals_status ON withdrawals(status);
CREATE INDEX IF NOT EXISTS idx_withdrawals_created_at ON withdrawals(created_at);
CREATE INDEX IF NOT EXISTS idx_withdrawals_trading_account ON withdrawals(trading_account_id);

CREATE INDEX IF NOT EXISTS idx_user_balances_user_id ON user_balances(user_id);
CREATE INDEX IF NOT EXISTS idx_user_balances_asset ON user_balances(asset);
CREATE INDEX IF NOT EXISTS idx_user_balances_trading_account ON user_balances(trading_account_id);

-- Create triggers for updated_at
CREATE TRIGGER update_deposits_updated_at BEFORE UPDATE ON deposits
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_withdrawals_updated_at BEFORE UPDATE ON withdrawals
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_user_balances_updated_at BEFORE UPDATE ON user_balances
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for easier querying
CREATE OR REPLACE VIEW user_transaction_history AS
SELECT 
    'deposit' as type,
    id,
    user_id,
    trading_account_id,
    asset,
    amount,
    status,
    transaction_signature,
    created_at,
    confirmed_at,
    NULL as destination_address
FROM deposits
UNION ALL
SELECT 
    'withdrawal' as type,
    id,
    user_id,
    trading_account_id,
    asset,
    amount,
    status,
    transaction_signature,
    created_at,
    confirmed_at,
    destination_address
FROM withdrawals
ORDER BY created_at DESC;

-- Create function to get user's total portfolio value
CREATE OR REPLACE FUNCTION get_user_portfolio_value(user_wallet_address text)
RETURNS TABLE (
    total_balance_usd numeric,
    total_locked_usd numeric,
    total_available_usd numeric,
    asset_breakdown jsonb
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        SUM(ub.balance * COALESCE(op.price, 1)) as total_balance_usd,
        SUM(ub.locked_balance * COALESCE(op.price, 1)) as total_locked_usd,
        SUM(ub.available_balance * COALESCE(op.price, 1)) as total_available_usd,
        jsonb_agg(
            jsonb_build_object(
                'asset', ub.asset,
                'balance', ub.balance,
                'locked_balance', ub.locked_balance,
                'available_balance', ub.available_balance,
                'usd_value', ub.balance * COALESCE(op.price, 1)
            )
        ) as asset_breakdown
    FROM user_balances ub
    JOIN users u ON ub.user_id = u.id
    LEFT JOIN oracle_prices op ON ub.asset = op.symbol
    WHERE u.wallet_address = user_wallet_address
    AND ub.balance > 0;
END;
$$ LANGUAGE plpgsql;

-- Insert default supported tokens (for reference)
INSERT INTO markets (symbol, base_asset, quote_asset, market_type, is_active) 
VALUES 
    ('SOL-USD', 'SOL', 'USD', 'spot', true),
    ('USDC-USD', 'USDC', 'USD', 'spot', true),
    ('USDT-USD', 'USDT', 'USD', 'spot', true),
    ('BTC-USD', 'BTC', 'USD', 'spot', true),
    ('ETH-USD', 'ETH', 'USD', 'spot', true)
ON CONFLICT (symbol) DO NOTHING;
