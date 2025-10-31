-- Multi-Account Management Tables
-- Run this migration to add trading account and delegate functionality

-- Add account type to users table
ALTER TABLE users ADD COLUMN IF NOT EXISTS master_account_id UUID;
ALTER TABLE users ADD COLUMN IF NOT EXISTS account_type VARCHAR(20) DEFAULT 'master';

-- Create trading accounts table
CREATE TABLE IF NOT EXISTS trading_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    master_account_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_index INTEGER NOT NULL,
    name VARCHAR(100) NOT NULL,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(master_account_id, account_index)
);

-- Create delegated accounts table
CREATE TABLE IF NOT EXISTS delegated_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    master_account_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    delegate_wallet_address TEXT NOT NULL,
    permissions JSONB DEFAULT '{"deposit": true, "trade": true, "cancel": true, "withdraw": false}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(main_account_id, delegate_wallet_address)
);

-- Add trading account references to existing tables
ALTER TABLE positions ADD COLUMN IF NOT EXISTS trading_account_id UUID REFERENCES trading_accounts(id);
ALTER TABLE orders ADD COLUMN IF NOT EXISTS trading_account_id UUID REFERENCES trading_accounts(id);
ALTER TABLE user_balances ADD COLUMN IF NOT EXISTS trading_account_id UUID REFERENCES trading_accounts(id);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trading_accounts_master ON trading_accounts(master_account_id);
CREATE INDEX IF NOT EXISTS idx_trading_accounts_active ON trading_accounts(is_active);
CREATE INDEX IF NOT EXISTS idx_delegated_accounts_master ON delegated_accounts(master_account_id);
CREATE INDEX IF NOT EXISTS idx_delegated_accounts_delegate ON delegated_accounts(delegate_wallet_address);
CREATE INDEX IF NOT EXISTS idx_positions_trading_account ON positions(trading_account_id);
CREATE INDEX IF NOT EXISTS idx_orders_trading_account ON orders(trading_account_id);
CREATE INDEX IF NOT EXISTS idx_user_balances_trading_account ON user_balances(trading_account_id);

-- Update triggers for new tables
CREATE TRIGGER update_trading_accounts_updated_at BEFORE UPDATE ON trading_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_delegated_accounts_updated_at BEFORE UPDATE ON delegated_accounts
    FOR EACH ROW EXECUTE FUNCTION update_updated_at_column();

-- Create views for easier querying
CREATE OR REPLACE VIEW active_trading_accounts AS
SELECT 
    ta.*,
    u.wallet_address as master_wallet_address
FROM trading_accounts ta
JOIN users u ON ta.master_account_id = u.id
WHERE ta.is_active = true;

CREATE OR REPLACE VIEW active_delegates AS
SELECT 
    da.*,
    u.wallet_address as master_wallet_address
FROM delegated_accounts da
JOIN users u ON da.master_account_id = u.id
WHERE da.is_active = true;

-- Create function to get user's total portfolio value across all trading accounts
CREATE OR REPLACE FUNCTION get_user_total_portfolio_value(user_wallet_address text)
RETURNS TABLE (
    total_balance_usd numeric,
    total_locked_usd numeric,
    total_available_usd numeric,
    sub_account_count integer
)
LANGUAGE sql
SECURITY DEFINER
AS $$
  WITH user_accounts AS (
    SELECT u.id as user_id
    FROM users u
    WHERE u.wallet_address = user_wallet_address
  ),
  sub_account_balances AS (
    SELECT 
      ub.asset,
      SUM(ub.balance) as total_balance,
      SUM(ub.locked_balance) as total_locked,
      SUM(ub.available_balance) as total_available
    FROM user_balances ub
    CROSS JOIN user_accounts ua
    WHERE ub.user_id = ua.user_id
    GROUP BY ub.asset
  ),
  sub_account_count AS (
    SELECT COUNT(*) as count
    FROM sub_accounts sa
    CROSS JOIN user_accounts ua
    WHERE sa.main_account_id = ua.user_id AND sa.is_active = true
  )
  SELECT 
    COALESCE(SUM(sab.total_balance), 0) as total_balance_usd,
    COALESCE(SUM(sab.total_locked), 0) as total_locked_usd,
    COALESCE(SUM(sab.total_available), 0) as total_available_usd,
    COALESCE(sac.count, 0) as sub_account_count
  FROM sub_account_balances sab
  CROSS JOIN sub_account_count sac;
$$;

-- Grant permissions
GRANT SELECT ON active_sub_accounts TO authenticated;
GRANT SELECT ON active_delegates TO authenticated;
GRANT EXECUTE ON FUNCTION get_user_total_portfolio_value(text) TO authenticated;

-- Insert sample data for testing (optional)
-- This creates a main account and one sub-account for testing
DO $$
DECLARE
    test_user_id UUID;
    test_sub_account_id UUID;
BEGIN
    -- Create test user if doesn't exist
    INSERT INTO users (wallet_address, account_type)
    VALUES ('11111111111111111111111111111112', 'main')
    ON CONFLICT (wallet_address) DO NOTHING;
    
    -- Get the test user ID
    SELECT id INTO test_user_id FROM users WHERE wallet_address = '11111111111111111111111111111112';
    
    -- Create a test sub-account
    INSERT INTO sub_accounts (main_account_id, sub_account_index, name)
    VALUES (test_user_id, 1, 'Main Trading Account')
    ON CONFLICT (main_account_id, sub_account_index) DO NOTHING;
    
    -- Get the sub-account ID
    SELECT id INTO test_sub_account_id FROM sub_accounts 
    WHERE main_account_id = test_user_id AND sub_account_index = 1;
    
    -- Create test balances
    INSERT INTO user_balances (user_id, sub_account_id, asset, balance, locked_balance)
    VALUES 
        (test_user_id, test_sub_account_id, 'USDC', 10000.0, 0.0),
        (test_user_id, test_sub_account_id, 'SOL', 100.0, 0.0)
    ON CONFLICT (user_id, sub_account_id, asset) DO NOTHING;
END $$;
