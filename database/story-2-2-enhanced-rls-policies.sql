-- Enhanced RLS Policies for Story 2.2 - Database Security Hardening
-- This script updates all RLS policies to use auth.uid() instead of auth.jwt() ->> 'wallet_address'
-- This provides better security by using Supabase's built-in user authentication

-- Drop existing policies
DROP POLICY IF EXISTS "Users can view own data" ON users;
DROP POLICY IF EXISTS "Users can view own balances" ON user_balances;
DROP POLICY IF EXISTS "Users can view own positions" ON positions;
DROP POLICY IF EXISTS "Users can view own orders" ON orders;
DROP POLICY IF EXISTS "Users can view own trades" ON trades;
DROP POLICY IF EXISTS "Users can view own stats" ON user_stats;
DROP POLICY IF EXISTS "Everyone can view active public channels" ON chat_channels;
DROP POLICY IF EXISTS "Users can create channels" ON chat_channels;
DROP POLICY IF EXISTS "Channel creators can update their channels" ON chat_channels;
DROP POLICY IF EXISTS "Everyone can view messages in public channels" ON chat_messages;
DROP POLICY IF EXISTS "Users can send messages" ON chat_messages;
DROP POLICY IF EXISTS "Users can edit their own messages" ON chat_messages;
DROP POLICY IF EXISTS "Users can delete their own messages" ON chat_messages;

-- Create enhanced policies using auth.uid()
-- Users can only see their own data
CREATE POLICY "Users can view own data" ON users
    FOR ALL USING (auth.uid() = id);

-- User balances policies
CREATE POLICY "Users can view own balances" ON user_balances
    FOR ALL USING (auth.uid() = user_id);

-- Positions policies
CREATE POLICY "Users can view own positions" ON positions
    FOR ALL USING (auth.uid() = user_id);

-- Orders policies
CREATE POLICY "Users can view own orders" ON orders
    FOR ALL USING (auth.uid() = user_id);

-- Trades policies
CREATE POLICY "Users can view own trades" ON trades
    FOR ALL USING (auth.uid() = user_id);

-- User stats policies
CREATE POLICY "Users can view own stats" ON user_stats
    FOR ALL USING (auth.uid() = user_id);

-- Chat channels policies
CREATE POLICY "Everyone can view active public channels" ON chat_channels
    FOR SELECT USING (is_active = true AND is_private = false);

CREATE POLICY "Users can create channels" ON chat_channels
    FOR INSERT WITH CHECK (auth.uid() = created_by);

CREATE POLICY "Channel creators can update their channels" ON chat_channels
    FOR UPDATE USING (auth.uid() = created_by);

-- Chat messages policies
CREATE POLICY "Everyone can view messages in public channels" ON chat_messages
    FOR SELECT USING (
        channel_id IN (SELECT id FROM chat_channels WHERE is_private = false AND is_active = true) AND
        deleted_at IS NULL
    );

CREATE POLICY "Users can send messages" ON chat_messages
    FOR INSERT WITH CHECK (auth.uid() = (
        SELECT id FROM users WHERE wallet_address = author_pubkey
    ));

CREATE POLICY "Users can edit their own messages" ON chat_messages
    FOR UPDATE USING (
        auth.uid() = (SELECT id FROM users WHERE wallet_address = author_pubkey) AND 
        deleted_at IS NULL
    );

CREATE POLICY "Users can delete their own messages" ON chat_messages
    FOR UPDATE USING (
        auth.uid() = (SELECT id FROM users WHERE wallet_address = author_pubkey)
    );

-- Additional security policies for trading accounts and sub accounts
-- Enable RLS on additional tables if not already enabled
ALTER TABLE trading_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE sub_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE delegated_accounts ENABLE ROW LEVEL SECURITY;
ALTER TABLE deposits ENABLE ROW LEVEL SECURITY;
ALTER TABLE withdrawals ENABLE ROW LEVEL SECURITY;
ALTER TABLE liquidations ENABLE ROW LEVEL SECURITY;

-- Trading accounts policies
CREATE POLICY "Users can view own trading accounts" ON trading_accounts
    FOR ALL USING (auth.uid() = master_account_id);

-- Sub accounts policies
CREATE POLICY "Users can view own sub accounts" ON sub_accounts
    FOR ALL USING (auth.uid() = main_account_id);

-- Delegated accounts policies
CREATE POLICY "Users can view own delegated accounts" ON delegated_accounts
    FOR ALL USING (auth.uid() = main_account_id);

-- Deposits policies
CREATE POLICY "Users can view own deposits" ON deposits
    FOR ALL USING (auth.uid() = user_id);

-- Withdrawals policies
CREATE POLICY "Users can view own withdrawals" ON withdrawals
    FOR ALL USING (auth.uid() = user_id);

-- Liquidations policies
CREATE POLICY "Users can view own liquidations" ON liquidations
    FOR ALL USING (auth.uid() = user_id);

-- Create indexes for performance on frequently queried columns
CREATE INDEX IF NOT EXISTS idx_users_auth_uid ON users(id);
CREATE INDEX IF NOT EXISTS idx_user_balances_user_id ON user_balances(user_id);
CREATE INDEX IF NOT EXISTS idx_positions_user_id ON positions(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_accounts_master_account_id ON trading_accounts(master_account_id);
CREATE INDEX IF NOT EXISTS idx_sub_accounts_main_account_id ON sub_accounts(main_account_id);
CREATE INDEX IF NOT EXISTS idx_delegated_accounts_main_account_id ON delegated_accounts(main_account_id);
CREATE INDEX IF NOT EXISTS idx_deposits_user_id ON deposits(user_id);
CREATE INDEX IF NOT EXISTS idx_withdrawals_user_id ON withdrawals(user_id);
CREATE INDEX IF NOT EXISTS idx_liquidations_user_id ON liquidations(user_id);

-- Create a function to validate user access
CREATE OR REPLACE FUNCTION validate_user_access(p_user_id UUID)
RETURNS BOOLEAN AS $$
BEGIN
    -- Check if the authenticated user matches the requested user_id
    RETURN auth.uid() = p_user_id;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create a function to get current user's wallet address
CREATE OR REPLACE FUNCTION get_current_user_wallet()
RETURNS TEXT AS $$
BEGIN
    -- Get the wallet address for the current authenticated user
    RETURN (SELECT wallet_address FROM users WHERE id = auth.uid());
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant necessary permissions
GRANT EXECUTE ON FUNCTION validate_user_access(UUID) TO authenticated;
GRANT EXECUTE ON FUNCTION get_current_user_wallet() TO authenticated;

-- Add comments for documentation
COMMENT ON POLICY "Users can view own data" ON users IS 'Users can only access their own user record using auth.uid()';
COMMENT ON POLICY "Users can view own balances" ON user_balances IS 'Users can only access their own balance records using auth.uid()';
COMMENT ON POLICY "Users can view own positions" ON positions IS 'Users can only access their own position records using auth.uid()';
COMMENT ON POLICY "Users can view own orders" ON orders IS 'Users can only access their own order records using auth.uid()';
COMMENT ON POLICY "Users can view own trades" ON trades IS 'Users can only access their own trade records using auth.uid()';
COMMENT ON POLICY "Users can view own stats" ON user_stats IS 'Users can only access their own statistics using auth.uid()';

-- Log the policy update
INSERT INTO system_events (event_type, event_data, severity)
VALUES (
    'rls_policy_update',
    '{"story": "2.2", "description": "Updated all RLS policies to use auth.uid() instead of auth.jwt() ->> wallet_address", "tables_updated": ["users", "user_balances", "positions", "orders", "trades", "user_stats", "chat_channels", "chat_messages", "trading_accounts", "sub_accounts", "delegated_accounts", "deposits", "withdrawals", "liquidations"]}',
    'info'
);
