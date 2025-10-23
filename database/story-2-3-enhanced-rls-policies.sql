-- Story 2.3: Enhanced RLS Policies for Database Security Hardening
-- This file contains the corrected RLS policies using auth.uid() pattern
-- instead of auth.jwt() for better security

-- Drop existing policies to replace them with secure versions
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

-- Create secure RLS policies using auth.uid() pattern
-- This ensures proper user context and prevents unauthorized access

-- Users table: Users can only access their own data
CREATE POLICY "secure_users_policy" ON users
    FOR ALL USING (auth.uid()::text = id::text);

-- User balances: Users can only access their own balances
CREATE POLICY "secure_user_balances_policy" ON user_balances
    FOR ALL USING (auth.uid()::text = user_id::text);

-- Positions: Users can only access their own positions
CREATE POLICY "secure_positions_policy" ON positions
    FOR ALL USING (auth.uid()::text = user_id::text);

-- Orders: Users can only access their own orders
CREATE POLICY "secure_orders_policy" ON orders
    FOR ALL USING (auth.uid()::text = user_id::text);

-- Trades: Users can only access their own trades
CREATE POLICY "secure_trades_policy" ON trades
    FOR ALL USING (auth.uid()::text = user_id::text);

-- User stats: Users can only access their own stats
CREATE POLICY "secure_user_stats_policy" ON user_stats
    FOR ALL USING (auth.uid()::text = user_id::text);

-- Chat channels: Public channels visible to all, private channels only to members
CREATE POLICY "secure_chat_channels_select" ON chat_channels
    FOR SELECT USING (
        is_active = true AND (
            is_private = false OR 
            auth.uid()::text = created_by::text
        )
    );

CREATE POLICY "secure_chat_channels_insert" ON chat_channels
    FOR INSERT WITH CHECK (auth.uid()::text = created_by::text);

CREATE POLICY "secure_chat_channels_update" ON chat_channels
    FOR UPDATE USING (auth.uid()::text = created_by::text);

-- Chat messages: Users can view messages in channels they have access to
CREATE POLICY "secure_chat_messages_select" ON chat_messages
    FOR SELECT USING (
        deleted_at IS NULL AND (
            channel_id IN (
                SELECT id FROM chat_channels 
                WHERE is_active = true AND (
                    is_private = false OR 
                    auth.uid()::text = created_by::text
                )
            )
        )
    );

CREATE POLICY "secure_chat_messages_insert" ON chat_messages
    FOR INSERT WITH CHECK (
        auth.uid()::text = (
            SELECT id::text FROM users WHERE wallet_address = author_pubkey
        )
    );

CREATE POLICY "secure_chat_messages_update" ON chat_messages
    FOR UPDATE USING (
        auth.uid()::text = (
            SELECT id::text FROM users WHERE wallet_address = author_pubkey
        ) AND deleted_at IS NULL
    );

-- Additional security policies for trading accounts and sub-accounts
-- These tables need to be created if they don't exist

-- Trading accounts table (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'trading_accounts') THEN
        ALTER TABLE trading_accounts ENABLE ROW LEVEL SECURITY;
        
        DROP POLICY IF EXISTS "secure_trading_accounts_policy" ON trading_accounts;
        CREATE POLICY "secure_trading_accounts_policy" ON trading_accounts
            FOR ALL USING (auth.uid()::text = master_account_id::text);
    END IF;
END $$;

-- Sub accounts table (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'sub_accounts') THEN
        ALTER TABLE sub_accounts ENABLE ROW LEVEL SECURITY;
        
        DROP POLICY IF EXISTS "secure_sub_accounts_policy" ON sub_accounts;
        CREATE POLICY "secure_sub_accounts_policy" ON sub_accounts
            FOR ALL USING (auth.uid()::text = main_account_id::text);
    END IF;
END $$;

-- Delegated accounts table (if exists)
DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM information_schema.tables WHERE table_name = 'delegated_accounts') THEN
        ALTER TABLE delegated_accounts ENABLE ROW LEVEL SECURITY;
        
        DROP POLICY IF EXISTS "secure_delegated_accounts_policy" ON delegated_accounts;
        CREATE POLICY "secure_delegated_accounts_policy" ON delegated_accounts
            FOR ALL USING (auth.uid()::text = main_account_id::text);
    END IF;
END $$;

-- Create indexes for performance optimization
CREATE INDEX IF NOT EXISTS idx_users_wallet_address ON users(wallet_address);
CREATE INDEX IF NOT EXISTS idx_user_balances_user_id ON user_balances(user_id);
CREATE INDEX IF NOT EXISTS idx_positions_user_id ON positions(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_user_stats_user_id ON user_stats(user_id);

-- Create function to get user ID from wallet address (for backward compatibility)
CREATE OR REPLACE FUNCTION get_user_id_from_wallet(wallet_address TEXT)
RETURNS UUID AS $$
BEGIN
    RETURN (SELECT id FROM users WHERE wallet_address = get_user_id_from_wallet.wallet_address);
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Grant necessary permissions
GRANT USAGE ON SCHEMA public TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO authenticated;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO authenticated;

-- Create audit log function for security monitoring
CREATE OR REPLACE FUNCTION log_database_access(
    table_name TEXT,
    operation TEXT,
    user_id UUID DEFAULT auth.uid()
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO audit_logs (table_name, operation, user_id, created_at)
    VALUES (table_name, operation, user_id, NOW());
EXCEPTION
    WHEN OTHERS THEN
        -- Ignore audit log errors to prevent breaking main operations
        NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create audit_logs table if it doesn't exist
CREATE TABLE IF NOT EXISTS audit_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    table_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Enable RLS on audit_logs
ALTER TABLE audit_logs ENABLE ROW LEVEL SECURITY;

-- Policy for audit_logs (only system can insert, users can view their own)
CREATE POLICY "secure_audit_logs_policy" ON audit_logs
    FOR ALL USING (
        auth.uid()::text = user_id::text OR 
        auth.role() = 'service_role'
    );

-- Performance monitoring function
CREATE OR REPLACE FUNCTION log_slow_query(
    query_text TEXT,
    execution_time_ms INTEGER,
    user_id UUID DEFAULT auth.uid()
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO performance_logs (query_text, execution_time_ms, user_id, created_at)
    VALUES (query_text, execution_time_ms, user_id, NOW());
EXCEPTION
    WHEN OTHERS THEN
        -- Ignore performance log errors
        NULL;
END;
$$ LANGUAGE plpgsql SECURITY DEFINER;

-- Create performance_logs table if it doesn't exist
CREATE TABLE IF NOT EXISTS performance_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    query_text TEXT NOT NULL,
    execution_time_ms INTEGER NOT NULL,
    user_id UUID,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Enable RLS on performance_logs
ALTER TABLE performance_logs ENABLE ROW LEVEL SECURITY;

-- Policy for performance_logs (only system can insert, users can view their own)
CREATE POLICY "secure_performance_logs_policy" ON performance_logs
    FOR ALL USING (
        auth.uid()::text = user_id::text OR 
        auth.role() = 'service_role'
    );

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_audit_logs_user_id ON audit_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit_logs(created_at);
CREATE INDEX IF NOT EXISTS idx_performance_logs_user_id ON performance_logs(user_id);
CREATE INDEX IF NOT EXISTS idx_performance_logs_created_at ON performance_logs(created_at);

-- Grant permissions for new tables
GRANT SELECT, INSERT, UPDATE, DELETE ON audit_logs TO authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON performance_logs TO authenticated;