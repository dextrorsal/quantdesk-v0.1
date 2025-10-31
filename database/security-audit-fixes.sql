-- =============================================
-- CRITICAL SECURITY FIXES FOR QUANTDESK
-- Based on Solana Expert Recommendations
-- =============================================

-- 1. LIQUIDATIONS TABLE - User Privacy
ALTER TABLE liquidations ENABLE ROW LEVEL SECURITY;

-- Users can only see their own liquidations
CREATE POLICY "Users can view own liquidations" ON liquidations
    FOR SELECT USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

-- Public can see aggregated liquidation stats (no user details)
CREATE POLICY "Public liquidation stats only" ON liquidations
    FOR SELECT USING (
        -- Only allow reads through specific views/functions that anonymize data
        false -- Force all reads through backend with service_role
    );

-- Service role can insert liquidations
CREATE POLICY "Service role can manage liquidations" ON liquidations
    FOR ALL USING (auth.role() = 'service_role');

-- 2. INSURANCE FUND - Admin Only
ALTER TABLE insurance_fund ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Public can read insurance fund balances" ON insurance_fund
    FOR SELECT USING (true);

CREATE POLICY "Only service role can modify insurance fund" ON insurance_fund
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

CREATE POLICY "Only service role can update insurance fund" ON insurance_fund
    FOR UPDATE USING (auth.role() = 'service_role');

-- 3. SYSTEM EVENTS - Admin Only
ALTER TABLE system_events ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Only service role can read system events" ON system_events
    FOR SELECT USING (auth.role() = 'service_role');

CREATE POLICY "Only service role can insert system events" ON system_events
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

-- 4. ADMIN USERS - Super Critical
ALTER TABLE admin_users ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Admin users cannot be read via Supabase client" ON admin_users
    FOR SELECT USING (false); -- Force all admin auth through backend

CREATE POLICY "Only service role can manage admin users" ON admin_users
    FOR ALL USING (auth.role() = 'service_role');

-- 5. ADMIN AUDIT LOGS - Super Critical
ALTER TABLE admin_audit_logs ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Only service role can read audit logs" ON admin_audit_logs
    FOR SELECT USING (auth.role() = 'service_role');

CREATE POLICY "Only service role can create audit logs" ON admin_audit_logs
    FOR INSERT WITH CHECK (auth.role() = 'service_role');

-- 6. JIT AUCTION TABLES - Public Read, Service Write
ALTER TABLE auctions ENABLE ROW LEVEL SECURITY;
ALTER TABLE auction_quotes ENABLE ROW LEVEL SECURITY;
ALTER TABLE auction_settlements ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Anyone can read auctions" ON auctions
    FOR SELECT USING (true);
CREATE POLICY "Service role can manage auctions" ON auctions
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Anyone can read auction quotes" ON auction_quotes
    FOR SELECT USING (true);
CREATE POLICY "Service role can manage quotes" ON auction_quotes
    FOR ALL USING (auth.role() = 'service_role');

CREATE POLICY "Anyone can read settlements" ON auction_settlements
    FOR SELECT USING (true);
CREATE POLICY "Service role can manage settlements" ON auction_settlements
    FOR ALL USING (auth.role() = 'service_role');

-- =============================================
-- SECURE VIEWS (Replace existing views)
-- =============================================

-- Drop insecure views
DROP VIEW IF EXISTS active_positions CASCADE;
DROP VIEW IF EXISTS pending_orders CASCADE;
DROP VIEW IF EXISTS user_portfolio CASCADE;
DROP VIEW IF EXISTS risk_dashboard CASCADE;

-- Create secure public market summary (aggregated data only)
CREATE OR REPLACE VIEW public_market_summary AS
SELECT 
    m.id,
    m.symbol,
    m.base_asset,
    m.quote_asset,
    m.is_active,
    COUNT(DISTINCT p.user_id) as active_traders,
    COALESCE(SUM(p.size * p.entry_price), 0) as total_open_interest,
    COALESCE(SUM(CASE WHEN p.side = 'long' THEN p.size * p.entry_price ELSE 0 END), 0) as long_oi,
    COALESCE(SUM(CASE WHEN p.side = 'short' THEN p.size * p.entry_price ELSE 0 END), 0) as short_oi
FROM markets m
LEFT JOIN positions p ON m.id = p.market_id AND p.size > 0 AND NOT p.is_liquidated
GROUP BY m.id, m.symbol, m.base_asset, m.quote_asset, m.is_active;

GRANT SELECT ON public_market_summary TO anon, authenticated;

-- Create secure authenticated-only user portfolio view
CREATE OR REPLACE VIEW my_portfolio AS
SELECT 
    p.*,
    m.symbol,
    m.base_asset,
    m.quote_asset
FROM positions p
JOIN users u ON p.user_id = u.id
JOIN markets m ON p.market_id = m.id
WHERE u.wallet_address = (auth.jwt() ->> 'wallet_address')
AND p.size > 0 
AND NOT p.is_liquidated;

GRANT SELECT ON my_portfolio TO authenticated;

-- =============================================
-- SECURE FUNCTIONS
-- =============================================

-- Secure update_user_balance with permission check
CREATE OR REPLACE FUNCTION update_user_balance(
    p_user_id UUID,
    p_asset TEXT,
    p_amount DECIMAL(20,8)
) RETURNS VOID 
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    -- Only service_role can call this
    IF auth.role() != 'service_role' THEN
        RAISE EXCEPTION 'Unauthorized: Only service role can update balances';
    END IF;
    
    INSERT INTO user_balances (user_id, asset, balance)
    VALUES (p_user_id, p_asset, p_amount)
    ON CONFLICT (user_id, asset)
    DO UPDATE SET 
        balance = user_balances.balance + p_amount,
        updated_at = NOW();
END;
$$;

-- Revoke public access to sensitive functions
REVOKE ALL ON FUNCTION update_user_balance FROM PUBLIC;
GRANT EXECUTE ON FUNCTION update_user_balance TO service_role;

-- Secure get_user_portfolio with auth check
CREATE OR REPLACE FUNCTION get_user_portfolio(user_wallet_address TEXT)
RETURNS TABLE (
    position_id UUID,
    market_symbol TEXT,
    side position_side,
    size DECIMAL(20,8),
    entry_price DECIMAL(20,8),
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8)
)
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    -- Only allow users to query their own portfolio
    IF auth.jwt() ->> 'wallet_address' != user_wallet_address THEN
        RAISE EXCEPTION 'Unauthorized: Cannot access other users portfolio';
    END IF;
    
    RETURN QUERY
    SELECT 
        p.id,
        m.symbol,
        p.side,
        p.size,
        p.entry_price,
        p.current_price,
        p.unrealized_pnl
    FROM positions p
    JOIN markets m ON p.market_id = m.id
    JOIN users u ON p.user_id = u.id
    WHERE u.wallet_address = user_wallet_address
    AND p.size > 0
    AND NOT p.is_liquidated;
END;
$$;

-- =============================================
-- VERIFY RLS IS ENABLED ON ALL TABLES
-- =============================================

DO $$
DECLARE
    r RECORD;
BEGIN
    FOR r IN 
        SELECT tablename FROM pg_tables 
        WHERE schemaname = 'public' 
        AND tablename NOT LIKE 'pg_%'
    LOOP
        EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', r.tablename);
        RAISE NOTICE 'Enabled RLS on table: %', r.tablename;
    END LOOP;
END $$;

-- =============================================
-- REVOKE DANGEROUS GRANTS
-- =============================================

-- Remove overly permissive grants from production schema
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM anon;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM anon;

-- Grant selective permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;

-- Public tables (read-only for anon)
GRANT SELECT ON markets TO anon, authenticated;
GRANT SELECT ON oracle_prices TO anon, authenticated;
GRANT SELECT ON funding_rates TO anon, authenticated;
GRANT SELECT ON mark_prices TO anon, authenticated;
GRANT SELECT ON market_stats TO anon, authenticated;
GRANT SELECT ON public_market_summary TO anon, authenticated;

-- Authenticated user tables (RLS enforced)
GRANT SELECT, INSERT, UPDATE ON users TO authenticated;
GRANT SELECT, INSERT, UPDATE ON user_balances TO authenticated;
GRANT SELECT, INSERT, UPDATE ON positions TO authenticated;
GRANT SELECT, INSERT, UPDATE ON orders TO authenticated;
GRANT SELECT, INSERT ON trades TO authenticated;
GRANT SELECT ON liquidations TO authenticated;
GRANT SELECT ON my_portfolio TO authenticated;

-- Service role has full access (for backend operations)
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO service_role;
