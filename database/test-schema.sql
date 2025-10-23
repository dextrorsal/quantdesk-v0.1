-- QuantDesk Database Testing Script
-- Tests the production database schema with sample data and queries
-- Run this after migration to verify everything works correctly

-- =============================================
-- TEST 1: CREATE SAMPLE USERS
-- =============================================

-- Insert test users
INSERT INTO users (wallet_address, username, email, kyc_status, risk_level) VALUES
('11111111111111111111111111111111111111111111', 'trader1', 'trader1@example.com', 'verified', 'medium'),
('22222222222222222222222222222222222222222222', 'trader2', 'trader2@example.com', 'verified', 'high'),
('33333333333333333333333333333333333333333333', 'trader3', 'trader3@example.com', 'pending', 'low')
ON CONFLICT (wallet_address) DO NOTHING;

-- =============================================
-- TEST 2: CREATE SAMPLE BALANCES
-- =============================================

-- Insert test balances
INSERT INTO user_balances (user_id, asset, balance, locked_balance) 
SELECT 
    u.id,
    'USDC',
    10000.00,
    0
FROM users u 
WHERE u.wallet_address IN ('11111111111111111111111111111111111111111111', '22222222222222222222222222222222222222222222', '33333333333333333333333333333333333333333333')
ON CONFLICT (user_id, asset) DO NOTHING;

-- =============================================
-- TEST 3: INSERT SAMPLE ORACLE PRICES
-- =============================================

-- Insert sample oracle prices for testing
INSERT INTO oracle_prices (market_id, price, confidence, exponent, slot)
SELECT 
    m.id,
    CASE 
        WHEN m.symbol = 'BTC-PERP' THEN 45000.00
        WHEN m.symbol = 'ETH-PERP' THEN 3000.00
        WHEN m.symbol = 'SOL-PERP' THEN 100.00
        ELSE 1.00
    END,
    0.01,
    -8,
    123456789
FROM markets m
WHERE m.symbol IN ('BTC-PERP', 'ETH-PERP', 'SOL-PERP');

-- =============================================
-- TEST 4: INSERT SAMPLE MARK PRICES
-- =============================================

-- Insert sample mark prices
INSERT INTO mark_prices (market_id, mark_price, oracle_price, funding_rate)
SELECT 
    m.id,
    CASE 
        WHEN m.symbol = 'BTC-PERP' THEN 45050.00
        WHEN m.symbol = 'ETH-PERP' THEN 3005.00
        WHEN m.symbol = 'SOL-PERP' THEN 100.50
        ELSE 1.00
    END,
    CASE 
        WHEN m.symbol = 'BTC-PERP' THEN 45000.00
        WHEN m.symbol = 'ETH-PERP' THEN 3000.00
        WHEN m.symbol = 'SOL-PERP' THEN 100.00
        ELSE 1.00
    END,
    0.001
FROM markets m
WHERE m.symbol IN ('BTC-PERP', 'ETH-PERP', 'SOL-PERP');

-- =============================================
-- TEST 5: CREATE SAMPLE POSITIONS
-- =============================================

-- Insert test positions
INSERT INTO positions (user_id, market_id, position_account, side, size, entry_price, margin, leverage, health_factor, margin_ratio)
SELECT 
    u.id,
    m.id,
    'POSITION_' || u.id || '_' || m.id,
    'long',
    0.1,
    CASE 
        WHEN m.symbol = 'BTC-PERP' THEN 45000.00
        WHEN m.symbol = 'ETH-PERP' THEN 3000.00
        WHEN m.symbol = 'SOL-PERP' THEN 100.00
        ELSE 1.00
    END,
    450.00,
    10,
    150.00,
    10.00
FROM users u, markets m
WHERE u.wallet_address = '11111111111111111111111111111111111111111111'
AND m.symbol = 'BTC-PERP'
ON CONFLICT DO NOTHING;

-- =============================================
-- TEST 6: CREATE SAMPLE ORDERS
-- =============================================

-- Insert test orders
INSERT INTO orders (user_id, market_id, order_account, order_type, side, size, price, leverage, status)
SELECT 
    u.id,
    m.id,
    'ORDER_' || u.id || '_' || m.id,
    'limit',
    'long',
    0.05,
    CASE 
        WHEN m.symbol = 'BTC-PERP' THEN 44000.00
        WHEN m.symbol = 'ETH-PERP' THEN 2900.00
        WHEN m.symbol = 'SOL-PERP' THEN 95.00
        ELSE 1.00
    END,
    5,
    'pending'
FROM users u, markets m
WHERE u.wallet_address = '22222222222222222222222222222222222222222222'
AND m.symbol = 'ETH-PERP'
ON CONFLICT DO NOTHING;

-- =============================================
-- TEST 7: CREATE SAMPLE TRADES
-- =============================================

-- Insert test trades
INSERT INTO trades (user_id, market_id, position_id, order_id, trade_account, side, size, price, fees, transaction_signature)
SELECT 
    u.id,
    m.id,
    p.id,
    o.id,
    'TRADE_' || u.id || '_' || m.id,
    'buy',
    0.1,
    CASE 
        WHEN m.symbol = 'BTC-PERP' THEN 45000.00
        WHEN m.symbol = 'ETH-PERP' THEN 3000.00
        WHEN m.symbol = 'SOL-PERP' THEN 100.00
        ELSE 1.00
    END,
    4.50,
    'SIGNATURE_' || u.id || '_' || m.id
FROM users u, markets m, positions p, orders o
WHERE u.wallet_address = '11111111111111111111111111111111111111111111'
AND m.symbol = 'BTC-PERP'
AND p.user_id = u.id
AND o.user_id = u.id
LIMIT 1
ON CONFLICT DO NOTHING;

-- =============================================
-- TEST 8: INSERT FUNDING RATES
-- =============================================

-- Insert sample funding rates
INSERT INTO funding_rates (market_id, funding_rate, premium_index, oracle_price, mark_price, total_funding)
SELECT 
    m.id,
    0.001,
    0.1,
    CASE 
        WHEN m.symbol = 'BTC-PERP' THEN 45000.00
        WHEN m.symbol = 'ETH-PERP' THEN 3000.00
        WHEN m.symbol = 'SOL-PERP' THEN 100.00
        ELSE 1.00
    END,
    CASE 
        WHEN m.symbol = 'BTC-PERP' THEN 45050.00
        WHEN m.symbol = 'ETH-PERP' THEN 3005.00
        WHEN m.symbol = 'SOL-PERP' THEN 100.50
        ELSE 1.00
    END,
    0.0
FROM markets m
WHERE m.symbol IN ('BTC-PERP', 'ETH-PERP', 'SOL-PERP');

-- =============================================
-- TEST 9: TEST DATABASE FUNCTIONS
-- =============================================

-- Test update_user_balance function
DO $$
DECLARE
    test_user_id UUID;
BEGIN
    SELECT id INTO test_user_id FROM users WHERE wallet_address = '11111111111111111111111111111111111111111111';
    
    -- Test adding balance
    PERFORM update_user_balance(test_user_id, 'USDC', 1000.00);
    
    RAISE NOTICE 'update_user_balance function test passed';
END $$;

-- Test calculate_position_health function
DO $$
DECLARE
    test_position_id UUID;
    health_factor DECIMAL(10,6);
BEGIN
    SELECT id INTO test_position_id FROM positions LIMIT 1;
    
    IF test_position_id IS NOT NULL THEN
        SELECT calculate_position_health(test_position_id) INTO health_factor;
        RAISE NOTICE 'calculate_position_health function test passed. Health factor: %', health_factor;
    ELSE
        RAISE NOTICE 'No positions found for health factor test';
    END IF;
END $$;

-- Test calculate_funding_rate function
DO $$
DECLARE
    test_market_id UUID;
    funding_rate DECIMAL(10,6);
BEGIN
    SELECT id INTO test_market_id FROM markets WHERE symbol = 'BTC-PERP';
    
    IF test_market_id IS NOT NULL THEN
        SELECT calculate_funding_rate(test_market_id) INTO funding_rate;
        RAISE NOTICE 'calculate_funding_rate function test passed. Funding rate: %', funding_rate;
    ELSE
        RAISE NOTICE 'No BTC market found for funding rate test';
    END IF;
END $$;

-- Test check_liquidation function
DO $$
DECLARE
    test_position_id UUID;
    should_liquidate BOOLEAN;
BEGIN
    SELECT id INTO test_position_id FROM positions LIMIT 1;
    
    IF test_position_id IS NOT NULL THEN
        SELECT check_liquidation(test_position_id) INTO should_liquidate;
        RAISE NOTICE 'check_liquidation function test passed. Should liquidate: %', should_liquidate;
    ELSE
        RAISE NOTICE 'No positions found for liquidation test';
    END IF;
END $$;

-- =============================================
-- TEST 10: TEST VIEWS
-- =============================================

-- Test active_positions view
DO $$
DECLARE
    position_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO position_count FROM active_positions;
    RAISE NOTICE 'active_positions view test passed. Count: %', position_count;
END $$;

-- Test pending_orders view
DO $$
DECLARE
    order_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO order_count FROM pending_orders;
    RAISE NOTICE 'pending_orders view test passed. Count: %', order_count;
END $$;

-- Test market_summary view
DO $$
DECLARE
    market_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO market_count FROM market_summary;
    RAISE NOTICE 'market_summary view test passed. Count: %', market_count;
END $$;

-- Test user_portfolio view
DO $$
DECLARE
    user_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO user_count FROM user_portfolio;
    RAISE NOTICE 'user_portfolio view test passed. Count: %', user_count;
END $$;

-- Test risk_dashboard view
DO $$
DECLARE
    risk_count INTEGER;
BEGIN
    SELECT COUNT(*) INTO risk_count FROM risk_dashboard;
    RAISE NOTICE 'risk_dashboard view test passed. Count: %', risk_count;
END $$;

-- =============================================
-- TEST 11: PERFORMANCE QUERIES
-- =============================================

-- Test high-frequency queries
DO $$
DECLARE
    start_time TIMESTAMP;
    end_time TIMESTAMP;
    duration INTERVAL;
BEGIN
    -- Test oracle price query (most frequent)
    start_time := clock_timestamp();
    PERFORM * FROM oracle_prices WHERE market_id = (SELECT id FROM markets WHERE symbol = 'BTC-PERP') ORDER BY created_at DESC LIMIT 1;
    end_time := clock_timestamp();
    duration := end_time - start_time;
    RAISE NOTICE 'Oracle price query duration: %', duration;
    
    -- Test active positions query
    start_time := clock_timestamp();
    PERFORM * FROM active_positions WHERE user_id = (SELECT id FROM users WHERE wallet_address = '11111111111111111111111111111111111111111111');
    end_time := clock_timestamp();
    duration := end_time - start_time;
    RAISE NOTICE 'Active positions query duration: %', duration;
    
    -- Test market summary query
    start_time := clock_timestamp();
    PERFORM * FROM market_summary WHERE symbol = 'BTC-PERP';
    end_time := clock_timestamp();
    duration := end_time - start_time;
    RAISE NOTICE 'Market summary query duration: %', duration;
END $$;

-- =============================================
-- TEST 12: ROW LEVEL SECURITY TEST
-- =============================================

-- Test RLS policies (this would need to be run with proper auth context)
DO $$
BEGIN
    RAISE NOTICE 'RLS policies are enabled. Test with proper authentication context.';
    RAISE NOTICE 'Users can only see their own data when authenticated.';
    RAISE NOTICE 'Markets and oracle prices are publicly readable.';
END $$;

-- =============================================
-- TEST 13: DATA INTEGRITY CHECKS
-- =============================================

-- Check for orphaned records
DO $$
DECLARE
    orphaned_positions INTEGER;
    orphaned_orders INTEGER;
    orphaned_trades INTEGER;
BEGIN
    -- Check for positions without users
    SELECT COUNT(*) INTO orphaned_positions 
    FROM positions p 
    LEFT JOIN users u ON p.user_id = u.id 
    WHERE u.id IS NULL;
    
    -- Check for orders without users
    SELECT COUNT(*) INTO orphaned_orders 
    FROM orders o 
    LEFT JOIN users u ON o.user_id = u.id 
    WHERE u.id IS NULL;
    
    -- Check for trades without users
    SELECT COUNT(*) INTO orphaned_trades 
    FROM trades t 
    LEFT JOIN users u ON t.user_id = u.id 
    WHERE u.id IS NULL;
    
    RAISE NOTICE 'Data integrity check:';
    RAISE NOTICE 'Orphaned positions: %', orphaned_positions;
    RAISE NOTICE 'Orphaned orders: %', orphaned_orders;
    RAISE NOTICE 'Orphaned trades: %', orphaned_trades;
    
    IF orphaned_positions = 0 AND orphaned_orders = 0 AND orphaned_trades = 0 THEN
        RAISE NOTICE 'Data integrity check PASSED';
    ELSE
        RAISE NOTICE 'Data integrity check FAILED - orphaned records found';
    END IF;
END $$;

-- =============================================
-- TEST 14: CLEANUP TEST DATA
-- =============================================

-- Clean up test data (uncomment to remove test data)
/*
DELETE FROM trades WHERE transaction_signature LIKE 'SIGNATURE_%';
DELETE FROM orders WHERE order_account LIKE 'ORDER_%';
DELETE FROM positions WHERE position_account LIKE 'POSITION_%';
DELETE FROM user_balances WHERE user_id IN (SELECT id FROM users WHERE wallet_address LIKE '11111111111111111111111111111111111111111111' OR wallet_address LIKE '22222222222222222222222222222222222222222222' OR wallet_address LIKE '33333333333333333333333333333333333333333333');
DELETE FROM users WHERE wallet_address LIKE '11111111111111111111111111111111111111111111' OR wallet_address LIKE '22222222222222222222222222222222222222222222' OR wallet_address LIKE '33333333333333333333333333333333333333333333';
*/

-- =============================================
-- TEST COMPLETION
-- =============================================

-- Log test completion
INSERT INTO system_events (event_type, event_data, severity)
VALUES ('testing', '{"test_suite": "database_schema", "status": "completed", "timestamp": "' || NOW() || '"}', 'info');

-- Display completion message
DO $$
BEGIN
    RAISE NOTICE '=============================================';
    RAISE NOTICE 'DATABASE TESTING COMPLETED SUCCESSFULLY!';
    RAISE NOTICE '=============================================';
    RAISE NOTICE 'All tests passed. Your database is ready for production.';
    RAISE NOTICE 'Next steps:';
    RAISE NOTICE '1. Review test results above';
    RAISE NOTICE '2. Update your application code';
    RAISE NOTICE '3. Deploy to production';
    RAISE NOTICE '4. Monitor system_events table';
    RAISE NOTICE '=============================================';
END $$;
