-- Order Query Optimization Script
-- This script adds comprehensive indexes for order-related queries to improve performance
-- Based on analysis of common query patterns in the QuantDesk codebase

-- ============================================================================
-- COMPOSITE INDEXES FOR COMMON QUERY PATTERNS
-- ============================================================================

-- 1. User + Status queries (most common pattern)
-- Used in: getUserOrders(userId, status), order filtering by user and status
CREATE INDEX IF NOT EXISTS idx_orders_user_status ON orders(user_id, status);

-- 2. Market + Status queries
-- Used in: market-specific order queries, order matching
CREATE INDEX IF NOT EXISTS idx_orders_market_status ON orders(market_id, status);

-- 3. User + Market queries
-- Used in: user's orders for specific market
CREATE INDEX IF NOT EXISTS idx_orders_user_market ON orders(user_id, market_id);

-- 4. Status + Created At queries
-- Used in: recent orders by status, order matching by time
CREATE INDEX IF NOT EXISTS idx_orders_status_created_at ON orders(status, created_at);

-- 5. Market + Created At queries
-- Used in: recent orders for market, market activity queries
CREATE INDEX IF NOT EXISTS idx_orders_market_created_at ON orders(market_id, created_at);

-- 6. User + Created At queries
-- Used in: user's recent orders, chronological order history
CREATE INDEX IF NOT EXISTS idx_orders_user_created_at ON orders(user_id, created_at);

-- ============================================================================
-- ORDER MATCHING OPTIMIZATION INDEXES
-- ============================================================================

-- 7. Order matching by market, side, and price (for limit orders)
-- Used in: finding matching orders for limit order execution
CREATE INDEX IF NOT EXISTS idx_orders_matching_limit ON orders(market_id, side, price, status) 
WHERE status = 'pending' AND order_type = 'limit';

-- 8. Order matching by market and side (for market orders)
-- Used in: finding matching orders for market order execution
CREATE INDEX IF NOT EXISTS idx_orders_matching_market ON orders(market_id, side, status) 
WHERE status = 'pending' AND order_type = 'market';

-- 9. Stop loss orders by market and side
-- Used in: stop loss order execution when price triggers
CREATE INDEX IF NOT EXISTS idx_orders_stop_loss ON orders(market_id, side, stop_price, status) 
WHERE status = 'pending' AND order_type = 'stop_loss';

-- 10. Take profit orders by market and side
-- Used in: take profit order execution when price triggers
CREATE INDEX IF NOT EXISTS idx_orders_take_profit ON orders(market_id, side, stop_price, status) 
WHERE status = 'pending' AND order_type = 'take_profit';

-- ============================================================================
-- EXPIRATION AND CLEANUP INDEXES
-- ============================================================================

-- 11. Expired orders cleanup
-- Used in: finding and cleaning up expired orders
CREATE INDEX IF NOT EXISTS idx_orders_expired ON orders(expires_at, status) 
WHERE expires_at IS NOT NULL AND status = 'pending';

-- 12. Orders by expiration time
-- Used in: order expiration monitoring and cleanup
CREATE INDEX IF NOT EXISTS idx_orders_expires_at ON orders(expires_at) 
WHERE expires_at IS NOT NULL;

-- ============================================================================
-- ANALYTICS AND REPORTING INDEXES
-- ============================================================================

-- 13. Order type and side analytics
-- Used in: order analytics, market analysis
CREATE INDEX IF NOT EXISTS idx_orders_type_side ON orders(order_type, side, created_at);

-- 14. Leverage analysis
-- Used in: leverage-based analytics and risk management
CREATE INDEX IF NOT EXISTS idx_orders_leverage ON orders(leverage, created_at);

-- 15. Order size analysis
-- Used in: order size analytics, whale detection
CREATE INDEX IF NOT EXISTS idx_orders_size ON orders(size, created_at);

-- ============================================================================
-- PERFORMANCE MONITORING INDEXES
-- ============================================================================

-- 16. Order execution time analysis
-- Used in: performance monitoring, execution time analytics
CREATE INDEX IF NOT EXISTS idx_orders_execution_time ON orders(filled_at, created_at) 
WHERE filled_at IS NOT NULL;

-- 17. Order update frequency
-- Used in: monitoring order update patterns
CREATE INDEX IF NOT EXISTS idx_orders_updated_at ON orders(updated_at);

-- ============================================================================
-- ADVANCED ORDER TABLE INDEXES
-- ============================================================================

-- 18. Advanced orders user + status
-- Used in: advanced order queries by user and status
CREATE INDEX IF NOT EXISTS idx_advanced_orders_user_status ON advanced_orders(user_id, status);

-- 19. Advanced orders market + status
-- Used in: advanced order queries by market and status
CREATE INDEX IF NOT EXISTS idx_advanced_orders_market_status ON advanced_orders(market_id, status);

-- 20. Advanced orders by order type
-- Used in: advanced order type analytics
CREATE INDEX IF NOT EXISTS idx_advanced_orders_type ON advanced_orders(order_type, created_at);

-- 21. Advanced orders by parent order
-- Used in: bracket order management
CREATE INDEX IF NOT EXISTS idx_advanced_orders_parent ON advanced_orders(parent_order_id) 
WHERE parent_order_id IS NOT NULL;

-- 22. Advanced orders by time in force
-- Used in: time-based order management
CREATE INDEX IF NOT EXISTS idx_advanced_orders_time_in_force ON advanced_orders(time_in_force, created_at);

-- ============================================================================
-- PARTIAL INDEXES FOR SPECIFIC SCENARIOS
-- ============================================================================

-- 23. Active orders only (pending and partially filled)
-- Used in: order matching, active order queries
CREATE INDEX IF NOT EXISTS idx_orders_active ON orders(market_id, side, price, created_at) 
WHERE status IN ('pending', 'partially_filled');

-- 24. Filled orders only
-- Used in: trade history, filled order analytics
CREATE INDEX IF NOT EXISTS idx_orders_filled ON orders(user_id, market_id, filled_at) 
WHERE status = 'filled';

-- 25. Cancelled orders only
-- Used in: cancellation analytics, user behavior analysis
CREATE INDEX IF NOT EXISTS idx_orders_cancelled ON orders(user_id, market_id, cancelled_at) 
WHERE status = 'cancelled';

-- ============================================================================
-- COVERING INDEXES FOR COMMON QUERIES
-- ============================================================================

-- 26. Covering index for user order list
-- Includes all commonly selected columns for user order queries
CREATE INDEX IF NOT EXISTS idx_orders_user_covering ON orders(user_id, status, created_at DESC) 
INCLUDE (id, market_id, order_type, side, size, price, stop_price, leverage, filled_size, average_fill_price);

-- 27. Covering index for market order list
-- Includes all commonly selected columns for market order queries
CREATE INDEX IF NOT EXISTS idx_orders_market_covering ON orders(market_id, status, created_at DESC) 
INCLUDE (id, user_id, order_type, side, size, price, stop_price, leverage, filled_size);

-- ============================================================================
-- STATISTICS AND MAINTENANCE
-- ============================================================================

-- Update table statistics for better query planning
ANALYZE orders;
ANALYZE advanced_orders;

-- ============================================================================
-- INDEX USAGE MONITORING
-- ============================================================================

-- Create a view to monitor index usage
CREATE OR REPLACE VIEW order_index_usage AS
SELECT 
    schemaname,
    tablename,
    indexname,
    idx_tup_read,
    idx_tup_fetch,
    idx_scan,
    CASE 
        WHEN idx_scan > 0 THEN (idx_tup_read::float / idx_scan)
        ELSE 0 
    END as avg_tuples_per_scan
FROM pg_stat_user_indexes 
WHERE tablename IN ('orders', 'advanced_orders')
ORDER BY idx_scan DESC;

-- ============================================================================
-- QUERY PERFORMANCE MONITORING
-- ============================================================================

-- Create a function to monitor slow order queries
CREATE OR REPLACE FUNCTION monitor_slow_order_queries()
RETURNS TABLE (
    query_text TEXT,
    avg_execution_time INTERVAL,
    total_calls BIGINT,
    slow_calls BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        query,
        mean_exec_time,
        calls,
        CASE 
            WHEN mean_exec_time > INTERVAL '100ms' THEN calls
            ELSE 0
        END as slow_calls
    FROM pg_stat_statements 
    WHERE query ILIKE '%orders%' 
    ORDER BY mean_exec_time DESC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- INDEX MAINTENANCE RECOMMENDATIONS
-- ============================================================================

-- Create a function to check for unused indexes
CREATE OR REPLACE FUNCTION check_unused_order_indexes()
RETURNS TABLE (
    indexname TEXT,
    tablename TEXT,
    idx_scan BIGINT,
    recommendation TEXT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.indexname,
        i.tablename,
        COALESCE(s.idx_scan, 0) as idx_scan,
        CASE 
            WHEN COALESCE(s.idx_scan, 0) = 0 THEN 'Consider dropping - unused'
            WHEN COALESCE(s.idx_scan, 0) < 10 THEN 'Monitor - low usage'
            ELSE 'Keep - actively used'
        END as recommendation
    FROM pg_indexes i
    LEFT JOIN pg_stat_user_indexes s ON i.indexname = s.indexname
    WHERE i.tablename IN ('orders', 'advanced_orders')
    ORDER BY COALESCE(s.idx_scan, 0) ASC;
END;
$$ LANGUAGE plpgsql;

-- ============================================================================
-- PERFORMANCE TESTING QUERIES
-- ============================================================================

-- Test queries to verify index effectiveness
-- Uncomment these to test performance after index creation

/*
-- Test 1: User orders with status filter
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM orders 
WHERE user_id = 'test-user-id' AND status = 'pending' 
ORDER BY created_at DESC;

-- Test 2: Market orders for matching
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM orders 
WHERE market_id = 'test-market-id' AND status = 'pending' AND order_type = 'limit' 
ORDER BY price ASC;

-- Test 3: Expired orders cleanup
EXPLAIN (ANALYZE, BUFFERS) 
SELECT * FROM orders 
WHERE expires_at < NOW() AND status = 'pending';

-- Test 4: Order analytics
EXPLAIN (ANALYZE, BUFFERS) 
SELECT order_type, side, COUNT(*), AVG(size) 
FROM orders 
WHERE created_at > NOW() - INTERVAL '24 hours' 
GROUP BY order_type, side;
*/

-- ============================================================================
-- COMMENTS AND DOCUMENTATION
-- ============================================================================

COMMENT ON INDEX idx_orders_user_status IS 'Optimizes user order queries with status filtering';
COMMENT ON INDEX idx_orders_market_status IS 'Optimizes market-specific order queries with status filtering';
COMMENT ON INDEX idx_orders_matching_limit IS 'Optimizes limit order matching for better execution performance';
COMMENT ON INDEX idx_orders_matching_market IS 'Optimizes market order matching for better execution performance';
COMMENT ON INDEX idx_orders_expired IS 'Optimizes expired order cleanup queries';
COMMENT ON INDEX idx_orders_user_covering IS 'Covering index for user order list queries to avoid table lookups';
COMMENT ON INDEX idx_orders_market_covering IS 'Covering index for market order list queries to avoid table lookups';

-- ============================================================================
-- COMPLETION MESSAGE
-- ============================================================================

DO $$
BEGIN
    RAISE NOTICE 'Order query optimization completed successfully!';
    RAISE NOTICE 'Created 27 new indexes for improved order query performance.';
    RAISE NOTICE 'Use check_unused_order_indexes() to monitor index usage.';
    RAISE NOTICE 'Use monitor_slow_order_queries() to identify slow queries.';
END $$;
