-- Portfolio Query Optimization Script
-- This script adds additional indexes and optimizations specifically for real-time portfolio calculations

-- =============================================
-- PORTFOLIO-SPECIFIC INDEXES
-- =============================================

-- Composite index for positions query (user_id + status) - most critical for portfolio calculations
CREATE INDEX IF NOT EXISTS idx_positions_user_status ON positions(user_id, status) 
WHERE status = 'open';

-- Composite index for collateral accounts query
CREATE INDEX IF NOT EXISTS idx_collateral_accounts_user ON collateral_accounts(user_id);

-- Optimized oracle prices index for latest price lookups
CREATE INDEX IF NOT EXISTS idx_oracle_prices_market_latest ON oracle_prices(market_id, created_at DESC) 
WHERE created_at >= NOW() - INTERVAL '1 hour'; -- Only index recent prices

-- =============================================
-- MATERIALIZED VIEWS FOR PORTFOLIO AGGREGATIONS
-- =============================================

-- Materialized view for user portfolio summary (refreshed every 30 seconds)
CREATE MATERIALIZED VIEW IF NOT EXISTS user_portfolio_summary AS
SELECT 
    u.id as user_id,
    u.wallet_address,
    COALESCE(SUM(ca.amount), 0) as total_collateral,
    COALESCE(COUNT(p.id), 0) as open_positions_count,
    COALESCE(SUM(p.size * p.entry_price), 0) as total_position_value,
    COALESCE(SUM(p.unrealized_pnl), 0) as total_unrealized_pnl,
    COALESCE(SUM(p.margin), 0) as total_margin,
    COALESCE(AVG(p.leverage), 0) as avg_leverage,
    COALESCE(MIN(p.health_factor), 100) as min_health_factor,
    NOW() as last_updated
FROM users u
LEFT JOIN collateral_accounts ca ON u.id = ca.user_id
LEFT JOIN positions p ON u.id = p.user_id AND p.status = 'open' AND p.size > 0
GROUP BY u.id, u.wallet_address;

-- Create unique index on materialized view
CREATE UNIQUE INDEX IF NOT EXISTS idx_user_portfolio_summary_user_id ON user_portfolio_summary(user_id);

-- =============================================
-- PORTFOLIO REFRESH FUNCTION
-- =============================================

-- Function to refresh portfolio summary materialized view
CREATE OR REPLACE FUNCTION refresh_user_portfolio_summary()
RETURNS void AS $$
BEGIN
    REFRESH MATERIALIZED VIEW CONCURRENTLY user_portfolio_summary;
    
    -- Log the refresh
    INSERT INTO system_events (event_type, severity, message, created_at)
    VALUES ('portfolio_refresh', 'info', 'User portfolio summary refreshed', NOW());
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- PORTFOLIO QUERY OPTIMIZATION FUNCTIONS
-- =============================================

-- Optimized function to get user portfolio data
CREATE OR REPLACE FUNCTION get_user_portfolio_data(p_user_id UUID)
RETURNS TABLE (
    user_id UUID,
    total_collateral NUMERIC,
    open_positions_count INTEGER,
    total_position_value NUMERIC,
    total_unrealized_pnl NUMERIC,
    total_margin NUMERIC,
    avg_leverage NUMERIC,
    min_health_factor NUMERIC,
    last_updated TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ups.user_id,
        ups.total_collateral,
        ups.open_positions_count,
        ups.total_position_value,
        ups.total_unrealized_pnl,
        ups.total_margin,
        ups.avg_leverage,
        ups.min_health_factor,
        ups.last_updated
    FROM user_portfolio_summary ups
    WHERE ups.user_id = p_user_id;
END;
$$ LANGUAGE plpgsql;

-- Optimized function to get user positions with current prices
CREATE OR REPLACE FUNCTION get_user_positions_with_prices(p_user_id UUID)
RETURNS TABLE (
    position_id UUID,
    market_id UUID,
    symbol TEXT,
    side TEXT,
    size NUMERIC,
    entry_price NUMERIC,
    current_price NUMERIC,
    unrealized_pnl NUMERIC,
    margin NUMERIC,
    leverage NUMERIC,
    health_factor NUMERIC
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        p.id as position_id,
        p.market_id,
        m.symbol,
        p.side,
        p.size,
        p.entry_price,
        COALESCE(op.price, p.mark_price) as current_price,
        p.unrealized_pnl,
        p.margin,
        p.leverage,
        p.health_factor
    FROM positions p
    JOIN markets m ON p.market_id = m.id
    LEFT JOIN LATERAL (
        SELECT price 
        FROM oracle_prices op2 
        WHERE op2.market_id = p.market_id 
        ORDER BY op2.created_at DESC 
        LIMIT 1
    ) op ON true
    WHERE p.user_id = p_user_id 
    AND p.status = 'open' 
    AND p.size > 0;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- AUTOMATIC REFRESH TRIGGER
-- =============================================

-- Function to refresh portfolio summary when positions change
CREATE OR REPLACE FUNCTION trigger_refresh_portfolio_summary()
RETURNS TRIGGER AS $$
BEGIN
    -- Refresh the materialized view for the affected user
    PERFORM refresh_user_portfolio_summary();
    RETURN COALESCE(NEW, OLD);
END;
$$ LANGUAGE plpgsql;

-- Create triggers for position changes
DROP TRIGGER IF EXISTS trigger_positions_portfolio_refresh ON positions;
CREATE TRIGGER trigger_positions_portfolio_refresh
    AFTER INSERT OR UPDATE OR DELETE ON positions
    FOR EACH ROW
    EXECUTE FUNCTION trigger_refresh_portfolio_summary();

-- Create triggers for collateral changes
DROP TRIGGER IF EXISTS trigger_collateral_portfolio_refresh ON collateral_accounts;
CREATE TRIGGER trigger_collateral_portfolio_refresh
    AFTER INSERT OR UPDATE OR DELETE ON collateral_accounts
    FOR EACH ROW
    EXECUTE FUNCTION trigger_refresh_portfolio_summary();

-- =============================================
-- PERFORMANCE MONITORING
-- =============================================

-- Function to get portfolio query performance stats
CREATE OR REPLACE FUNCTION get_portfolio_performance_stats()
RETURNS TABLE (
    query_name TEXT,
    avg_execution_time_ms NUMERIC,
    total_executions BIGINT,
    last_execution TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'portfolio_summary_query' as query_name,
        COALESCE(AVG(mean_exec_time), 0) * 1000 as avg_execution_time_ms,
        COALESCE(SUM(calls), 0) as total_executions,
        NOW() as last_execution
    FROM pg_stat_statements 
    WHERE query LIKE '%user_portfolio_summary%'
    UNION ALL
    SELECT 
        'positions_query' as query_name,
        COALESCE(AVG(mean_exec_time), 0) * 1000 as avg_execution_time_ms,
        COALESCE(SUM(calls), 0) as total_executions,
        NOW() as last_execution
    FROM pg_stat_statements 
    WHERE query LIKE '%positions%user_id%';
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- INITIAL DATA SETUP
-- =============================================

-- Refresh the materialized view initially
REFRESH MATERIALIZED VIEW user_portfolio_summary;

-- Create a scheduled job to refresh the materialized view every 30 seconds
-- Note: This would typically be handled by a cron job or application scheduler
-- For now, we'll create a function that can be called periodically

-- =============================================
-- INDEX USAGE ANALYSIS
-- =============================================

-- Function to analyze index usage for portfolio queries
CREATE OR REPLACE FUNCTION analyze_portfolio_index_usage()
RETURNS TABLE (
    index_name TEXT,
    table_name TEXT,
    index_size TEXT,
    index_scans BIGINT,
    tuples_read BIGINT,
    tuples_fetched BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        i.indexname as index_name,
        i.tablename as table_name,
        pg_size_pretty(pg_relation_size(i.indexname::regclass)) as index_size,
        s.idx_scan as index_scans,
        s.idx_tup_read as tuples_read,
        s.idx_tup_fetch as tuples_fetched
    FROM pg_indexes i
    JOIN pg_stat_user_indexes s ON i.indexname = s.indexrelname
    WHERE i.tablename IN ('positions', 'collateral_accounts', 'oracle_prices', 'user_portfolio_summary')
    ORDER BY s.idx_scan DESC;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- CLEANUP AND MAINTENANCE
-- =============================================

-- Function to clean up old oracle prices (keep only last 24 hours)
CREATE OR REPLACE FUNCTION cleanup_old_oracle_prices()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM oracle_prices 
    WHERE created_at < NOW() - INTERVAL '24 hours';
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    
    -- Log the cleanup
    INSERT INTO system_events (event_type, severity, message, created_at)
    VALUES ('cleanup', 'info', 'Cleaned up ' || deleted_count || ' old oracle price records', NOW());
    
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- =============================================
-- COMMENTS AND DOCUMENTATION
-- =============================================

COMMENT ON MATERIALIZED VIEW user_portfolio_summary IS 'Pre-computed portfolio summaries for fast real-time updates';
COMMENT ON FUNCTION get_user_portfolio_data(UUID) IS 'Optimized function to get user portfolio data using materialized view';
COMMENT ON FUNCTION get_user_positions_with_prices(UUID) IS 'Optimized function to get user positions with current market prices';
COMMENT ON FUNCTION refresh_user_portfolio_summary() IS 'Function to refresh the portfolio summary materialized view';
COMMENT ON FUNCTION cleanup_old_oracle_prices() IS 'Function to clean up old oracle price data to maintain performance';

-- =============================================
-- GRANT PERMISSIONS
-- =============================================

-- Grant necessary permissions for the application user
-- GRANT SELECT ON user_portfolio_summary TO quantdesk_app;
-- GRANT EXECUTE ON FUNCTION get_user_portfolio_data(UUID) TO quantdesk_app;
-- GRANT EXECUTE ON FUNCTION get_user_positions_with_prices(UUID) TO quantdesk_app;
-- GRANT EXECUTE ON FUNCTION refresh_user_portfolio_summary() TO quantdesk_app;
-- GRANT EXECUTE ON FUNCTION cleanup_old_oracle_prices() TO quantdesk_app;
