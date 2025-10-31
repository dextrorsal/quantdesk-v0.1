-- QuantDesk Hybrid Points System Database Migration
-- Dynamic allocation inspired by Drift + Hyperliquid models
-- Proper allocation and incentives without fixed rates

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create points allocation periods table
CREATE TABLE IF NOT EXISTS points_allocation_periods (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    period_name VARCHAR(100) NOT NULL, -- 'Season 1', 'Season 2', etc.
    period_type VARCHAR(50) NOT NULL, -- 'foundation', 'engagement', 'innovation'
    start_date TIMESTAMP NOT NULL,
    end_date TIMESTAMP NOT NULL,
    total_points_pool BIGINT NOT NULL, -- Total points to distribute this period
    is_active BOOLEAN DEFAULT false,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create activity multipliers table
CREATE TABLE IF NOT EXISTS activity_multipliers (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    period_id UUID REFERENCES points_allocation_periods(id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL, -- 'trading_volume', 'deposits', 'staking', etc.
    base_multiplier DECIMAL(5,2) NOT NULL DEFAULT 1.00,
    early_user_bonus DECIMAL(5,2) NOT NULL DEFAULT 1.00,
    active_user_bonus DECIMAL(5,2) NOT NULL DEFAULT 1.00,
    community_bonus DECIMAL(5,2) NOT NULL DEFAULT 1.00,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(period_id, activity_type)
);

-- Create trading metrics table for core points system
CREATE TABLE IF NOT EXISTS trading_metrics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    total_trading_volume DECIMAL(20,2) DEFAULT 0.00,
    maker_volume DECIMAL(20,2) DEFAULT 0.00,
    taker_volume DECIMAL(20,2) DEFAULT 0.00,
    total_deposits DECIMAL(20,2) DEFAULT 0.00,
    total_withdrawals DECIMAL(20,2) DEFAULT 0.00,
    net_deposits DECIMAL(20,2) DEFAULT 0.00,
    staking_amount DECIMAL(20,2) DEFAULT 0.00,
    insurance_fund_stake DECIMAL(20,2) DEFAULT 0.00,
    total_trades INTEGER DEFAULT 0,
    active_trading_days INTEGER DEFAULT 0,
    last_trade_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id)
);

-- Create trading activity log for detailed tracking
CREATE TABLE IF NOT EXISTS trading_activity_log (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    period_id UUID REFERENCES points_allocation_periods(id) ON DELETE CASCADE,
    activity_type VARCHAR(50) NOT NULL, -- 'trade', 'deposit', 'withdrawal', 'staking', 'insurance_fund'
    amount DECIMAL(20,2) NOT NULL,
    market_symbol VARCHAR(20), -- For trades
    side VARCHAR(10), -- 'long', 'short', 'buy', 'sell', 'maker', 'taker'
    leverage DECIMAL(5,2), -- For trades
    points_earned INTEGER DEFAULT 0,
    multiplier_applied DECIMAL(5,2) DEFAULT 1.00,
    bonus_type VARCHAR(50), -- 'early_user', 'active_user', 'community'
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_trading_metrics_user_id ON trading_metrics(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_trading_volume ON trading_metrics(total_trading_volume DESC);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_net_deposits ON trading_metrics(net_deposits DESC);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_staking_amount ON trading_metrics(staking_amount DESC);
CREATE INDEX IF NOT EXISTS idx_trading_metrics_insurance_fund_stake ON trading_metrics(insurance_fund_stake DESC);
CREATE INDEX IF NOT EXISTS idx_trading_activity_log_user_id ON trading_activity_log(user_id);
CREATE INDEX IF NOT EXISTS idx_trading_activity_log_activity_type ON trading_activity_log(activity_type);
CREATE INDEX IF NOT EXISTS idx_trading_activity_log_created_at ON trading_activity_log(created_at);
CREATE INDEX IF NOT EXISTS idx_trading_activity_log_period_id ON trading_activity_log(period_id);
CREATE INDEX IF NOT EXISTS idx_points_allocation_periods_is_active ON points_allocation_periods(is_active);
CREATE INDEX IF NOT EXISTS idx_points_allocation_periods_period_type ON points_allocation_periods(period_type);
CREATE INDEX IF NOT EXISTS idx_activity_multipliers_period_id ON activity_multipliers(period_id);
CREATE INDEX IF NOT EXISTS idx_activity_multipliers_activity_type ON activity_multipliers(activity_type);

-- Create function to update trading metrics
CREATE OR REPLACE FUNCTION update_trading_metrics(
    user_uuid UUID,
    trading_volume DECIMAL(20,2) DEFAULT 0,
    maker_volume DECIMAL(20,2) DEFAULT 0,
    taker_volume DECIMAL(20,2) DEFAULT 0,
    deposit_amount DECIMAL(20,2) DEFAULT 0,
    withdrawal_amount DECIMAL(20,2) DEFAULT 0,
    staking_amount DECIMAL(20,2) DEFAULT 0,
    insurance_fund_stake DECIMAL(20,2) DEFAULT 0,
    trade_count INTEGER DEFAULT 0
)
RETURNS VOID AS $$
BEGIN
    INSERT INTO trading_metrics (
        user_id,
        total_trading_volume,
        maker_volume,
        taker_volume,
        total_deposits,
        total_withdrawals,
        net_deposits,
        staking_amount,
        insurance_fund_stake,
        total_trades,
        active_trading_days,
        last_trade_at,
        updated_at
    ) VALUES (
        user_uuid,
        trading_volume,
        maker_volume,
        taker_volume,
        deposit_amount,
        withdrawal_amount,
        deposit_amount - withdrawal_amount, -- net_deposits
        staking_amount,
        insurance_fund_stake,
        trade_count,
        1, -- active_trading_days
        NOW(),
        NOW()
    )
    ON CONFLICT (user_id) DO UPDATE SET
        total_trading_volume = trading_metrics.total_trading_volume + trading_volume,
        maker_volume = trading_metrics.maker_volume + maker_volume,
        taker_volume = trading_metrics.taker_volume + taker_volume,
        total_deposits = trading_metrics.total_deposits + deposit_amount,
        total_withdrawals = trading_metrics.total_withdrawals + withdrawal_amount,
        net_deposits = trading_metrics.net_deposits + deposit_amount - withdrawal_amount,
        staking_amount = GREATEST(trading_metrics.staking_amount, staking_amount),
        insurance_fund_stake = GREATEST(trading_metrics.insurance_fund_stake, insurance_fund_stake),
        total_trades = trading_metrics.total_trades + trade_count,
        active_trading_days = CASE 
            WHEN DATE(trading_metrics.last_trade_at) != CURRENT_DATE THEN trading_metrics.active_trading_days + 1
            ELSE trading_metrics.active_trading_days
        END,
        last_trade_at = NOW(),
        updated_at = NOW();
END;
$$ LANGUAGE plpgsql;

-- Create function to calculate dynamic points based on period allocation
CREATE OR REPLACE FUNCTION calculate_dynamic_points(
    activity_type VARCHAR(50),
    amount DECIMAL(20,2),
    period_id UUID,
    market_symbol VARCHAR(20) DEFAULT NULL,
    side VARCHAR(10) DEFAULT NULL,
    leverage DECIMAL(5,2) DEFAULT NULL,
    user_created_at TIMESTAMP DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    points INTEGER := 0;
    base_multiplier DECIMAL(5,2) := 1.00;
    early_user_bonus DECIMAL(5,2) := 1.00;
    active_user_bonus DECIMAL(5,2) := 1.00;
    community_bonus DECIMAL(5,2) := 1.00;
    total_multiplier DECIMAL(5,2) := 1.00;
    is_early_user BOOLEAN := false;
    is_active_user BOOLEAN := false;
    period_start TIMESTAMP;
BEGIN
    -- Get period start date
    SELECT start_date INTO period_start
    FROM points_allocation_periods
    WHERE id = period_id;
    
    -- Check if user is early user (joined before period start)
    IF user_created_at IS NOT NULL AND user_created_at < period_start THEN
        is_early_user := true;
    END IF;
    
    -- Check if user is active user (has recent activity)
    -- This would be determined by recent trading activity
    is_active_user := true; -- Simplified for now
    
    -- Get multipliers for this activity type and period
    SELECT 
        am.base_multiplier,
        am.early_user_bonus,
        am.active_user_bonus,
        am.community_bonus
    INTO 
        base_multiplier,
        early_user_bonus,
        active_user_bonus,
        community_bonus
    FROM activity_multipliers am
    WHERE am.period_id = calculate_dynamic_points.period_id
        AND am.activity_type = calculate_dynamic_points.activity_type;
    
    -- Calculate total multiplier
    total_multiplier := base_multiplier;
    
    IF is_early_user THEN
        total_multiplier := total_multiplier * early_user_bonus;
    END IF;
    
    IF is_active_user THEN
        total_multiplier := total_multiplier * active_user_bonus;
    END IF;
    
    -- Community bonus (simplified - could be based on referrals, etc.)
    total_multiplier := total_multiplier * community_bonus;
    
    -- Calculate base points based on activity type (similar to Drift's approach)
    CASE activity_type
        WHEN 'trade' THEN
            -- Trading volume points: Dynamic based on period allocation
            points := FLOOR(amount * 0.01 * total_multiplier); -- 1% of amount as base
            -- Bonus for maker volume
            IF side = 'maker' THEN
                points := FLOOR(points * 1.5); -- 50% bonus for makers
            END IF;
            -- Leverage bonus
            IF leverage > 1 THEN
                points := FLOOR(points * (1 + (leverage - 1) * 0.1)); -- 10% bonus per leverage point
            END IF;
            
        WHEN 'deposit' THEN
            -- Deposit points: Higher multiplier than trading
            points := FLOOR(amount * 0.02 * total_multiplier); -- 2% of amount as base
            
        WHEN 'staking' THEN
            -- Staking points: Highest multiplier
            points := FLOOR(amount * 0.03 * total_multiplier); -- 3% of amount as base
            
        WHEN 'insurance_fund' THEN
            -- Insurance fund points: Highest multiplier
            points := FLOOR(amount * 0.05 * total_multiplier); -- 5% of amount as base
            
        WHEN 'withdrawal' THEN
            -- Small penalty for withdrawals to encourage retention
            points := FLOOR(-amount * 0.005 * total_multiplier); -- 0.5% penalty
            
        ELSE
            points := 0;
    END CASE;
    
    -- Ensure points are not negative
    IF points < 0 THEN
        points := 0;
    END IF;
    
    RETURN points;
END;
$$ LANGUAGE plpgsql;

-- Create function to process trading activity and award points
CREATE OR REPLACE FUNCTION process_trading_activity(
    user_uuid UUID,
    activity_type VARCHAR(50),
    amount DECIMAL(20,2),
    market_symbol VARCHAR(20) DEFAULT NULL,
    side VARCHAR(10) DEFAULT NULL,
    leverage DECIMAL(5,2) DEFAULT NULL
)
RETURNS INTEGER AS $$
DECLARE
    points_earned INTEGER;
    activity_id UUID;
    current_period_id UUID;
    user_created_at TIMESTAMP;
    total_multiplier DECIMAL(5,2);
    bonus_type VARCHAR(50);
BEGIN
    -- Get current active period
    SELECT id INTO current_period_id
    FROM points_allocation_periods
    WHERE is_active = true
    ORDER BY start_date DESC
    LIMIT 1;
    
    -- If no active period, return 0
    IF current_period_id IS NULL THEN
        RETURN 0;
    END IF;
    
    -- Get user creation date
    SELECT created_at INTO user_created_at
    FROM users
    WHERE id = user_uuid;
    
    -- Calculate points for this activity
    points_earned := calculate_dynamic_points(
        activity_type, 
        amount, 
        current_period_id, 
        market_symbol, 
        side, 
        leverage, 
        user_created_at
    );
    
    -- Determine bonus type
    bonus_type := 'base';
    IF user_created_at < (SELECT start_date FROM points_allocation_periods WHERE id = current_period_id) THEN
        bonus_type := 'early_user';
    END IF;
    
    -- Log the activity
    INSERT INTO trading_activity_log (
        user_id,
        period_id,
        activity_type,
        amount,
        market_symbol,
        side,
        leverage,
        points_earned,
        multiplier_applied,
        bonus_type
    ) VALUES (
        user_uuid,
        current_period_id,
        activity_type,
        amount,
        market_symbol,
        side,
        leverage,
        points_earned,
        1.00, -- Will be calculated properly in the function
        bonus_type
    ) RETURNING id INTO activity_id;
    
    -- Update trading metrics
    CASE activity_type
        WHEN 'trade' THEN
            PERFORM update_trading_metrics(
                user_uuid,
                trading_volume := amount,
                maker_volume := CASE WHEN side = 'maker' THEN amount ELSE 0 END,
                taker_volume := CASE WHEN side = 'taker' THEN amount ELSE 0 END,
                trade_count := 1
            );
        WHEN 'deposit' THEN
            PERFORM update_trading_metrics(
                user_uuid,
                deposit_amount := amount
            );
        WHEN 'withdrawal' THEN
            PERFORM update_trading_metrics(
                user_uuid,
                withdrawal_amount := amount
            );
        WHEN 'staking' THEN
            PERFORM update_trading_metrics(
                user_uuid,
                staking_amount := amount
            );
        WHEN 'insurance_fund' THEN
            PERFORM update_trading_metrics(
                user_uuid,
                insurance_fund_stake := amount
            );
    END CASE;
    
    -- Award points to user
    IF points_earned > 0 THEN
        INSERT INTO points_transactions (
            user_id,
            points,
            transaction_type,
            source,
            description,
            metadata
        ) VALUES (
            user_uuid,
            points_earned,
            'earned',
            'trading_activity',
            'Trading activity: ' || activity_type || ' - ' || amount || ' USD',
            jsonb_build_object(
                'activity_type', activity_type,
                'amount', amount,
                'market_symbol', market_symbol,
                'side', side,
                'leverage', leverage,
                'activity_log_id', activity_id,
                'period_id', current_period_id,
                'bonus_type', bonus_type
            )
        );
    END IF;
    
    RETURN points_earned;
END;
$$ LANGUAGE plpgsql;

-- Create function to get trading metrics leaderboard
CREATE OR REPLACE FUNCTION get_trading_metrics_leaderboard(
    metric_type VARCHAR(50) DEFAULT 'trading_volume',
    limit_count INTEGER DEFAULT 10
)
RETURNS TABLE(
    rank BIGINT,
    user_id UUID,
    wallet_address VARCHAR(44),
    username VARCHAR(50),
    metric_value DECIMAL(20,2),
    total_trades BIGINT,
    active_trading_days BIGINT,
    total_points BIGINT
) AS $$
BEGIN
    CASE metric_type
        WHEN 'trading_volume' THEN
            RETURN QUERY
            SELECT 
                ROW_NUMBER() OVER (ORDER BY tm.total_trading_volume DESC) as rank,
                u.id as user_id,
                u.wallet_address,
                u.username,
                tm.total_trading_volume as metric_value,
                tm.total_trades,
                tm.active_trading_days,
                u.total_points
            FROM trading_metrics tm
            JOIN users u ON tm.user_id = u.id
            WHERE u.is_active = true AND tm.total_trading_volume > 0
            ORDER BY tm.total_trading_volume DESC
            LIMIT limit_count;
            
        WHEN 'deposits' THEN
            RETURN QUERY
            SELECT 
                ROW_NUMBER() OVER (ORDER BY tm.net_deposits DESC) as rank,
                u.id as user_id,
                u.wallet_address,
                u.username,
                tm.net_deposits as metric_value,
                tm.total_trades,
                tm.active_trading_days,
                u.total_points
            FROM trading_metrics tm
            JOIN users u ON tm.user_id = u.id
            WHERE u.is_active = true AND tm.net_deposits > 0
            ORDER BY tm.net_deposits DESC
            LIMIT limit_count;
            
        WHEN 'staking' THEN
            RETURN QUERY
            SELECT 
                ROW_NUMBER() OVER (ORDER BY tm.staking_amount DESC) as rank,
                u.id as user_id,
                u.wallet_address,
                u.username,
                tm.staking_amount as metric_value,
                tm.total_trades,
                tm.active_trading_days,
                u.total_points
            FROM trading_metrics tm
            JOIN users u ON tm.user_id = u.id
            WHERE u.is_active = true AND tm.staking_amount > 0
            ORDER BY tm.staking_amount DESC
            LIMIT limit_count;
            
        WHEN 'insurance_fund' THEN
            RETURN QUERY
            SELECT 
                ROW_NUMBER() OVER (ORDER BY tm.insurance_fund_stake DESC) as rank,
                u.id as user_id,
                u.wallet_address,
                u.username,
                tm.insurance_fund_stake as metric_value,
                tm.total_trades,
                tm.active_trading_days,
                u.total_points
            FROM trading_metrics tm
            JOIN users u ON tm.user_id = u.id
            WHERE u.is_active = true AND tm.insurance_fund_stake > 0
            ORDER BY tm.insurance_fund_stake DESC
            LIMIT limit_count;
            
        ELSE
            RETURN QUERY
            SELECT 
                ROW_NUMBER() OVER (ORDER BY tm.total_trading_volume DESC) as rank,
                u.id as user_id,
                u.wallet_address,
                u.username,
                tm.total_trading_volume as metric_value,
                tm.total_trades,
                tm.active_trading_days,
                u.total_points
            FROM trading_metrics tm
            JOIN users u ON tm.user_id = u.id
            WHERE u.is_active = true AND tm.total_trading_volume > 0
            ORDER BY tm.total_trading_volume DESC
            LIMIT limit_count;
    END CASE;
END;
$$ LANGUAGE plpgsql;

-- Create function to get trading analytics
CREATE OR REPLACE FUNCTION get_trading_analytics()
RETURNS TABLE(
    total_trading_volume DECIMAL(20,2),
    total_maker_volume DECIMAL(20,2),
    total_taker_volume DECIMAL(20,2),
    total_net_deposits DECIMAL(20,2),
    total_staking_amount DECIMAL(20,2),
    total_insurance_fund_stake DECIMAL(20,2),
    active_traders BIGINT,
    total_trading_points BIGINT
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(tm.total_trading_volume), 0) as total_trading_volume,
        COALESCE(SUM(tm.maker_volume), 0) as total_maker_volume,
        COALESCE(SUM(tm.taker_volume), 0) as total_taker_volume,
        COALESCE(SUM(tm.net_deposits), 0) as total_net_deposits,
        COALESCE(SUM(tm.staking_amount), 0) as total_staking_amount,
        COALESCE(SUM(tm.insurance_fund_stake), 0) as total_insurance_fund_stake,
        COUNT(DISTINCT tm.user_id) as active_traders,
        COALESCE(SUM(u.total_points), 0) as total_trading_points
    FROM trading_metrics tm
    JOIN users u ON tm.user_id = u.id
    WHERE u.is_active = true;
END;
$$ LANGUAGE plpgsql;

-- Create view for trading dashboard
CREATE OR REPLACE VIEW trading_dashboard AS
SELECT 
    u.id as user_id,
    u.wallet_address,
    u.username,
    u.total_points,
    u.level,
    COALESCE(tm.total_trading_volume, 0) as total_trading_volume,
    COALESCE(tm.maker_volume, 0) as maker_volume,
    COALESCE(tm.taker_volume, 0) as taker_volume,
    COALESCE(tm.net_deposits, 0) as net_deposits,
    COALESCE(tm.staking_amount, 0) as staking_amount,
    COALESCE(tm.insurance_fund_stake, 0) as insurance_fund_stake,
    COALESCE(tm.total_trades, 0) as total_trades,
    COALESCE(tm.active_trading_days, 0) as active_trading_days,
    tm.last_trade_at
FROM users u
LEFT JOIN trading_metrics tm ON u.id = tm.user_id
WHERE u.is_active = true;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO quantdesk_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO quantdesk_user;

COMMENT ON TABLE trading_metrics IS 'Core trading metrics for points calculation based on Drift airdrop criteria';
COMMENT ON TABLE trading_activity_log IS 'Detailed log of all trading activities for points calculation';
COMMENT ON TABLE points_allocation_periods IS 'Points allocation periods for dynamic distribution';
COMMENT ON TABLE activity_multipliers IS 'Activity multipliers for different periods and user types';
COMMENT ON FUNCTION calculate_dynamic_points IS 'Calculate points based on dynamic allocation and user bonuses';
COMMENT ON FUNCTION process_trading_activity IS 'Process trading activity and award points automatically';

-- Create function to initialize the points system with default periods
CREATE OR REPLACE FUNCTION initialize_points_system()
RETURNS VOID AS $$
DECLARE
    season1_id UUID;
    season2_id UUID;
    season3_id UUID;
BEGIN
    -- Create Season 1: Foundation (Drift-inspired)
    INSERT INTO points_allocation_periods (
        period_name,
        period_type,
        start_date,
        end_date,
        total_points_pool,
        is_active
    ) VALUES (
        'Season 1: Foundation',
        'foundation',
        NOW(),
        NOW() + INTERVAL '3 months',
        1000000, -- 1M points pool
        true
    ) RETURNING id INTO season1_id;
    
    -- Create Season 2: Engagement (Hyperliquid-inspired)
    INSERT INTO points_allocation_periods (
        period_name,
        period_type,
        start_date,
        end_date,
        total_points_pool,
        is_active
    ) VALUES (
        'Season 2: Engagement',
        'engagement',
        NOW() + INTERVAL '3 months',
        NOW() + INTERVAL '6 months',
        1500000, -- 1.5M points pool
        false
    ) RETURNING id INTO season2_id;
    
    -- Create Season 3: Innovation (QuantDesk-exclusive)
    INSERT INTO points_allocation_periods (
        period_name,
        period_type,
        start_date,
        end_date,
        total_points_pool,
        is_active
    ) VALUES (
        'Season 3: Innovation',
        'innovation',
        NOW() + INTERVAL '6 months',
        NOW() + INTERVAL '9 months',
        2000000, -- 2M points pool
        false
    ) RETURNING id INTO season3_id;
    
    -- Set up Season 1 multipliers (Drift-inspired)
    INSERT INTO activity_multipliers (period_id, activity_type, base_multiplier, early_user_bonus, active_user_bonus, community_bonus) VALUES
    (season1_id, 'trading_volume', 1.00, 2.00, 1.50, 1.20),
    (season1_id, 'deposits', 2.00, 2.50, 1.75, 1.30),
    (season1_id, 'staking', 3.00, 3.50, 2.00, 1.40),
    (season1_id, 'insurance_fund', 5.00, 6.00, 2.50, 1.50);
    
    -- Set up Season 2 multipliers (Hyperliquid-inspired)
    INSERT INTO activity_multipliers (period_id, activity_type, base_multiplier, early_user_bonus, active_user_bonus, community_bonus) VALUES
    (season2_id, 'trading_volume', 1.20, 1.80, 1.60, 1.40),
    (season2_id, 'deposits', 2.20, 2.20, 1.80, 1.50),
    (season2_id, 'staking', 3.20, 3.20, 2.20, 1.60),
    (season2_id, 'insurance_fund', 5.20, 5.20, 2.80, 1.70);
    
    -- Set up Season 3 multipliers (QuantDesk-exclusive)
    INSERT INTO activity_multipliers (period_id, activity_type, base_multiplier, early_user_bonus, active_user_bonus, community_bonus) VALUES
    (season3_id, 'trading_volume', 1.50, 1.50, 1.80, 1.60),
    (season3_id, 'deposits', 2.50, 2.00, 2.00, 1.70),
    (season3_id, 'staking', 3.50, 2.50, 2.20, 1.80),
    (season3_id, 'insurance_fund', 5.50, 3.00, 2.50, 1.90);
    
END;
$$ LANGUAGE plpgsql;

-- Initialize the points system with default periods
SELECT initialize_points_system();

-- Migration completed successfully
SELECT 'QuantDesk Hybrid Points System database migration completed successfully!' as status;
