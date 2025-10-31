-- QuantDesk MIKEY AI Integration Database Migration
-- Enhanced implementation exceeding Drift's capabilities
-- "More Open Than Drift" competitive positioning

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create MIKEY AI sessions table
CREATE TABLE IF NOT EXISTS mikey_sessions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    tier VARCHAR(20) NOT NULL CHECK (tier IN ('basic', 'pro', 'vip')),
    started_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP NOT NULL,
    is_active BOOLEAN DEFAULT true,
    usage_count INTEGER DEFAULT 0,
    max_usage INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

-- Create MIKEY AI usage tracking table
CREATE TABLE IF NOT EXISTS mikey_usage (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    session_id UUID REFERENCES mikey_sessions(id) ON DELETE CASCADE,
    feature_used VARCHAR(100) NOT NULL,
    points_earned INTEGER DEFAULT 0,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create MIKEY AI features table
CREATE TABLE IF NOT EXISTS mikey_features (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    name VARCHAR(100) UNIQUE NOT NULL,
    description TEXT,
    tier_required VARCHAR(20) NOT NULL CHECK (tier_required IN ('basic', 'pro', 'vip')),
    points_cost INTEGER DEFAULT 0,
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create MIKEY AI analytics table
CREATE TABLE IF NOT EXISTS mikey_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    session_id UUID REFERENCES mikey_sessions(id) ON DELETE CASCADE,
    feature_used VARCHAR(100) NOT NULL,
    usage_duration INTEGER DEFAULT 0, -- in seconds
    success BOOLEAN DEFAULT true,
    error_message TEXT,
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_mikey_sessions_user_id ON mikey_sessions(user_id);
CREATE INDEX IF NOT EXISTS idx_mikey_sessions_tier ON mikey_sessions(tier);
CREATE INDEX IF NOT EXISTS idx_mikey_sessions_expires_at ON mikey_sessions(expires_at);
CREATE INDEX IF NOT EXISTS idx_mikey_sessions_is_active ON mikey_sessions(is_active);
CREATE INDEX IF NOT EXISTS idx_mikey_usage_session_id ON mikey_usage(session_id);
CREATE INDEX IF NOT EXISTS idx_mikey_usage_feature_used ON mikey_usage(feature_used);
CREATE INDEX IF NOT EXISTS idx_mikey_usage_created_at ON mikey_usage(created_at);
CREATE INDEX IF NOT EXISTS idx_mikey_features_tier_required ON mikey_features(tier_required);
CREATE INDEX IF NOT EXISTS idx_mikey_features_is_active ON mikey_features(is_active);
CREATE INDEX IF NOT EXISTS idx_mikey_analytics_user_id ON mikey_analytics(user_id);
CREATE INDEX IF NOT EXISTS idx_mikey_analytics_session_id ON mikey_analytics(session_id);
CREATE INDEX IF NOT EXISTS idx_mikey_analytics_feature_used ON mikey_analytics(feature_used);

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_mikey_sessions_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_mikey_sessions_updated_at
    BEFORE UPDATE ON mikey_sessions
    FOR EACH ROW EXECUTE FUNCTION update_mikey_sessions_updated_at();

-- Insert default MIKEY AI features
INSERT INTO mikey_features (name, description, tier_required, points_cost, is_active) VALUES
('Market Analysis', 'Comprehensive market analysis using AI', 'basic', 0, true),
('Trading Suggestions', 'AI-powered trading recommendations', 'basic', 0, true),
('Portfolio Optimization', 'AI-driven portfolio optimization', 'pro', 0, true),
('Risk Management', 'Advanced risk assessment and management', 'pro', 0, true),
('Sentiment Analysis', 'Social media and news sentiment analysis', 'basic', 0, true),
('Custom Strategies', 'Develop custom trading strategies', 'vip', 0, true),
('Real-time Predictions', 'Live market predictions', 'vip', 0, true),
('Advanced Analytics', 'Deep market analytics and insights', 'vip', 0, true)
ON CONFLICT (name) DO NOTHING;

-- Create view for MIKEY AI user statistics
CREATE OR REPLACE VIEW mikey_user_stats AS
SELECT 
    u.id as user_id,
    u.wallet_address,
    u.username,
    u.total_points,
    u.level,
    COUNT(DISTINCT ms.id) as total_sessions,
    COUNT(DISTINCT mu.id) as total_feature_usage,
    COALESCE(SUM(mu.points_earned), 0) as total_points_earned,
    MAX(ms.tier) as highest_tier_used,
    MAX(ms.expires_at) as last_session_expiry
FROM users u
LEFT JOIN mikey_sessions ms ON u.id = ms.user_id
LEFT JOIN mikey_usage mu ON ms.id = mu.session_id
WHERE u.is_active = true
GROUP BY u.id, u.wallet_address, u.username, u.total_points, u.level;

-- Create function to check MIKEY AI access
CREATE OR REPLACE FUNCTION check_mikey_access(user_uuid UUID)
RETURNS TABLE(
    has_access BOOLEAN,
    tier VARCHAR(20),
    expires_at TIMESTAMP,
    usage_remaining INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CASE WHEN ms.id IS NOT NULL THEN true ELSE false END as has_access,
        ms.tier,
        ms.expires_at,
        (ms.max_usage - ms.usage_count) as usage_remaining
    FROM mikey_sessions ms
    WHERE ms.user_id = user_uuid 
        AND ms.is_active = true 
        AND ms.expires_at > NOW()
    ORDER BY ms.started_at DESC
    LIMIT 1;
END;
$$ LANGUAGE plpgsql;

-- Create function to get MIKEY AI usage statistics
CREATE OR REPLACE FUNCTION get_mikey_usage_stats(user_uuid UUID)
RETURNS TABLE(
    feature_used VARCHAR(100),
    usage_count BIGINT,
    total_points_earned BIGINT,
    last_used TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        mu.feature_used,
        COUNT(*) as usage_count,
        SUM(mu.points_earned) as total_points_earned,
        MAX(mu.created_at) as last_used
    FROM mikey_sessions ms
    JOIN mikey_usage mu ON ms.id = mu.session_id
    WHERE ms.user_id = user_uuid
    GROUP BY mu.feature_used
    ORDER BY usage_count DESC;
END;
$$ LANGUAGE plpgsql;

-- Create function to clean up expired MIKEY AI sessions
CREATE OR REPLACE FUNCTION cleanup_expired_mikey_sessions()
RETURNS INTEGER AS $$
DECLARE
    expired_count INTEGER;
BEGIN
    UPDATE mikey_sessions 
    SET is_active = false, updated_at = NOW()
    WHERE expires_at < NOW() AND is_active = true;
    
    GET DIAGNOSTICS expired_count = ROW_COUNT;
    
    RETURN expired_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to get MIKEY AI tier requirements
CREATE OR REPLACE FUNCTION get_mikey_tier_requirements()
RETURNS TABLE(
    tier VARCHAR(20),
    points_required INTEGER,
    duration_days INTEGER,
    max_usage INTEGER,
    features TEXT[]
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        'basic'::VARCHAR(20) as tier,
        500 as points_required,
        30 as duration_days,
        100 as max_usage,
        ARRAY['Market Analysis', 'Trading Suggestions', 'Sentiment Analysis'] as features
    UNION ALL
    SELECT 
        'pro'::VARCHAR(20) as tier,
        1000 as points_required,
        30 as duration_days,
        500 as max_usage,
        ARRAY['Portfolio Optimization', 'Risk Management', 'Custom Indicators', 'Backtesting'] as features
    UNION ALL
    SELECT 
        'vip'::VARCHAR(20) as tier,
        2000 as points_required,
        30 as duration_days,
        1000 as max_usage,
        ARRAY['Custom Strategies', 'Real-time Predictions', 'Advanced Analytics', 'Priority Support'] as features;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO quantdesk_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO quantdesk_user;

COMMENT ON TABLE mikey_sessions IS 'MIKEY AI access sessions for users';
COMMENT ON TABLE mikey_usage IS 'MIKEY AI feature usage tracking';
COMMENT ON TABLE mikey_features IS 'Available MIKEY AI features and their requirements';
COMMENT ON TABLE mikey_analytics IS 'MIKEY AI usage analytics and performance metrics';

-- Migration completed successfully
SELECT 'QuantDesk MIKEY AI Integration database migration completed successfully!' as status;
