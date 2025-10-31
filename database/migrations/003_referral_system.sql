-- QuantDesk Referral System Database Migration
-- Enhanced implementation exceeding Drift's capabilities
-- "More Open Than Drift" competitive positioning with 20% referral rewards

-- Enable UUID extension
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create referral codes table
CREATE TABLE IF NOT EXISTS referral_codes (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    code VARCHAR(20) UNIQUE NOT NULL,
    platform VARCHAR(20) NOT NULL CHECK (platform IN ('wallet', 'telegram', 'discord', 'twitter')),
    platform_id VARCHAR(100), -- Telegram username, Discord ID, Twitter handle, etc.
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP DEFAULT NOW(),
    expires_at TIMESTAMP,
    usage_count INTEGER DEFAULT 0,
    max_usage INTEGER DEFAULT NULL, -- NULL for unlimited
    metadata JSONB DEFAULT '{}'
);

-- Create referral relationships table
CREATE TABLE IF NOT EXISTS referral_relationships (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    referrer_id UUID REFERENCES users(id) ON DELETE CASCADE,
    referred_id UUID REFERENCES users(id) ON DELETE CASCADE,
    referral_code_id UUID REFERENCES referral_codes(id) ON DELETE CASCADE,
    platform VARCHAR(20) NOT NULL CHECK (platform IN ('wallet', 'telegram', 'discord', 'twitter')),
    platform_context JSONB DEFAULT '{}', -- Additional platform-specific data
    created_at TIMESTAMP DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    UNIQUE(referred_id) -- Each user can only be referred once
);

-- Create referral rewards table
CREATE TABLE IF NOT EXISTS referral_rewards (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    referral_relationship_id UUID REFERENCES referral_relationships(id) ON DELETE CASCADE,
    referrer_id UUID REFERENCES users(id) ON DELETE CASCADE,
    referred_id UUID REFERENCES users(id) ON DELETE CASCADE,
    points_earned INTEGER NOT NULL,
    points_percentage DECIMAL(5,2) NOT NULL DEFAULT 20.00, -- 20% referral reward
    source_transaction_id UUID REFERENCES points_transactions(id) ON DELETE CASCADE,
    reward_type VARCHAR(50) NOT NULL DEFAULT 'referral_bonus',
    created_at TIMESTAMP DEFAULT NOW(),
    metadata JSONB DEFAULT '{}'
);


-- Create referral analytics table
CREATE TABLE IF NOT EXISTS referral_analytics (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    total_referrals INTEGER DEFAULT 0,
    active_referrals INTEGER DEFAULT 0,
    total_points_earned INTEGER DEFAULT 0,
    referral_conversion_rate DECIMAL(5,2) DEFAULT 0.00,
    platform_breakdown JSONB DEFAULT '{}', -- Breakdown by platform
    last_referral_at TIMESTAMP,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(user_id)
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_referral_codes_user_id ON referral_codes(user_id);
CREATE INDEX IF NOT EXISTS idx_referral_codes_code ON referral_codes(code);
CREATE INDEX IF NOT EXISTS idx_referral_codes_platform ON referral_codes(platform);
CREATE INDEX IF NOT EXISTS idx_referral_codes_is_active ON referral_codes(is_active);
CREATE INDEX IF NOT EXISTS idx_referral_relationships_referrer_id ON referral_relationships(referrer_id);
CREATE INDEX IF NOT EXISTS idx_referral_relationships_referred_id ON referral_relationships(referred_id);
CREATE INDEX IF NOT EXISTS idx_referral_relationships_platform ON referral_relationships(platform);
CREATE INDEX IF NOT EXISTS idx_referral_relationships_created_at ON referral_relationships(created_at);
CREATE INDEX IF NOT EXISTS idx_referral_rewards_referrer_id ON referral_rewards(referrer_id);
CREATE INDEX IF NOT EXISTS idx_referral_rewards_referred_id ON referral_rewards(referred_id);
CREATE INDEX IF NOT EXISTS idx_referral_rewards_created_at ON referral_rewards(created_at);
CREATE INDEX IF NOT EXISTS idx_referral_analytics_user_id ON referral_analytics(user_id);
CREATE INDEX IF NOT EXISTS idx_referral_analytics_total_referrals ON referral_analytics(total_referrals DESC);

-- Create trigger to update referral analytics
CREATE OR REPLACE FUNCTION update_referral_analytics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update referrer analytics
    INSERT INTO referral_analytics (user_id, total_referrals, active_referrals, last_referral_at, updated_at)
    VALUES (NEW.referrer_id, 1, 1, NEW.created_at, NOW())
    ON CONFLICT (user_id) DO UPDATE SET
        total_referrals = referral_analytics.total_referrals + 1,
        active_referrals = referral_analytics.active_referrals + 1,
        last_referral_at = NEW.created_at,
        updated_at = NOW();

    -- Update referral code usage count
    UPDATE referral_codes 
    SET usage_count = usage_count + 1
    WHERE id = NEW.referral_code_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_referral_analytics
    AFTER INSERT ON referral_relationships
    FOR EACH ROW EXECUTE FUNCTION update_referral_analytics();

-- Create trigger to update referral rewards analytics
CREATE OR REPLACE FUNCTION update_referral_rewards_analytics()
RETURNS TRIGGER AS $$
BEGIN
    -- Update referrer's total points earned from referrals
    UPDATE referral_analytics 
    SET total_points_earned = total_points_earned + NEW.points_earned,
        updated_at = NOW()
    WHERE user_id = NEW.referrer_id;

    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_referral_rewards_analytics
    AFTER INSERT ON referral_rewards
    FOR EACH ROW EXECUTE FUNCTION update_referral_rewards_analytics();

-- Create function to generate referral code
CREATE OR REPLACE FUNCTION generate_referral_code(user_uuid UUID, platform_name VARCHAR(20))
RETURNS VARCHAR(20) AS $$
DECLARE
    base_code VARCHAR(20);
    final_code VARCHAR(20);
    counter INTEGER := 0;
BEGIN
    -- Generate base code from user ID
    base_code := SUBSTRING(user_uuid::text, 1, 8);
    
    -- Add platform prefix
    CASE platform_name
        WHEN 'wallet' THEN final_code := 'W' || base_code;
        WHEN 'telegram' THEN final_code := 'T' || base_code;
        WHEN 'discord' THEN final_code := 'D' || base_code;
        WHEN 'twitter' THEN final_code := 'X' || base_code;
        ELSE final_code := 'R' || base_code;
    END CASE;
    
    -- Ensure uniqueness
    WHILE EXISTS (SELECT 1 FROM referral_codes WHERE code = final_code) LOOP
        counter := counter + 1;
        final_code := final_code || counter::text;
    END LOOP;
    
    RETURN final_code;
END;
$$ LANGUAGE plpgsql;

-- Create function to process referral reward
CREATE OR REPLACE FUNCTION process_referral_reward(
    referred_user_uuid UUID,
    points_amount INTEGER
)
RETURNS INTEGER AS $$
DECLARE
    referral_rel RECORD;
    reward_points INTEGER;
    reward_percentage DECIMAL(5,2) := 20.00; -- 20% referral reward
BEGIN
    -- Find the referral relationship
    SELECT * INTO referral_rel
    FROM referral_relationships
    WHERE referred_id = referred_user_uuid
        AND is_active = true
    LIMIT 1;
    
    -- If no referral relationship found, return 0
    IF NOT FOUND THEN
        RETURN 0;
    END IF;
    
    -- Calculate reward points (20% of earned points)
    reward_points := FLOOR(points_amount * (reward_percentage / 100));
    
    -- If reward points is 0 or negative, return 0
    IF reward_points <= 0 THEN
        RETURN 0;
    END IF;
    
    -- Award points to referrer
    INSERT INTO points_transactions (
        user_id,
        points,
        transaction_type,
        source,
        description,
        metadata
    ) VALUES (
        referral_rel.referrer_id,
        reward_points,
        'earned',
        'referral_reward',
        'Referral reward: 20% of referred user points',
        jsonb_build_object(
            'referred_user_id', referred_user_uuid,
            'original_points', points_amount,
            'reward_percentage', reward_percentage,
            'referral_relationship_id', referral_rel.id
        )
    );
    
    -- Record the referral reward
    INSERT INTO referral_rewards (
        referral_relationship_id,
        referrer_id,
        referred_id,
        points_earned,
        points_percentage,
        reward_type,
        metadata
    ) VALUES (
        referral_rel.id,
        referral_rel.referrer_id,
        referred_user_uuid,
        reward_points,
        reward_percentage,
        'referral_bonus',
        jsonb_build_object(
            'original_points', points_amount,
            'platform', referral_rel.platform
        )
    );
    
    RETURN reward_points;
END;
$$ LANGUAGE plpgsql;

-- Create function to get referral statistics
CREATE OR REPLACE FUNCTION get_referral_stats(user_uuid UUID)
RETURNS TABLE(
    total_referrals BIGINT,
    active_referrals BIGINT,
    total_points_earned BIGINT,
    referral_conversion_rate DECIMAL(5,2),
    platform_breakdown JSONB,
    last_referral_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(ra.total_referrals, 0) as total_referrals,
        COALESCE(ra.active_referrals, 0) as active_referrals,
        COALESCE(ra.total_points_earned, 0) as total_points_earned,
        COALESCE(ra.referral_conversion_rate, 0.00) as referral_conversion_rate,
        COALESCE(ra.platform_breakdown, '{}'::jsonb) as platform_breakdown,
        ra.last_referral_at
    FROM referral_analytics ra
    WHERE ra.user_id = user_uuid;
END;
$$ LANGUAGE plpgsql;

-- Create function to get referral leaderboard
CREATE OR REPLACE FUNCTION get_referral_leaderboard(limit_count INTEGER DEFAULT 10)
RETURNS TABLE(
    rank BIGINT,
    user_id UUID,
    wallet_address VARCHAR(44),
    username VARCHAR(50),
    total_referrals BIGINT,
    total_points_earned BIGINT,
    platform_breakdown JSONB
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        ROW_NUMBER() OVER (ORDER BY ra.total_referrals DESC, ra.total_points_earned DESC) as rank,
        u.id as user_id,
        u.wallet_address,
        u.username,
        ra.total_referrals,
        ra.total_points_earned,
        ra.platform_breakdown
    FROM referral_analytics ra
    JOIN users u ON ra.user_id = u.id
    WHERE u.is_active = true
    ORDER BY ra.total_referrals DESC, ra.total_points_earned DESC
    LIMIT limit_count;
END;
$$ LANGUAGE plpgsql;

-- Create function to validate referral code
CREATE OR REPLACE FUNCTION validate_referral_code(code_to_validate VARCHAR(20))
RETURNS TABLE(
    is_valid BOOLEAN,
    user_id UUID,
    platform VARCHAR(20),
    platform_id VARCHAR(100),
    usage_count INTEGER,
    max_usage INTEGER,
    expires_at TIMESTAMP
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        CASE 
            WHEN rc.id IS NOT NULL 
                AND rc.is_active = true 
                AND (rc.expires_at IS NULL OR rc.expires_at > NOW())
                AND (rc.max_usage IS NULL OR rc.usage_count < rc.max_usage)
            THEN true 
            ELSE false 
        END as is_valid,
        rc.user_id,
        rc.platform,
        rc.platform_id,
        rc.usage_count,
        rc.max_usage,
        rc.expires_at
    FROM referral_codes rc
    WHERE rc.code = code_to_validate;
END;
$$ LANGUAGE plpgsql;

-- Create view for referral dashboard
CREATE OR REPLACE VIEW referral_dashboard AS
SELECT 
    u.id as user_id,
    u.wallet_address,
    u.username,
    u.total_points,
    u.level,
    COALESCE(ra.total_referrals, 0) as total_referrals,
    COALESCE(ra.active_referrals, 0) as active_referrals,
    COALESCE(ra.total_points_earned, 0) as referral_points_earned,
    COALESCE(ra.referral_conversion_rate, 0.00) as conversion_rate,
    ra.platform_breakdown,
    ra.last_referral_at,
    COUNT(DISTINCT rc.id) as referral_codes_count
FROM users u
LEFT JOIN referral_analytics ra ON u.id = ra.user_id
LEFT JOIN referral_codes rc ON u.id = rc.user_id AND rc.is_active = true
WHERE u.is_active = true
GROUP BY u.id, u.wallet_address, u.username, u.total_points, u.level, 
         ra.total_referrals, ra.active_referrals, ra.total_points_earned, 
         ra.referral_conversion_rate, ra.platform_breakdown, ra.last_referral_at;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO quantdesk_user;
-- GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO quantdesk_user;

COMMENT ON TABLE referral_codes IS 'Referral codes for different platforms (wallet, telegram, discord, twitter)';
COMMENT ON TABLE referral_relationships IS 'Referral relationships between users';
COMMENT ON TABLE referral_rewards IS 'Referral rewards and points earned by referrers';
COMMENT ON TABLE referral_analytics IS 'Referral analytics and statistics per user';

-- Migration completed successfully
SELECT 'QuantDesk Referral System database migration completed successfully!' as status;
