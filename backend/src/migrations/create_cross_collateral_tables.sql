-- Create collateral_accounts table
CREATE TABLE IF NOT EXISTS collateral_accounts (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    asset_type VARCHAR(10) NOT NULL CHECK (asset_type IN ('SOL', 'USDC', 'BTC', 'ETH', 'USDT')),
    amount DECIMAL(20, 8) NOT NULL CHECK (amount >= 0),
    value_usd DECIMAL(20, 2) NOT NULL CHECK (value_usd >= 0),
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    is_active BOOLEAN DEFAULT true,
    
    -- Additional fields for cross-collateralization
    price_usd DECIMAL(20, 2), -- Current price of the asset
    utilization_rate DECIMAL(5, 4) DEFAULT 0 CHECK (utilization_rate >= 0 AND utilization_rate <= 1), -- How much is being used
    available_amount DECIMAL(20, 8) DEFAULT 0 CHECK (available_amount >= 0), -- Available for withdrawal
    
    -- Metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Constraints
    CONSTRAINT chk_available_amount CHECK (available_amount <= amount),
    CONSTRAINT chk_value_calculation CHECK (value_usd = amount * COALESCE(price_usd, 0))
);

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_collateral_accounts_user_id ON collateral_accounts(user_id);
CREATE INDEX IF NOT EXISTS idx_collateral_accounts_asset_type ON collateral_accounts(asset_type);
CREATE INDEX IF NOT EXISTS idx_collateral_accounts_active ON collateral_accounts(is_active);
CREATE INDEX IF NOT EXISTS idx_collateral_accounts_user_asset ON collateral_accounts(user_id, asset_type) WHERE is_active = true;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_collateral_accounts_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER trigger_update_collateral_accounts_updated_at
    BEFORE UPDATE ON collateral_accounts
    FOR EACH ROW
    EXECUTE FUNCTION update_collateral_accounts_updated_at();

-- Create view for collateral portfolio summary
CREATE OR REPLACE VIEW collateral_portfolio_summary AS
SELECT 
    user_id,
    COUNT(*) as total_accounts,
    SUM(value_usd) as total_value_usd,
    SUM(available_amount * COALESCE(price_usd, 0)) as total_available_usd,
    SUM(value_usd - (available_amount * COALESCE(price_usd, 0))) as total_utilized_usd,
    CASE 
        WHEN SUM(value_usd) > 0 THEN 
            SUM(value_usd - (available_amount * COALESCE(price_usd, 0))) / SUM(value_usd)
        ELSE 0 
    END as utilization_rate,
    CASE 
        WHEN SUM(value_usd - (available_amount * COALESCE(price_usd, 0))) > 0 THEN 
            SUM(value_usd) / SUM(value_usd - (available_amount * COALESCE(price_usd, 0)))
        ELSE 0 
    END as health_factor
FROM collateral_accounts
WHERE is_active = true
GROUP BY user_id;

-- Create view for collateral by asset type
CREATE OR REPLACE VIEW collateral_by_asset_type AS
SELECT 
    asset_type,
    COUNT(*) as account_count,
    COUNT(DISTINCT user_id) as unique_users,
    SUM(amount) as total_amount,
    SUM(value_usd) as total_value_usd,
    AVG(utilization_rate) as avg_utilization_rate,
    MIN(price_usd) as min_price,
    MAX(price_usd) as max_price,
    AVG(price_usd) as avg_price
FROM collateral_accounts
WHERE is_active = true
GROUP BY asset_type;

-- Create table for collateral swaps history
CREATE TABLE IF NOT EXISTS collateral_swaps (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    user_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    from_asset VARCHAR(10) NOT NULL CHECK (from_asset IN ('SOL', 'USDC', 'BTC', 'ETH', 'USDT')),
    to_asset VARCHAR(10) NOT NULL CHECK (to_asset IN ('SOL', 'USDC', 'BTC', 'ETH', 'USDT')),
    from_amount DECIMAL(20, 8) NOT NULL CHECK (from_amount > 0),
    to_amount DECIMAL(20, 8) NOT NULL CHECK (to_amount > 0),
    exchange_rate DECIMAL(20, 8) NOT NULL CHECK (exchange_rate > 0),
    fee DECIMAL(20, 8) NOT NULL CHECK (fee >= 0),
    transaction_id VARCHAR(100),
    status VARCHAR(20) DEFAULT 'completed' CHECK (status IN ('pending', 'completed', 'failed')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    CONSTRAINT chk_different_assets CHECK (from_asset != to_asset)
);

-- Create indexes for collateral swaps
CREATE INDEX IF NOT EXISTS idx_collateral_swaps_user_id ON collateral_swaps(user_id);
CREATE INDEX IF NOT EXISTS idx_collateral_swaps_created_at ON collateral_swaps(created_at);
CREATE INDEX IF NOT EXISTS idx_collateral_swaps_status ON collateral_swaps(status);

-- Create table for collateral utilization tracking
CREATE TABLE IF NOT EXISTS collateral_utilization (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    collateral_account_id UUID NOT NULL REFERENCES collateral_accounts(id) ON DELETE CASCADE,
    position_id UUID, -- Reference to position using this collateral
    utilized_amount DECIMAL(20, 8) NOT NULL CHECK (utilized_amount > 0),
    utilization_type VARCHAR(20) NOT NULL CHECK (utilization_type IN ('position', 'loan', 'margin')),
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    released_at TIMESTAMP WITH TIME ZONE,
    
    CONSTRAINT chk_released_after_created CHECK (released_at IS NULL OR released_at >= created_at)
);

-- Create indexes for collateral utilization
CREATE INDEX IF NOT EXISTS idx_collateral_utilization_account_id ON collateral_utilization(collateral_account_id);
CREATE INDEX IF NOT EXISTS idx_collateral_utilization_position_id ON collateral_utilization(position_id);
CREATE INDEX IF NOT EXISTS idx_collateral_utilization_active ON collateral_utilization(collateral_account_id) WHERE released_at IS NULL;

-- Create function to calculate collateral health
CREATE OR REPLACE FUNCTION calculate_collateral_health(p_user_id UUID)
RETURNS TABLE (
    total_value_usd DECIMAL(20, 2),
    total_utilized_usd DECIMAL(20, 2),
    utilization_rate DECIMAL(5, 4),
    health_factor DECIMAL(10, 4),
    is_healthy BOOLEAN
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        COALESCE(SUM(ca.value_usd), 0) as total_value_usd,
        COALESCE(SUM(ca.value_usd - (ca.available_amount * COALESCE(ca.price_usd, 0))), 0) as total_utilized_usd,
        CASE 
            WHEN SUM(ca.value_usd) > 0 THEN 
                SUM(ca.value_usd - (ca.available_amount * COALESCE(ca.price_usd, 0))) / SUM(ca.value_usd)
            ELSE 0 
        END as utilization_rate,
        CASE 
            WHEN SUM(ca.value_usd - (ca.available_amount * COALESCE(ca.price_usd, 0))) > 0 THEN 
                SUM(ca.value_usd) / SUM(ca.value_usd - (ca.available_amount * COALESCE(ca.price_usd, 0)))
            ELSE 0 
        END as health_factor,
        CASE 
            WHEN SUM(ca.value_usd) > 0 AND 
                 SUM(ca.value_usd - (ca.available_amount * COALESCE(ca.price_usd, 0))) / SUM(ca.value_usd) < 0.8 
            THEN true 
            ELSE false 
        END as is_healthy
    FROM collateral_accounts ca
    WHERE ca.user_id = p_user_id AND ca.is_active = true;
END;
$$ LANGUAGE plpgsql;

-- Create function to get max borrowable amount
CREATE OR REPLACE FUNCTION get_max_borrowable_amount(p_user_id UUID)
RETURNS DECIMAL(20, 2) AS $$
DECLARE
    max_borrowable DECIMAL(20, 2) := 0;
    asset_config RECORD;
    asset_value DECIMAL(20, 2);
BEGIN
    -- Calculate max borrowable for each asset type
    FOR asset_config IN 
        SELECT DISTINCT asset_type, value_usd
        FROM collateral_accounts 
        WHERE user_id = p_user_id AND is_active = true
    LOOP
        -- Get LTV for this asset type
        CASE asset_config.asset_type
            WHEN 'SOL' THEN asset_value := asset_config.value_usd * 0.8;
            WHEN 'USDC' THEN asset_value := asset_config.value_usd * 0.95;
            WHEN 'BTC' THEN asset_value := asset_config.value_usd * 0.85;
            WHEN 'ETH' THEN asset_value := asset_config.value_usd * 0.85;
            WHEN 'USDT' THEN asset_value := asset_config.value_usd * 0.95;
            ELSE asset_value := 0;
        END CASE;
        
        max_borrowable := max_borrowable + asset_value;
    END LOOP;
    
    RETURN max_borrowable;
END;
$$ LANGUAGE plpgsql;

-- Grant permissions (adjust as needed for your setup)
-- GRANT SELECT, INSERT, UPDATE, DELETE ON collateral_accounts TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON collateral_swaps TO your_app_user;
-- GRANT SELECT, INSERT, UPDATE, DELETE ON collateral_utilization TO your_app_user;
-- GRANT SELECT ON collateral_portfolio_summary TO your_app_user;
-- GRANT SELECT ON collateral_by_asset_type TO your_app_user;
-- GRANT EXECUTE ON FUNCTION calculate_collateral_health(UUID) TO your_app_user;
-- GRANT EXECUTE ON FUNCTION get_max_borrowable_amount(UUID) TO your_app_user;
