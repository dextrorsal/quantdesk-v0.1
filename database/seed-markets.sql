-- QuantDesk Market Seeding Script
-- Comprehensive list of Solana perpetual markets for DEX

-- Major Cryptocurrencies
INSERT INTO markets (symbol, base_asset, quote_asset, pyth_price_feed_id, max_leverage, initial_margin_ratio, maintenance_margin_ratio, is_active) VALUES
-- Core Crypto
('BTC-PERP', 'BTC', 'USDT', 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J', 100, 500, 300, true),
('ETH-PERP', 'ETH', 'USDT', 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB', 50, 500, 300, true),
('SOL-PERP', 'SOL', 'USDT', 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG', 50, 500, 300, true),

-- Major Altcoins
('BNB-PERP', 'BNB', 'USDT', '4bzs4FQ7jXJg8jLz1v5JzKzKzKzKzKzKzKzKzKzKzKzKz', 30, 500, 300, true),
('MATIC-PERP', 'MATIC', 'USDT', '5z3EqYQo9HiRjs7rBErM1Lz4Mq7qd1gGy6kdaBdGzKzKz', 30, 500, 300, true),
('AVAX-PERP', 'AVAX', 'USDT', '6zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 30, 500, 300, true),
('ARB-PERP', 'ARB', 'USDT', '7zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 30, 500, 300, true),

-- Solana Ecosystem Tokens
('RAY-PERP', 'RAY', 'USDT', 'AnLf8tVYCMuxgm5agCakMQPcYHjtx6kLDnF7AfWXThAL', 20, 500, 300, true),
('SRM-PERP', 'SRM', 'USDT', '3vxLXJqLqF3JG5TCbYycbKWRBbCJQLx2TfExGGCfQCx1', 20, 500, 300, true),
('MNGO-PERP', 'MNGO', 'USDT', '79wm3jjcPr6RaNQ4DGvP5KxG1mNd3gEBsg6kgNVd7Lk', 20, 500, 300, true),
('ORCA-PERP', 'ORCA', 'USDT', '4ivThsXhqieTBvHxDvRKFj2pMQmnkkuY81kf4RweGCLM', 20, 500, 300, true),
('JUP-PERP', 'JUP', 'USDT', 'g6eRCbboSw8tH2jG2QmR2L2L2L2L2L2L2L2L2L2L2L2L2', 20, 500, 300, true),
('STEP-PERP', 'STEP', 'USDT', '8zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),
('COPE-PERP', 'COPE', 'USDT', '9zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),
('FIDA-PERP', 'FIDA', 'USDT', '8zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),

-- Popular Meme Coins
('BONK-PERP', 'BONK', 'USDT', '8ihFLu5FimgTQ1Unh4dVyEHUGodJ5gJQCrQf4KUVB9bN', 10, 1000, 500, true),
('WIF-PERP', 'WIF', 'USDT', '6ABgrEZk8urs6kJ1JNdC1sspH5zKXRqxy8sg3ZG2cQps', 10, 1000, 500, true),
('POPCAT-PERP', 'POPCAT', 'USDT', '7zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 10, 1000, 500, true),
('MYRO-PERP', 'MYRO', 'USDT', '8zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 10, 1000, 500, true),
('PEPE-PERP', 'PEPE', 'USDT', '9zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 10, 1000, 500, true),
('DOGE-PERP', 'DOGE', 'USDT', '3LfA1yQN4xq8N4xq8N4xq8N4xq8N4xq8N4xq8N4xq8N4xq', 10, 1000, 500, true),
('SHIB-PERP', 'SHIB', 'USDT', '4zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 10, 1000, 500, true),

-- DeFi Tokens
('UNI-PERP', 'UNI', 'USDT', '5zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),
('AAVE-PERP', 'AAVE', 'USDT', '6zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),
('COMP-PERP', 'COMP', 'USDT', '7zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),
('MKR-PERP', 'MKR', 'USDT', '8zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),
('CRV-PERP', 'CRV', 'USDT', '9zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),

-- Gaming Tokens
('AXS-PERP', 'AXS', 'USDT', '5zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 15, 500, 300, true),
('SAND-PERP', 'SAND', 'USDT', '6zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 15, 500, 300, true),
('MANA-PERP', 'MANA', 'USDT', '7zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 15, 500, 300, true),
('GALA-PERP', 'GALA', 'USDT', '8zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 15, 500, 300, true),

-- Layer 2 Tokens
('OP-PERP', 'OP', 'USDT', '9zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 25, 500, 300, true),
('IMX-PERP', 'IMX', 'USDT', '5zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 25, 500, 300, true),
('LRC-PERP', 'LRC', 'USDT', '6zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 25, 500, 300, true),

-- AI Tokens
('FET-PERP', 'FET', 'USDT', '7zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),
('AGIX-PERP', 'AGIX', 'USDT', '8zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),
('OCEAN-PERP', 'OCEAN', 'USDT', '9zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 20, 500, 300, true),

-- Stablecoins (for arbitrage)
('USDC-PERP', 'USDC', 'USDT', 'Gnt27xtC473ZT2Mw5u8wZ68Z3gULkSTb5DuxJy7eJotD', 5, 2000, 1000, true),
('DAI-PERP', 'DAI', 'USDT', '5zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 5, 2000, 1000, true),

-- Commodities
('GOLD-PERP', 'GOLD', 'USDT', '6zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 10, 1000, 500, true),
('SILVER-PERP', 'SILVER', 'USDT', '7zKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKzKz', 10, 1000, 500, true)

ON CONFLICT (symbol) DO UPDATE SET
    max_leverage = EXCLUDED.max_leverage,
    initial_margin_ratio = EXCLUDED.initial_margin_ratio,
    maintenance_margin_ratio = EXCLUDED.maintenance_margin_ratio,
    is_active = EXCLUDED.is_active,
    updated_at = NOW();

-- Add market metadata for frontend
UPDATE markets SET metadata = jsonb_build_object(
    'category', CASE 
        WHEN base_asset IN ('BTC', 'ETH', 'SOL', 'BNB', 'MATIC', 'AVAX', 'ARB') THEN 'major-crypto'
        WHEN base_asset IN ('RAY', 'SRM', 'MNGO', 'ORCA', 'JUP', 'STEP', 'COPE', 'FIDA') THEN 'solana-ecosystem'
        WHEN base_asset IN ('BONK', 'WIF', 'POPCAT', 'MYRO', 'PEPE', 'DOGE', 'SHIB') THEN 'meme-coins'
        WHEN base_asset IN ('UNI', 'AAVE', 'COMP', 'MKR', 'CRV') THEN 'defi'
        WHEN base_asset IN ('AXS', 'SAND', 'MANA', 'GALA') THEN 'gaming'
        WHEN base_asset IN ('OP', 'IMX', 'LRC') THEN 'layer2'
        WHEN base_asset IN ('FET', 'AGIX', 'OCEAN') THEN 'ai'
        WHEN base_asset IN ('USDC', 'DAI') THEN 'stablecoins'
        WHEN base_asset IN ('GOLD', 'SILVER') THEN 'commodities'
        ELSE 'other'
    END,
    'description', CASE 
        WHEN base_asset = 'BTC' THEN 'Bitcoin - The original cryptocurrency'
        WHEN base_asset = 'ETH' THEN 'Ethereum - Smart contract platform'
        WHEN base_asset = 'SOL' THEN 'Solana - High-performance blockchain'
        WHEN base_asset = 'BONK' THEN 'Bonk - Solana meme coin'
        WHEN base_asset = 'WIF' THEN 'Dogwifhat - Popular Solana meme'
        WHEN base_asset = 'RAY' THEN 'Raydium - Solana DEX'
        WHEN base_asset = 'JUP' THEN 'Jupiter - Solana aggregator'
        ELSE base_asset || ' perpetual contract'
    END,
    'logo_url', 'https://cdn.quantdesk.com/logos/' || LOWER(base_asset) || '.png',
    'volume_24h', 0,
    'price_change_24h', 0,
    'market_cap', 0
) WHERE metadata IS NULL OR metadata = '{}';

-- Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_markets_category ON markets USING GIN ((metadata->>'category'));
CREATE INDEX IF NOT EXISTS idx_markets_base_asset ON markets(base_asset);
CREATE INDEX IF NOT EXISTS idx_markets_quote_asset ON markets(quote_asset);
CREATE INDEX IF NOT EXISTS idx_markets_active ON markets(is_active);

-- Create a view for frontend market data
CREATE OR REPLACE VIEW market_data AS
SELECT 
    m.*,
    m.metadata->>'category' as category,
    m.metadata->>'description' as description,
    m.metadata->>'logo_url' as logo_url,
    (m.metadata->>'volume_24h')::numeric as volume_24h,
    (m.metadata->>'price_change_24h')::numeric as price_change_24h,
    (m.metadata->>'market_cap')::numeric as market_cap
FROM markets m
WHERE m.is_active = true
ORDER BY 
    CASE m.metadata->>'category'
        WHEN 'major-crypto' THEN 1
        WHEN 'solana-ecosystem' THEN 2
        WHEN 'defi' THEN 3
        WHEN 'meme-coins' THEN 4
        WHEN 'gaming' THEN 5
        WHEN 'layer2' THEN 6
        WHEN 'ai' THEN 7
        WHEN 'stablecoins' THEN 8
        WHEN 'commodities' THEN 9
        ELSE 10
    END,
    m.base_asset;
