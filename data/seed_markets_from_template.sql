-- Seed markets from template feed list (PERP + USD)
-- Excludes MNGO and MATIC by request
-- Idempotent: upserts on unique(symbol)

-- Ensure unique index on symbol
DO $$
BEGIN
  IF NOT EXISTS (
    SELECT 1 FROM pg_indexes WHERE schemaname = 'public' AND indexname = 'idx_markets_symbol_unique'
  ) THEN
    BEGIN
      CREATE UNIQUE INDEX idx_markets_symbol_unique ON public.markets(symbol);
    EXCEPTION WHEN others THEN
      -- ignore if another migration created it concurrently
      NULL;
    END;
  END IF;
END $$;

WITH base(symbol, base_asset, quote_asset, category, max_leverage) AS (
  VALUES
    -- core
    ('BTC-PERP','BTC','USDT','perp',20),
    ('ETH-PERP','ETH','USDT','perp',20),
    ('SOL-PERP','SOL','USDT','perp',20),
    ('BTC/USD','BTC','USD','spot',1),
    ('ETH/USD','ETH','USD','spot',1),
    ('SOL/USD','SOL','USD','spot',1),

    -- template symbols (no MNGO, no MATIC)
    ('AAVE-PERP','AAVE','USDT','perp',20), ('AAVE/USD','AAVE','USD','spot',1),
    ('ASTER-PERP','ASTER','USDT','perp',20), ('ASTER/USD','ASTER','USD','spot',1),
    ('AI16Z-PERP','AI16Z','USDT','perp',20), ('AI16Z/USD','AI16Z','USD','spot',1),
    ('AVAX-PERP','AVAX','USDT','perp',20), ('AVAX/USD','AVAX','USD','spot',1),
    ('BNB-PERP','BNB','USDT','perp',20),   ('BNB/USD','BNB','USD','spot',1),
    ('BONK-PERP','BONK','USDT','perp',10), ('BONK/USD','BONK','USD','spot',1),
    ('DOGE-PERP','DOGE','USDT','perp',10), ('DOGE/USD','DOGE','USD','spot',1),
    ('FARTCOIN-PERP','FARTCOIN','USDT','perp',5), ('FARTCOIN/USD','FARTCOIN','USD','spot',1),
    ('GOAT-PERP','GOAT','USDT','perp',5),       ('GOAT/USD','GOAT','USD','spot',1),
    ('JUP-PERP','JUP','USDT','perp',10),        ('JUP/USD','JUP','USD','spot',1),
    ('HYPE-PERP','HYPE','USDT','perp',10),      ('HYPE/USD','HYPE','USD','spot',1),
    ('LINK-PERP','LINK','USDT','perp',20),      ('LINK/USD','LINK','USD','spot',1),
    ('POL-PERP','POL','USDT','perp',10),        ('POL/USD','POL','USD','spot',1),
    ('MYRO-PERP','MYRO','USDT','perp',5),       ('MYRO/USD','MYRO','USD','spot',1),
    ('ORCA-PERP','ORCA','USDT','perp',10),      ('ORCA/USD','ORCA','USD','spot',1),
    ('PENGU-PERP','PENGU','USDT','perp',5),     ('PENGU/USD','PENGU','USD','spot',1),
    ('PEPE-PERP','PEPE','USDT','perp',5),       ('PEPE/USD','PEPE','USD','spot',1),
    ('POPCAT-PERP','POPCAT','USDT','perp',5),   ('POPCAT/USD','POPCAT','USD','spot',1),
    ('RAY-PERP','RAY','USDT','perp',10),        ('RAY/USD','RAY','USD','spot',1),
    ('SPX-PERP','SPX','USDT','perp',5),         ('SPX/USD','SPX','USD','spot',1),
    ('SUI-PERP','SUI','USDT','perp',10),        ('SUI/USD','SUI','USD','spot',1),
    ('TRUMP-PERP','TRUMP','USDT','perp',5),     ('TRUMP/USD','TRUMP','USD','spot',1),
    ('UNI-PERP','UNI','USDT','perp',20),        ('UNI/USD','UNI','USD','spot',1),
    ('WIF-PERP','WIF','USDT','perp',5),         ('WIF/USD','WIF','USD','spot',1),
    ('XRP-PERP','XRP','USDT','perp',10),        ('XRP/USD','XRP','USD','spot',1),
    ('ZEC-PERP','ZEC','USDT','perp',10),        ('ZEC/USD','ZEC','USD','spot',1),
    -- stables as spot only
    ('USDC/USD','USDC','USD','spot',1),
    ('USDT/USD','USDT','USD','spot',1),
    ('USDE/USD','USDE','USD','spot',1),
    ('USD1/USD','USD1','USD','spot',1),
    ('PYUSD/USD','PYUSD','USD','spot',1)
)
INSERT INTO public.markets (
  symbol, base_asset, quote_asset, is_active, max_leverage, initial_margin_ratio,
  maintenance_margin_ratio, tick_size, step_size, min_order_size, max_order_size,
  funding_interval, current_funding_rate, category, created_at
)
SELECT
  b.symbol,
  b.base_asset,
  b.quote_asset,
  TRUE,
  b.max_leverage,
  COALESCE(NULLIF(b.category,'perp'), NULL)::text IS DISTINCT FROM 'perp'::text ? 0.0 : 0.05,
  COALESCE(NULLIF(b.category,'perp'), NULL)::text IS DISTINCT FROM 'perp'::text ? 0.0 : 0.03,
  0.01, 0.001,
  0.001, 1000000,
  CASE WHEN b.category='perp' THEN 3600 ELSE NULL END,
  0,
  b.category,
  NOW()
FROM base b
ON CONFLICT (symbol) DO UPDATE SET
  base_asset = EXCLUDED.base_asset,
  quote_asset = EXCLUDED.quote_asset,
  is_active = TRUE,
  max_leverage = EXCLUDED.max_leverage,
  category = EXCLUDED.category,
  updated_at = NOW();


