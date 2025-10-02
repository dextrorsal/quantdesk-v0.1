-- QuantDesk RLS Security Policies
-- Run these commands in your Supabase SQL Editor

-- 1. Enable RLS on all tables
ALTER TABLE markets ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE positions ENABLE ROW LEVEL SECURITY;
ALTER TABLE orders ENABLE ROW LEVEL SECURITY;
ALTER TABLE trades ENABLE ROW LEVEL SECURITY;
ALTER TABLE oracle_prices ENABLE ROW LEVEL SECURITY;
ALTER TABLE funding_rates ENABLE ROW LEVEL SECURITY;

-- 2. Create policies for markets (public read, admin write)
CREATE POLICY "Anyone can read markets" ON markets
FOR SELECT USING (true);

CREATE POLICY "Only authenticated users can insert markets" ON markets
FOR INSERT WITH CHECK (auth.role() = 'authenticated');

CREATE POLICY "Only authenticated users can update markets" ON markets
FOR UPDATE USING (auth.role() = 'authenticated');

-- 3. Create policies for users (own data only)
CREATE POLICY "Users can read own profile" ON users
FOR SELECT USING (auth.uid()::text = id::text);

CREATE POLICY "Users can update own profile" ON users
FOR UPDATE USING (auth.uid()::text = id::text);

CREATE POLICY "Users can insert own profile" ON users
FOR INSERT WITH CHECK (auth.uid()::text = id::text);

-- 4. Create policies for positions (own data only)
CREATE POLICY "Users can read own positions" ON positions
FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert own positions" ON positions
FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update own positions" ON positions
FOR UPDATE USING (auth.uid()::text = user_id::text);

-- 5. Create policies for orders (own data only)
CREATE POLICY "Users can read own orders" ON orders
FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert own orders" ON orders
FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

CREATE POLICY "Users can update own orders" ON orders
FOR UPDATE USING (auth.uid()::text = user_id::text);

-- 6. Create policies for trades (own data only)
CREATE POLICY "Users can read own trades" ON trades
FOR SELECT USING (auth.uid()::text = user_id::text);

CREATE POLICY "Users can insert own trades" ON trades
FOR INSERT WITH CHECK (auth.uid()::text = user_id::text);

-- 7. Create policies for oracle_prices (public read, system write)
CREATE POLICY "Anyone can read oracle prices" ON oracle_prices
FOR SELECT USING (true);

CREATE POLICY "Only service role can insert oracle prices" ON oracle_prices
FOR INSERT WITH CHECK (auth.role() = 'service_role');

-- 8. Create policies for funding_rates (public read, system write)
CREATE POLICY "Anyone can read funding rates" ON funding_rates
FOR SELECT USING (true);

CREATE POLICY "Only service role can insert funding rates" ON funding_rates
FOR INSERT WITH CHECK (auth.role() = 'service_role');

-- 9. Create a function to check if user is authenticated
CREATE OR REPLACE FUNCTION auth.user_id()
RETURNS text
LANGUAGE sql
SECURITY DEFINER
AS $$
  SELECT auth.uid()::text;
$$;

-- 10. Create indexes for better performance
CREATE INDEX IF NOT EXISTS idx_positions_user_id ON positions(user_id);
CREATE INDEX IF NOT EXISTS idx_orders_user_id ON orders(user_id);
CREATE INDEX IF NOT EXISTS idx_trades_user_id ON trades(user_id);
CREATE INDEX IF NOT EXISTS idx_oracle_prices_market_id ON oracle_prices(market_id);
CREATE INDEX IF NOT EXISTS idx_funding_rates_market_id ON funding_rates(market_id);

-- 11. Create a view for public market data (no sensitive info)
CREATE OR REPLACE VIEW public_markets AS
SELECT 
  id,
  symbol,
  base_asset,
  quote_asset,
  is_active,
  max_leverage,
  initial_margin_ratio,
  maintenance_margin_ratio,
  tick_size,
  step_size,
  min_order_size,
  max_order_size,
  funding_interval,
  current_funding_rate,
  created_at
FROM markets
WHERE is_active = true;

-- Grant access to the view
GRANT SELECT ON public_markets TO anon, authenticated;

-- 12. Create a function to get user's portfolio (secure)
CREATE OR REPLACE FUNCTION get_user_portfolio(user_wallet_address text)
RETURNS TABLE (
  position_id text,
  market_symbol text,
  side text,
  size numeric,
  entry_price numeric,
  current_price numeric,
  pnl numeric
)
LANGUAGE sql
SECURITY DEFINER
AS $$
  SELECT 
    p.id::text,
    m.symbol,
    p.side,
    p.size,
    p.entry_price,
    COALESCE(op.price, 0) as current_price,
    (p.size * (COALESCE(op.price, 0) - p.entry_price)) as pnl
  FROM positions p
  JOIN markets m ON p.market_id = m.id
  LEFT JOIN oracle_prices op ON m.id = op.market_id
  WHERE p.user_id = (
    SELECT id FROM users WHERE wallet_address = user_wallet_address
  )
  AND p.size > 0
  AND NOT p.is_liquidated;
$$;

-- Grant access to the function
GRANT EXECUTE ON FUNCTION get_user_portfolio(text) TO authenticated;
