# RLS Policy Implementation

## Expert-Recommended RLS Policies

### 1. Public Data (No RLS Needed)
```sql
-- Market specifications
CREATE POLICY "Public market access" ON markets
    FOR SELECT USING (true);

-- Oracle prices
CREATE POLICY "Public oracle prices" ON oracle_prices
    FOR SELECT USING (true);

-- Funding rates
CREATE POLICY "Public funding rates" ON funding_rates
    FOR SELECT USING (true);

-- Market statistics
CREATE POLICY "Public market stats" ON market_stats
    FOR SELECT USING (true);
```

### 2. User-Specific Data (RLS Required)
```sql
-- User positions
CREATE POLICY "Users can view own positions" ON positions
    FOR SELECT USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

-- User orders
CREATE POLICY "Users can view own orders" ON orders
    FOR SELECT USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

-- User trades
CREATE POLICY "Users can view own trades" ON trades
    FOR SELECT USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

-- User liquidations
CREATE POLICY "Users can view own liquidations" ON liquidations
    FOR SELECT USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));
```

### 3. Admin-Only Data (Service Role Only)
```sql
-- Admin users
CREATE POLICY "Admin users service role only" ON admin_users
    FOR ALL USING (auth.role() = 'service_role');

-- Admin audit logs
CREATE POLICY "Admin audit logs service role only" ON admin_audit_logs
    FOR ALL USING (auth.role() = 'service_role');

-- System events
CREATE POLICY "System events service role only" ON system_events
    FOR ALL USING (auth.role() = 'service_role');
```
