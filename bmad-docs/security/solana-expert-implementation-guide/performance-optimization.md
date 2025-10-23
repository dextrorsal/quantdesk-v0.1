# Performance Optimization

## Expert Recommendations

### 1. Database Indexing
```sql
-- Critical indexes for performance
CREATE INDEX idx_positions_user_market ON positions(user_id, market_id);
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
CREATE INDEX idx_trades_user_timestamp ON trades(user_id, created_at);
CREATE INDEX idx_oracle_prices_symbol_timestamp ON oracle_prices(symbol, timestamp);
CREATE INDEX idx_markets_active ON markets(is_active) WHERE is_active = true;
```

### 2. Query Optimization
```sql
-- Optimized user portfolio query
CREATE OR REPLACE FUNCTION get_user_portfolio(user_wallet_address TEXT)
RETURNS TABLE (
    position_id UUID,
    market_symbol TEXT,
    side position_side,
    size DECIMAL(20,8),
    entry_price DECIMAL(20,8),
    current_price DECIMAL(20,8),
    unrealized_pnl DECIMAL(20,8)
)
SECURITY DEFINER
SET search_path = public
LANGUAGE plpgsql
AS $$
BEGIN
    -- Only allow users to query their own portfolio
    IF auth.jwt() ->> 'wallet_address' != user_wallet_address THEN
        RAISE EXCEPTION 'Unauthorized: Cannot access other users portfolio';
    END IF;
    
    RETURN QUERY
    SELECT 
        p.id,
        m.symbol,
        p.side,
        p.size,
        p.entry_price,
        p.current_price,
        p.unrealized_pnl
    FROM positions p
    JOIN markets m ON p.market_id = m.id
    JOIN users u ON p.user_id = u.id
    WHERE u.wallet_address = user_wallet_address
    AND p.size > 0
    AND NOT p.is_liquidated;
END;
$$;
```

### 3. Caching Strategy
```typescript
// Redis caching for frequently accessed data
export class CacheService {
  private redis: Redis;

  async cacheMarketData(symbol: string, data: any) {
    await this.redis.setex(`market:${symbol}`, 300, JSON.stringify(data)); // 5 min cache
  }

  async getCachedMarketData(symbol: string) {
    const cached = await this.redis.get(`market:${symbol}`);
    return cached ? JSON.parse(cached) : null;
  }

  async cacheUserPortfolio(walletAddress: string, portfolio: any) {
    await this.redis.setex(`portfolio:${walletAddress}`, 60, JSON.stringify(portfolio)); // 1 min cache
  }
}
```
