# Key Architectural Decisions

## 1. Hybrid Data Architecture (Recommended)

**Decision**: Use a hybrid approach combining on-chain state mirroring with off-chain historical data storage.

**Rationale**: 
- Fast UI queries require cached current state
- Historical data enables analytics and reporting
- On-chain data provides single source of truth
- Off-chain data enables complex queries and filtering

**Implementation**:

### Current State (Mirror in PostgreSQL)
```sql
-- Essential current state for fast UI queries
- positions: Mirror active positions for order book display
- orders: Mirror active orders for trading interface
- market_data: Mirror current market state for UI performance
- user_balances: Mirror for fast balance checks
```

### Historical Data (Store in PostgreSQL)
```sql
-- Complete historical data for analytics
- trades: All trade history for analytics
- order_history: Complete order lifecycle tracking
- liquidations: Historical liquidation data
- funding_rates: Historical funding calculations
- oracle_prices: Historical price data for charts/analytics
```

## 2. Oracle Price Caching Strategy

**Expert Recommendation**: Store Pyth oracle prices in PostgreSQL with public read access.

**Rationale**:
- Oracle prices are public on-chain data
- Caching improves performance and reduces RPC calls
- Public read access is appropriate for oracle data
- Keep current prices synchronized with on-chain state

**Implementation**:
```sql
-- Oracle prices table with public read access
CREATE TABLE oracle_prices (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    symbol TEXT NOT NULL,
    price DECIMAL(20,8) NOT NULL,
    confidence DECIMAL(20,8),
    timestamp TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    slot BIGINT,
    source TEXT DEFAULT 'pyth'
);

-- Public read policy for oracle data
CREATE POLICY "Public can read oracle prices" ON oracle_prices
    FOR SELECT USING (true);
```

## 3. User Privacy vs Blockchain Transparency

**Key Insight**: User positions and liquidations are public on Solana blockchain, but database should protect user privacy.

**Database Approach**:
- Restrict individual user data with RLS policies
- Make aggregated data (total open interest, funding rates) public
- Protect user privacy even though on-chain data is public

**Implementation**:
```sql
-- User-specific data with RLS
CREATE POLICY "Users can view own positions" ON positions
    FOR SELECT USING (auth.jwt() ->> 'wallet_address' = (
        SELECT wallet_address FROM users WHERE id = user_id
    ));

-- Public aggregated data
CREATE VIEW public_market_summary AS
SELECT 
    m.symbol,
    COUNT(DISTINCT p.user_id) as active_traders,
    SUM(p.size * p.entry_price) as total_open_interest
FROM markets m
LEFT JOIN positions p ON m.id = p.market_id
GROUP BY m.symbol;
```
