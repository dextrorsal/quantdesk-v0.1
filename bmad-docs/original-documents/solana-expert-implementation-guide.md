# Solana Expert Implementation Guide

## Overview
This document captures the expert recommendations from Solana/Anchor specialists for implementing a secure, scalable perpetual DEX architecture. These guidelines ensure proper separation of on-chain and off-chain data, optimal performance, and security best practices.

## Expert Consultation Summary

### Consultation Date: January 2025
### Experts Consulted:
- **Solana Expert via MCP**: On-chain vs off-chain data architecture
- **Anchor Framework Expert via MCP**: Event-driven synchronization and RLS policies

## Key Architectural Decisions

### 1. Hybrid Data Architecture (Recommended)

**Decision**: Use a hybrid approach combining on-chain state mirroring with off-chain historical data storage.

**Rationale**: 
- Fast UI queries require cached current state
- Historical data enables analytics and reporting
- On-chain data provides single source of truth
- Off-chain data enables complex queries and filtering

**Implementation**:

#### Current State (Mirror in PostgreSQL)
```sql
-- Essential current state for fast UI queries
- positions: Mirror active positions for order book display
- orders: Mirror active orders for trading interface
- market_data: Mirror current market state for UI performance
- user_balances: Mirror for fast balance checks
```

#### Historical Data (Store in PostgreSQL)
```sql
-- Complete historical data for analytics
- trades: All trade history for analytics
- order_history: Complete order lifecycle tracking
- liquidations: Historical liquidation data
- funding_rates: Historical funding calculations
- oracle_prices: Historical price data for charts/analytics
```

### 2. Oracle Price Caching Strategy

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

### 3. User Privacy vs Blockchain Transparency

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

## Synchronization Strategy

### Event-Driven Architecture (Recommended)

**Expert Recommendation**: Use Anchor events for all significant state changes.

**Benefits**:
- Most reliable and scalable synchronization method
- Real-time updates without polling
- Efficient data transfer
- Built-in retry mechanisms

**Implementation**:

#### 1. Anchor Events Definition
```rust
#[event]
pub struct OrderFilled {
    pub market: Pubkey,
    pub position_id: u64,
    pub order_id: u64,
    pub price: u64,
    pub quantity: u64,
    pub taker_fee: u64,
    pub maker_fee: u64,
}

#[event]
pub struct PositionUpdated {
    pub position_id: u64,
    pub user: Pubkey,
    pub market: Pubkey,
    pub size: u64,
    pub entry_price: u64,
    pub unrealized_pnl: i64,
}

#[event]
pub struct LiquidationExecuted {
    pub position_id: u64,
    pub user: Pubkey,
    pub market: Pubkey,
    pub liquidated_size: u64,
    pub liquidation_price: u64,
    pub penalty_fee: u64,
}
```

#### 2. Off-Chain Event Listener
```typescript
// Event listener service
import { Program } from "@coral-xyz/anchor";
import { Connection } from "@solana/web3.js";

export class EventListener {
  private program: Program;
  private connection: Connection;

  async subscribeToEvents() {
    // Order filled events
    this.program.addEventListener("OrderFilled", async (event, slot, signature) => {
      await this.handleOrderFilled(event, signature);
    });

    // Position updated events
    this.program.addEventListener("PositionUpdated", async (event, slot, signature) => {
      await this.handlePositionUpdated(event, signature);
    });

    // Liquidation events
    this.program.addEventListener("LiquidationExecuted", async (event, slot, signature) => {
      await this.handleLiquidation(event, signature);
    });
  }

  private async handleOrderFilled(event: any, signature: string) {
    // Insert into trades table
    await this.database.query(`
      INSERT INTO trades (signature, market_id, user_id, side, size, price, fees, timestamp)
      VALUES ($1, $2, $3, $4, $5, $6, $7, NOW())
    `, [signature, event.market, event.user, event.side, event.quantity, event.price, event.taker_fee]);
  }
}
```

#### 3. Database Event Storage
```sql
-- Event storage tables
CREATE TABLE program_events (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    signature VARCHAR(64) UNIQUE NOT NULL,
    slot BIGINT NOT NULL,
    event_type TEXT NOT NULL,
    event_data JSONB NOT NULL,
    processed BOOLEAN DEFAULT false,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Index for efficient querying
CREATE INDEX idx_program_events_signature ON program_events(signature);
CREATE INDEX idx_program_events_slot ON program_events(slot);
CREATE INDEX idx_program_events_type ON program_events(event_type);
```

## RLS Policy Implementation

### Expert-Recommended RLS Policies

#### 1. Public Data (No RLS Needed)
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

#### 2. User-Specific Data (RLS Required)
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

#### 3. Admin-Only Data (Service Role Only)
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

## Performance Optimization

### Expert Recommendations

#### 1. Database Indexing
```sql
-- Critical indexes for performance
CREATE INDEX idx_positions_user_market ON positions(user_id, market_id);
CREATE INDEX idx_orders_user_status ON orders(user_id, status);
CREATE INDEX idx_trades_user_timestamp ON trades(user_id, created_at);
CREATE INDEX idx_oracle_prices_symbol_timestamp ON oracle_prices(symbol, timestamp);
CREATE INDEX idx_markets_active ON markets(is_active) WHERE is_active = true;
```

#### 2. Query Optimization
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

#### 3. Caching Strategy
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

## Security Best Practices

### Expert-Recommended Security Measures

#### 1. Principle of Least Privilege
```sql
-- Revoke overly permissive grants
REVOKE ALL ON ALL TABLES IN SCHEMA public FROM anon;
REVOKE ALL ON ALL FUNCTIONS IN SCHEMA public FROM anon;
REVOKE ALL ON ALL SEQUENCES IN SCHEMA public FROM anon;

-- Grant selective permissions
GRANT USAGE ON SCHEMA public TO anon, authenticated;

-- Public tables (read-only for anon)
GRANT SELECT ON markets TO anon, authenticated;
GRANT SELECT ON oracle_prices TO anon, authenticated;

-- Authenticated user tables (RLS enforced)
GRANT SELECT, INSERT, UPDATE ON users TO authenticated;
GRANT SELECT, INSERT, UPDATE ON positions TO authenticated;
```

#### 2. Service Role Isolation
```sql
-- Service role has full access for backend operations
GRANT ALL ON ALL TABLES IN SCHEMA public TO service_role;
GRANT ALL ON ALL SEQUENCES IN SCHEMA public TO service_role;
GRANT ALL ON ALL FUNCTIONS IN SCHEMA public TO service_role;
```

#### 3. Input Validation
```typescript
// Joi validation schemas
const positionSchema = Joi.object({
  market_id: Joi.string().uuid().required(),
  side: Joi.string().valid('long', 'short').required(),
  size: Joi.number().positive().required(),
  leverage: Joi.number().min(1).max(100).required()
});

const orderSchema = Joi.object({
  market_id: Joi.string().uuid().required(),
  side: Joi.string().valid('buy', 'sell').required(),
  type: Joi.string().valid('market', 'limit', 'stop_loss').required(),
  size: Joi.number().positive().required(),
  price: Joi.number().positive().optional()
});
```

## Error Handling and Monitoring

### Expert Recommendations

#### 1. Comprehensive Error Handling
```typescript
// Error handling middleware
export const errorHandler = (error: Error, req: Request, res: Response, next: NextFunction) => {
  logger.error('Database error:', {
    error: error.message,
    stack: error.stack,
    query: req.body,
    user: req.user?.id
  });

  // Don't expose internal errors to clients
  if (error instanceof DatabaseError) {
    return res.status(500).json({
      error: 'Database operation failed',
      code: 'DATABASE_ERROR'
    });
  }

  // Handle specific error types
  if (error.message.includes('RLS')) {
    return res.status(403).json({
      error: 'Access denied',
      code: 'ACCESS_DENIED'
    });
  }

  res.status(500).json({
    error: 'Internal server error',
    code: 'INTERNAL_ERROR'
  });
};
```

#### 2. Event Processing Monitoring
```typescript
// Event processing health check
export class EventProcessor {
  private processedEvents = new Map<string, number>();
  private failedEvents = new Map<string, number>();

  async processEvent(event: any, signature: string) {
    try {
      await this.handleEvent(event);
      this.processedEvents.set(signature, Date.now());
    } catch (error) {
      this.failedEvents.set(signature, Date.now());
      logger.error('Event processing failed:', { signature, error });
      throw error;
    }
  }

  getHealthStatus() {
    return {
      processed: this.processedEvents.size,
      failed: this.failedEvents.size,
      success_rate: this.processedEvents.size / (this.processedEvents.size + this.failedEvents.size)
    };
  }
}
```

## Implementation Checklist

### Phase 1: Database Security
- [ ] Apply RLS policies to all sensitive tables
- [ ] Create secure views for public data
- [ ] Implement service role isolation
- [ ] Run security verification tests

### Phase 2: Event-Driven Synchronization
- [ ] Define Anchor events for all state changes
- [ ] Implement off-chain event listener
- [ ] Create event storage tables
- [ ] Set up event processing pipeline

### Phase 3: Performance Optimization
- [ ] Add critical database indexes
- [ ] Implement Redis caching
- [ ] Optimize frequently used queries
- [ ] Set up performance monitoring

### Phase 4: Security Hardening
- [ ] Implement input validation
- [ ] Set up comprehensive error handling
- [ ] Configure monitoring and alerting
- [ ] Regular security audits

## Future Considerations

### Scalability
- Consider database partitioning for large tables
- Implement read replicas for analytics queries
- Use connection pooling for high concurrency

### Advanced Features
- Implement real-time notifications via WebSockets
- Add advanced analytics and reporting
- Consider implementing a GraphQL API for complex queries

### Monitoring
- Set up comprehensive logging
- Implement health checks for all services
- Monitor database performance and query times
- Track event processing latency

## Conclusion

This implementation guide provides a comprehensive roadmap for building a secure, scalable perpetual DEX following Solana expert recommendations. The hybrid architecture ensures optimal performance while maintaining security and data integrity.

Key takeaways:
1. **Hybrid Architecture**: Mirror current state, store historical data
2. **Event-Driven Sync**: Use Anchor events for reliable synchronization
3. **RLS Security**: Implement comprehensive row-level security
4. **Performance First**: Optimize queries and implement caching
5. **Security Hardened**: Follow principle of least privilege

Regular reviews and updates of this guide ensure continued alignment with best practices and evolving requirements.
