# Solana Expert Recommendations for Database Architecture

## On-Chain vs Off-Chain Data Separation

### Recommended Architecture: Hybrid Approach

**Current State (Mirror in PostgreSQL):**
- **Positions**: Mirror essential current state for fast UI queries
- **Orders**: Mirror active orders for order book display
- **Market Data**: Mirror current market state for UI performance
- **User Balances**: Mirror for fast balance checks

**Historical Data (Store in PostgreSQL):**
- **Trades**: All trade history for analytics
- **Order History**: Complete order lifecycle tracking
- **Liquidations**: Historical liquidation data
- **Funding Rates**: Historical funding calculations
- **Oracle Prices**: Historical price data for charts/analytics

### Oracle Price Caching Strategy

**Recommendation**: Store Pyth oracle prices in PostgreSQL with public read access
- **Rationale**: Oracle prices are public on-chain data, caching improves performance
- **Best Practice**: Keep current prices synchronized with on-chain state
- **Security**: Public read access is appropriate for oracle data

### User Privacy vs Blockchain Transparency

**Key Insight**: User positions and liquidations are public on Solana blockchain
- **Database Approach**: Restrict individual user data with RLS policies
- **Public Data**: Make aggregated data (total open interest, funding rates) public
- **Rationale**: While on-chain data is public, database should protect user privacy

## Synchronization Strategy

### Event-Driven Architecture (Recommended)

**Anchor Events**: Emit events for all significant state changes
- OrderFilled, PositionUpdated, LiquidationExecuted, etc.
- Include all relevant data in events to avoid additional RPC calls

**Off-Chain Listener**: 
- Subscribe to program events using Anchor event listeners
- Parse events and update PostgreSQL with proper transactions
- Implement idempotency and error handling

### Example Event Structure
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
```

## RLS Policy Recommendations

### Public Data (No RLS needed)
- `markets` - Market specifications
- `oracle_prices` - Public price data
- `funding_rates` - Public funding calculations
- `market_stats` - Aggregated market statistics

### User-Specific Data (RLS Required)
- `positions` - Users can only see their own positions
- `orders` - Users can only see their own orders
- `trades` - Users can only see their own trades
- `liquidations` - Users can only see their own liquidations
- `user_balances` - Users can only see their own balances

### Admin-Only Data (Service Role Only)
- `admin_users` - Admin authentication data
- `admin_audit_logs` - Admin action logs
- `system_events` - System debugging information

## Implementation Priority

1. **Phase 1**: Implement comprehensive RLS policies
2. **Phase 2**: Set up event-driven synchronization
3. **Phase 3**: Optimize for performance and scalability

## Security Considerations

- **Principle of Least Privilege**: Grant minimal necessary permissions
- **Service Role**: Use service_role for backend operations only
- **Public Access**: Limit to truly public data (markets, prices)
- **User Privacy**: Protect individual user data even if on-chain data is public
