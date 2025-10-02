# QuantDesk Database Schema Documentation

## 🏗️ Architecture Overview

**Database**: Supabase PostgreSQL 17  
**Connection**: Direct Postgres via `DATABASE_URL` (fast path) + Supabase JS client (auth/realtime)  
**Extensions**: `uuid-ossp`, `pgcrypto`  
**RLS**: Enabled for user data isolation  

## 📊 Core Tables

### 1. **Users & Authentication**
```sql
users (wallet-based auth)
├── id (UUID, PK)
├── wallet_address (TEXT, UNIQUE) -- Primary identifier
├── username (TEXT)
├── email (TEXT)
├── kyc_status (TEXT) -- pending, verified, rejected
├── risk_level (TEXT) -- low, medium, high
├── total_volume (DECIMAL)
├── total_trades (INTEGER)
└── metadata (JSONB)
```

### 2. **Markets & Trading**
```sql
markets
├── id (UUID, PK)
├── symbol (TEXT, UNIQUE) -- BTC-PERP, ETH-PERP
├── base_asset, quote_asset (TEXT)
├── program_id (TEXT) -- Solana program
├── market_account, oracle_account (TEXT)
├── max_leverage (INTEGER)
├── initial_margin_ratio, maintenance_margin_ratio (INTEGER)
├── tick_size, step_size, min_order_size, max_order_size (DECIMAL)
├── funding_interval (INTEGER) -- seconds
├── current_funding_rate (DECIMAL)
└── metadata (JSONB)

user_balances (collateral)
├── id (UUID, PK)
├── user_id (UUID, FK → users)
├── asset (TEXT) -- USDC, SOL
├── balance (DECIMAL)
├── locked_balance (DECIMAL)
└── available_balance (DECIMAL) -- GENERATED COLUMN

positions
├── id (UUID, PK)
├── user_id (UUID, FK → users)
├── market_id (UUID, FK → markets)
├── position_account (TEXT) -- Solana account
├── side (ENUM) -- long, short
├── size, entry_price, current_price (DECIMAL)
├── margin, leverage (DECIMAL, INTEGER)
├── unrealized_pnl, realized_pnl, funding_fees (DECIMAL)
├── is_liquidated (BOOLEAN)
├── liquidation_price, health_factor (DECIMAL)
└── closed_at (TIMESTAMP)

orders
├── id (UUID, PK)
├── user_id (UUID, FK → users)
├── market_id (UUID, FK → markets)
├── order_account (TEXT) -- Solana account
├── order_type (ENUM) -- market, limit, stop_loss, etc.
├── side (ENUM) -- long, short
├── size, price, stop_price, trailing_distance (DECIMAL)
├── leverage (INTEGER)
├── status (ENUM) -- pending, filled, cancelled, etc.
├── filled_size, remaining_size, average_fill_price (DECIMAL)
└── expires_at, filled_at, cancelled_at (TIMESTAMP)

trades
├── id (UUID, PK)
├── user_id (UUID, FK → users)
├── market_id (UUID, FK → markets)
├── position_id (UUID, FK → positions)
├── order_id (UUID, FK → orders)
├── trade_account (TEXT) -- Solana account
├── side (ENUM) -- buy, sell
├── size, price, fees, pnl (DECIMAL)
├── value (DECIMAL) -- GENERATED COLUMN
└── metadata (JSONB)
```

### 3. **Time-Series Data**
```sql
oracle_prices (price feeds)
├── id (UUID, PK)
├── market_id (UUID, FK → markets)
├── price, confidence (DECIMAL)
├── exponent (INTEGER)
└── created_at (TIMESTAMP)

funding_rates (funding history)
├── id (UUID, PK)
├── market_id (UUID, FK → markets)
├── funding_rate, premium_index (DECIMAL)
├── oracle_price, mark_price (DECIMAL)
├── total_funding (DECIMAL)
└── created_at (TIMESTAMP)

liquidations
├── id (UUID, PK)
├── user_id (UUID, FK → users)
├── market_id (UUID, FK → markets)
├── position_id (UUID, FK → positions)
├── liquidator_address (TEXT)
├── liquidation_type (ENUM) -- market, backstop
├── liquidated_size, liquidation_price, liquidation_fee (DECIMAL)
├── remaining_margin (DECIMAL)
└── metadata (JSONB)
```

### 4. **Analytics & Statistics**
```sql
market_stats (daily rollups)
├── id (UUID, PK)
├── market_id (UUID, FK → markets)
├── date (DATE)
├── open_price, high_price, low_price, close_price (DECIMAL)
├── volume, volume_usd (DECIMAL)
├── trades_count (INTEGER)
├── open_interest, funding_rate (DECIMAL)
└── UNIQUE(market_id, date)

user_stats (daily rollups)
├── id (UUID, PK)
├── user_id (UUID, FK → users)
├── date (DATE)
├── total_volume, total_volume_usd (DECIMAL)
├── trades_count (INTEGER)
├── realized_pnl, unrealized_pnl, fees_paid (DECIMAL)
└── UNIQUE(user_id, date)

system_events (monitoring)
├── id (UUID, PK)
├── event_type (TEXT) -- liquidation, funding, oracle_update, error
├── event_data (JSONB)
├── severity (TEXT) -- info, warning, error, critical
└── created_at (TIMESTAMP)
```

### 5. **Admin & Audit**
```sql
admin_users (admin panel)
├── id (UUID, PK)
├── username (TEXT, UNIQUE)
├── password_hash (TEXT)
├── role (TEXT)
├── permissions (JSONB)
├── is_active (BOOLEAN)
└── created_by (UUID, FK → admin_users)

admin_audit_logs (admin actions)
├── id (UUID, PK)
├── admin_user_id (UUID, FK → admin_users)
├── action (TEXT)
├── resource (TEXT)
├── details (JSONB)
├── ip_address (INET)
└── user_agent (TEXT)
```

## 🔍 Views (Pre-computed Queries)

### 1. **Trading Views**
```sql
active_positions
-- JOIN positions + users + markets
-- WHERE size > 0 AND NOT is_liquidated

pending_orders  
-- JOIN orders + users + markets
-- WHERE status = 'pending' AND expires_at > NOW()

market_summary
-- JOIN markets + positions
-- GROUP BY market with open_interest, active_traders, avg_leverage

safe_markets
-- SELECT FROM markets WHERE is_active = true
```

## 🚀 Performance Indexes

### **Time-Series Indexes** (Critical for speed)
```sql
-- Oracle prices (most queried)
idx_oracle_prices_market_time: (market_id, created_at DESC)

-- Trades (user + market queries)
idx_trades_user_id: (user_id)
idx_trades_market_id: (market_id)
idx_trades_created_at: (created_at)

-- Orders (status + expiration)
idx_orders_status: (status)
idx_orders_expires_at: (expires_at)
idx_orders_market_id: (market_id)

-- Positions (health monitoring)
idx_positions_health_factor: (health_factor)
idx_positions_is_liquidated: (is_liquidated)
```

### **Lookup Indexes**
```sql
-- Users
idx_users_wallet_address: (wallet_address) -- UNIQUE

-- Markets  
idx_markets_symbol: (symbol) -- UNIQUE
idx_markets_is_active: (is_active)

-- Balances
idx_user_balances_user_id: (user_id)
idx_user_balances_asset: (asset)
```

## 🔒 Security (Row Level Security)

### **User Data Isolation**
```sql
-- Users can only see their own data
CREATE POLICY "Users can view own data" ON users
    FOR ALL USING (auth.jwt() ->> 'wallet_address' = wallet_address);

-- Same for balances, positions, orders, trades, user_stats
```

### **Admin Access**
- Admin tables have no RLS (admin-only access)
- Admin users managed separately from trading users

## 📈 Data Flow

### **Trading Flow**
```
1. User connects wallet → users table
2. Deposit collateral → user_balances table  
3. Place order → orders table
4. Order fills → trades table + positions table
5. Price updates → oracle_prices table
6. Funding events → funding_rates table
7. Liquidations → liquidations table
```

### **Analytics Flow**
```
1. Raw data → time-series tables
2. Daily rollups → market_stats, user_stats tables
3. Views → pre-computed aggregations
4. Admin monitoring → system_events table
```

## 🎯 Query Patterns

### **Hot Queries** (Optimized)
```sql
-- Latest price for market
SELECT price FROM oracle_prices 
WHERE market_id = ? ORDER BY created_at DESC LIMIT 1;

-- User's active positions  
SELECT * FROM active_positions WHERE user_id = ?;

-- Market summary
SELECT * FROM market_summary WHERE symbol = ?;

-- User's recent trades
SELECT * FROM trades 
WHERE user_id = ? ORDER BY created_at DESC LIMIT 50;
```

### **Analytics Queries**
```sql
-- Price history
SELECT price, created_at FROM oracle_prices 
WHERE market_id = ? AND created_at >= ? 
ORDER BY created_at DESC;

-- Volume by day
SELECT date, volume_usd FROM market_stats 
WHERE market_id = ? ORDER BY date DESC;
```

## 🔧 Maintenance

### **Cleanup Jobs** (pg_cron)
```sql
-- Archive old oracle prices (keep 30 days)
DELETE FROM oracle_prices WHERE created_at < NOW() - INTERVAL '30 days';

-- Archive old system events (keep 7 days)  
DELETE FROM system_events WHERE created_at < NOW() - INTERVAL '7 days';
```

### **Monitoring**
- Watch `idx_oracle_prices_market_time` usage
- Monitor `health_factor` indexes for liquidations
- Track `system_events` for errors

---

## 🚨 REDUNDANCY ISSUES TO FIX

1. **Remove duplicate `users` table** (keep QuantDesk version)
2. **Consolidate audit tables** (keep `admin_audit_logs`)
3. **Add missing JIT liquidity tables** (`auctions`, `auction_quotes`, `auction_settlements`)

This schema supports:
- ✅ High-frequency trading
- ✅ Real-time price feeds  
- ✅ Position management
- ✅ Risk monitoring
- ✅ Analytics & reporting
- ✅ Admin operations
