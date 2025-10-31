# 🚀 QuantDesk Production Database Setup Guide

## Overview

This guide will help you set up a production-ready database for your Solana perpetual trading DEX using PostgreSQL and Supabase. The database schema includes all features needed for a professional trading platform.

## 🎯 What You'll Get

✅ **Complete Perpetual DEX Schema** - All tables, indexes, and functions needed  
✅ **High-Performance Design** - Optimized for high-frequency trading  
✅ **Security Features** - Row Level Security (RLS) and proper access controls  
✅ **Risk Management** - Liquidation, funding, and insurance fund systems  
✅ **Real-time Capabilities** - TimescaleDB hypertables for time-series data  
✅ **Monitoring & Analytics** - Built-in system events and statistics  
✅ **JIT Liquidity** - Auction system for large trades  
✅ **Admin Panel Support** - Complete admin and audit functionality  

## 📋 Prerequisites

- PostgreSQL 15+ or Supabase account
- TimescaleDB extension (for time-series optimization)
- Basic knowledge of SQL and database administration

## 🚀 Quick Start

### Option 1: Supabase (Recommended)

1. **Create Supabase Project**
   ```bash
   # Go to https://supabase.com
   # Create new project
   # Note your project URL and API keys
   ```

2. **Run Production Schema**
   ```bash
   # Copy your Supabase connection details
   cp env.example .env
   # Edit .env with your Supabase credentials
   
   # Run the production schema
   psql "your-supabase-connection-string" -f database/production-schema.sql
   ```

3. **Run Migration (if upgrading existing database)**
   ```bash
   # If you have existing data, run migration instead
   psql "your-supabase-connection-string" -f database/migration-to-production.sql
   ```

4. **Test the Setup**
   ```bash
   # Run tests to verify everything works
   psql "your-supabase-connection-string" -f database/test-schema.sql
   ```

### Option 2: Local PostgreSQL

1. **Install PostgreSQL with TimescaleDB**
   ```bash
   # Ubuntu/Debian
   sudo apt install postgresql postgresql-contrib
   sudo apt install timescaledb-postgresql-15
   
   # macOS
   brew install postgresql timescaledb
   
   # Start PostgreSQL
   sudo systemctl start postgresql
   ```

2. **Create Database**
   ```bash
   sudo -u postgres psql
   CREATE DATABASE quantdesk;
   CREATE USER quantdesk_user WITH PASSWORD 'secure_password';
   GRANT ALL PRIVILEGES ON DATABASE quantdesk TO quantdesk_user;
   \q
   ```

3. **Run Schema**
   ```bash
   psql -h localhost -U quantdesk_user -d quantdesk -f database/production-schema.sql
   ```

## 🔧 Configuration

### Environment Variables

Copy `env.example` to `.env` and configure:

```bash
# Database
DATABASE_URL=postgresql://user:password@host:port/database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_supabase_anon_key
SUPABASE_SERVICE_ROLE_KEY=your_supabase_service_role_key

# Solana
SOLANA_NETWORK=devnet  # or mainnet-beta for production
RPC_URL=https://api.devnet.solana.com
PROGRAM_ID=your_program_id

# Oracle
BTC_ORACLE_ACCOUNT=HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J
ETH_ORACLE_ACCOUNT=JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB
SOL_ORACLE_ACCOUNT=H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG
```

### Supabase Configuration

1. **Enable Extensions**
   - Go to Supabase Dashboard → Database → Extensions
   - Enable: `uuid-ossp`, `pgcrypto`, `timescaledb`

2. **Configure RLS**
   - The migration script automatically enables RLS
   - Users can only see their own data
   - Markets and prices are publicly readable

3. **Set up Realtime**
   - Supabase automatically provides real-time subscriptions
   - Subscribe to table changes for live updates

## 📊 Database Schema Overview

### Core Trading Tables

| Table | Purpose | Key Features |
|-------|---------|--------------|
| `users` | User accounts | Wallet-based auth, KYC, risk levels |
| `markets` | Trading markets | Leverage limits, margin requirements |
| `user_balances` | Collateral | Available/locked balance tracking |
| `positions` | Active positions | P&L, health factor, liquidation price |
| `orders` | Order management | Multiple order types, expiration |
| `trades` | Trade history | Execution details, fees, P&L |

### Risk Management

| Table | Purpose | Key Features |
|-------|---------|--------------|
| `funding_rates` | Funding history | Automatic/manual funding |
| `liquidations` | Liquidation events | Market/backstop liquidations |
| `insurance_fund` | Risk buffer | Per-market insurance |
| `risk_alerts` | Risk monitoring | Real-time risk notifications |

### Oracle & Pricing

| Table | Purpose | Key Features |
|-------|---------|--------------|
| `oracle_prices` | Price feeds | Pyth oracle integration |
| `mark_prices` | Calculated prices | Oracle + funding adjustments |
| `market_stats` | Daily statistics | Volume, OI, funding rates |

### Advanced Features

| Table | Purpose | Key Features |
|-------|---------|--------------|
| `auctions` | JIT liquidity | Large trade auctions |
| `auction_quotes` | Market maker quotes | Competitive pricing |
| `system_events` | Monitoring | Error tracking, alerts |
| `admin_users` | Admin panel | Role-based access |

## 🔍 Key Features

### 1. High-Performance Design

- **TimescaleDB Hypertables** for time-series data
- **Optimized Indexes** for high-frequency queries
- **Generated Columns** for computed values
- **Partitioning** for large datasets

### 2. Risk Management

- **Health Factor Monitoring** - Real-time position health
- **Liquidation System** - Automatic risk management
- **Insurance Fund** - Risk buffer for liquidations
- **Funding Rate System** - Perpetual contract funding

### 3. Security

- **Row Level Security (RLS)** - User data isolation
- **Wallet-based Authentication** - No passwords needed
- **Audit Logging** - Complete action tracking
- **Role-based Access** - Admin vs user permissions

### 4. Real-time Capabilities

- **Live Price Feeds** - Oracle price updates
- **Position Monitoring** - Real-time P&L
- **Order Book** - Live order management
- **Risk Alerts** - Instant notifications

## 🧪 Testing

### Run Test Suite

```bash
# Test the complete schema
psql "your-connection-string" -f database/test-schema.sql
```

### Test Results

The test suite will verify:
- ✅ All tables created correctly
- ✅ Indexes working properly
- ✅ Functions executing correctly
- ✅ Views returning data
- ✅ RLS policies enforced
- ✅ Performance benchmarks met

### Sample Queries

```sql
-- Get latest prices
SELECT * FROM oracle_prices 
WHERE market_id = (SELECT id FROM markets WHERE symbol = 'BTC-PERP') 
ORDER BY created_at DESC LIMIT 1;

-- Get user's active positions
SELECT * FROM active_positions 
WHERE user_id = 'user-uuid';

-- Get market summary
SELECT * FROM market_summary 
WHERE symbol = 'BTC-PERP';

-- Check risk dashboard
SELECT * FROM risk_dashboard 
WHERE risk_level IN ('high', 'critical');
```

## 📈 Performance Optimization

### Indexes

The schema includes optimized indexes for:
- **Time-series queries** - Oracle prices, trades
- **User lookups** - Positions, orders, balances
- **Market queries** - Statistics, summaries
- **Risk monitoring** - Health factors, alerts

### TimescaleDB Benefits

- **Automatic partitioning** by time
- **Compression** for historical data
- **Continuous aggregates** for statistics
- **Retention policies** for data cleanup

### Query Optimization

```sql
-- Use indexes for time-based queries
SELECT * FROM oracle_prices 
WHERE market_id = ? AND created_at >= NOW() - INTERVAL '1 hour'
ORDER BY created_at DESC;

-- Use views for complex aggregations
SELECT * FROM market_summary WHERE symbol = ?;

-- Use functions for calculations
SELECT calculate_position_health(position_id) FROM positions;
```

## 🔒 Security Best Practices

### 1. Environment Security

```bash
# Never commit .env files
echo ".env" >> .gitignore
echo "*.env" >> .gitignore

# Use strong secrets
JWT_SECRET=$(openssl rand -hex 32)
```

### 2. Database Security

- **Enable SSL** for all connections
- **Use connection pooling** (Supabase handles this)
- **Regular backups** (Supabase automatic)
- **Monitor access logs**

### 3. Application Security

- **Validate all inputs** before database queries
- **Use parameterized queries** to prevent SQL injection
- **Implement rate limiting** on API endpoints
- **Monitor for suspicious activity**

## 🚨 Monitoring & Maintenance

### System Events

Monitor the `system_events` table for:
- **Liquidation events** - Risk management alerts
- **Oracle updates** - Price feed monitoring
- **Error tracking** - Application issues
- **Performance metrics** - Query performance

### Regular Maintenance

```sql
-- Clean up old data (run weekly)
DELETE FROM system_events 
WHERE created_at < NOW() - INTERVAL '30 days';

-- Update statistics (run daily)
ANALYZE;

-- Check for slow queries
SELECT * FROM pg_stat_statements 
ORDER BY total_time DESC LIMIT 10;
```

### Backup Strategy

- **Supabase**: Automatic daily backups
- **Local**: Regular pg_dump backups
- **Test restores** monthly

## 🆘 Troubleshooting

### Common Issues

1. **TimescaleDB Extension Missing**
   ```sql
   CREATE EXTENSION IF NOT EXISTS timescaledb;
   ```

2. **RLS Policies Not Working**
   ```sql
   -- Check if RLS is enabled
   SELECT schemaname, tablename, rowsecurity 
   FROM pg_tables WHERE tablename = 'users';
   ```

3. **Performance Issues**
   ```sql
   -- Check index usage
   SELECT * FROM pg_stat_user_indexes;
   
   -- Analyze slow queries
   SELECT * FROM pg_stat_statements;
   ```

4. **Connection Issues**
   ```bash
   # Test connection
   psql "your-connection-string" -c "SELECT version();"
   ```

### Getting Help

- Check the `system_events` table for errors
- Review Supabase logs in dashboard
- Use `EXPLAIN ANALYZE` for slow queries
- Monitor TimescaleDB metrics

## 🎉 Next Steps

After setting up the database:

1. **Update Application Code** - Use new schema structure
2. **Deploy to Production** - Use production environment variables
3. **Set up Monitoring** - Configure alerts and dashboards
4. **Test Trading Functions** - Verify all features work correctly
5. **Go Live** - Start accepting real users

## 📚 Additional Resources

- [PostgreSQL Documentation](https://www.postgresql.org/docs/)
- [TimescaleDB Documentation](https://docs.timescale.com/)
- [Supabase Documentation](https://supabase.com/docs)
- [Solana Program Library](https://spl.solana.com/)

---

**Your database is now ready for production perpetual trading! 🚀**
