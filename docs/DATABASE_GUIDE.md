# 🗄️ Database Guide - QuantDesk Trading System

## 🎯 Current Database Status ✅

### Data Successfully Stored
- **Total Candles**: 885,391 records
- **Exchanges**: Binance, Coinbase
- **Symbols**: BTC/USDT, ETH/USDT, SOL/USDT
- **Timeframes**: 5m, 15m, 1h, 4h, 1d
- **Date Range**: July 2024 - July 2025 (1 year of data)

### Database Schema Structure
```
market_data/
├── exchanges (2 records)
│   ├── id (Primary Key)
│   └── name (binance, coinbase)
├── markets (6 records)
│   ├── id (Primary Key)
│   ├── exchange_id (Foreign Key → exchanges.id)
│   ├── symbol (BTC/USDT, ETH/USDT, SOL/USDT)
│   ├── base_asset (BTC, ETH, SOL)
│   ├── quote_asset (USDT)
│   └── type (SPOT)
└── candles (885,391 records)
    ├── id (Primary Key)
    ├── market_id (Foreign Key → markets.id)
    ├── resolution (5m, 15m, 1h, 4h, 1d)
    ├── ts (Timestamp with timezone)
    ├── open, high, low, close (Price data)
    └── volume (Trading volume)
```

## 📊 Understanding Your Data (SQL vs CSV)

### CSV Approach (What you're used to)
```python
# Old way - CSV files
import pandas as pd
df = pd.read_csv('btc_data.csv')
# Data is flat, no relationships
```

### SQL Approach (Professional way)
```sql
-- New way - Relational database
SELECT 
    c.ts, c.open, c.high, c.low, c.close, c.volume,
    m.symbol, e.name as exchange, c.resolution
FROM market_data.candles c
JOIN market_data.markets m ON c.market_id = m.id
JOIN market_data.exchanges e ON m.exchange_id = e.id
WHERE m.symbol = 'BTC/USDT' 
    AND c.resolution = '15m'
    AND c.ts >= '2024-01-01'
ORDER BY c.ts DESC
LIMIT 100;
```

## 🔧 Database Optimization for Algo Trading

### 1. Performance Indexes (Already Created)
```sql
-- These indexes make queries fast
CREATE INDEX idx_candles_market_resolution ON market_data.candles(market_id, resolution);
CREATE INDEX idx_candles_ts ON market_data.candles(ts);
CREATE INDEX idx_markets_symbol ON market_data.markets(symbol);
```

### 2. Useful Views for Common Queries ✅
Views have been created to make data access easier:

```sql
-- Easy data access view
market_data.candle_data
-- Latest prices view
market_data.latest_prices
-- Data summary view
market_data.data_summary
```

### 3. Simple Data Loading (Like CSV)
Instead of CSV files, you can now load data directly from the database:

```python
# Old way (CSV)
import pandas as pd
df = pd.read_csv('btc_data.csv')

# New way (Database) - Just as simple!
from src.data.database_loader import load_trading_data
import asyncio

async def load_data():
    df = await load_trading_data("BTC/USDT", "15m", "binance", 30)
    return df

# Run it
df = asyncio.run(load_data())
```

## 🚀 Next Steps for Model Training

### 1. Train Models on Your New Data
You now have 885,391 candles from reliable sources (Binance/Coinbase). Let's retrain your models:

```bash
# Retrain Lorentzian Classifier on 1 year of 15m data
python scripts/train_model.py --symbol BTC/USDT --resolution 15m --days 365

# Compare all strategies on the new dataset
python scripts/run_comparison.py --use-database
```

### 2. Database vs CSV Benefits
- **CSV**: Static files, no relationships, manual updates
- **Database**: Live data, relationships, automated updates, fast queries

### 3. Professional Algo Trading Setup
Your database is now set up like professional trading systems:
- ✅ Relational schema with proper relationships
- ✅ Indexed for fast queries
- ✅ Views for common operations
- ✅ Data from multiple exchanges
- ✅ 1+ year of historical data
- ✅ Automated data fetching pipeline

## 📋 Quick Reference

### Check Your Data
```sql
-- See all available data
SELECT * FROM market_data.data_summary;

-- Get latest prices
SELECT * FROM market_data.latest_prices;

-- Load specific data
SELECT * FROM market_data.candle_data 
WHERE symbol = 'BTC/USDT' AND resolution = '15m'
ORDER BY ts DESC LIMIT 100;
```

### Add More Data
```bash
# Fetch more data
python scripts/fetch_reliable_data.py

# Add new symbols
# Edit SYMBOLS list in fetch_reliable_data.py
```

### Model Training
```bash
# Train on database data
python scripts/train_model.py --use-database

# Backtest strategies
python scripts/run_comparison.py --use-database
```

## 🎯 Success Metrics
- ✅ 885,391 candles stored
- ✅ 3 symbols (BTC, ETH, SOL)
- ✅ 2 exchanges (Binance, Coinbase)
- ✅ 5 timeframes (5m, 15m, 1h, 4h, 1d)
- ✅ 1 year of historical data
- ✅ Professional database schema
- ✅ Fast query performance
- ✅ Easy data loading interface

**Your database is now ready for professional algo trading! 🚀**
