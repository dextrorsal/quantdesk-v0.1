# CSV Storage System Documentation

## Overview

The CSV Storage System provides a robust, scalable solution for storing and loading candlestick (OHLCV) data in CSV format. **This is the CURRENT system** that replaces the previous database (Neon/Supabase) approach due to limitations and costs. It offers better performance, unlimited storage, and easier data management.

### Why CSV Storage?
- **No Database Limits**: Unlimited storage without Neon/Supabase constraints
- **Better Performance**: Direct file access vs. SQL queries
- **Cost Effective**: No database hosting costs
- **Simpler Management**: Easy backup, restore, and data portability
- **Offline Capable**: Works without internet connection

## Features

- **Per-day CSV files** organized by exchange/pair/interval/year/month
- **Automatic folder creation** and data deduplication
- **Flexible data loading** with date ranges and multiple pairs
- **Data validation** and integrity checks
- **Resampling capabilities** for different timeframes
- **Automated daily fetching** with cron jobs
- **Comprehensive logging** and error handling

## Directory Structure

```
data/ohlcv/
├── {exchange}/                    # Exchange name (e.g., binance, bitget)
│   ├── {pair}/                    # Trading pair (e.g., BTC/USDT, ETH/USDT)
│   │   ├── {interval}/            # Timeframe (e.g., 1m, 5m, 1h, 1d)
│   │   │   ├── {YYYY}/            # Year
│   │   │   │   ├── {MM}/          # Month
│   │   │   │   │   ├── {YYYY-MM-DD}.csv  # Daily CSV file
│   │   │   │   │   └── ...
│   │   │   │   └── ...
│   │   │   └── ...
│   │   └── ...
│   └── ...
└── ...
```

## Components

### 1. CSV Storage (`src/data/csv_storage.py`)

The core storage class that handles saving and loading candlestick data.

**Key Methods:**
- `store_candles()` - Store candlestick data with deduplication
- `load_candles()` - Load data for a specific date range
- `resample_candles()` - Resample data to different timeframes
- `verify_data_integrity()` - Validate data quality

**Example:**
```python
from src.data.csv_storage import CSVStorage, StorageConfig

# Initialize storage
config = StorageConfig(data_path="data/ohlcv")
storage = CSVStorage(config)

# Store data
candles = [
    {
        "timestamp": "2025-01-15T10:00:00",
        "open": 65000.0,
        "high": 65100.0,
        "low": 64900.0,
        "close": 65050.0,
        "volume": 12.345
    }
]
await storage.store_candles("binance", "BTC/USDT", "1h", candles)

# Load data
df = await storage.load_candles(
    "binance", "BTC/USDT", "1h",
    start_time=datetime(2025, 1, 15),
    end_time=datetime(2025, 1, 16)
)
```

### 2. Data Loader (`src/data/data_loader.py`)

High-level interface for loading data with various filtering options.

**Key Methods:**
- `load_data()` - Load data for specific exchange/pair/interval
- `load_multiple_pairs()` - Load data for multiple combinations
- `load_recent_data()` - Load recent data for the last N days
- `get_available_data()` - Get summary of available data

**Example:**
```python
from src.data.data_loader import DataLoader

# Initialize loader
loader = DataLoader("data/ohlcv")

# Load specific data
df = await loader.load_data(
    exchange="binance",
    pair="BTC/USDT",
    interval="1h",
    start_date="2025-01-01",
    end_date="2025-01-31"
)

# Load multiple pairs
data = await loader.load_multiple_pairs(
    exchanges=["binance", "bitget"],
    pairs=["BTC/USDT", "ETH/USDT"],
    interval="1h",
    start_date="2025-01-01",
    end_date="2025-01-31"
)
```

### 3. Data Fetcher (`scripts/fetch_to_csv.py`)

Script for fetching historical data from exchanges and storing in CSV format.

**Usage:**
```bash
python scripts/fetch_to_csv.py
```

**Configuration:**
- Edit the script to modify exchanges, pairs, timeframes, and date ranges
- Supports all major exchanges via CCXT library
- Automatic rate limiting and error handling

### 4. Daily Fetcher (`scripts/daily_fetch.py`)

Automated script for daily data fetching with your preferred exchanges and pairs.

**Features:**
- Decentralized Exchanges: Drift Protocol, Jupiter, Raydium, Orca (Solana DEXs)
- Multi-Chain DEXs: Uniswap, PancakeSwap, SushiSwap
- Centralized Exchanges: Bitget, Binance, Coinbase, MEXC, KuCoin, Kraken
- 44 trading pairs including major cryptocurrencies and meme coins
- 6 timeframes: 1m, 5m, 15m, 1h, 4h, 1d
- Automatic error handling and retry logic
- Progress tracking and comprehensive logging

**Usage:**
```bash
python scripts/daily_fetch.py
```

### 5. Cron Setup (`scripts/setup_daily_fetch.sh`)

Shell script to set up automated daily fetching using cron jobs.

**Usage:**
```bash
# Install daily fetch cron job
bash scripts/setup_daily_fetch.sh install

# Check status
bash scripts/setup_daily_fetch.sh status

# Remove cron job
bash scripts/setup_daily_fetch.sh remove
```

## Configuration

### Exchanges and Pairs

The daily fetcher is configured with your preferred exchanges and pairs:

**Exchanges:**
- Decentralized: Drift Protocol, Jupiter, Raydium, Orca (Solana DEXs)
- Multi-Chain DEXs: Uniswap, PancakeSwap, SushiSwap
- Centralized: Bitget, Binance, Coinbase, MEXC, KuCoin, Kraken

**Pairs (44 total):**
- Major: BTC, ETH, SOL, XRP, ADA, LINK, AVAX, BNB
- DeFi: UNI, AAVE, CAKE, ORCA, JUP, DRIFT
- Meme: PEPE, DOGE, SHIB, FLOKI, BRETT, WIF, BOME
- And many more...

**Timeframes:**
- 1m, 5m, 15m, 1h, 4h, 1d

### Storage Configuration

The `StorageConfig` class allows customization:

```python
@dataclass
class StorageConfig:
    data_path: Path = Path("data/ohlcv")      # Base storage path
    use_compression: bool = False              # Enable file compression
    backup_enabled: bool = False               # Enable backups
    backup_path: Optional[Path] = None         # Backup location
```

## Usage Examples

### 1. Basic Data Loading

```python
from src.data.data_loader import load_data
from datetime import datetime

# Load data for a specific range
df = await load_data(
    exchange="binance",
    pair="BTC/USDT",
    interval="1h",
    start_date="2025-01-01",
    end_date="2025-01-31"
)

print(f"Loaded {len(df)} candles")
print(f"Price range: ${df['low'].min():.2f} - ${df['high'].max():.2f}")
```

### 2. Multiple Pairs Loading

```python
from src.data.data_loader import DataLoader

loader = DataLoader()

# Load data for multiple exchanges and pairs
data = await loader.load_multiple_pairs(
    exchanges=["binance", "bitget"],
    pairs=["BTC/USDT", "ETH/USDT", "SOL/USDT"],
    interval="1h",
    start_date="2025-01-01",
    end_date="2025-01-31"
)

for key, df in data.items():
    print(f"{key}: {len(df)} candles")
```

### 3. Recent Data Loading

```python
from src.data.data_loader import DataLoader

loader = DataLoader()

# Load last 30 days of data
df = await loader.load_recent_data(
    exchange="binance",
    pair="BTC/USDT",
    interval="1h",
    days=30
)
```

### 4. Data Resampling

```python
from src.data.csv_storage import CSVStorage, StorageConfig

storage = CSVStorage(StorageConfig())

# Load 1-minute data
df_1m = await storage.load_candles(
    "binance", "BTC/USDT", "1m",
    start_time=datetime(2025, 1, 1),
    end_time=datetime(2025, 1, 2)
)

# Resample to 1-hour
df_1h = await storage.resample_candles(df_1m, "1h")
```

## Testing

### Test Scripts

1. **`scripts/test_csv_storage.py`** - Test CSV storage functionality
2. **`scripts/test_data_loader.py`** - Test data loader functionality
3. **`scripts/test_simple_loader.py`** - Simple data loading test
4. **`scripts/test_correct_pair.py`** - Test with correct pair formats

### Running Tests

```bash
# Test CSV storage
python scripts/test_csv_storage.py

# Test data loader
python scripts/test_data_loader.py

# Test simple loading
python scripts/test_simple_loader.py
```

## Performance

### Storage Efficiency

- **Per-day files**: Small, fast-to-read files
- **Compression**: Optional gzip compression for space savings
- **Deduplication**: Automatic removal of duplicate candles
- **Indexing**: Fast file-based access without database overhead

### Loading Performance

- **Parallel loading**: Support for loading multiple files simultaneously
- **Memory efficient**: Load only requested date ranges
- **Caching**: Optional caching for frequently accessed data

## Monitoring and Logging

### Log Files

- **Daily fetch logs**: `logs/daily_fetch_YYYYMMDD.log`
- **Cron logs**: `logs/cron.log`
- **Application logs**: Console and file output

### Data Integrity

The system includes comprehensive data validation:

- **OHLC relationship checks**: High ≥ max(Open, Close), Low ≤ min(Open, Close)
- **Negative value detection**: Warns about invalid price/volume data
- **Missing value detection**: Identifies gaps in data
- **Duplicate detection**: Removes duplicate timestamps

## Migration from Database

### Benefits of CSV Storage

1. **Unlimited storage**: No database size limits
2. **Better performance**: Direct file access vs. SQL queries
3. **Easier backup**: Simple file copying
4. **No network dependencies**: Works offline
5. **Cost effective**: No database hosting costs
6. **Data portability**: Easy to move between systems

### Migration Process

1. **Export existing data** from database
2. **Convert to CSV format** using the storage system
3. **Update scripts** to use CSV storage instead of database
4. **Test thoroughly** with new system
5. **Deploy and monitor**

## Troubleshooting

### Common Issues

1. **No data found**: Check file paths and pair formats
2. **Import errors**: Ensure `src/` is in Python path
3. **Permission errors**: Check file/directory permissions
4. **Memory issues**: Load smaller date ranges for large datasets

### Debug Mode

Enable debug logging for detailed troubleshooting:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Future Enhancements

### Planned Features

1. **Data compression**: Automatic gzip compression for old data
2. **Backup system**: Automated backups to cloud storage
3. **Data validation**: Enhanced integrity checks
4. **Performance optimization**: Parallel processing for large datasets
5. **Web interface**: Dashboard for data management
6. **Real-time updates**: WebSocket integration for live data

### Extensibility

The system is designed to be easily extensible:

- **New exchanges**: Add to CCXT configuration
- **New pairs**: Update pair lists
- **New timeframes**: Add to timeframe configuration
- **Custom storage**: Extend StorageConfig for custom backends

## Support

For issues and questions:

1. Check the logs in `logs/` directory
2. Run test scripts to verify functionality
3. Review this documentation
4. Check file permissions and paths
5. Verify Python dependencies are installed

## Conclusion

The CSV Storage System provides a robust, scalable solution for candlestick data management. It offers better performance than database storage while maintaining data integrity and providing comprehensive tools for data fetching, loading, and analysis.

The system is production-ready and includes automated daily fetching, comprehensive error handling, and extensive testing capabilities. 