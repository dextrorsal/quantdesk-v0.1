# ⚙️ Configuration Guide

This guide covers configuring QuantDesk for your trading setup.

## 🔑 Exchange API Setup

### Bitget (Recommended for Live Trading)

1. **Create Account**: Sign up at [bitget.com](https://bitget.com)
2. **Enable API**: Go to Account → API Management
3. **Create API Key**: 
   - Name: "QuantDesk Trading"
   - Permissions: Enable "Spot Trading" and "Read"
   - IP Restrictions: Add your IP address
4. **Get Credentials**:
   - API Key
   - Secret Key  
   - Passphrase

### Binance (For Market Data)

1. **Create Account**: Sign up at [binance.com](https://binance.com)
2. **Enable API**: Go to Account → API Management
3. **Create API Key**:
   - Name: "QuantDesk Data"
   - Permissions: Enable "Read Info" only
4. **Get Credentials**:
   - API Key
   - Secret Key

### Coinbase (Alternative Data Source)

1. **Create Account**: Sign up at [coinbase.com](https://coinbase.com)
2. **Enable API**: Go to Settings → API
3. **Create API Key**:
   - Name: "QuantDesk Data"
   - Permissions: Read-only
4. **Get Credentials**:
   - API Key
   - Secret Key

## 📝 Environment Configuration

### Create .env File
```bash
# Copy example configuration
cp .env.example .env

# Edit configuration
nano .env  # or use your preferred editor
```

### Complete .env Template
```env
# ===========================================
# QuantDesk TRADING SYSTEM CONFIGURATION
# ===========================================

# Bitget API (for live trading)
BITGET_API_KEY=your_bitget_api_key_here
BITGET_SECRET_KEY=your_bitget_secret_key_here
BITGET_PASSPHRASE=your_bitget_passphrase_here

# Binance API (for market data)
BINANCE_API_KEY=your_binance_api_key_here
BINANCE_SECRET_KEY=your_binance_secret_key_here

# Coinbase API (alternative data source)
COINBASE_API_KEY=your_coinbase_api_key_here
COINBASE_SECRET_KEY=your_coinbase_secret_key_here

# Trading Configuration
DEFAULT_LEVERAGE=10
MAX_POSITION_SIZE=0.1
RISK_PER_TRADE=0.02
STOP_LOSS_PCT=0.05
TAKE_PROFIT_PCT=0.15

# Data Configuration
DATA_PATH=data/ohlcv
BACKUP_ENABLED=true
COMPRESSION_ENABLED=false

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=logs/quantdesk.log
MAX_LOG_SIZE=10MB
LOG_RETENTION_DAYS=30

# Web UI Configuration
WEB_UI_PORT=3000
API_PORT=8000
WEBSOCKET_PORT=8001

# Performance Configuration
GPU_ACCELERATION=true
BATCH_SIZE=32
MAX_WORKERS=4
CACHE_SIZE=1000
```

## 🎯 Trading Configuration

### Risk Management Settings
```env
# Position Sizing
DEFAULT_LEVERAGE=10          # Leverage multiplier (1-100)
MAX_POSITION_SIZE=0.1        # Max position as % of account
RISK_PER_TRADE=0.02          # Risk per trade (2% of account)

# Stop Loss & Take Profit
STOP_LOSS_PCT=0.05           # Stop loss percentage (5%)
TAKE_PROFIT_PCT=0.15         # Take profit percentage (15%)

# Portfolio Limits
MAX_POSITIONS=5              # Maximum concurrent positions
PORTFOLIO_HEAT_LIMIT=0.1     # Max portfolio risk (10%)
```

### Strategy Configuration
```env
# ML Strategy Settings
LORENTZIAN_LOOKBACK=50       # Lorentzian classifier lookback
LAG_THRESHOLD=1.5            # Lag-based strategy threshold
LOGISTIC_LEARNING_RATE=0.0009 # Logistic regression learning rate

# Backtesting Settings
BACKTEST_INITIAL_CAPITAL=10000 # Starting capital for backtests
BACKTEST_COMMISSION=0.001     # Trading commission (0.1%)
BACKTEST_SLIPPAGE=0.0005      # Market slippage (0.05%)
```

## 📊 Data Configuration

### Storage Settings
```env
# Data Storage
DATA_PATH=data/ohlcv         # Where to store market data
BACKUP_ENABLED=true          # Enable automatic backups
COMPRESSION_ENABLED=false    # Compress data files
CLEANUP_DAYS=365             # Keep data for 1 year

# Data Sources
PREFERRED_EXCHANGE=bitget     # Primary data source
FALLBACK_EXCHANGES=binance,coinbase # Backup data sources
UPDATE_FREQUENCY=1m           # Data update frequency
```

### Symbol Configuration
```env
# Trading Pairs
MAJOR_PAIRS=BTCUSDT,ETHUSDT,SOLUSDT,ADAUSDT
MEME_COINS=DOGEUSDT,SHIBUSDT,PEPEUSDT
DEFI_TOKENS=UNIUSDT,AAVEUSDT,COMPUSDT
EXOTIC_PAIRS=NEARUSDT,AVAXUSDT,DOTUSDT
```

## 🖥️ Web UI Configuration

### Server Settings
```env
# Web Interface
WEB_UI_PORT=3000             # Frontend port
API_PORT=8000                # Backend API port
WEBSOCKET_PORT=8001          # WebSocket port
HOST=0.0.0.0                 # Bind to all interfaces

# Security
CORS_ORIGINS=http://localhost:3000,http://localhost:5173
API_RATE_LIMIT=100           # Requests per minute
SESSION_TIMEOUT=3600          # Session timeout (1 hour)
```

### Dashboard Settings
```env
# Dashboard Configuration
DEFAULT_TIMEFRAME=1h         # Default chart timeframe
CHART_CANDLES=1000           # Number of candles to display
UPDATE_INTERVAL=5            # Data update interval (seconds)
THEME=dark                   # UI theme (dark/light)
```

## 🔧 Advanced Configuration

### Performance Tuning
```env
# GPU Configuration
GPU_ACCELERATION=true        # Enable GPU acceleration
CUDA_DEVICE=0                # GPU device number
BATCH_SIZE=32                # ML model batch size
MAX_WORKERS=4                # Number of worker threads

# Memory Management
CACHE_SIZE=1000              # Cache size (MB)
MEMORY_LIMIT=8192            # Memory limit (MB)
GARBAGE_COLLECTION=true      # Enable garbage collection
```

### Logging Configuration
```env
# Logging Settings
LOG_LEVEL=INFO               # Log level (DEBUG/INFO/WARNING/ERROR)
LOG_FILE=logs/quantdesk.log   # Log file location
MAX_LOG_SIZE=10MB            # Maximum log file size
LOG_RETENTION_DAYS=30        # Keep logs for 30 days
CONSOLE_LOGGING=true         # Show logs in console
```

## ✅ Verify Configuration

### Test API Connections
```bash
# Test Bitget connection
python -c "
import os
from src.exchanges.bitget.bitget_trader import BitgetTrader
try:
    trader = BitgetTrader()
    print('✅ Bitget API connected')
except Exception as e:
    print(f'❌ Bitget API error: {e}')
"

# Test Binance connection
python -c "
import os
from src.exchanges.binance.binance import BinanceDataProvider
try:
    provider = BinanceDataProvider()
    print('✅ Binance API connected')
except Exception as e:
    print(f'❌ Binance API error: {e}')
"
```

### Test Data Access
```bash
# Test data loading
python -c "
from src.data.data_loader import DataLoader
loader = DataLoader('data/ohlcv')
data = loader.load_data('bitget', 'BTCUSDT', '1h', '2024-01-01', '2024-01-02')
print(f'✅ Data loaded: {len(data)} records')
"
```

## 🚨 Common Configuration Issues

### "API Key Invalid"
- Check API key is correct
- Verify API permissions
- Ensure IP restrictions allow your IP

### "Permission Denied"
- Check API key has trading permissions
- Verify account is verified
- Check exchange account status

### "Rate Limit Exceeded"
- Reduce request frequency
- Use multiple API keys
- Implement request queuing

### "Data Not Found"
- Check data path exists
- Verify exchange symbols
- Run data fetching script

## 🎯 Next Steps

After configuration:

1. **Test Setup**: [Quick Start Guide](quick-start.md)
2. **Web Interface**: [Web UI Guide](../user-guide/web-ui.md)
3. **Trading Strategies**: [Strategy Guide](../user-guide/trading-strategies.md)

---

*Previous: [Installation Guide](installation.md) | Next: [Quick Start Guide](quick-start.md)*
