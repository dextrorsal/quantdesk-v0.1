# 📈 Exchange Integrations

This directory contains integrations with various cryptocurrency exchanges and protocols.

## 🔍 Overview

The exchange integrations are organized into two main categories:
1. Trading & Data Exchanges - Full-featured integrations supporting both trading and data fetching
2. Data Providers - Exchanges used primarily for market data and price feeds

## 📁 Structure

```
exchanges/
├── trading/           # Trading-enabled exchanges
│   └── jupiter/      # Jupiter DEX integration
│       ├── client.py     # Core trading client
│       ├── data.py       # Market data handling
│       └── auth.py       # Authentication
├── data/             # Data-only exchange integrations
│   ├── binance.py    # Binance market data
│   ├── coinbase.py   # Coinbase market data
│   └── bitget.py     # BitGet market data
├── base.py           # Base exchange interfaces
├── auth/             # Shared authentication utilities
├── __init__.py       # Package initialization
└── README.md         # This file
```

## 🚀 Trading & Data Exchanges

### Jupiter DEX
Decentralized exchange integration for token swaps:
```python
from src.exchanges.trading.jupiter.client import JupiterClient
from src.exchanges.trading.jupiter.data import JupiterDataProvider

# Initialize clients
client = JupiterClient()
data_provider = JupiterDataProvider()

# Get price data
price = await data_provider.get_token_price("SOL")

# Execute swap
quote = await client.get_quote(
    input_mint="SOL",
    output_mint="USDC",
    amount=1.0
)
await client.execute_swap(quote)
```

## 📊 Data Providers

### Binance
Market data integration:
```python
from src.exchanges.data.binance import BinanceDataProvider

provider = BinanceDataProvider()

# Fetch historical data
candles = await provider.fetch_historical_candles(
    symbol="SOLUSDT",
    interval="1h",
    limit=1000
)

# Stream live data
async for candle in provider.stream_candles("SOLUSDT", "1m"):
    print(f"New candle: {candle}")
```

### Coinbase
Market data integration:
```python
from src.exchanges.data.coinbase import CoinbaseDataProvider

provider = CoinbaseDataProvider()

# Fetch historical data
candles = await provider.fetch_historical_candles(
    product_id="SOL-USD",
    granularity=3600
)

# Get current price
price = await provider.get_current_price("SOL-USD")
```

## 🔧 Configuration

The exchange integrations use configuration from your `.env` file:
```env
# Solana Configuration
SOLANA_RPC_URL=https://api.mainnet-beta.solana.com

# Exchange API Keys
BINANCE_API_KEY=your_binance_key
BINANCE_API_SECRET=your_binance_secret
COINBASE_API_KEY=your_coinbase_key
COINBASE_API_SECRET=your_coinbase_secret
BITGET_API_KEY=your_bitget_key
BITGET_API_SECRET=your_bitget_secret

# Optional: Custom API endpoints
JUPITER_API_URL=https://...
```

## 🧪 Testing

Each exchange integration has its own test suite:
```bash
# Run all exchange tests
python -m pytest tests/exchanges/

# Run trading exchange tests
python -m pytest tests/exchanges/trading/test_jupiter.py

# Run data provider tests
python -m pytest tests/exchanges/data/test_binance.py
python -m pytest tests/exchanges/data/test_coinbase.py
```

## 📚 Resources

- [Jupiter API Documentation](https://docs.jup.ag/)
- [Binance API Documentation](https://binance-docs.github.io/apidocs/)
- [Coinbase API Documentation](https://docs.cloud.coinbase.com/)
- [BitGet API Documentation](https://bitgetlimited.github.io/apidoc/) 