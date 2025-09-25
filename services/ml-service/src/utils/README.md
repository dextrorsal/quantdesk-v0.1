# 🛠 Utility Functions

This directory contains utility functions and helpers used throughout the Quantify platform.

## 🔍 Overview

The utility components provide:
- Wallet management and encryption
- Technical indicators
- Logging setup
- Strategy utilities
- Price services
- Solana RPC utilities

## 📁 Structure

```
utils/
├── wallet/            # Wallet management
│   ├── wallet_manager.py    # Core wallet management
│   ├── encryption.py        # Wallet encryption
│   ├── phantom_utils.py     # Phantom wallet integration
│   ├── sol_rpc.py          # Solana RPC utilities
│   ├── sol_wallet.py       # Solana wallet implementation
│   ├── price_service.py    # Price feed services
│   └── watch_mode_gui.py   # Wallet monitoring GUI
├── indicators/        # Technical indicators
│   ├── base_indicator.py   # Base indicator class
│   ├── adx.py             # Average Directional Index
│   ├── bollinger_bands.py # Bollinger Bands
│   ├── cci.py            # Commodity Channel Index
│   ├── knn.py            # K-Nearest Neighbors
│   ├── lorentzian.py     # Lorentzian Classification
│   ├── macd.py           # Moving Average Convergence Divergence
│   ├── rsi.py            # Relative Strength Index
│   ├── stochastic.py     # Stochastic Oscillator
│   ├── supertrend.py     # SuperTrend
│   └── williams_r.py     # Williams %R
├── strategy/          # Strategy utilities
├── log_setup.py      # Logging configuration
├── __init__.py       # Package initialization
└── README.md         # This file
```

## 🔐 Wallet Management

### Features
- Secure wallet encryption
- Multiple wallet support
- Phantom wallet integration
- Transaction monitoring
- Price feed integration
- GUI monitoring tools

### Example
```python
from src.utils.wallet import WalletManager
from src.utils.wallet.encryption import encrypt_wallet

# Initialize wallet manager
wallet_manager = WalletManager()

# Create and encrypt wallet
wallet = wallet_manager.create_wallet()
encrypted = encrypt_wallet(wallet, password="your_password")

# Monitor transactions
await wallet_manager.monitor_transactions(wallet.public_key)
```

## 📊 Technical Indicators

### Available Indicators
- ADX (Average Directional Index)
- Bollinger Bands
- CCI (Commodity Channel Index)
- KNN (K-Nearest Neighbors)
- Lorentzian Classification
- MACD (Moving Average Convergence Divergence)
- RSI (Relative Strength Index)
- Stochastic Oscillator
- SuperTrend
- Williams %R

### Example
```python
from src.utils.indicators import (
    RSI,
    BollingerBands,
    MACD,
    SuperTrend
)

# Calculate RSI
rsi = RSI(period=14)
rsi_values = rsi.calculate(prices)

# Calculate Bollinger Bands
bb = BollingerBands(period=20, std_dev=2)
upper, middle, lower = bb.calculate(prices)

# Calculate MACD
macd = MACD(fast=12, slow=26, signal=9)
macd_line, signal_line, histogram = macd.calculate(prices)

# Calculate SuperTrend
supertrend = SuperTrend(period=10, multiplier=3)
trend, direction = supertrend.calculate(high, low, close)
```

## 📝 Logging

### Setting up Logging
```python
from src.utils.log_setup import setup_logger

# Setup logger
logger = setup_logger(
    name="trading",
    log_level="INFO",
    log_file="trading.log"
)

# Use logger
logger.info("Starting trading session")
logger.error("Error executing trade", extra={"market": "SOL"})
```

## 💰 Price Services

### Features
- Real-time price feeds
- Multiple data sources
- Price aggregation
- Historical prices
- Price alerts

### Example
```python
from src.utils.wallet.price_service import PriceService

# Initialize price service
price_service = PriceService()

# Get real-time price
sol_price = await price_service.get_price("SOL")

# Set price alert
await price_service.set_alert(
    symbol="SOL",
    price=100.0,
    condition="above"
)
```

## 🌐 Solana RPC

### Features
- RPC connection management
- Transaction building
- Account monitoring
- Program interaction
- Error handling

### Example
```python
from src.utils.wallet.sol_rpc import SolanaRPC

# Initialize RPC client
rpc = SolanaRPC(url="https://api.mainnet-beta.solana.com")

# Get account info
account_info = await rpc.get_account_info(pubkey)

# Send transaction
signature = await rpc.send_transaction(transaction)
```

## 🎯 Strategy Utilities

### Available Strategies
- Base Strategy (base.py)
- Solana Spot Trading (sol_spot.py)
- Machine Learning Strategies (ml_stratagies.py)
- Segmented Trading (segmented.py)
- Multi-Indicator Strategy (multi_indicator_strategy.py)

### Example
```python
from src.utils.strategy import (
    BaseStrategy,
    SolSpotStrategy,
    MLStrategy,
    SegmentedStrategy,
    MultiIndicatorStrategy
)

# Initialize Solana spot strategy
spot_strategy = SolSpotStrategy(
    market="SOL",
    indicators=[RSI(), MACD()],
    risk_params={"max_position": 10}
)

# Initialize ML strategy
ml_strategy = MLStrategy(
    model_path="models/sol_classifier.pkl",
    features=["rsi", "macd", "bb"]
)

# Initialize segmented strategy
segmented = SegmentedStrategy(
    timeframes=["1m", "5m", "15m"],
    sub_strategies=[spot_strategy, ml_strategy]
)

# Run strategy
async def run_trading():
    signals = await segmented.generate_signals(market_data)
    for signal in signals:
        await execute_trade(signal)
```

## 🧪 Testing

```bash
# Run all utility tests
python -m pytest tests/utils/

# Run specific component tests
python -m pytest tests/utils/wallet/
python -m pytest tests/utils/indicators/
```

## 📚 Resources

- [Wallet Management Guide](docs/utils/wallet.md)
- [Technical Indicators Guide](docs/utils/indicators.md)
- [Logging Guide](docs/utils/logging.md)
- [Price Services Guide](docs/utils/price_service.md) 