# 📈 Trading Components

This directory contains the trading logic, strategies, and execution components of the Quantify platform.

## 🔍 Overview

The trading components provide:
- Strategy implementation
- Order execution
- Position management
- Risk management
- Performance tracking

## 📁 Structure

```
trading/
├── strategies/         # Trading strategy implementations
├── execution/         # Order execution and management
├── risk/             # Risk management components
├── __init__.py       # Package initialization
└── README.md         # This file
```

## 🚀 Quick Start

```python
from src.trading.strategies import SimpleStrategy
from src.trading.execution import OrderExecutor
from src.core.models import TimeRange

# Initialize strategy
strategy = SimpleStrategy(
    market="SOL",
    timeframe="1h",
    risk_percentage=1.0
)

# Initialize executor
executor = OrderExecutor(client=None)  # Drift logic removed

# Run strategy
signals = await strategy.generate_signals(time_range)
for signal in signals:
    await executor.execute_order(signal)
```

## 📊 Trading Strategies

### Available Strategies
- Simple Moving Average Crossover
- RSI Mean Reversion
- Custom strategy templates

### Creating New Strategies
```python
from src.trading.strategies.base import BaseStrategy

class MyStrategy(BaseStrategy):
    def __init__(self, params):
        super().__init__()
        self.params = params
        
    async def generate_signals(self, time_range: TimeRange):
        # Your strategy logic here
        pass
```

## 🎯 Order Execution

### Features
- Smart order routing
- Position sizing
- Order types (market, limit)
- Slippage protection
- Error handling

### Example
```python
from src.trading.execution import OrderExecutor

executor = OrderExecutor(client=None)  # Drift logic removed
await executor.place_order(
    market="SOL",
    side="long",
    size=1.0,
    price=100.0,
    order_type="limit"
)
```

## ⚠️ Risk Management

### Features
- Position size limits
- Loss limits
- Exposure tracking
- Risk metrics calculation

### Example
```python
from src.trading.risk import RiskManager

risk_manager = RiskManager(
    max_position_size=5.0,
    max_loss_percentage=2.0
)

# Check if order is within risk limits
is_safe = risk_manager.check_order(order)
```

## 📊 Performance Tracking

### Features
- Trade history
- Performance metrics
- Risk metrics
- Equity curves

### Example
```python
from src.trading.performance import PerformanceTracker

tracker = PerformanceTracker()
metrics = tracker.calculate_metrics(trades)
print(f"Sharpe Ratio: {metrics.sharpe_ratio}")
```

## 🧪 Testing

```bash
# Run all trading tests
python -m pytest tests/trading/

# Run specific strategy tests
python -m pytest tests/trading/strategies/
```

## 📚 Resources

- [Trading Strategy Guide](docs/trading/strategies.md)
- [Risk Management Guide](docs/trading/risk.md)
- [Performance Metrics Guide](docs/trading/performance.md)