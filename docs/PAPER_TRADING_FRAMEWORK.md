# Paper Trading Framework Documentation

## Overview

The Paper Trading Framework is a comprehensive backtesting and paper trading system that integrates all your existing strategies, models, features, and indicators. It provides a unified interface for testing and optimizing trading strategies with realistic market simulation.

## Features

- **Multi-Strategy Support**: Works with all existing strategies (Lag-based, Lorentzian, Logistic Regression, Chandelier Exit)
- **GPU-Accelerated**: Uses your custom GPU-accelerated features and indicators
- **Realistic Simulation**: Includes fees, slippage, and order execution simulation
- **Comprehensive Metrics**: Calculates win rate, Sharpe ratio, drawdown, and more
- **Parameter Optimization**: Grid search optimization for strategy parameters
- **Strategy Comparison**: Compare multiple strategies side-by-side
- **Data Integration**: Works with your existing CSV storage system
- **Visualization**: Automatic plotting of results and performance metrics

## Quick Start

### 1. Basic Backtest

```python
from src.ml.paper_trading_framework import PaperTradingFramework, BacktestConfig

# Initialize framework
config = BacktestConfig(
    initial_capital=10000,
    commission=0.001,
    slippage=0.0005
)
framework = PaperTradingFramework(config)

# Run backtest
results = await framework.backtest_strategy(
    strategy_name='lorentzian',
    data=your_data
)

# Print results
print(f"Total Return: {results['metrics'].total_return:.2f}%")
print(f"Win Rate: {results['metrics'].win_rate:.1%}")
```

### 2. Using the CLI

```bash
# Run backtest for Lorentzian strategy
python scripts/paper_trading_cli.py backtest --strategy lorentzian --symbols BTC --days 30

# Optimize parameters
python scripts/paper_trading_cli.py optimize --strategy lorentzian --symbols BTC --days 60 --param-ranges "lookback_bars=30:50:70,k_neighbors=10:20:30"

# Compare strategies
python scripts/paper_trading_cli.py compare --strategies lorentzian logistic_regression chandelier_exit --symbols BTC --days 30
```

### 3. Run Examples

```bash
# Run comprehensive examples
python scripts/example_paper_trading.py
```

## Supported Strategies

### 1. Lag-Based Strategy (`lag_based`)

**Description**: Trades follower assets based on lagging price movements after significant moves in leader assets.

**Parameters**:
- `threshold`: Minimum leader move percentage (default: 1.5)
- `max_lag_minutes`: Maximum lag time to consider (default: 60)
- `min_correlation`: Minimum correlation coefficient (default: 0.3)
- `risk_per_trade`: Risk per trade percentage (default: 0.02)

**Usage**:
```python
# Multi-asset data required
data = {
    'BTC': btc_data,  # Leader
    'ETH': eth_data,  # Leader
    'SOL': sol_data,  # Leader
    'WIF': wif_data,  # Follower
    'PEPE': pepe_data  # Follower
}

results = await framework.backtest_strategy(
    strategy_name='lag_based',
    data=data,
    params={'threshold': 2.0, 'max_lag_minutes': 45}
)
```

### 2. Lorentzian Classifier (`lorentzian`)

**Description**: Uses Lorentzian distance metric for pattern recognition and signal generation.

**Parameters**:
- `lookback_bars`: Number of historical bars (default: 50)
- `prediction_bars`: Bars into future to predict (default: 4)
- `k_neighbors`: Number of nearest neighbors (default: 20)

**Usage**:
```python
results = await framework.backtest_strategy(
    strategy_name='lorentzian',
    data=single_asset_data,
    params={'lookback_bars': 40, 'k_neighbors': 15}
)
```

### 3. Logistic Regression (`logistic_regression`)

**Description**: TradingView-style logistic regression with GPU acceleration.

**Parameters**:
- `lookback`: Lookback window size (default: 3)
- `learning_rate`: Learning rate for gradient descent (default: 0.0009)
- `iterations`: Training iterations (default: 1000)
- `threshold`: Signal threshold (default: 0.5)

**Usage**:
```python
results = await framework.backtest_strategy(
    strategy_name='logistic_regression',
    data=single_asset_data,
    params={'lookback': 5, 'learning_rate': 0.001}
)
```

### 4. Chandelier Exit (`chandelier_exit`)

**Description**: Trailing stop strategy based on ATR with ML enhancements.

**Parameters**:
- `atr_period`: ATR calculation period (default: 22)
- `atr_multiplier`: ATR multiplier for stops (default: 3.0)
- `use_close`: Use close prices for calculations (default: True)

**Usage**:
```python
results = await framework.backtest_strategy(
    strategy_name='chandelier_exit',
    data=single_asset_data,
    params={'atr_period': 14, 'atr_multiplier': 2.5}
)
```

## Configuration Options

### BacktestConfig

```python
@dataclass
class BacktestConfig:
    initial_capital: float = 10000.0      # Starting capital
    commission: float = 0.001             # Commission rate (0.1%)
    slippage: float = 0.0005              # Slippage rate (0.05%)
    max_positions: int = 3                # Max concurrent positions
    risk_per_trade: float = 0.02          # Risk per trade (2%)
    stop_loss_pct: float = 0.05           # Stop loss percentage (5%)
    take_profit_pct: float = 0.10         # Take profit percentage (10%)
    min_confidence: float = 0.3           # Minimum confidence for entry
    use_gpu: bool = True                  # Use GPU acceleration
    save_results: bool = True             # Save results to files
    results_dir: str = "results/paper_trading"  # Results directory
```

## Performance Metrics

The framework calculates comprehensive performance metrics:

- **Total Return**: Overall percentage return
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Ratio of gross profit to gross loss
- **Sharpe Ratio**: Risk-adjusted return measure
- **Max Drawdown**: Maximum peak-to-trough decline
- **Average Win/Loss**: Average profit/loss per trade
- **Best/Worst Trade**: Largest profit and loss
- **Trade Duration**: Average time in trades

## Parameter Optimization

### Grid Search Optimization

```python
# Define parameter ranges
param_ranges = {
    'lookback_bars': [30, 50, 70],
    'k_neighbors': [10, 20, 30],
    'prediction_bars': [2, 4, 6]
}

# Run optimization
results = await framework.optimize_strategy_parameters(
    strategy_name='lorentzian',
    data=data,
    param_ranges=param_ranges,
    metric='total_return'  # or 'sharpe_ratio', 'win_rate', etc.
)

print(f"Best Parameters: {results['best_params']}")
print(f"Best Return: {results['best_metric']:.2f}%")
```

### CLI Optimization

```bash
python scripts/paper_trading_cli.py optimize \
    --strategy lorentzian \
    --symbols BTC \
    --days 60 \
    --param-ranges "lookback_bars=30:50:70,k_neighbors=10:20:30" \
    --metric total_return
```

## Strategy Comparison

### Compare Multiple Strategies

```python
strategies = ['lorentzian', 'logistic_regression', 'chandelier_exit']
comparison_results = {}

for strategy in strategies:
    results = await framework.backtest_strategy(
        strategy_name=strategy,
        data=data
    )
    comparison_results[strategy] = results

# Print comparison table
framework._print_comparison_results(comparison_results)
```

### CLI Comparison

```bash
python scripts/paper_trading_cli.py compare \
    --strategies lorentzian logistic_regression chandelier_exit \
    --symbols BTC \
    --days 30 \
    --save
```

## Data Loading

The framework integrates with your existing CSV storage system:

```python
from src.data.csv_storage import CSVStorage, StorageConfig

storage = CSVStorage(StorageConfig(data_path="data/historical/processed"))

# Load data for backtesting
data = await storage.load_candles(
    exchange='bitget',
    symbol='BTCUSDT',
    timeframe='1h',
    start_date=datetime.now() - timedelta(days=30),
    end_date=datetime.now()
)
```

## Results and Output

### Saved Files

The framework automatically saves results to `results/paper_trading/`:

- **Metrics**: JSON files with performance metrics
- **Trades**: CSV files with detailed trade history
- **Equity Curve**: CSV files with portfolio value over time
- **Plots**: PNG files with performance visualizations

### Example Output Structure

```
results/paper_trading/
├── lorentzian_metrics_20241201_143022.json
├── lorentzian_trades_20241201_143022.csv
├── lorentzian_equity_20241201_143022.csv
├── lorentzian_backtest_20241201_143022.png
├── comparison_20241201_143022.json
└── optimization_results_20241201_143022.csv
```

## Advanced Usage

### Custom Risk Management

```python
config = BacktestConfig(
    initial_capital=50000,
    commission=0.0005,      # Lower commission
    slippage=0.0002,        # Lower slippage
    max_positions=5,        # More positions
    risk_per_trade=0.01,    # Lower risk per trade
    stop_loss_pct=0.03,     # Tighter stop loss
    take_profit_pct=0.06    # Lower take profit
)
```

### Multi-Timeframe Analysis

```python
# Test different timeframes
timeframes = ['1h', '4h', '1d']
results_by_timeframe = {}

for tf in timeframes:
    data = await storage.load_candles(
        exchange='bitget',
        symbol='BTCUSDT',
        timeframe=tf,
        start_date=start_date,
        end_date=end_date
    )
    
    results = await framework.backtest_strategy(
        strategy_name='lorentzian',
        data=data
    )
    results_by_timeframe[tf] = results
```

### Walk-Forward Analysis

```python
# Split data into training and testing periods
train_data = data[:len(data)//2]
test_data = data[len(data)//2:]

# Optimize on training data
opt_results = await framework.optimize_strategy_parameters(
    strategy_name='lorentzian',
    data=train_data,
    param_ranges=param_ranges
)

# Test on out-of-sample data
test_results = await framework.backtest_strategy(
    strategy_name='lorentzian',
    data=test_data,
    params=opt_results['best_params']
)
```

## Troubleshooting

### Common Issues

1. **No Data Loaded**: Check symbol names and data availability
2. **GPU Memory Issues**: Reduce batch size or use CPU
3. **Strategy Errors**: Check parameter compatibility
4. **Import Errors**: Ensure all dependencies are installed

### Debug Mode

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with detailed logging
results = await framework.backtest_strategy(
    strategy_name='lorentzian',
    data=data
)
```

## Integration with Existing Systems

The framework is designed to work seamlessly with your existing:

- **Data Pipeline**: Uses your CSV storage system
- **GPU Features**: Leverages your custom GPU-accelerated indicators
- **Strategies**: Integrates all existing strategy implementations
- **Configuration**: Works with your existing config files

## Next Steps

1. **Run Examples**: Start with `python scripts/example_paper_trading.py`
2. **Test Your Data**: Use the CLI to test with your historical data
3. **Optimize Strategies**: Find optimal parameters for your strategies
4. **Compare Performance**: Compare different strategies and timeframes
5. **Extend Framework**: Add new strategies or features as needed

## Support

For questions or issues:
- Check the example scripts for usage patterns
- Review the CLI help: `python scripts/paper_trading_cli.py --help`
- Examine the generated results and plots
- Check the logs in `logs/paper_trading.log` 