# Lorentzian Live Trader for Bitget

This document describes the Lorentzian Live Trader implementation for Bitget exchange, which uses the Lorentzian classifier model for generating trading signals across multiple timeframes.

## Overview

The Lorentzian Live Trader is a trading bot that:

1. Uses the Lorentzian classifier model to generate trading signals
2. Supports multiple timeframes (1m, 5m, 15m)
3. Implements risk management with stop-loss and take-profit
4. Tracks trading performance and positions
5. Works with Bitget futures trading API

## Features

- **Multi-timeframe Analysis**: Analyzes price action across multiple timeframes and uses a majority vote system to make trading decisions.
- **Position Management**: Tracks open positions and manages position sizing based on account balance and leverage.
- **Risk Management**: Implements stop-loss and take-profit orders for each trade.
- **Performance Tracking**: Records all trades and calculates performance metrics.
- **Test Mode**: Includes a test mode for verifying functionality without placing actual trades.

## Usage

```bash
# Run with default settings (SOL, 5m timeframe, 3x leverage)
python scripts/lorentzian_live_trader.py

# Run with custom settings
python scripts/lorentzian_live_trader.py --symbol SOL --timeframes 1m 5m 15m --leverage 3 --position-size 0.1 --stop-loss 0.02 --take-profit 0.04

# Run in test mode (no actual trades)
python scripts/lorentzian_live_trader.py --symbol SOL --timeframes 5m --test
```

## Parameters

- `--symbol`: Trading symbol (e.g., SOL)
- `--timeframes`: Timeframes to analyze (e.g., 1m, 5m, 15m)
- `--leverage`: Leverage to use for trading
- `--position-size`: Position size as percentage of account
- `--stop-loss`: Stop loss percentage
- `--take-profit`: Take profit percentage
- `--test`: Run in test mode without placing actual trades

## Model Parameters

The Lorentzian classifier model is configured with different parameters for each timeframe:

- **1m**: lookback=30, prediction_bars=3, k_neighbors=20
- **5m**: lookback=20, prediction_bars=4, k_neighbors=25
- **15m**: lookback=15, prediction_bars=5, k_neighbors=30

## Trading Logic

1. **Signal Generation**:
   - For each timeframe, the Lorentzian classifier generates buy/sell signals
   - A majority vote determines the overall signal

2. **Entry Rules**:
   - If no position is open and majority vote is buy → Open long
   - If no position is open and majority vote is sell → Open short

3. **Exit Rules**:
   - If long position is open and majority vote is sell → Close long
   - If short position is open and majority vote is buy → Close short
   - Stop-loss and take-profit orders are placed for each position

4. **Position Sizing**:
   - Position size = (Account balance × Position size percentage) × Leverage
   - Default is 10% of account with 3x leverage

## File Structure

- `scripts/lorentzian_live_trader.py`: Main script for the live trader
- `scripts/run_lorentzian_trader.sh`: Shell script to run the trader with optimal settings
- `logs/lorentzian_live_trader.log`: Log file for the trader
- `logs/trades/SOL_trades.json`: JSON file containing trade history and performance metrics
- `models/lorentzian_SOL_5m.pt`: Saved model file for SOL on 5m timeframe

## Requirements

- Python 3.11+
- PyTorch
- Pandas
- Bitget API credentials (set in environment variables)

## Environment Variables

The following environment variables need to be set:

```
BITGET_API_KEY=your_api_key
BITGET_SECRET_KEY=your_secret_key
BITGET_PASSPHRASE=your_passphrase
```

## Performance Monitoring

The trader saves all trades and performance metrics to a JSON file in the `logs/trades` directory. This file can be used to analyze the performance of the trader over time.

## Future Improvements

- Add more sophisticated signal filtering
- Implement dynamic position sizing based on volatility
- Add support for more exchanges
- Implement a web interface for monitoring performance
