# Automated Lag-Based Trading System

## Overview

This document describes the **Automated Lag-Based Trading System** added to the QuantDesk project. This system fully automates the lag-based strategy you previously traded by hand, with support for leverage, daily optimization, paper trading, and extensibility for other strategies.

---

## Features

- **Daily Parameter Optimization**: Automatically finds the best move thresholds and leverage for each leader-follower pair.
- **Leverage Support**: Simulates trading with leverage (1x–20x) for high-frequency strategies.
- **Paper Trading Environment**: All trades are simulated with realistic fees and slippage, so you can test safely.
- **Risk Management**: Position sizing, stop loss, take profit, and max leverage controls.
- **Signal Generation & Logging**: Every signal and trade is logged for review and analysis.
- **Performance Tracking**: Tracks win rate, profit factor, drawdown, and more.
- **Extensible**: Easily add new strategies (e.g., Lorentzian) or connect to a real exchange for live trading.

---

## Usage

### 1. **Optimize Parameters**
Run daily or whenever you want to update thresholds and leverage:
```bash
python scripts/automated_lag_trading_system.py optimize
```

### 2. **Run the Automated System (Paper Trading)**
```bash
python scripts/automated_lag_trading_system.py run --paper-trading
```
- This will fetch new data, generate signals, and paper trade in a loop (default: every 60 seconds).

### 3. **Check System Status**
```bash
python scripts/automated_lag_trading_system.py status
```

---

## Configuration

Edit `configs/automated_trading_config.json` to adjust:
- **Leverage and threshold ranges**
- **Risk management settings** (max positions, stop loss, take profit, etc.)
- **Which strategies are enabled**
- **Which assets are leaders/followers**
- **Paper trading settings**

---

## How It Works

1. **Optimization**: Scans a range of thresholds and leverages for all leader-follower pairs, finding the best settings for each.
2. **Signal Generation**: Monitors BTC, ETH, SOL for big moves, then looks for lag in meme/DeFi coins.
3. **Paper Trading**: Executes trades in a simulated environment, logging all actions and results.
4. **Performance Tracking**: Logs and summarizes all trades, signals, and portfolio performance.
5. **Extensibility**: Add new strategies or connect to a real exchange by extending the framework.

---

## Results & Logs

- **Signals**: `results/automated_trading/signals/`
- **Trades**: `results/automated_trading/trades/`
- **Performance**: `results/automated_trading/performance/`
- **Optimization**: `results/automated_trading/optimization/`

---

## Extending the System

- **Add New Strategies**: Implement your strategy and add it to the config.
- **Go Live**: Replace the paper trading engine with your real exchange client.
- **Schedule Retraining**: The system can be run as a cron job or background service for continuous optimization and trading.

---

## Example: Adding a New Follower

1. Edit `configs/automated_trading_config.json`:
   ```json
   "followers": [
     "PEPE", "WIF", "FARTCOIN", "NEWCOIN"  // Add your new coin here
   ]
   ```
2. Rerun optimization and the trading system.

---

## FAQ

**Q: Can I use this for real trading?**  
A: Yes, just swap out the paper trading engine for your exchange client and set `paper_trading.enabled` to `false`.

**Q: How do I add my own strategy?**  
A: Implement your strategy class, add it to the config, and hook it into the main system.

**Q: Where do I see my results?**  
A: All signals, trades, and performance metrics are saved in the `results/automated_trading/` directory.

---

## See Also
- [lag-based.md](./lag-based.md) — for the original strategy design and research notes.
- [configs/automated_trading_config.json](../configs/automated_trading_config.json) — for all system settings.

---

*This document will be updated as new features and strategies are added. For questions or help, see the main README or ask your AI assistant!* 