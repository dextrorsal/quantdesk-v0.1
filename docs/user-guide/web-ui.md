# 🖥️ Web UI User Guide

The QuantDesk Web UI is your command center for managing your AI trading system. This guide shows you how to use all the features.

## 🚀 Getting Started

### Start the Web Interface
```bash
# Start the backend server
./web_ui/start_integrated.sh

# Start the frontend (in another terminal)
./web_ui/start_frontend_integrated.sh

# Open your browser to http://localhost:3000
```

### First Time Setup
1. **Open Dashboard**: Go to http://localhost:3000
2. **Check Connection**: Look for "WebSocket: Connected" status
3. **Verify Data**: See your available exchanges and symbols
4. **Test Strategies**: View your ML strategies and performance

## 📊 Dashboard Overview

### Main Dashboard Features
- **📈 Live Market Data**: Real-time crypto prices
- **🤖 ML Strategies**: Your AI trading strategies
- **📊 Performance Metrics**: Win rates and returns
- **💰 Account Balance**: Your exchange balance
- **🎯 Trading Controls**: Start/stop trading

### Navigation
- **Dashboard**: Main overview and system status
- **Strategies**: Manage and configure ML strategies
- **Backtesting**: Test strategies on historical data
- **Data Management**: Browse and manage market data
- **Settings**: Configure system settings

## 🤖 Strategy Management

### View Available Strategies
1. **Go to Strategies Tab**
2. **See Your ML Strategies**:
   - **Lorentzian Classifier** - 53.5% win rate, best performer
   - **Lag-based Strategy** - Multi-asset momentum
   - **Logistic Regression** - Machine learning classifier
   - **Chandelier Exit** - Risk management strategy

### Strategy Performance
Each strategy shows:
- **Win Rate**: Percentage of profitable trades
- **Total Return**: Overall performance
- **Max Drawdown**: Worst loss period
- **Total Trades**: Number of trades executed

### Configure Strategies
1. **Select Strategy**: Click on strategy name
2. **Adjust Parameters**: Modify settings
3. **Save Configuration**: Apply changes
4. **Test Changes**: Run backtest to verify

## 📈 Backtesting

### Run a Backtest
1. **Go to Backtesting Tab**
2. **Select Strategy**: Choose from dropdown
3. **Choose Exchange**: Select data source
4. **Pick Symbol**: Choose trading pair (e.g., BTCUSDT)
5. **Set Timeframe**: Choose interval (1h, 4h, 1d)
6. **Set Duration**: How many days to test
7. **Click "Run Backtest"**

### Understanding Results
- **Total Return**: Overall profit/loss percentage
- **Win Rate**: Percentage of winning trades
- **Max Drawdown**: Worst loss period
- **Sharpe Ratio**: Risk-adjusted return
- **Total Trades**: Number of trades executed

### Example Backtest
```
Strategy: Lorentzian Classifier
Exchange: Bitget
Symbol: BTCUSDT
Timeframe: 1h
Duration: 30 days

Results:
- Total Return: +2.3%
- Win Rate: 54.2%
- Max Drawdown: -3.1%
- Total Trades: 47
```

## 💰 Live Trading

### Connect Exchange Account
1. **Go to Settings Tab**
2. **Enter API Credentials**:
   - API Key
   - Secret Key
   - Passphrase
3. **Test Connection**: Verify account access
4. **Save Settings**: Store credentials securely

### Start Live Trading
1. **Go to Dashboard**
2. **Select Strategy**: Choose your strategy
3. **Set Position Size**: How much to risk per trade
4. **Click "Start Trading"**
5. **Monitor Performance**: Watch live trades

### Trading Controls
- **Start/Stop**: Enable or disable trading
- **Position Size**: Adjust risk per trade
- **Stop Loss**: Set maximum loss per trade
- **Take Profit**: Set profit target

## 📊 Market Data

### View Live Data
- **Real-time Prices**: Current market prices
- **24h Change**: Price movement in last 24 hours
- **Volume**: Trading volume
- **High/Low**: 24h price range

### Browse Historical Data
1. **Go to Data Management Tab**
2. **Select Exchange**: Choose data source
3. **Pick Symbol**: Choose trading pair
4. **Set Timeframe**: Choose interval
5. **View Data**: Browse historical candles

### Data Quality
- **Exchanges**: 6 exchanges (Bitget, Binance, Coinbase, etc.)
- **Symbols**: 44+ trading pairs
- **Timeframes**: 1m, 5m, 15m, 1h, 4h, 1d
- **Coverage**: 1+ year of historical data

## ⚙️ Settings & Configuration

### System Settings
- **Data Path**: Where market data is stored
- **Log Level**: How detailed logging should be
- **Update Frequency**: How often to refresh data
- **Theme**: Dark or light interface

### Trading Settings
- **Default Leverage**: Leverage multiplier
- **Max Position Size**: Maximum position size
- **Risk Per Trade**: Risk percentage per trade
- **Stop Loss**: Default stop loss percentage

### API Settings
- **Exchange APIs**: Configure exchange connections
- **Rate Limits**: Set API request limits
- **Timeouts**: Set connection timeouts
- **Retries**: Set retry attempts

## 🔍 Monitoring & Alerts

### Real-time Monitoring
- **WebSocket Connection**: Live data updates
- **Strategy Status**: Active/inactive strategies
- **Position Status**: Open positions
- **Account Balance**: Current balance

### Performance Tracking
- **Daily P&L**: Daily profit/loss
- **Win Rate**: Current win rate
- **Drawdown**: Current drawdown
- **Sharpe Ratio**: Risk-adjusted performance

### Alerts & Notifications
- **Trade Alerts**: When trades are executed
- **Error Alerts**: When errors occur
- **Performance Alerts**: When metrics change
- **System Alerts**: When system issues occur

## 🚨 Troubleshooting

### Common Issues

#### "WebSocket Disconnected"
- Check if backend is running
- Restart backend server
- Check firewall settings

#### "No Market Data"
- Verify data fetching is working
- Check exchange API connections
- Run data fetching script

#### "Strategy Not Working"
- Check strategy configuration
- Verify data is available
- Check strategy parameters

#### "Trading Not Executing"
- Verify exchange API credentials
- Check account permissions
- Ensure sufficient balance

### Getting Help
- **Check Logs**: Look at system logs
- **Restart Services**: Restart backend/frontend
- **Verify Configuration**: Check settings
- **Contact Support**: Get help if needed

## 🎯 Best Practices

### Risk Management
- **Start Small**: Begin with small position sizes
- **Use Stop Losses**: Always set stop losses
- **Monitor Performance**: Watch your strategies
- **Diversify**: Don't put all money in one strategy

### System Management
- **Regular Backups**: Backup your configuration
- **Monitor Logs**: Check for errors regularly
- **Update Data**: Keep market data current
- **Test Changes**: Test before going live

### Performance Optimization
- **Monitor Resources**: Watch CPU/memory usage
- **Optimize Parameters**: Tune strategy parameters
- **Regular Backtests**: Test strategies regularly
- **Keep Updated**: Update system regularly

## 🎉 You're Ready!

You now know how to use the QuantDesk Web UI to:

- ✅ **Monitor Markets**: View real-time data
- ✅ **Manage Strategies**: Configure ML strategies
- ✅ **Run Backtests**: Test performance
- ✅ **Execute Trades**: Trade live with your strategies
- ✅ **Monitor Performance**: Track your results

**Happy Trading! 🚀**

---

*Previous: [Configuration Guide](../getting-started/configuration.md) | Next: [Trading Strategies Guide](trading-strategies.md)*
