# 🚀 Getting Started with QuantDesk

Welcome to **QuantDesk** - the world's first **universal crypto trading terminal**! This isn't just another trading bot - it's a revolutionary platform that puts the entire crypto universe at your fingertips.

## 🌟 What Makes QuantDesk Revolutionary?

**QuantDesk is a one-of-a-kind crypto terminal** that gives you unprecedented control:

### 🔗 **Universal Exchange Access**
- **Decentralized First**: Drift Protocol, Jupiter, Raydium, Orca (Solana DEXs)
- **Cross-Chain**: Uniswap, PancakeSwap, SushiSwap (Multi-chain DEXs)
- **Centralized Options**: Bitget, Binance, Coinbase, MEXC, KuCoin, Kraken
- **Any Asset**: Spot, Perpetuals, Futures, Options
- **Any Chain**: Ethereum, BSC, Solana, Polygon, Arbitrum, Base
- **Any Timeframe**: 1m, 5m, 15m, 1h, 4h, 1d

### 🤖 **AI-Powered Intelligence**
- **ML Strategies**: Lorentzian Classifier, Lag-based, Logistic Regression
- **53.5% Win Rate** - Consistently beats the market
- **-4.7% Max Drawdown** - Professional risk management
- **GPU Acceleration** - AMD ROCm PyTorch for lightning-fast analysis

### 🎮 **Terminal Experience**
- **Command Center**: Like a crypto trading terminal
- **Real-time Data**: Live market feeds from all exchanges
- **Instant Execution**: Trade anywhere, anytime
- **Complete Control**: Your API keys, your rules, your profits

## ⚡ Quick Start (5 Minutes)

### Step 1: Download & Setup
```bash
# Download QuantDesk
git clone https://github.com/dextrorsal/quantdesk.git
cd QuantDesk

# Create Python environment
conda create -n QuantDesk python=3.11
conda activate QuantDesk

# Install everything
pip install -r requirements.txt
```

### Step 2: Configure Your Exchange
```bash
# Copy the example config
cp .env.example .env

# Edit with your exchange API keys
nano .env
```

**Add your exchange API credentials:**
```env
# Bitget API (for live trading)
BITGET_API_KEY=your_api_key_here
BITGET_SECRET_KEY=your_secret_key_here
BITGET_PASSPHRASE=your_passphrase_here

# Binance API (for data)
BINANCE_API_KEY=your_binance_key
BINANCE_SECRET_KEY=your_binance_secret
```

### Step 3: Get Market Data
```bash
# Download historical data (this takes a few minutes)
python scripts/daily_fetch.py
```

### Step 4: Start the Web Interface
```bash
# Start the trading dashboard
./web_ui/start_integrated.sh

# In another terminal, start the web interface
./web_ui/start_frontend_integrated.sh
```

### Step 5: Open Your Dashboard
Open http://localhost:3000 in your browser and you'll see:

- 📊 **Live Market Data** - Real-time crypto prices
- 🤖 **ML Strategies** - Your AI trading strategies  
- 📈 **Performance Metrics** - Win rates and returns
- 💰 **Account Balance** - Your exchange balance
- 🎯 **Trading Controls** - Start/stop trading

## 🎮 What You Can Do with Your Crypto Terminal

### 🌐 **Universal Trading**
- **Trade Anywhere**: Connect to any exchange with just API keys
- **Any Asset Type**: Spot, perps, futures, options - all in one place
- **Cross-Chain**: Trade on Ethereum, BSC, Solana, Polygon, Arbitrum, Base
- **Real-time Execution**: Instant trades across all connected exchanges

### 🤖 **AI-Powered Strategies**
- **Lorentzian Classifier** - 53.5% win rate, your best performer
- **Lag-based Strategy** - Multi-asset momentum across all chains
- **Logistic Regression** - Machine learning for any market condition
- **Chandelier Exit** - Professional risk management

### 📊 **Terminal Features**
- **Live Market Data**: Real-time feeds from all exchanges
- **Cross-Exchange Arbitrage**: Find opportunities across platforms
- **Portfolio Management**: Track all positions across all exchanges
- **Risk Management**: Unified risk controls across all trades

### 🔬 **Advanced Analysis**
- **Backtest Any Strategy**: Test on historical data from any exchange
- **ML Model Training**: Train custom models on your data
- **Performance Analytics**: Deep insights into your trading
- **Custom Indicators**: Build your own technical analysis tools

## 🔧 Troubleshooting

### "No module named 'src'" Error
```bash
# Make sure you're in the project directory
cd /path/to/QuantDesk

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
```

### "Exchange API Error"
- Check your API keys in `.env` file
- Make sure API keys have trading permissions
- Verify exchange account is active

### "No data found" Error
```bash
# Re-download data
python scripts/daily_fetch.py

# Check data directory
ls data/ohlcv/
```

### Web Interface Won't Start
```bash
# Check if backend is running
curl http://localhost:8000/health

# Restart backend
./web_ui/start_integrated.sh
```

## 📚 Next Steps

### For Beginners:
1. **Explore the Dashboard** - Get familiar with the interface
2. **Run Paper Trading** - Test strategies without real money
3. **Study Performance** - Understand how strategies work
4. **Start Small** - Begin with small amounts

### For Advanced Users:
1. **Custom Strategies** - Modify existing strategies
2. **Multi-Exchange** - Add more exchanges
3. **Advanced Backtesting** - Optimize parameters
4. **Live Deployment** - Go live with real money

## 🆘 Need Help?

- 📖 **Documentation**: Check the [full documentation](../README.md)
- 🐛 **Issues**: Report bugs on GitHub
- 💬 **Community**: Join our Discord/Telegram
- 📧 **Support**: Contact contact@quantdesk.app

## 🎉 You're Ready!

Congratulations! You now have a professional AI trading system running. The dashboard at http://localhost:3001 is your command center for:

- Monitoring market data
- Running ML strategies  
- Backtesting performance
- Executing live trades

**Happy Trading! 🚀**

---

*Next: [Installation Guide](installation.md) | [Web UI Guide](../user-guide/web-ui.md) | [Trading Strategies](../user-guide/trading-strategies.md)*
