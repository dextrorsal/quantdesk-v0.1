# Quick Start Guide

Welcome to QuantDesk - **The Bloomberg Terminal for Crypto**. This guide will get you up and running in under 10 minutes.

## 🎯 What You'll Build

By the end of this guide, you'll have:
- ✅ QuantDesk running locally
- ✅ Connected to Solana devnet
- ✅ Made your first trade
- ✅ Viewed real-time analytics

## 📋 Prerequisites

Before we begin, ensure you have:

- **Node.js 18+** and npm
- **Rust 1.70+** and Cargo
- **Solana CLI tools** installed
- **Anchor Framework** installed
- **Git** for cloning the repository

### Quick Prerequisites Check

```bash
# Check Node.js version
node --version  # Should be 18+

# Check Rust version
rustc --version  # Should be 1.70+

# Check Solana CLI
solana --version  # Should be 1.17+

# Check Anchor
anchor --version  # Should be 0.31+
```

## 🚀 Step 1: Clone and Setup

### Clone the Repository

```bash
git clone https://github.com/quantdesk/quantdesk.git
cd quantdesk
```

### Install Dependencies

```bash
# Install backend dependencies
cd backend && npm install

# Install frontend dependencies
cd ../frontend && npm install

# Install smart contract dependencies
cd ../contracts/smart-contracts && npm install
```

## ⚙️ Step 2: Environment Configuration

### Create Environment File

```bash
# Copy the environment template
cp env.example .env
```

### Configure Environment Variables

Edit `.env` with your configuration:

```env
# Solana Configuration
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_WALLET_PATH=~/.config/solana/id.json

# Supabase Configuration (Optional for local development)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Backend Configuration
PORT=3002
NODE_ENV=development
JWT_SECRET=your_jwt_secret

# Frontend Configuration
VITE_API_URL=http://localhost:3002
VITE_SOLANA_RPC_URL=https://api.devnet.solana.com
```

## 🏃‍♂️ Step 3: Start the Platform

### Terminal 1: Start Backend

```bash
cd backend
npm run dev
```

You should see:
```
🚀 QuantDesk Backend API running on port 3002
📊 Environment: development
🔗 Frontend URL: http://localhost:3000
📡 WebSocket enabled: Yes
💰 Starting Pyth Oracle price feed service...
✅ Pyth Oracle price feed service started
```

### Terminal 2: Start Frontend

```bash
cd frontend
npm run dev
```

You should see:
```
  VITE v5.0.0  ready in 500 ms

  ➜  Local:   http://localhost:3001/
  ➜  Network: use --host to expose
```

### Terminal 3: Start Solana Validator (Optional)

For local testing with smart contracts:

```bash
solana-test-validator
```

## 🌐 Step 4: Access the Platform

### Open QuantDesk

Navigate to **http://localhost:3001** in your browser.

You should see the QuantDesk trading interface with:
- **Market Overview**: Real-time price feeds
- **Trading Panel**: Order placement interface
- **Portfolio**: Position and balance overview
- **Analytics**: Market and trading metrics

### Connect Your Wallet

1. Click **"Connect Wallet"** in the top right
2. Select your preferred wallet (Phantom, Solflare, etc.)
3. Approve the connection
4. Your wallet address will appear in the interface

## 📊 Step 5: Explore the Features

### View Market Data

- **Real-time Prices**: Live BTC, ETH, SOL perpetual prices
- **Market Depth**: Order book visualization
- **Volume Analytics**: 24h volume and trading activity

### Place Your First Order

1. **Select Market**: Choose BTC-PERP, ETH-PERP, or SOL-PERP
2. **Set Parameters**:
   - Side: Long or Short
   - Size: Position size
   - Leverage: Up to 100x
   - Order Type: Market or Limit
3. **Review**: Check your order details
4. **Submit**: Confirm and place the order

### Monitor Your Position

- **Real-time P&L**: Live profit/loss updates
- **Risk Metrics**: Margin requirements and liquidation risk
- **Position History**: Complete trading history

## 🔍 Step 6: Explore Analytics

### Access Grafana Dashboard

If you have Grafana configured:

1. Navigate to **http://localhost:3000**
2. Login to Grafana
3. Import the QuantDesk dashboard
4. View real-time trading metrics

### API Endpoints

Test the API directly:

```bash
# Health check
curl http://localhost:3002/health

# Get markets
curl http://localhost:3002/api/supabase-oracle/markets

# Get trading metrics
curl http://localhost:3002/api/metrics/trading
```

## 🎉 Congratulations!

You've successfully set up QuantDesk! Here's what you've accomplished:

- ✅ **Platform Running**: Full-stack application operational
- ✅ **Wallet Connected**: Ready for trading
- ✅ **Market Data**: Real-time price feeds active
- ✅ **Trading Ready**: Can place orders and manage positions
- ✅ **Analytics Available**: Monitoring and metrics accessible

## 🚀 Next Steps

### For Traders
- [Trading Guide](../trading/overview.md) - Master advanced trading features
- [Risk Management](../trading/risk-management.md) - Learn risk management strategies
- [Portfolio Analytics](../trading/portfolio.md) - Optimize your portfolio

### For Developers
- [API Documentation](../api/overview.md) - Integrate with our APIs
- [Smart Contract Integration](../api/smart-contracts.md) - Direct blockchain integration
- [SDK Libraries](../api/sdks.md) - Use our official SDKs

### For Institutions
- [Enterprise Features](../deployment/production.md) - Production deployment
- [Compliance Guide](../security/compliance.md) - Regulatory compliance
- [Professional Support](mailto:contact@quantdesk.io) - Enterprise support

## 🆘 Need Help?

- **Discord Community**: [Join our Discord](https://discord.gg/quantdesk)
- **GitHub Issues**: [Report issues](https://github.com/quantdesk/quantdesk/issues)
- **Documentation**: [Browse all docs](../README.md)
- **Email Support**: [contact@quantdesk.io](mailto:contact@quantdesk.io)

## 🔧 Troubleshooting

### Common Issues

**Backend won't start**:
```bash
# Check if port 3002 is available
lsof -i :3002

# Kill any conflicting processes
pkill -f "node.*3002"
```

**Frontend won't start**:
```bash
# Clear npm cache
npm cache clean --force

# Reinstall dependencies
rm -rf node_modules package-lock.json
npm install
```

**Wallet connection issues**:
- Ensure your wallet is unlocked
- Check that you're on the correct network (Devnet)
- Try refreshing the page

**Smart contract errors**:
```bash
# Restart Solana validator
pkill solana-test-validator
solana-test-validator
```

---

**Ready for more?** Check out our [Trading Guide](../trading/overview.md) or [API Documentation](../api/overview.md)!
