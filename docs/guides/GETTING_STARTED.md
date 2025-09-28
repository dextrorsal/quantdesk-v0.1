# Getting Started with QuantDesk

Welcome to QuantDesk! This guide will help you get up and running with our institutional-grade trading platform.

## 🚀 **Quick Start (5 minutes)**

### 1. Prerequisites

Make sure you have the following installed:
- **Node.js** 18+ ([Download](https://nodejs.org/))
- **Git** ([Download](https://git-scm.com/))
- **Solana CLI** ([Install Guide](https://docs.solana.com/cli/install-solana-cli-tools))

### 2. Clone and Install

```bash
# Clone the repository
git clone https://github.com/dextrorsal/quantdesk.git
cd quantdesk

# Install backend dependencies
cd backend
npm install

# Install frontend dependencies
cd ../frontend
npm install
```

### 3. Environment Setup

```bash
# Backend environment
cd backend
cp .env.example .env
# Edit .env with your configuration

# Frontend environment
cd ../frontend
cp .env.example .env
# Edit .env with your configuration
```

### 4. Start Development

```bash
# Terminal 1: Backend
cd backend
npm run dev

# Terminal 2: Frontend
cd frontend
npm run dev
```

### 5. Access the Application

- **Frontend**: http://localhost:5173
- **Backend API**: http://localhost:3002
- **API Docs**: http://localhost:3002/api/docs

## 🔧 **Configuration Guide**

### Backend Configuration (.env)

```env
# Server Configuration
PORT=3002
NODE_ENV=development

# Database (Supabase)
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Solana Configuration
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_WALLET_PATH=~/.config/solana/id.json

# JWT Configuration
JWT_SECRET=your_jwt_secret_key_here
JWT_EXPIRES_IN=24h

# Optional API Keys
PYTH_NETWORK_API_KEY=your_pyth_key
COINGECKO_API_KEY=your_coingecko_key
```

### Frontend Configuration (.env)

```env
VITE_API_URL=http://localhost:3002
VITE_SOLANA_RPC_URL=https://api.devnet.solana.com
VITE_WALLET_ADAPTER_NETWORK=devnet
```

## 📊 **First Steps**

### 1. Explore the Markets

Visit the markets page to see available trading pairs:
- BTC-PERP, ETH-PERP, SOL-PERP
- AVAX-PERP, MATIC-PERP, ARB-PERP
- OP-PERP, DOGE-PERP, ADA-PERP
- DOT-PERP, LINK-PERP

### 2. Connect Your Wallet

1. Click "Connect Wallet" in the top right
2. Choose from supported wallets:
   - Phantom
   - Solflare
   - Backpack
   - Glow
3. Approve the connection

### 3. View Your Portfolio

Once connected, you can:
- View your current positions
- Check portfolio performance
- Monitor risk metrics
- Analyze trading history

### 4. Place Your First Order

1. Select a market (e.g., BTC-PERP)
2. Choose order type (Market, Limit, Stop-Loss)
3. Set size and price
4. Review and confirm

## 🛠️ **Development Workflow**

### Running Tests

```bash
# Backend tests
cd backend
npm test

# Frontend tests
cd frontend
npm test

# Smart contract tests
cd contracts/smart-contracts
anchor test
```

### Code Quality

```bash
# Linting
npm run lint

# Type checking
npm run type-check

# Formatting
npm run format
```

### Building for Production

```bash
# Backend build
cd backend
npm run build

# Frontend build
cd frontend
npm run build
```

## 📚 **API Usage Examples**

### Basic API Calls

```javascript
// Get available markets
const markets = await fetch('http://localhost:3002/api/supabase-oracle/markets')
  .then(res => res.json());

// Get portfolio metrics (requires auth)
const metrics = await fetch('http://localhost:3002/api/portfolio/metrics', {
  headers: {
    'Authorization': 'Bearer your_jwt_token'
  }
}).then(res => res.json());
```

### WebSocket Connection

```javascript
const ws = new WebSocket('ws://localhost:3002/ws');

ws.onopen = () => {
  // Subscribe to market data
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'market_data',
    marketId: 'market_id_here'
  }));
};

ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  console.log('Market update:', data);
};
```

## 🔍 **Troubleshooting**

### Common Issues

**1. "Connection refused" errors**
- Make sure the backend server is running on port 3002
- Check if the port is already in use

**2. "Authentication required" errors**
- Ensure you're logged in and have a valid JWT token
- Check if the token has expired

**3. Wallet connection issues**
- Make sure you have a Solana wallet installed
- Check if you're on the correct network (devnet/mainnet)

**4. Build errors**
- Run `npm install` in both backend and frontend directories
- Check Node.js version (requires 18+)

### Getting Help

- **Documentation**: [GitBook](https://quantdesk.gitbook.io)
- **Issues**: [GitHub Issues](https://github.com/dextrorsal/quantdesk/issues)
- **Discord**: [Community Discord](https://discord.gg/quantdesk)
- **Email**: contact@quantdesk.app

## 🎯 **Next Steps**

### For Developers

1. **Explore the Codebase**
   - Check out the smart contracts in `contracts/`
   - Review the API routes in `backend/src/routes/`
   - Examine the React components in `frontend/src/`

2. **Contribute**
   - Read our [Contributing Guidelines](CONTRIBUTING.md)
   - Pick up an issue from our [GitHub Issues](https://github.com/dextrorsal/quantdesk/issues)
   - Submit a pull request

3. **Build Integrations**
   - Use our REST API for custom applications
   - Implement WebSocket connections for real-time data
   - Create trading bots using our SDK

### For Traders

1. **Learn the Platform**
   - Try different order types
   - Experiment with cross-collateralization
   - Explore portfolio analytics

2. **Advanced Features**
   - Set up risk management alerts
   - Use advanced order types
   - Monitor JIT liquidity opportunities

3. **Join the Community**
   - Share strategies and insights
   - Get help from other traders
   - Provide feedback for improvements

## 📖 **Additional Resources**

- **API Documentation**: [API.md](docs/API.md)
- **Architecture Overview**: [ARCHITECTURE.md](docs/ARCHITECTURE.md)
- **Smart Contract Guide**: [SMART_CONTRACTS.md](docs/SMART_CONTRACTS.md)
- **Deployment Guide**: [DEPLOYMENT.md](docs/DEPLOYMENT.md)
- **Security Guide**: [SECURITY.md](docs/SECURITY.md)

---

**Welcome to QuantDesk - The Bloomberg Terminal for Crypto!** 🚀

*Start building the future of decentralized trading today.*
