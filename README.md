# QuantDesk - The Bloomberg Terminal for Crypto

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Solana](https://img.shields.io/badge/Solana-9945FF?logo=solana&logoColor=white)](https://solana.com/)
[![Anchor](https://img.shields.io/badge/Anchor-9945FF?logo=anchor&logoColor=white)](https://www.anchor-lang.com/)

**The first institutional-grade decentralized perpetual trading platform** that brings the power and sophistication of traditional finance terminals to the crypto ecosystem. Built on Solana, QuantDesk combines the reliability of traditional finance with the innovation of DeFi.

## 🎯 Why "Bloomberg Terminal for Crypto"?

Just as Bloomberg Terminal revolutionized traditional finance by providing real-time data, analytics, and trading capabilities in one unified platform, QuantDesk brings the same level of sophistication to crypto trading:

- **Real-time Market Data**: Live price feeds, order book depth, and market analytics
- **Professional Trading Tools**: Advanced order types, risk management, and portfolio analytics
- **Institutional Features**: JIT liquidity, professional APIs, and comprehensive monitoring
- **Enterprise-Grade Infrastructure**: Scalable, secure, and reliable platform architecture

## 🚀 Features

### Core Trading Features
- **Perpetual Trading**: Trade BTC, ETH, SOL perpetuals with up to 100x leverage
- **Advanced Order Types**: Market, limit, stop-loss, and take-profit orders
- **Real-time Price Feeds**: Integrated Pyth Network oracle for accurate pricing
- **Risk Management**: Automated liquidation and position management
- **Multi-wallet Support**: Connect with Phantom, Solflare, and other Solana wallets

### Platform Features
- **Modern UI/UX**: Beautiful, responsive React frontend with Tailwind CSS
- **Real-time Updates**: WebSocket connections for live market data
- **Analytics Dashboard**: Comprehensive Grafana integration for monitoring
- **JIT Liquidity**: Just-in-time liquidity auctions for optimal execution
- **Admin Panel**: Complete administrative interface for platform management

### Technical Features
- **Smart Contracts**: Rust-based Anchor programs for on-chain logic
- **Backend API**: TypeScript/Node.js backend with comprehensive REST APIs
- **Database**: Supabase integration for scalable data storage
- **Monitoring**: Grafana dashboards for system and trading metrics
- **Testing**: Comprehensive test suites for smart contracts and APIs

## 📋 Prerequisites

- Node.js 18+ and npm
- Rust 1.70+ and Cargo
- Solana CLI tools
- Anchor Framework
- Git

## 🛠️ Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/quantdesk.git
cd quantdesk
```

### 2. Install Dependencies
```bash
# Install backend dependencies
cd backend && npm install

# Install frontend dependencies
cd ../frontend && npm install

# Install smart contract dependencies
cd ../contracts/smart-contracts && npm install
```

### 3. Environment Setup
```bash
# Copy environment template
cp env.example .env

# Edit .env with your configuration
# Required: SUPABASE_URL, SUPABASE_ANON_KEY, SOLANA_RPC_URL
```

### 4. Start Development Servers
```bash
# Terminal 1: Start backend
cd backend && npm run dev

# Terminal 2: Start frontend
cd frontend && npm run dev

# Terminal 3: Start Solana validator (for local testing)
solana-test-validator
```

### 5. Access the Platform
- **Frontend**: http://localhost:3001
- **Backend API**: http://localhost:3002
- **Grafana Dashboard**: http://localhost:3000 (if configured)

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend       │    │   Smart         │
│   (React)       │◄──►│   (Node.js)     │◄──►│   Contracts     │
│   Port: 3001    │    │   Port: 3002    │    │   (Anchor)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Grafana       │    │   Supabase      │    │   Solana        │
│   Dashboard     │    │   Database      │    │   Blockchain    │
│   Port: 3000    │    │   (PostgreSQL)  │    │   (Devnet)      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
quantdesk/
├── frontend/                 # React frontend application
│   ├── src/
│   │   ├── components/      # Reusable UI components
│   │   ├── pages/          # Page components
│   │   ├── services/       # API services
│   │   └── providers/     # Context providers
│   └── package.json
├── backend/                # Node.js backend API
│   ├── src/
│   │   ├── routes/         # API route handlers
│   │   ├── services/       # Business logic services
│   │   ├── middleware/     # Express middleware
│   │   └── utils/          # Utility functions
│   └── package.json
├── contracts/              # Solana smart contracts
│   └── smart-contracts/
│       ├── programs/       # Anchor programs
│       ├── tests/          # Contract tests
│       └── migrations/     # Deployment scripts
├── docs/                   # Documentation
│   ├── ARCHITECTURE.md
│   ├── API.md
│   └── DEPLOYMENT.md
└── scripts/               # Utility scripts
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file in the root directory:

```env
# Solana Configuration
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_WALLET_PATH=~/.config/solana/id.json

# Supabase Configuration
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

## 🧪 Testing

### Smart Contract Tests
```bash
cd contracts/smart-contracts
anchor test
```

### Backend Tests
```bash
cd backend
npm test
```

### Frontend Tests
```bash
cd frontend
npm test
```

## 🚀 Deployment

### Development Deployment
```bash
# Deploy smart contracts to devnet
cd contracts/smart-contracts
anchor deploy

# Start production servers
cd backend && npm start
cd frontend && npm run build && npm run preview
```

### Production Deployment
See [DEPLOYMENT.md](docs/DEPLOYMENT.md) for detailed production deployment instructions.

## 📊 Monitoring

QuantDesk includes comprehensive monitoring through Grafana:

- **Trading Metrics**: Volume, active traders, positions, TVL
- **System Performance**: API response times, memory usage, CPU
- **Market Analytics**: Volume distribution, leverage analysis
- **Real-time Alerts**: Configurable alerts for critical metrics

Access Grafana at `http://localhost:3000` (when configured).

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs/](docs/)
- **Issues**: [GitHub Issues](https://github.com/yourusername/quantdesk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/quantdesk/discussions)

## 🗺️ Roadmap

- [ ] Cross-chain support (Ethereum, Polygon)
- [ ] Advanced order types (TWAP, Iceberg)
- [ ] Mobile application
- [ ] Institutional trading features
- [ ] Advanced analytics and reporting

## 🙏 Acknowledgments

- [Solana](https://solana.com/) - The blockchain platform
- [Anchor](https://www.anchor-lang.com/) - Solana development framework
- [Pyth Network](https://pyth.network/) - Price feed oracles
- [Supabase](https://supabase.com/) - Backend-as-a-Service
- [Grafana](https://grafana.com/) - Monitoring and analytics

---

**Built with ❤️ by the QuantDesk team**