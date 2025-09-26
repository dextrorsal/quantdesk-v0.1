# QuantDesk - The Bloomberg Terminal for Crypto

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Solana](https://img.shields.io/badge/Solana-9945FF?logo=solana&logoColor=white)](https://solana.com/)
[![Anchor](https://img.shields.io/badge/Anchor-9945FF?logo=anchor&logoColor=white)](https://www.anchor-lang.com/)

**The first institutional-grade decentralized perpetual trading platform** that brings the power and sophistication of traditional finance terminals to the crypto ecosystem. Built on Solana, QuantDesk combines the reliability of traditional finance with the innovation of DeFi.

## 🎯 **Why "Bloomberg Terminal for Crypto"?**

Just as Bloomberg Terminal revolutionized traditional finance by providing real-time data, analytics, and trading capabilities in one unified platform, QuantDesk brings the same level of sophistication to crypto trading:

- **Real-time Market Data**: Live price feeds, order book depth, and market analytics
- **Professional Trading Tools**: Advanced order types, risk management, and portfolio analytics
- **Institutional Features**: Enterprise-grade APIs, compliance reporting, and comprehensive monitoring
- **Enterprise-Grade Infrastructure**: Scalable, secure, and reliable platform architecture

## ✨ **Key Features**

### 🚀 **Core Trading Features**
- **Perpetual Trading**: Trade crypto perpetuals with up to 100x leverage
- **Advanced Order Types**: Stop-loss, take-profit, trailing stops, iceberg orders
- **Cross-Collateralization**: Use multiple assets as collateral (SOL, USDC, BTC, ETH, USDT, AVAX, MATIC, ARB, OP, DOGE, ADA, DOT, LINK)
- **Real-time Execution**: Sub-second order execution with professional-grade matching

### 📊 **Professional Analytics**
- **Portfolio Analytics**: Sharpe ratio, VaR, correlation analysis, risk-adjusted returns
- **Risk Management**: Real-time risk monitoring, alerts, stress testing
- **Performance Metrics**: Comprehensive trading performance analytics
- **Market Intelligence**: Advanced market data and insights

### 🏢 **Institutional Features**
- **Enterprise APIs**: Professional-grade REST and WebSocket APIs
- **Compliance Reporting**: Audit trails, regulatory compliance, KYC/AML
- **Multi-sig Support**: Enterprise wallet security
- **SLA Guarantees**: Service level agreements for institutional clients

### 💧 **Advanced Liquidity**
- **JIT Liquidity**: Just-In-Time liquidity auctions for better execution
- **Market Making**: Automated market making strategies
- **Price Improvement**: Competitive pricing mechanisms
- **Liquidity Mining**: Reward programs for liquidity providers

## 🛠️ **Technology Stack**

- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **Backend**: Node.js, TypeScript, Express.js
- **Blockchain**: Solana, Anchor Framework
- **Database**: PostgreSQL (Supabase)
- **APIs**: RESTful APIs, WebSocket, GraphQL
- **Monitoring**: Grafana, Prometheus, Custom Metrics

## 🚀 **Quick Start**

### Prerequisites
- Node.js 18+ 
- Solana CLI tools
- Git

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/your-username/quantdesk.git
cd quantdesk
```

2. **Install dependencies**
```bash
# Backend
cd backend
npm install

# Frontend
cd ../frontend
npm install
```

3. **Environment Setup**
```bash
# Copy environment files
cp backend/.env.example backend/.env
cp frontend/.env.example frontend/.env

# Configure your environment variables
# See Configuration section below
```

4. **Start Development Servers**
```bash
# Terminal 1: Backend
cd backend
npm run dev

# Terminal 2: Frontend
cd frontend
npm run dev
```

5. **Access the Application**
- Frontend: http://localhost:5173
- Backend API: http://localhost:3002
- API Documentation: http://localhost:3002/api/docs

## ⚙️ **Configuration**

### Environment Variables

**Backend (.env)**
```env
# Server Configuration
PORT=3002
NODE_ENV=development

# Database
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_anon_key

# Solana Configuration
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_WALLET_PATH=~/.config/solana/id.json

# JWT Configuration
JWT_SECRET=your_jwt_secret
JWT_EXPIRES_IN=24h

# API Keys (Optional)
PYTH_NETWORK_API_KEY=your_pyth_key
COINGECKO_API_KEY=your_coingecko_key
```

**Frontend (.env)**
```env
VITE_API_URL=http://localhost:3002
VITE_SOLANA_RPC_URL=https://api.devnet.solana.com
VITE_WALLET_ADAPTER_NETWORK=devnet
```

## 📚 **API Documentation**

### Core Endpoints

**Markets**
```bash
GET /api/supabase-oracle/markets
# Get all available trading markets
```

**Orders**
```bash
POST /api/orders
# Place a new order
GET /api/orders
# Get user orders
```

**Positions**
```bash
GET /api/positions
# Get user positions
POST /api/positions
# Open new position
```

**Portfolio Analytics**
```bash
GET /api/portfolio/metrics
# Get portfolio performance metrics
GET /api/portfolio/risk
# Get risk analysis
```

### Authentication

All protected endpoints require authentication:
```bash
Authorization: Bearer <your_jwt_token>
```

### WebSocket Support

Real-time updates via WebSocket:
```javascript
const ws = new WebSocket('ws://localhost:3002/ws');
ws.onmessage = (event) => {
  const data = JSON.parse(event.data);
  // Handle real-time updates
};
```

## 🏗️ **Architecture**

### System Overview
```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   Backend API   │    │   Solana        │
│   (React)       │◄──►│   (Node.js)     │◄──►│   Blockchain    │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
                       ┌─────────────────┐
                       │   Database      │
                       │   (Supabase)    │
                       └─────────────────┘
```

### Key Components

- **Smart Contracts**: Solana programs for trading logic
- **Backend Services**: REST APIs, WebSocket, business logic
- **Frontend**: React-based trading interface
- **Database**: PostgreSQL for user data and analytics
- **Monitoring**: Grafana dashboards and metrics

## 🔧 **Development**

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

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### Development Workflow
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Areas for Contribution
- **Frontend Components**: UI/UX improvements
- **API Documentation**: Better examples and guides
- **Testing**: Additional test coverage
- **Performance**: Optimization and monitoring
- **Documentation**: Guides and tutorials

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 **Support**

- **Documentation**: [GitBook](https://quantdesk.gitbook.io)
- **Issues**: [GitHub Issues](https://github.com/your-username/quantdesk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-username/quantdesk/discussions)
- **Discord**: [Community Discord](https://discord.gg/quantdesk)

## 🗺️ **Roadmap**

### Phase 1: Core Platform ✅
- [x] Basic trading functionality
- [x] Advanced order types
- [x] Cross-collateralization
- [x] Portfolio analytics

### Phase 2: Professional Features 🔄
- [x] Risk management
- [x] JIT liquidity
- [ ] Mobile applications
- [ ] Advanced analytics

### Phase 3: Enterprise & Scale 📋
- [ ] Institutional features
- [ ] Cross-chain support
- [ ] Governance token
- [ ] Advanced compliance

## 🌟 **Acknowledgments**

- **Solana Foundation** for the amazing blockchain platform
- **Anchor Framework** for Solana development tools
- **Supabase** for backend infrastructure
- **Community Contributors** for feedback and support

---

**Built with ❤️ by the QuantDesk team**

*QuantDesk - Bringing institutional-grade trading to decentralized finance*