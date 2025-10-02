# QuantDesk Open Source Package

## 🎯 **What's Included**

This open source package provides a **comprehensive foundation** for building institutional-grade decentralized trading platforms. It includes:

### ✅ **Core Features (Public)**
- **Basic Trading**: Market orders, limit orders, position management
- **Multi-Asset Support**: 11 perpetual markets (BTC, ETH, SOL, AVAX, MATIC, ARB, OP, DOGE, ADA, DOT, LINK)
- **Cross-Collateralization**: Use multiple assets as collateral
- **Portfolio Analytics**: Basic performance metrics and risk analysis
- **API Documentation**: Complete REST API documentation
- **WebSocket Support**: Real-time market data and updates

### 🔒 **Advanced Features (Private)**
- **Advanced Risk Management**: Institutional-grade risk monitoring
- **JIT Liquidity**: Just-In-Time liquidity auctions
- **Market Making**: Automated market making strategies
- **Enterprise APIs**: Professional-grade integrations
- **Compliance Tools**: Audit trails and regulatory reporting

## 🚀 **Quick Start**

### 1. **Clone and Setup**
```bash
git clone https://github.com/dextrorsal/quantdesk.git
cd quantdesk
chmod +x setup-demo.sh
./setup-demo.sh
```

### 2. **Configure Environment**
```bash
# Edit backend/.env
SUPABASE_URL=your_supabase_url
SUPABASE_ANON_KEY=your_supabase_key
JWT_SECRET=your_jwt_secret

# Edit frontend/.env
VITE_API_URL=http://localhost:3002
VITE_SOLANA_RPC_URL=https://api.devnet.solana.com
```

### 3. **Start Development**
```bash
# Terminal 1: Backend
cd backend && npm run dev

# Terminal 2: Frontend
cd frontend && npm run dev
```

### 4. **Access Application**
- **Frontend**: http://localhost:5173
- **Backend**: http://localhost:3002
- **API Docs**: http://localhost:3002/api/docs

## 📚 **Documentation**

- **[Getting Started](docs/GETTING_STARTED.md)** - Complete setup guide
- **[API Documentation](docs/API.md)** - REST API reference
- **[Contributing](CONTRIBUTING.md)** - How to contribute
- **[Architecture](docs/ARCHITECTURE.md)** - System overview

## 🧪 **Demo Examples**

### Basic Trading Demo
```bash
node examples/basic-trading-demo.js
```

### API Usage Examples
```javascript
// Get markets
const markets = await fetch('http://localhost:3002/api/supabase-oracle/markets')
  .then(res => res.json());

// Get portfolio metrics (requires auth)
const metrics = await fetch('http://localhost:3002/api/portfolio/metrics', {
  headers: { 'Authorization': 'Bearer your_token' }
}).then(res => res.json());
```

### WebSocket Connection
```javascript
const ws = new WebSocket('ws://localhost:3002/ws');
ws.onopen = () => {
  ws.send(JSON.stringify({
    type: 'subscribe',
    channel: 'market_data',
    marketId: 'market_id'
  }));
};
```

## 🏗️ **Architecture**

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

## 🔧 **Technology Stack**

- **Frontend**: React, TypeScript, Vite, Tailwind CSS
- **Backend**: Node.js, TypeScript, Express.js
- **Blockchain**: Solana, Anchor Framework
- **Database**: PostgreSQL (Supabase)
- **APIs**: RESTful APIs, WebSocket, GraphQL
- **Monitoring**: Grafana, Prometheus, Custom Metrics

## 📊 **Available Markets**

| Symbol | Base Asset | Quote Asset | Max Leverage |
|--------|------------|-------------|--------------|
| BTC-PERP | BTC | USDT | 100x |
| ETH-PERP | ETH | USDT | 100x |
| SOL-PERP | SOL | USDT | 100x |
| AVAX-PERP | AVAX | USDT | 50x |
| MATIC-PERP | MATIC | USDT | 50x |
| ARB-PERP | ARB | USDT | 50x |
| OP-PERP | OP | USDT | 50x |
| DOGE-PERP | DOGE | USDT | 25x |
| ADA-PERP | ADA | USDT | 25x |
| DOT-PERP | DOT | USDT | 25x |
| LINK-PERP | LINK | USDT | 25x |

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

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
- **Issues**: [GitHub Issues](https://github.com/dextrorsal/quantdesk/issues)
- **Discussions**: [GitHub Discussions](https://github.com/dextrorsal/quantdesk/discussions)
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
