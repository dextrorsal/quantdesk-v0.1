# QuantDesk Protocol - Public Demo

This repository contains the public documentation and demo scripts for QuantDesk Protocol, a high-frequency trading platform built on Solana.

## 🚀 What is QuantDesk?

QuantDesk is a revolutionary trading platform that combines:
- **Solana-based perpetual DEX** for high-speed trading
- **AI-powered trading agents** for intelligent market analysis
- **Real-time data ingestion** from multiple sources
- **Advanced risk management** and portfolio optimization

## 📁 Repository Structure

```
├── docs/                    # Complete documentation
├── examples/               # Demo scripts and examples
├── scripts/               # Utility scripts
├── database/              # Database schema and setup
├── contracts/             # Smart contract interfaces (public)
├── sdk/                   # TypeScript SDK (public)
├── public-demo/           # Public demo components
└── README.md              # This file
```

## 🎯 Core Components (Proprietary)

The following components contain proprietary trading algorithms and are not included in this public repository:

- **Frontend Trading Interface** (`/frontend/`)
- **Backend API Services** (`/backend/`)
- **MIKEY-AI Trading Agent** (`/MIKEY-AI/`)
- **Data Ingestion Pipeline** (`/data-ingestion/`)
- **Admin Dashboard** (`/admin-dashboard/`)

## 🛠️ Getting Started

### Prerequisites

- Node.js 18+
- Solana CLI
- PostgreSQL (or Supabase)
- Redis

### Quick Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/quantdesk-protocol.git
   cd quantdesk-protocol
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Configure environment**
   ```bash
   cp env.example .env
   # Edit .env with your configuration
   ```

4. **Run demo scripts**
   ```bash
   npm run demo:trading
   npm run demo:ai
   npm run demo:data
   ```

## 📊 Demo Scripts

### Trading Demo
```bash
npm run demo:trading
```
Demonstrates:
- Wallet connection
- Account initialization
- Order placement
- Position management

### AI Agent Demo
```bash
npm run demo:ai
```
Shows:
- AI market analysis
- Trading signal generation
- Risk assessment

### Data Pipeline Demo
```bash
npm run demo:data
```
Displays:
- Real-time data ingestion
- Market data processing
- Analytics generation

## 🔧 SDK Usage

```typescript
import { QuantDeskSDK } from '@quantdesk/sdk';

const sdk = new QuantDeskSDK({
  rpcUrl: 'https://api.devnet.solana.com',
  wallet: yourWallet
});

// Initialize trading account
const account = await sdk.initializeAccount();

// Place a trade
const order = await sdk.placeOrder({
  symbol: 'SOL-PERP',
  side: 'long',
  size: 1.0,
  leverage: 3
});
```

## 📈 Supported Markets

- **SOL-PERP** - Solana perpetual futures
- **BTC-PERP** - Bitcoin perpetual futures  
- **ETH-PERP** - Ethereum perpetual futures
- **Custom tokens** - Any SPL token with sufficient liquidity

## 🔒 Security

- All private keys and API keys are excluded from this repository
- Smart contracts are audited and verified
- Follows Solana security best practices
- Implements proper access controls

## 📚 Documentation

- [Architecture Overview](docs/ARCHITECTURE.md)
- [API Reference](docs/api/)
- [Smart Contract Guide](docs/SMART_CONTRACTS.md)
- [Deployment Guide](docs/DEPLOYMENT.md)
- [Contributing](docs/CONTRIBUTING.md)

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](docs/CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🆘 Support

- **Documentation**: [docs.quantdesk.io](https://docs.quantdesk.io)
- **Discord**: [discord.gg/quantdesk](https://discord.gg/quantdesk)
- **Twitter**: [@QuantDesk](https://twitter.com/quantdesk)

## ⚠️ Disclaimer

This software is for educational and demonstration purposes. Trading cryptocurrencies involves substantial risk of loss. Use at your own risk.

---

**Built with ❤️ by the QuantDesk Team**
