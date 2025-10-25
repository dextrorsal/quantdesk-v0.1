# QuantDesk
## The Bloomberg Terminal for Crypto Trading

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![TypeScript](https://img.shields.io/badge/TypeScript-007ACC?logo=typescript&logoColor=white)](https://www.typescriptlang.org/)
[![React](https://img.shields.io/badge/React-20232A?logo=react&logoColor=61DAFB)](https://reactjs.org/)
[![Solana](https://img.shields.io/badge/Solana-9945FF?logo=solana&logoColor=white)](https://solana.com/)

> **QuantDesk** is a sophisticated Solana-based perpetual DEX platform featuring multi-service architecture, AI-powered trading assistance, and real-time data ingestion.

---

## 🎯 **What is QuantDesk?**

QuantDesk is a **production-ready Solana perpetual DEX platform** that eliminates the "16 tabs problem" by providing institutional-grade trading tools with AI-powered insights in a unified interface.

### **Key Features**
- **Multi-Service Architecture:** Backend, Frontend, MIKEY-AI, and Data Ingestion services
- **AI-Powered Trading:** LangChain integration with real-time market intelligence
- **Enterprise Security:** Multi-layer security with comprehensive protection
- **Professional Interface:** Bloomberg Terminal-level sophistication for crypto trading

---

## 🚀 **Quick Start**

### **Prerequisites**
- Node.js 20+
- pnpm (package manager)
- Solana CLI tools
- Rust (for smart contracts)

### **Installation**
```bash
# Clone the repository
git clone https://github.com/quantdesk/quantdesk.git
cd quantdesk

# Install dependencies
pnpm install

# Start all services
pnpm run dev
```

### **Services**
| Service | Port | Description |
|---------|------|-------------|
| **Frontend** | 3001 | React trading interface |
| **Backend** | 3002 | API gateway and services |
| **MIKEY-AI** | 3000 | AI trading assistant |
| **Data Ingestion** | 3003 | Real-time market data |

---

## 🏗️ **Architecture**

### **Multi-Service Design**
- **Frontend:** React/Vite/TypeScript trading interface
- **Backend:** Node.js/Express API gateway with Supabase integration
- **MIKEY-AI:** LangChain-powered AI trading agent
- **Data Ingestion:** Real-time market data processing pipeline
- **Smart Contracts:** Rust/Anchor Solana programs

### **Technology Stack**
- **Frontend:** React 18, Vite, Tailwind CSS, TypeScript
- **Backend:** Node.js 20+, Express.js, TypeScript, pnpm
- **Smart Contracts:** Rust, Anchor Framework, Solana
- **Database:** Supabase (PostgreSQL)
- **Oracle:** Pyth Network
- **AI:** LangChain, Multi-LLM routing

---

## 🛡️ **Enterprise-Grade Security**

### **Multi-Layer Circuit Breaker System**
- **Price Deviation Protection:** Triggers on 5% price movements
- **Liquidation Rate Control:** Limits to 100 liquidations per 5-minute period
- **Oracle Health Monitoring:** Dynamic staleness detection with multi-oracle fallback
- **Advanced Keeper Authorization:** Multi-signature requirements with performance monitoring

### **Security Features**
- **Rate Limiting:** Tiered rate limits to prevent abuse
- **Input Validation:** Comprehensive input sanitization
- **Error Handling:** Secure error management without information leakage
- **Audit Logging:** Complete transaction and access logging

---

## 🤖 **AI Integration**

### **MIKEY-AI Features**
- **Market Analysis:** Real-time market sentiment and trend analysis
- **Trading Recommendations:** AI-powered trading suggestions and strategies
- **Risk Assessment:** Automated portfolio risk evaluation
- **Multi-LLM Routing:** GPT-4, Claude, and other models for optimal responses
- **Natural Language Processing:** Chat-based trading assistance

---

## 📁 **Project Structure**

```
quantdesk/
├── frontend/          # React trading interface
├── backend/           # API services and gateway
├── MIKEY-AI/          # AI trading agent
├── data-ingestion/    # Real-time data pipeline
├── contracts/         # Solana smart contracts
├── examples/          # Community examples and demos
├── sdk/              # Public SDK components
├── scripts/          # Utility scripts
└── docs/             # Documentation
```

## 🌐 **Open Source Components**

The following components are available for community use:

### **Examples** (`examples/`)
- **Frontend UI Components** - React components for trading interfaces
- **Backend API Services** - Node.js service patterns and utilities
- **MIKEY-AI Agents** - LangChain agent implementations
- **Data Ingestion** - Real-time data processing examples
- **Smart Contract Interactions** - Solana program interaction patterns
- **Community Trading Bot** - Complete automated trading bot example
- **API Integration** - Comprehensive API client examples
- **UI Component Library** - Professional trading interface components

### **SDK** (`sdk/`)
- **TypeScript SDK** - Complete SDK for QuantDesk integration
- **API Client** - RESTful API client with TypeScript support
- **Utilities** - Helper functions and common patterns
- **Type Definitions** - Comprehensive TypeScript types

### **Scripts** (`scripts/`)
- **Development Scripts** - Environment setup and development utilities
- **Deployment Scripts** - Production deployment automation
- **Security Scripts** - Security auditing and vulnerability scanning
- **Database Scripts** - Database management and seeding utilities

---

## 📋 **Documentation**

### **Architecture & Services**
- **[Architecture Guide](ARCHITECTURE_GUIDE.md)** - Complete system architecture overview
- **[Services Overview](SERVICES_OVERVIEW.md)** - Multi-service architecture details
- **[Frontend Service](FRONTEND_SERVICE.md)** - React trading interface documentation
- **[Backend Service](BACKEND_SERVICE.md)** - API services and gateway documentation
- **[MIKEY-AI Service](MIKEY_AI_SERVICE.md)** - AI trading agent documentation

### **Community Resources**
- **[Examples](examples/README.md)** - Code examples and demos
- **[SDK](sdk/README.md)** - Software Development Kit documentation
- **[Scripts](scripts/README.md)** - Utility scripts and automation tools

### **Security & Contributing**
- **[Security Policy](SECURITY.md)** - Security policies and vulnerability reporting
- **[Contributing Guidelines](CONTRIBUTING.md)** - How to contribute to the project
- **[License](LICENSE)** - Apache License 2.0

---

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guidelines](CONTRIBUTING.md) for details.

### **Development Workflow**
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

---

## 📞 **Support**

- **Documentation:** Check the service-specific READMEs
- **Issues:** [GitHub Issues](https://github.com/quantdesk/quantdesk/issues)
- **Discussions:** [GitHub Discussions](https://github.com/quantdesk/quantdesk/discussions)
- **Security:** [SECURITY.md](SECURITY.md) for vulnerability reporting

---

## 📄 **License**

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

---

## 🙏 **Acknowledgments**

- **Solana Foundation** - For the Solana blockchain infrastructure
- **Pyth Network** - For reliable price oracle services
- **Supabase** - For database infrastructure
- **LangChain** - For AI agent framework
- **Drift Protocol** - For inspiration and patterns

---

**Built with ❤️ by the QuantDesk team**

For more information, visit [quantdesk.com](https://quantdesk.com)