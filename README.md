# QuantDesk - Professional Perpetual Trading Platform

🚀 **A comprehensive Solana-based perpetual DEX with advanced trading features, AI integration, and professional-grade infrastructure.**

## 🏗️ Project Structure

```
quantdesk/
├── 📱 frontend/           # React + Vite frontend application
├── 🔧 backend/            # Node.js + Express API server
├── 👨‍💼 admin-dashboard/    # Admin management interface
├── 📊 data-ingestion/     # Real-time data pipeline
├── 🤖 MIKEY-AI/          # AI trading assistant
├── 📄 contracts/         # Solana smart contracts
├── 🗄️ database/          # Database schemas and migrations
├── 📚 docs/              # Comprehensive documentation
├── 🛠️ scripts/           # Utility scripts and tools
├── ⚙️ config/            # Configuration files
└── 🧪 tests/             # Test suites
```

## 🚀 Quick Start

### Prerequisites
- Node.js 20+
- Docker (optional)
- Solana CLI tools

### Installation
```bash
# Install all dependencies
npm run install:all

# Start development servers
npm run dev
```

### Services
- **Frontend**: http://localhost:3000
- **Backend**: http://localhost:3001
- **Admin Dashboard**: http://localhost:3002
- **Data Ingestion**: Port 3003

## 📚 Documentation

- **[Getting Started](docs/getting-started/)** - Setup and installation guides
- **[Architecture](docs/architecture/)** - System design and technical overview
- **[API Documentation](docs/api/)** - REST API reference
- **[Deployment](docs/deployment/)** - Production deployment guides
- **[Project Status](docs/project-status/)** - Current development status

## 🛠️ Development

### Available Scripts
```bash
npm run dev              # Start all services in development
npm run build            # Build all components
npm run test             # Run all tests
npm run lint             # Lint all code
npm run type-check       # TypeScript type checking
```

### CI/CD Pipeline
- ✅ **Automated builds** on push to main
- ✅ **Type checking** and linting
- ✅ **Docker image builds**
- ✅ **Deployment ready**

## 🌐 Deployment

### Backend (Railway)
- **Status**: ✅ Configured
- **Config**: `backend/railway.json`
- **Auto-deploy**: On push to main

### Frontend (Vercel)
- **Status**: ✅ Configured  
- **Auto-deploy**: On push to main

## 🔧 Key Features

- **Perpetual Trading**: Advanced DEX with leverage
- **AI Integration**: MIKEY-AI trading assistant
- **Real-time Data**: Live price feeds and market data
- **Admin Dashboard**: Comprehensive management interface
- **Smart Contracts**: Solana program integration
- **Professional UI**: Modern, responsive design

## 📊 Project Status

- ✅ **CI/CD Pipeline**: Fixed and working
- ✅ **Frontend Build**: Successful
- ⚠️ **Backend**: TypeScript errors (non-blocking)
- ⚠️ **Admin Dashboard**: TypeScript errors (non-blocking)
- ✅ **Deployment**: Railway + Vercel configured

## 🤝 Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## 📄 License

MIT License - see [LICENSE](LICENSE) file.

---

**Built with ❤️ for the Solana ecosystem**