# 🌐 QuantDesk Open Source Setup Guide

## 🎯 Quick Start for Open Source Contributors

This guide helps you set up QuantDesk for development and contribution.

## 📋 Prerequisites

- **Node.js 20+** - [Download](https://nodejs.org/)
- **pnpm** - `npm install -g pnpm`
- **Rust** - [Install Rust](https://rustup.rs/)
- **Solana CLI** - [Install Solana CLI](https://docs.solana.com/cli/install-solana-cli-tools)
- **Git** - [Install Git](https://git-scm.com/)

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/quantdesk/quantdesk.git
cd quantdesk
```

### 2. Install Dependencies
```bash
# Install all dependencies
pnpm install

# Or install individually
pnpm --filter=backend install
pnpm --filter=frontend install
pnpm --filter=MIKEY-AI install
pnpm --filter=data-ingestion install
```

### 3. Environment Setup
```bash
# Copy environment template
cp env.template .env

# Edit .env with your configuration
# See Configuration section below
```

### 4. Start Development Servers
```bash
# Start all services
pnpm run dev

# Or start individually
pnpm --filter=backend dev
pnpm --filter=frontend dev
pnpm --filter=MIKEY-AI dev
pnpm --filter=data-ingestion dev
```

## ⚙️ Configuration

### Required API Keys

You'll need API keys for the following services:

#### 🤖 AI Services (for MIKEY-AI)
- **OpenAI API Key** - [Get OpenAI API Key](https://platform.openai.com/api-keys)
- **Google API Key** - [Get Google API Key](https://console.cloud.google.com/)
- **Cohere API Key** - [Get Cohere API Key](https://dashboard.cohere.ai/)
- **Anthropic API Key** - [Get Anthropic API Key](https://console.anthropic.com/)

#### 🗄️ Database (Supabase)
- **Supabase URL** - [Create Supabase Project](https://supabase.com/)
- **Supabase Anon Key** - From your Supabase project settings
- **Supabase Service Role Key** - From your Supabase project settings

#### 🔗 Blockchain (Solana)
- **Solana RPC URL** - Use devnet: `https://api.devnet.solana.com`
- **Wallet Files** - Generate with `solana-keygen new`

#### 📊 Monitoring (Optional)
- **Sentry DSN** - [Get Sentry DSN](https://sentry.io/)

### Environment Variables

Edit your `.env` file with the following structure:

```bash
# AI Services
OPENAI_API_KEY=sk-your-openai-key-here
GOOGLE_API_KEY=your-google-key-here
COHERE_API_KEY=your-cohere-key-here
ANTHROPIC_API_KEY=your-anthropic-key-here

# Database
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your-anon-key-here
SUPABASE_SERVICE_ROLE_KEY=your-service-role-key-here

# Blockchain
SOLANA_NETWORK=devnet
SOLANA_RPC_URL=https://api.devnet.solana.com
QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

# Security (generate strong random strings)
JWT_SECRET=your-jwt-secret-here
SESSION_SECRET=your-session-secret-here
CSRF_SECRET=your-csrf-secret-here
```

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   MIKEY-AI      │    │  Data Ingestion │
│   (React)       │    │   (LangChain)   │    │   (Pipeline)    │
│   Port: 3001    │    │   Port: 3000    │    │   Port: 3003    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                │
                    ┌───────────▼───────────┐
                    │      Backend API      │
                    │    (Node.js/Express)  │
                    │      Port: 3002       │
                    └───────────┬───────────┘
                                │
                ┌───────────────┼───────────────┐
                │               │               │
        ┌───────▼───────┐ ┌────▼────┐ ┌────────▼────────┐
        │   Supabase    │ │  Pyth   │ │ Solana Smart     │
        │ (PostgreSQL)  │ │ Oracle  │ │   Contracts      │
        │               │ │ Network │ │                  │
        └───────────────┘ └─────────┘ └──────────────────┘
```

## 🧪 Testing

### Run Tests
```bash
# Backend tests
cd backend && pnpm test

# Frontend tests
cd frontend && pnpm test

# Smart contract tests
cd contracts && anchor test

# All tests
pnpm run test:all
```

### Test Coverage
```bash
# Backend coverage
cd backend && pnpm run test:coverage

# Frontend coverage
cd frontend && pnpm run test:coverage
```

## 🔧 Development Tools

### Recommended VS Code Extensions
- **TypeScript** - TypeScript support
- **ESLint** - Code linting
- **Prettier** - Code formatting
- **GitLens** - Git integration
- **Thunder Client** - API testing
- **Rust Analyzer** - Rust support
- **Solana** - Solana development

### Useful Commands
```bash
# Type checking
pnpm run type-check

# Linting
pnpm run lint

# Format code
pnpm run format

# Build all services
pnpm run build

# Clean dependencies
pnpm run clean
```

## 📚 Available Services

| Service | Port | Description | Technology |
|---------|------|-------------|------------|
| **Frontend** | 3001 | React trading interface | React, Vite, TypeScript |
| **Backend** | 3002 | API gateway and services | Node.js, Express, TypeScript |
| **MIKEY-AI** | 3000 | AI trading assistant | LangChain, TypeScript |
| **Data Ingestion** | 3003 | Real-time market data | Node.js, Pipeline |

## 🌐 Open Source Components

### What's Available
- ✅ **UI Components** - Reusable React components
- ✅ **API Services** - Complete API service layer
- ✅ **Utilities** - Common utility functions
- ✅ **Data Processors** - Data ingestion patterns
- ✅ **AI Interfaces** - AI agent interfaces
- ✅ **Documentation** - Comprehensive guides
- ✅ **Examples** - Working code samples
- ✅ **Scripts** - Development utilities

### What's Protected
- 🔒 **Trading Algorithms** - Proprietary strategies
- 🔒 **AI Models** - Trained models and weights
- 🔒 **Business Logic** - Revenue-generating code
- 🔒 **Smart Contracts** - Trading contract logic
- 🔒 **Analytics** - Performance metrics

## 🤝 Contributing

### Getting Started
1. **Fork** the repository
2. **Clone** your fork locally
3. **Create** a feature branch
4. **Make** your changes
5. **Test** your changes
6. **Submit** a pull request

### Development Workflow
1. **Create Issue** - Discuss your changes first
2. **Fork & Branch** - Create a feature branch
3. **Develop** - Implement your changes
4. **Test** - Ensure all tests pass
5. **Document** - Update documentation if needed
6. **Submit PR** - Submit pull request for review

### Code Standards
- **TypeScript** - Use TypeScript for all new code
- **ESLint** - Follow ESLint configuration
- **Prettier** - Use Prettier for formatting
- **Tests** - Write tests for new features
- **Documentation** - Document public APIs

## 🐛 Troubleshooting

### Common Issues

#### Service Won't Start
```bash
# Check if ports are available
lsof -i :3001
lsof -i :3002
lsof -i :3000
lsof -i :3003

# Kill processes using ports
kill -9 $(lsof -t -i:3001)
```

#### Dependencies Issues
```bash
# Clear node_modules and reinstall
rm -rf node_modules
rm -rf */node_modules
pnpm install
```

#### Solana Issues
```bash
# Check Solana configuration
solana config get

# Check wallet balance
solana balance

# Check program deployment
solana program show C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw
```

#### Database Issues
```bash
# Check Supabase connection
curl -H "apikey: YOUR_ANON_KEY" https://YOUR_PROJECT.supabase.co/rest/v1/
```

### Debug Commands
```bash
# Check service health
curl http://localhost:3002/api/health

# Check frontend
curl http://localhost:3001

# Check MIKEY-AI
curl http://localhost:3000/api/health

# Check data ingestion
curl http://localhost:3003/api/health
```

## 📞 Support

### Getting Help
- **GitHub Issues** - Bug reports and feature requests
- **GitHub Discussions** - General questions
- **Discord** - Real-time community support
- **Email** - Private or sensitive issues

### Documentation
- **README** - Start with the main README
- **API Docs** - Check API documentation
- **Examples** - Look at code examples
- **Wiki** - Check the project wiki

## 🎉 You're Ready!

You now have QuantDesk running locally and are ready to contribute to the open source project!

### Next Steps
1. **Explore** the codebase
2. **Run** the tests
3. **Try** the examples
4. **Join** the community
5. **Contribute** your first PR

---

**Happy Coding!** 🚀

For more information, visit [quantdesk.com](https://quantdesk.com) or join our [Discord community](https://discord.gg/quantdesk).
