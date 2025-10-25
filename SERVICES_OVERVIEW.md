# QuantDesk Services Architecture

## 🏗️ Multi-Service Architecture Overview

QuantDesk is built as a sophisticated multi-service platform with each service handling specific aspects of the perpetual DEX ecosystem. This document provides an overview of each service and its role in the overall system.

## 📊 Service Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Frontend      │    │   MIKEY-AI      │    │  Data Ingestion │
│   (React)       │    │   (LangChain)   │    │   (Pipeline)    │
│   Port: 3001    │    │   Port: 3000    │    │   Port: 3003    │
└─────────┬───────┘    └─────────┬───────┘    └─────────┬───────┘
          │                      │                      │
          └──────────────────────┼──────────────────────┘
                                 │
                    ┌─────────────┴─────────────┐
                    │        Backend            │
                    │   (Node.js/Express)       │
                    │      Port: 3002           │
                    └─────────────┬─────────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │     Smart Contracts       │
                    │    (Solana/Anchor)        │
                    │      On-Chain             │
                    └───────────────────────────┘
```

## 🔧 Core Services

### 1. Frontend Service (Port 3001)
**Technology**: React 18, Vite, TypeScript, Tailwind CSS

**Purpose**: 
- User-facing trading interface
- Real-time portfolio management
- Order placement and position tracking
- Market data visualization

**Key Features**:
- Responsive trading dashboard
- Real-time price charts
- Portfolio overview
- Order management interface
- Wallet integration

**Public Examples**: `examples/frontend-ui-components.tsx`

### 2. Backend Service (Port 3002)
**Technology**: Node.js, Express.js, TypeScript

**Purpose**:
- API gateway and business logic
- Database management
- Oracle price feed integration
- User authentication and authorization

**Key Features**:
- RESTful API endpoints
- WebSocket real-time updates
- Database abstraction layer
- Oracle price normalization
- Rate limiting and security

**Public Examples**: `examples/backend-api-services.ts`

### 3. MIKEY-AI Service (Port 3000)
**Technology**: LangChain, TypeScript, Multi-LLM routing

**Purpose**:
- AI-powered trading assistance
- Market analysis and insights
- Natural language trading commands
- Risk assessment and recommendations

**Key Features**:
- Multi-LLM routing (OpenAI, Anthropic, Google)
- Trading strategy analysis
- Market sentiment analysis
- Risk management suggestions
- Natural language processing

**Public Examples**: `examples/mikey-ai-agents.ts`

### 4. Data Ingestion Service (Port 3003)
**Technology**: Node.js, Pipeline Architecture

**Purpose**:
- Real-time market data collection
- Social media sentiment tracking
- Price feed aggregation
- Data normalization and storage

**Key Features**:
- Multi-source data collection
- Real-time processing pipelines
- Data quality validation
- Historical data management
- Event-driven architecture

**Public Examples**: `examples/data-ingestion-processors.ts`

### 5. Smart Contracts (On-Chain)
**Technology**: Rust, Anchor Framework, Solana

**Purpose**:
- On-chain trading logic
- Position management
- Collateral handling
- Liquidation mechanisms

**Key Features**:
- Perpetual futures trading
- Cross-margin collateral system
- Automated liquidation
- Oracle price integration
- Risk management

**Public Examples**: `examples/smart-contract-interactions.ts`

## 🔗 Service Communication

### API Gateway Pattern
The Backend service acts as the central API gateway, coordinating communication between all services:

- **Frontend ↔ Backend**: REST API + WebSocket
- **MIKEY-AI ↔ Backend**: REST API for data access
- **Data Ingestion ↔ Backend**: Event streaming
- **Backend ↔ Smart Contracts**: Solana RPC calls

### Data Flow
1. **Data Ingestion** collects market data and social sentiment
2. **Backend** processes and normalizes data, stores in database
3. **MIKEY-AI** analyzes data and provides insights
4. **Frontend** displays real-time data and trading interface
5. **Smart Contracts** execute on-chain trading logic

## 🛠️ Development Setup

### Prerequisites
- Node.js 20+
- pnpm package manager
- Solana CLI tools
- Anchor Framework

### Quick Start
```bash
# Install dependencies
pnpm install

# Start all services
pnpm run dev

# Or start individual services
cd frontend && pnpm run dev
cd backend && pnpm run dev
cd MIKEY-AI && pnpm run dev
cd data-ingestion && pnpm run dev
```

### Environment Configuration
Each service requires specific environment variables. See `.env.example` for the complete configuration template.

## 📚 Examples and Integration

### Available Examples
- **API Integration**: `examples/api-integration-example.ts`
- **Trading Bot**: `examples/community-trading-bot.ts`
- **UI Components**: `examples/ui-component-library.tsx`
- **Smart Contract**: `examples/smart-contract-interactions.ts`

### Integration Patterns
- **REST API**: Standard HTTP endpoints for data access
- **WebSocket**: Real-time updates for trading interface
- **Event Streaming**: Asynchronous data processing
- **Cross-Service**: Service-to-service communication

## 🔒 Security Considerations

### API Security
- JWT-based authentication
- Rate limiting per endpoint
- Input validation and sanitization
- CORS configuration

### Smart Contract Security
- PDA-based account isolation
- Signer verification
- Oracle price validation
- Liquidation protection

## 📈 Scalability

### Horizontal Scaling
- Stateless service design
- Load balancer ready
- Database connection pooling
- Caching strategies

### Performance Optimization
- Efficient data structures
- Minimal database queries
- Real-time data streaming
- Optimized Solana transactions

## 🤝 Contributing

Each service follows consistent patterns for:
- Code organization
- Error handling
- Logging
- Testing
- Documentation

See `CONTRIBUTING.md` for detailed contribution guidelines.

## 📞 Support

For questions about specific services:
- **Frontend**: React/Vite documentation
- **Backend**: Express.js/Node.js documentation  
- **MIKEY-AI**: LangChain documentation
- **Smart Contracts**: Anchor/Solana documentation
- **Data Ingestion**: Pipeline architecture patterns

---

*This architecture enables QuantDesk to provide a sophisticated, scalable, and maintainable perpetual DEX platform while keeping each service focused on its core responsibilities.*
