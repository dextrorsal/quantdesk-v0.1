# QuantDesk Architecture Guide

## 🏗️ Complete Service Architecture

This guide provides a comprehensive overview of QuantDesk's multi-service architecture, helping developers understand how each service works and how they integrate together.

## 📊 System Overview

QuantDesk is built as a sophisticated multi-service platform where each service handles specific aspects of the perpetual DEX ecosystem:

```
┌─────────────────────────────────────────────────────────────────┐
│                        QuantDesk Platform                       │
├─────────────────────────────────────────────────────────────────┤
│  Frontend (React)     │  MIKEY-AI (LangChain)  │  Data Pipeline │
│  Port: 3001          │  Port: 3000            │  Port: 3003    │
│  • Trading UI        │  • AI Assistant        │  • Market Data │
│  • Real-time Charts  │  • Market Analysis     │  • Sentiment   │
│  • Portfolio Mgmt    │  • Risk Assessment     │  • Price Feeds │
└───────────┬───────────┴───────────┬───────────┴───────────┬─────┘
            │                        │                       │
            └────────────────────────┼───────────────────────┘
                                     │
                    ┌─────────────────┴─────────────────┐
                    │           Backend Service          │
                    │        (Node.js/Express)           │
                    │            Port: 3002              │
                    │  • API Gateway                    │
                    │  • Business Logic                 │
                    │  • Database Management            │
                    │  • Oracle Integration             │
                    └─────────────────┬─────────────────┘
                                      │
                    ┌─────────────────┴─────────────────┐
                    │        Smart Contracts             │
                    │       (Solana/Anchor)              │
                    │         On-Chain                    │
                    │  • Trading Logic                   │
                    │  • Position Management             │
                    │  • Collateral System               │
                    │  • Liquidation Engine              │
                    └─────────────────────────────────────┘
```

## 🔧 Service Details

### 1. Frontend Service
**Purpose**: User-facing trading interface and experience

**Key Responsibilities**:
- Real-time trading dashboard
- Order placement and management
- Portfolio visualization
- Wallet integration
- Responsive design

**Technology Stack**:
- React 18 with concurrent features
- Vite for fast development
- TypeScript for type safety
- Tailwind CSS for styling
- WebSocket for real-time updates

**Integration Points**:
- Backend API for data operations
- WebSocket for real-time updates
- Solana RPC for blockchain interactions

### 2. Backend Service
**Purpose**: Central API gateway and business logic

**Key Responsibilities**:
- RESTful API endpoints
- WebSocket real-time communication
- Database management and abstraction
- User authentication and authorization
- Oracle price feed integration
- Cross-service coordination

**Technology Stack**:
- Node.js 20+ runtime
- Express.js web framework
- TypeScript for type safety
- Supabase for database and auth
- Pyth Network for oracle feeds

**Integration Points**:
- Frontend via REST API and WebSocket
- MIKEY-AI via REST API
- Data Ingestion via event streaming
- Smart Contracts via Solana RPC

### 3. MIKEY-AI Service
**Purpose**: AI-powered trading assistance and analysis

**Key Responsibilities**:
- Natural language trading commands
- Market analysis and insights
- Risk assessment and recommendations
- Trading strategy suggestions
- Personalized user assistance

**Technology Stack**:
- LangChain for LLM orchestration
- Multi-LLM routing (OpenAI, Anthropic, Google)
- TypeScript for type safety
- Vector databases for memory
- WebSocket for real-time communication

**Integration Points**:
- Backend API for data access
- Frontend for user interaction
- External APIs for market data

### 4. Data Ingestion Service
**Purpose**: Real-time market data collection and processing

**Key Responsibilities**:
- Multi-source data collection
- Real-time processing pipelines
- Data quality validation
- Historical data management
- Event-driven data distribution

**Technology Stack**:
- Node.js for processing
- Pipeline architecture
- Event streaming
- Data validation frameworks
- Caching systems

**Integration Points**:
- Backend for data distribution
- External data sources
- Database for storage

### 5. Smart Contracts
**Purpose**: On-chain trading logic and position management

**Key Responsibilities**:
- Perpetual futures trading
- Cross-margin collateral system
- Automated liquidation
- Oracle price integration
- Risk management

**Technology Stack**:
- Rust programming language
- Anchor framework
- Solana blockchain
- Pyth oracle integration
- Program-derived addresses (PDAs)

**Integration Points**:
- Backend via Solana RPC
- Oracle networks for price feeds
- User wallets for transactions

## 🔄 Data Flow Architecture

### Real-time Data Flow
1. **Data Ingestion** collects market data from multiple sources
2. **Backend** processes, validates, and stores data
3. **MIKEY-AI** analyzes data for insights and recommendations
4. **Frontend** displays real-time data to users
5. **Smart Contracts** execute on-chain trading logic

### User Interaction Flow
1. **User** interacts with Frontend interface
2. **Frontend** sends requests to Backend API
3. **Backend** processes business logic and database operations
4. **Smart Contracts** execute on-chain transactions
5. **WebSocket** streams updates back to Frontend

### AI Assistant Flow
1. **User** asks question via Frontend
2. **MIKEY-AI** processes natural language request
3. **Backend** provides relevant market data
4. **MIKEY-AI** generates AI-powered response
5. **Frontend** displays response to user

## 🛠️ Development Setup

### Prerequisites
- Node.js 20+
- pnpm package manager
- Solana CLI tools
- Anchor Framework
- Supabase account

### Quick Start
```bash
# Clone repository
git clone https://github.com/dextrorsal/quantdesk-v0.1.git
cd quantdesk-v0.1

# Install dependencies
pnpm install

# Setup environment
cp .env.example .env
# Fill in your environment variables

# Start all services
pnpm run dev
```

### Individual Service Setup
```bash
# Frontend
cd frontend && pnpm install && pnpm run dev

# Backend
cd backend && pnpm install && pnpm run dev

# MIKEY-AI
cd MIKEY-AI && pnpm install && pnpm run dev

# Data Ingestion
cd data-ingestion && pnpm install && pnpm run dev

# Smart Contracts
cd contracts && anchor build && anchor test
```

## 🔧 Configuration

### Environment Variables
Each service requires specific environment variables. See the individual service documentation for detailed configuration:

- **Frontend**: `FRONTEND_SERVICE.md`
- **Backend**: `BACKEND_SERVICE.md`
- **MIKEY-AI**: `MIKEY_AI_SERVICE.md`
- **Smart Contracts**: See `contracts/README.md`

### Service Communication
Services communicate through:
- **REST APIs** for synchronous operations
- **WebSocket** for real-time updates
- **Event Streaming** for asynchronous processing
- **Solana RPC** for blockchain interactions

## 📚 Examples and Integration

### Available Examples
- **API Integration**: `examples/api-integration-example.ts`
- **Trading Bot**: `examples/community-trading-bot.ts`
- **UI Components**: `examples/ui-component-library.tsx`
- **Smart Contract**: `examples/smart-contract-interactions.ts`
- **Data Processing**: `examples/data-ingestion-processors.ts`
- **AI Agents**: `examples/mikey-ai-agents.ts`

### Integration Patterns
- **Service-to-Service**: REST API communication
- **Real-time Updates**: WebSocket connections
- **Event-Driven**: Asynchronous processing
- **Cross-Chain**: Solana blockchain integration

## 🔒 Security Architecture

### API Security
- JWT-based authentication
- Rate limiting per endpoint
- Input validation and sanitization
- CORS configuration

### Smart Contract Security
- PDA-based account isolation
- Signer verification for all operations
- Oracle price validation
- Liquidation protection mechanisms

### Data Protection
- Encryption at rest and in transit
- Secure API key management
- Environment isolation
- Comprehensive audit logging

## 📈 Scalability Considerations

### Horizontal Scaling
- Stateless service design
- Load balancer ready architecture
- Database connection pooling
- Caching strategies

### Performance Optimization
- Efficient data structures
- Minimal database queries
- Real-time data streaming
- Optimized Solana transactions

### Monitoring and Observability
- Service health checks
- Performance metrics
- Error tracking
- User analytics

## 🤝 Contributing

### Development Workflow
1. Choose the service you want to work on
2. Read the specific service documentation
3. Set up the development environment
4. Implement your changes with tests
5. Update documentation
6. Submit pull request

### Code Standards
- TypeScript strict mode
- ESLint and Prettier configuration
- Comprehensive testing
- Documentation requirements

## 📞 Support and Resources

### Service-Specific Documentation
- **Frontend**: React, Vite, Tailwind CSS documentation
- **Backend**: Express.js, Node.js, Supabase documentation
- **MIKEY-AI**: LangChain, OpenAI, Anthropic documentation
- **Smart Contracts**: Anchor, Solana, Rust documentation

### Community Resources
- GitHub Issues for bug reports
- Discussions for questions
- Documentation for learning
- Examples for integration

---

*This architecture enables QuantDesk to provide a sophisticated, scalable, and maintainable perpetual DEX platform while keeping each service focused on its core responsibilities and allowing for independent development and scaling.*
