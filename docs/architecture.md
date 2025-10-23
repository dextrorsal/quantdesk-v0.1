# QuantDesk Architecture Documentation

## Executive Summary

QuantDesk implements a sophisticated multi-service architecture optimized for Solana-based perpetual trading. The platform features enterprise-grade security, real-time data ingestion, AI-powered trading assistance, and professional-grade infrastructure designed for institutional and retail traders.

## Architecture Overview

### Multi-Service Architecture

| Service | Port | Technology | Purpose |
|---------|------|------------|---------|
| **Backend** | 3002 | Node.js/Express/TypeScript | API Gateway, Database, Oracle |
| **Frontend** | 3001 | React/Vite/TypeScript | Trading Interface, Portfolio |
| **MIKEY-AI** | 3000 | LangChain/TypeScript | AI Trading Agent |
| **Data Ingestion** | 3003 | Node.js/Pipeline | Real-time Data Collection |

### Core Design Principles

1. **Backend-Centric Oracle**: Pyth prices fetched by backend, normalized and cached
2. **Consolidated Database Service**: Single abstraction layer prevents direct Supabase usage
3. **Multi-Service Coordination**: Services communicate via backend API gateway
4. **Enterprise-Grade Security**: Multi-layer security with comprehensive monitoring

## Technology Stack

### Core Technologies
- **Backend**: Node.js 20+, Express.js, TypeScript, pnpm
- **Frontend**: React 18, Vite, Tailwind CSS, TypeScript
- **Smart Contracts**: Rust, Anchor Framework, Solana
- **Database**: Supabase (PostgreSQL)
- **Oracle**: Pyth Network
- **AI**: LangChain, Multi-LLM routing

### Deployment Infrastructure
- **Frontend/Backend**: Vercel
- **Optional**: Railway for backend
- **Smart Contracts**: Solana devnet/mainnet
- **Monitoring**: Grafana dashboards

## Service Architecture

### Backend Service (Port 3002)
- **API Gateway**: Centralized API management
- **Database Service**: Supabase abstraction layer
- **Oracle Integration**: Pyth Network price feeds
- **Authentication**: Multi-factor authentication
- **Rate Limiting**: Tiered rate limits
- **Error Handling**: Custom error classes

### Frontend Service (Port 3001)
- **Trading Interface**: Professional trading terminal
- **Portfolio Management**: Real-time portfolio tracking
- **User Dashboard**: Account management and analytics
- **Responsive Design**: Mobile and desktop optimized

### MIKEY-AI Service (Port 3000)
- **AI Trading Agent**: LangChain-powered assistant
- **Market Analysis**: Real-time market intelligence
- **Trading Recommendations**: AI-powered suggestions
- **Multi-LLM Routing**: Intelligent LLM selection

### Data Ingestion Service (Port 3003)
- **Real-time Data**: Market data collection
- **Data Processing**: Data normalization and storage
- **Pipeline Management**: Data flow orchestration
- **Monitoring**: Data quality and pipeline health

## Security Architecture

### Enterprise-Grade Security Implementation
- **Multi-Layer Security**: Defense in depth strategy
- **Multi-Factor Authentication**: Enhanced user verification
- **Dynamic Oracle Staleness Protection**: Real-time oracle validation
- **Comprehensive Monitoring**: Real-time security monitoring
- **Audit Trails**: Complete transaction logging

### Security Validation
- **QA Score**: 95/100 security validation
- **Status**: PASS - Enterprise-grade security achieved
- **Last Updated**: October 18, 2025

## Database Architecture

### Supabase Integration
- **PostgreSQL**: Primary database
- **Real-time Subscriptions**: Live data updates
- **Row Level Security**: Data access control
- **API Integration**: RESTful and GraphQL APIs

### Database Schema

#### Core Tables
- **users**: Wallet-based authentication with KYC, risk levels, referral tracking
- **markets**: Perpetual contract configuration with Pyth price feeds
- **positions**: Trading positions with health factors and liquidation tracking
- **orders**: Advanced order types (market, limit, stop-loss, take-profit, trailing-stop)
- **trades**: Trade execution history with PnL tracking
- **user_balances**: Multi-asset balance management
- **funding_payments**: Funding rate calculations and payments
- **liquidations**: Liquidation events and insurance fund usage
- **chat_channels**: Multi-channel chat system
- **chat_messages**: Message history with mentions and system announcements
- **system_events**: Monitoring and debugging events

#### Data Types
- **Custom ENUMs**: `position_side`, `order_type`, `order_status`, `trade_side`, `liquidation_type`
- **JSONB Fields**: Flexible metadata storage for extensibility
- **TimescaleDB**: Time-series data for market data and analytics
- **UUID Primary Keys**: Distributed system compatibility

#### Performance Optimizations
- **Indexes**: Optimized for trading queries (user_id, market_id, created_at)
- **Partitioning**: Time-based partitioning for high-volume tables
- **Connection Pooling**: Efficient database connection management

## Smart Contract Architecture

### Solana Program Structure
- **Program ID**: `HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso`
- **Anchor Framework**: Rust-based smart contracts
- **Modular Design**: Organized instruction modules by domain

### Contract Components
- **Market Management**: `initialize_market()`, `update_oracle_price()`, `settle_funding()`
- **Position Management**: `open_position()`, `close_position()` - Critical trading functions
- **Order Management**: `place_order()`, `cancel_order()`, `execute_order()` - Advanced order types
- **Collateral Management**: `deposit_native_sol()`, `withdraw_native_sol()` - SOL operations
- **Vault Management**: `initialize_token_vault()`, `deposit_tokens()`, `withdraw_tokens()` - Token operations
- **User Account Management**: `create_user_account()`, `update_user_account()` - Account lifecycle
- **Security Management**: Enterprise-grade security controls and validation

### Account Structures
- **UserAccount**: User state management with sub-accounts
- **TokenVault**: Token vault with authority and deposit tracking
- **ProtocolSolVault**: Protocol-level SOL vault management
- **Market**: Market configuration and state
- **Position**: Position data with health factors
- **Order**: Order management with advanced types

## AI Architecture

### MIKEY-AI Implementation
- **LangChain Framework**: AI agent orchestration
- **Multi-LLM Routing**: Intelligent model selection
- **Tool Integration**: Trading and analysis tools
- **Memory Management**: Context and conversation memory

### AI Capabilities
- **Market Analysis**: Real-time market intelligence
- **Trading Recommendations**: AI-powered suggestions
- **Risk Assessment**: Automated risk analysis
- **User Assistance**: Natural language interaction

### AI Service Architecture
- **Port 3000**: Dedicated AI service
- **API Integration**: `/api/ai/*`, `/api/chat/*` endpoints
- **WebSocket Support**: Real-time AI interactions
- **Tool Ecosystem**: Trading tools, market analysis, portfolio management
- **Memory Persistence**: Conversation and context management

## Monitoring and Observability

### Monitoring Stack
- **Grafana Dashboards**: Real-time monitoring
- **Prometheus Metrics**: System performance tracking
- **Log Aggregation**: Centralized logging
- **Alert Management**: Proactive alerting

### Key Metrics
- **Performance**: Response times, throughput
- **Security**: Authentication, authorization
- **Business**: Trading volume, user activity
- **Infrastructure**: Resource utilization, health

## Deployment Architecture

### Production Deployment
- **Frontend**: Vercel CDN distribution
- **Backend**: Vercel serverless functions
- **Database**: Supabase managed PostgreSQL
- **Smart Contracts**: Solana mainnet deployment

### Development Environment
- **Local Development**: Docker Compose setup
- **Testing**: Comprehensive test suites
- **CI/CD**: Automated deployment pipelines
- **Quality Gates**: QA validation checkpoints

## Scalability Considerations

### Horizontal Scaling
- **Microservices**: Independent service scaling
- **Load Balancing**: Traffic distribution
- **Caching**: Redis caching layer
- **CDN**: Content delivery optimization

### Performance Optimization
- **Database Indexing**: Query optimization
- **Connection Pooling**: Database connection management
- **Caching Strategies**: Multi-level caching
- **Code Splitting**: Frontend optimization

## Integration Architecture

### External Integrations
- **Pyth Network**: Oracle price feeds
- **Solana RPC**: Blockchain interaction
- **Supabase**: Database and authentication
- **Vercel**: Deployment and hosting

### API Architecture

#### Core API Endpoints
- **Authentication**: `/api/auth/*` - SIWS, OAuth, JWT management
- **Trading**: `/api/positions/*`, `/api/orders/*`, `/api/trades/*` - Position and order management
- **Markets**: `/api/markets/*` - Market data and configuration
- **Oracle**: `/api/oracle/*` - Pyth Network price feeds
- **Portfolio**: `/api/portfolio/*` - User portfolio analytics
- **AI Agent**: `/api/ai/*`, `/api/chat/*` - MIKEY-AI integration
- **Admin**: `/api/admin/*` - Administrative functions
- **Webhooks**: `/api/webhooks/*` - Event subscriptions

#### Real-time Communication
- **WebSocket**: Live market data, position updates, order status
- **Socket.IO**: Chat system, notifications, system events
- **Rate Limiting**: Tiered limits (100/min for trading, 1000/min for data)

#### API Documentation
- **OpenAPI/Swagger**: `/api/docs/swagger` - Complete API specification
- **AI-Optimized**: Structured for AI assistant integration
- **Development Endpoints**: `/api/dev/*` - Architecture introspection

## Security Considerations

### Data Protection
- **Encryption**: Data encryption at rest and in transit
- **Access Control**: Role-based access management
- **Audit Logging**: Comprehensive activity logging
- **Compliance**: Regulatory compliance measures

### Smart Contract Security
- **Code Audits**: Comprehensive security audits
- **Input Validation**: Comprehensive input validation
- **Access Control**: Program-level access control
- **Upgrade Mechanisms**: Secure upgrade procedures

## Future Architecture Considerations

### Planned Enhancements
- **Mobile Applications**: Native mobile apps
- **Cross-Chain Integration**: Multi-blockchain support
- **Advanced AI**: Enhanced AI capabilities
- **Institutional Features**: Enterprise-grade features

### Scalability Roadmap
- **Microservices Evolution**: Further service decomposition
- **Event-Driven Architecture**: Asynchronous processing
- **Advanced Caching**: Multi-tier caching strategies
- **Global Distribution**: Multi-region deployment

## Current Implementation Status

### Production Ready Components
- ✅ **Backend Service**: Full API implementation with 50+ endpoints
- ✅ **Frontend Service**: Complete trading interface with portfolio management
- ✅ **Smart Contracts**: Deployed on Solana devnet with comprehensive instruction set
- ✅ **Database Schema**: Production-ready PostgreSQL schema with TimescaleDB
- ✅ **Security Architecture**: Enterprise-grade security with 95/100 QA score
- ✅ **Oracle Integration**: Pyth Network integration with real-time price feeds
- ✅ **AI Service**: MIKEY-AI with LangChain integration

### Deployment Status
- **Frontend**: Deployed on Vercel (Port 3001)
- **Backend**: Deployed on Vercel (Port 3002)
- **AI Service**: Deployed on Railway (Port 3000)
- **Data Ingestion**: Local deployment (Port 3003)
- **Smart Contracts**: Deployed on Solana devnet
- **Database**: Supabase PostgreSQL with production schema

### Active Features
- **Trading**: Position management, order placement, real-time execution
- **Portfolio**: Multi-asset portfolio tracking and analytics
- **Chat System**: Multi-channel chat with mentions and system announcements
- **Admin Panel**: Complete administrative interface
- **Webhooks**: Event subscription system
- **API Documentation**: OpenAPI/Swagger specification
- **Monitoring**: Grafana dashboards and system metrics

---

**Architecture Status**: Production Ready  
**Last Updated**: October 18, 2025  
**Next Review**: November 2025  
**Implementation**: 95% Complete - Enterprise Grade
