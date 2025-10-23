I'll execute the document-project task to create comprehensive project documentation.


[1 tool called]

Now I'll execute the document-project task. First, let me check if there's a PRD or requirements document in the project context.


[4 tools called]

Perfect! I found a comprehensive PRD document. Now I'll conduct the project analysis focusing on the areas relevant to the Solana DEX Trading Platform enhancement described in the PRD.


[5 tools called]

Now let me examine the key entry points and architecture files to understand the current system structure:


[4 tools called]

Now let me examine the database schema and key configuration files:


[3 tools called]

Now let me examine some key service files and the data ingestion service:


[3 tools called]

Now let me examine some key frontend components and the existing architecture documentation:


[3 tools called]

Now I have comprehensive understanding of the QuantDesk project. Let me create the brownfield architecture document focusing on the areas relevant to the Solana DEX Trading Platform enhancement described in the PRD.

# QuantDesk Solana DEX Trading Platform Brownfield Architecture Document

## Introduction

This document captures the CURRENT STATE of the QuantDesk Solana DEX Trading Platform codebase, including technical debt, workarounds, and real-world patterns. It serves as a reference for AI agents working on enhancements to create a unified trading experience that eliminates the need for traders to manage multiple tabs and platforms.

### Document Scope

**Focused on areas relevant to: Solana DEX Trading Platform with Social Media Integration, News Sentiment Analysis, Alpha Channel Integration, and Unified Dashboard**

### Change Log

| Date   | Version | Description                 | Author    |
| ------ | ------- | --------------------------- | --------- |
| 2025-01-27 | 1.0     | Initial brownfield analysis | Winston (Architect) |

## Quick Reference - Key Files and Entry Points

### Critical Files for Understanding the System

- **Main Entry**: `backend/src/server.ts` - Express server with comprehensive API
- **Configuration**: `backend/src/config/environment.ts`, `.env` files
- **Core Business Logic**: `backend/src/services/supabaseDatabase.ts`, `backend/src/services/pythOracleService.ts`
- **API Definitions**: `backend/src/routes/` - 50+ API endpoints
- **Database Models**: `database/schema.sql` - PostgreSQL schema with TimescaleDB
- **Key Algorithms**: `backend/src/services/pythOracleService.ts` - Oracle price processing
- **Smart Contracts**: `contracts/programs/quantdesk-perp-dex/src/lib.rs` - Solana program
- **Frontend Entry**: `frontend/src/main.tsx` - React application with wallet integration
- **AI Service**: `MIKEY-AI/src/api/index.ts` - LangChain-powered AI agent

### Enhancement Impact Areas

Based on the PRD requirements, these files/modules will be affected by the planned enhancement:

- **Backend API Routes**: New endpoints for social media, news, alpha channels
- **Database Schema**: New tables for news articles, social media posts, alpha channel messages
- **Frontend Components**: New unified dashboard components
- **AI Service**: Enhanced sentiment analysis and alpha channel processing
- **Data Ingestion**: New collectors for Twitter, Discord, Telegram, news feeds

## High Level Architecture

### Technical Summary

QuantDesk is a sophisticated multi-service Solana DEX platform with enterprise-grade security, real-time data ingestion, AI-powered trading assistance, and professional-grade infrastructure. The platform is designed to eliminate the need for traders to manage multiple tabs and platforms.

### Actual Tech Stack (from package.json analysis)

| Category  | Technology | Version | Notes                      |
| --------- | ---------- | ------- | -------------------------- |
| Runtime   | Node.js    | 20+     | Backend services           |
| Framework | Express    | 4.18.2  | Backend API server         |
| Frontend  | React      | 18.2.0  | Vite-based frontend        |
| Database  | PostgreSQL | 13+     | Supabase with TimescaleDB  |
| Blockchain| Solana     | 1.87.0  | Web3.js integration        |
| Smart Contracts | Rust | 1.70+ | Anchor Framework           |
| AI        | LangChain  | 0.3.15  | MIKEY-AI service           |
| Oracle    | Pyth Network | 2.0.0 | Real-time price feeds      |
| Cache     | Redis      | 4.6.10  | Session and data caching   |
| Package Manager | pnpm | Latest | Used throughout project     |

### Repository Structure Reality Check

- **Type**: Monorepo with multi-service architecture
- **Package Manager**: pnpm (CRITICAL: Always use pnpm, never npm)
- **Notable**: Each service has independent package.json with shared dependencies

## Source Tree and Module Organization

### Project Structure (Actual)

```text
quantdesk-1.0.6/
├── backend/                    # Node.js/Express API server (Port 3002)
│   ├── src/
│   │   ├── server.ts          # Main Express server with 50+ routes
│   │   ├── services/          # Business logic services
│   │   │   ├── supabaseDatabase.ts    # CRITICAL: Single database abstraction
│   │   │   ├── pythOracleService.ts  # Oracle price processing
│   │   │   └── websocket.ts          # Real-time communication
│   │   ├── routes/            # API route handlers
│   │   ├── middleware/         # Auth, rate limiting, error handling
│   │   └── config/            # Environment configuration
│   └── package.json           # Backend dependencies
├── frontend/                  # React/Vite frontend (Port 3001)
│   ├── src/
│   │   ├── main.tsx          # React application entry
│   │   ├── App.tsx           # Main app with wallet integration
│   │   ├── components/       # React components
│   │   └── contexts/         # React context providers
│   └── package.json          # Frontend dependencies
├── MIKEY-AI/                 # AI service (Port 3000)
│   ├── src/api/index.ts      # LangChain AI agent
│   └── package.json          # AI dependencies
├── data-ingestion/           # Data pipeline (Port 3003)
│   ├── src/collectors/       # Data collection services
│   └── package.json          # Data ingestion dependencies
├── contracts/                # Solana smart contracts
│   ├── programs/quantdesk-perp-dex/src/lib.rs  # Main Solana program
│   └── package.json          # Anchor dependencies
├── database/                 # Database schemas
│   └── schema.sql            # PostgreSQL schema with TimescaleDB
└── docs/                     # Comprehensive documentation
    ├── architecture.md       # System architecture
    └── prd.md               # Product requirements document
```

### Key Modules and Their Purpose

- **Backend Service**: `backend/src/server.ts` - Centralized API gateway with comprehensive trading functionality
- **Database Service**: `backend/src/services/supabaseDatabase.ts` - CRITICAL: Single abstraction layer, prevents direct Supabase usage
- **Oracle Service**: `backend/src/services/pythOracleService.ts` - Real-time Pyth Network price feeds with caching
- **Smart Contracts**: `contracts/programs/quantdesk-perp-dex/src/lib.rs` - Solana program with 50+ instructions
- **Frontend App**: `frontend/src/App.tsx` - React app with Solana wallet integration
- **AI Service**: `MIKEY-AI/src/api/index.ts` - LangChain-powered trading assistant
- **Data Pipeline**: `data-ingestion/src/collectors/` - Real-time data collection services

## Data Models and APIs

### Data Models

Instead of duplicating, reference actual model files:

- **User Model**: See `backend/src/services/supabaseDatabase.ts` - User interface with wallet authentication
- **Market Model**: See `backend/src/services/supabaseDatabase.ts` - Market interface with Pyth integration
- **Position Model**: See `backend/src/services/supabaseDatabase.ts` - Position interface with health factors
- **Order Model**: See `backend/src/services/supabaseDatabase.ts` - Order interface with advanced types
- **Trade Model**: See `backend/src/services/supabaseDatabase.ts` - Trade interface with PnL tracking
- **Database Schema**: See `database/schema.sql` - Complete PostgreSQL schema

### API Specifications

- **OpenAPI Spec**: Available at `/api/docs/swagger` endpoint
- **Development Endpoints**: `/api/dev/*` - Architecture introspection for AI assistants
- **Core Trading APIs**: `/api/positions/*`, `/api/orders/*`, `/api/trades/*` - Position and order management
- **Oracle APIs**: `/api/oracle/*` - Pyth Network price feeds
- **AI APIs**: `/api/ai/*`, `/api/chat/*` - MIKEY-AI integration
- **Admin APIs**: `/api/admin/*` - Administrative functions

## Technical Debt and Known Issues

### Critical Technical Debt

1. **Missing Social Media Integration**: No Twitter API, Discord, or Telegram integration yet - needs implementation
2. **Missing News Integration**: No real-time news aggregation or sentiment analysis - needs implementation
3. **Missing Unified Dashboard**: No single interface combining all data sources - needs implementation
4. **Data Ingestion Service**: Currently minimal implementation, needs expansion for social media feeds
5. **AI Service**: Basic implementation, needs enhancement for sentiment analysis and alpha channel processing

### Workarounds and Gotchas

- **Package Manager**: ALWAYS use pnpm, never npm - this is critical for the project
- **Database Access**: Always use `databaseService` from `backend/src/services/supabaseDatabase.ts`, never direct Supabase calls
- **Oracle Prices**: Prices are in scientific notation and already normalized by backend - don't apply exponent again
- **Environment Variables**: Backend loads from `backend/.env`, not root `.env`
- **Smart Contract**: Program ID is `C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw` on devnet
- **WebSocket**: Uses both Socket.IO and native WebSocket for different purposes

## Integration Points and External Dependencies

### External Services

| Service  | Purpose  | Integration Type | Key Files                      |
| -------- | -------- | ---------------- | ------------------------------ |
| Pyth Network | Oracle prices | WebSocket + REST API | `backend/src/services/pythOracleService.ts` |
| Supabase | Database | REST API | `backend/src/services/supabaseDatabase.ts` |
| Solana RPC | Blockchain | Web3.js | `frontend/src/App.tsx` |
| Twitter API | Social media | REST API | **NOT IMPLEMENTED YET** |
| Discord API | Alpha channels | Bot API | **NOT IMPLEMENTED YET** |
| Telegram API | Alpha channels | Bot API | **NOT IMPLEMENTED YET** |
| News APIs | News aggregation | REST API | **NOT IMPLEMENTED YET** |

### Internal Integration Points

- **Frontend-Backend Communication**: REST API on port 3002, expects specific headers
- **AI Service Integration**: Backend calls MIKEY-AI service on port 3000
- **Data Ingestion**: Independent service on port 3003, communicates via Redis
- **Smart Contract Integration**: Frontend uses Anchor framework to interact with Solana program

## Development and Deployment

### Local Development Setup

1. **Prerequisites**: Node.js 20+, pnpm, Solana CLI tools
2. **Installation**: `npm run install:all` (uses pnpm internally)
3. **Start Services**: `npm run dev` (starts all services)
4. **Individual Services**:
   - Backend: `cd backend && pnpm run dev`
   - Frontend: `cd frontend && pnpm run dev`
   - MIKEY-AI: `cd MIKEY-AI && pnpm run dev`
   - Data Ingestion: `cd data-ingestion && pnpm run dev`

### Build and Deployment Process

- **Build Command**: `npm run build` (builds all services)
- **Deployment**: 
  - Frontend: Vercel (automatic on push)
  - Backend: Vercel (automatic on push)
  - AI Service: Railway (configured)
  - Smart Contracts: Solana devnet/mainnet
- **Environments**: Development (local), Production (Vercel/Railway)

## Testing Reality

### Current Test Coverage

- **Unit Tests**: Minimal coverage, mostly in smart contracts
- **Integration Tests**: Basic API testing in backend
- **E2E Tests**: None implemented
- **Smart Contract Tests**: Comprehensive test suite in `contracts/tests/`
- **Manual Testing**: Primary QA method for trading functionality

### Running Tests

```bash
# Smart contract tests
cd contracts && anchor test

# Backend tests
cd backend && pnpm test

# Frontend tests
cd frontend && pnpm test
```

## Enhancement Impact Analysis

### Files That Will Need Modification

Based on the PRD requirements for social media integration, news sentiment analysis, and alpha channel integration:

- `backend/src/server.ts` - Add new API routes for social media, news, alpha channels
- `backend/src/services/supabaseDatabase.ts` - Add methods for new data types
- `database/schema.sql` - Add tables for news articles, social media posts, alpha channel messages
- `frontend/src/App.tsx` - Add new routes for unified dashboard
- `frontend/src/components/` - Create new components for unified data display
- `MIKEY-AI/src/api/index.ts` - Enhance AI service for sentiment analysis
- `data-ingestion/src/collectors/` - Add new collectors for Twitter, Discord, Telegram, news

### New Files/Modules Needed

- `backend/src/routes/socialMedia.ts` - Social media API endpoints
- `backend/src/routes/news.ts` - News aggregation API endpoints
- `backend/src/routes/alphaChannels.ts` - Alpha channel API endpoints
- `backend/src/services/sentimentAnalysis.ts` - Sentiment analysis service
- `frontend/src/components/UnifiedDashboard.tsx` - Unified data dashboard
- `data-ingestion/src/collectors/twitter-collector.js` - Twitter data collection
- `data-ingestion/src/collectors/discord-collector.js` - Discord data collection
- `data-ingestion/src/collectors/telegram-collector.js` - Telegram data collection
- `data-ingestion/src/collectors/news-collector.js` - News data collection

### Integration Considerations

- Will need to integrate with existing auth middleware for API access
- Must follow existing response format in API endpoints
- Need to integrate with existing WebSocket system for real-time updates
- Must use existing `databaseService` abstraction layer
- Need to integrate with existing AI service for sentiment analysis
- Must follow existing error handling patterns

## Appendix - Useful Commands and Scripts

### Frequently Used Commands

```bash
# Start all services
npm run dev

# Start individual services
cd backend && pnpm run dev
cd frontend && pnpm run dev
cd MIKEY-AI && pnpm run dev
cd data-ingestion && pnpm run dev

# Build all services
npm run build

# Run tests
cd contracts && anchor test
cd backend && pnpm test

# Database operations
cd backend && pnpm run init:devnet
```

### Debugging and Troubleshooting

- **Logs**: Check `backend/logs/` for application logs
- **Debug Mode**: Set `DEBUG=app:*` for verbose logging
- **API Testing**: Use `/api/dev/*` endpoints for system introspection
- **Common Issues**: 
  - Always use pnpm, not npm
  - Check environment variables in `backend/.env`
  - Verify Solana program deployment on devnet
  - Check Supabase connection and RLS policies

### AI Development Endpoints

The backend provides special endpoints optimized for AI assistants:

```bash
# Get system architecture
curl http://localhost:3002/api/dev/codebase-structure

# Get market data structure
curl http://localhost:3002/api/dev/market-summary

# Get API documentation
curl http://localhost:3002/api/docs/swagger
```

---

**Document Status**: Complete - Ready for Enhancement Implementation  
**Last Updated**: January 27, 2025  
**Next Review**: After Phase 2 Implementation  
**Implementation**: Core Platform Ready (85%), AI Tools Integration Phase 2 (0%)

This brownfield architecture document provides a comprehensive understanding of the current QuantDesk system state, enabling AI agents to effectively implement the social media integration, news sentiment analysis, alpha channel integration, and unified dashboard features described in the PRD.