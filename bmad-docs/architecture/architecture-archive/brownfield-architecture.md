# QuantDesk Brownfield Architecture Document

## Introduction

This document captures the CURRENT STATE of the QuantDesk codebase, including technical debt, workarounds, and real-world patterns. It serves as a reference for AI agents working on the AI Portfolio Rebalancing enhancement.

### Document Scope

**Focused on areas relevant to: AI Portfolio Rebalancing Enhancement**

The planned enhancement will require coordination between:
- **Backend**: Portfolio analytics API, AI integration endpoints
- **Frontend**: AI recommendation dashboard, real-time portfolio updates  
- **MIKEY-AI**: Portfolio analysis engine, rebalancing recommendations
- **Smart Contracts**: Automated execution engine, batch order processing
- **Database**: Portfolio data storage, analytics tracking
- **Oracle**: Real-time price feeds for rebalancing decisions

### Change Log

| Date   | Version | Description                 | Author    |
| ------ | ------- | --------------------------- | --------- |
| 2025-10-18 | 1.0     | Initial brownfield analysis | Winston (Architect) |

## Quick Reference - Key Files and Entry Points

### Critical Files for Understanding the System

- **Main Entry**: `backend/src/server.ts` - Express API server with 34+ route handlers
- **Configuration**: `backend/src/config/environment.ts`, `.env` files
- **Core Business Logic**: `backend/src/services/supabaseDatabase.ts` - Consolidated database service
- **API Definitions**: `backend/src/routes/` - 34+ specialized route files
- **Database Models**: `backend/src/services/supabaseDatabase.ts` - TypeScript interfaces
- **Key Algorithms**: `backend/src/services/pythOracleService.ts` - Oracle price integration

### Enhancement Impact Areas

**Files that will be affected by AI Portfolio Rebalancing:**

- `backend/src/routes/portfolioAnalytics.ts` - Portfolio analytics API (EXISTS)
- `backend/src/routes/ai.ts` - AI service endpoints (EXISTS) 
- `backend/src/services/portfolioAnalyticsService.ts` - Portfolio analytics service (EXISTS)
- `frontend/src/components/PortfolioDashboard.tsx` - Portfolio management UI (EXISTS)
- `MIKEY-AI/src/agents/TradingAgent.ts` - AI trading agent (EXISTS)
- `MIKEY-AI/src/services/QuantDeskTools.ts` - QuantDesk API integration (EXISTS)
- `contracts/programs/quantdesk-perp-dex/src/instructions/order_management.rs` - Order execution (EXISTS)

## High Level Architecture

### Technical Summary

QuantDesk is a sophisticated multi-service Solana perpetual DEX platform with AI integration. The system uses a backend-centric architecture where the Express API server acts as the central hub, coordinating between frontend, AI services, smart contracts, and external data sources.

### Actual Tech Stack (from package.json analysis)

| Category  | Technology | Version | Notes                      |
| --------- | ---------- | ------- | -------------------------- |
| Runtime   | Node.js    | 20+     | Backend services           |
| Framework | Express    | 4.18.2  | Backend API server         |
| Frontend  | React      | 18.2.0  | Trading interface          |
| Database  | Supabase   | 2.58.0  | PostgreSQL with abstraction layer |
| Blockchain| Solana     | 1.87.0  | Smart contracts via Anchor |
| AI        | LangChain  | 0.3.15  | MIKEY-AI trading agent     |
| Oracle    | Pyth       | 2.0.0   | Price feeds via Hermes    |
| Caching   | Redis      | 4.6.10  | Session and data caching   |
| Real-time | Socket.io  | 4.7.4   | WebSocket communication    |

### Repository Structure Reality Check

- **Type**: Monorepo with pnpm workspace
- **Package Manager**: pnpm (NOT npm - critical constraint)
- **Notable**: Multi-service architecture with shared dependencies

## Source Tree and Module Organization

### Project Structure (Actual)

```text
quantdesk-1.0.6/
├── backend/                    # Node.js/Express API server (Port 3002)
│   ├── src/
│   │   ├── routes/            # 34+ specialized route handlers
│   │   ├── services/          # 25+ specialized services
│   │   ├── middleware/         # Auth, rate limiting, error handling
│   │   └── server.ts          # Main entry point
├── frontend/                   # React trading interface (Port 3001)
│   ├── src/
│   │   ├── components/        # 45+ React components
│   │   ├── contexts/          # React context providers
│   │   └── stores/            # Zustand state management
├── MIKEY-AI/                   # LangChain AI agent (Port 3000)
│   ├── src/
│   │   ├── agents/            # AI agent implementations
│   │   ├── services/          # AI service layer
│   │   └── api/               # AI API server
├── contracts/                  # Solana smart contracts
│   ├── programs/
│   │   └── quantdesk-perp-dex/ # Anchor program
├── data-ingestion/             # Real-time data pipeline (Port 3003)
├── admin-dashboard/            # Admin management interface
└── database/                   # Database schemas and migrations
```

### Key Modules and Their Purpose

- **Backend API Gateway**: `backend/src/server.ts` - Central coordination hub
- **Database Service**: `backend/src/services/supabaseDatabase.ts` - CRITICAL: Single abstraction layer, prevents direct Supabase usage
- **Oracle Integration**: `backend/src/services/pythOracleService.ts` - Pyth Network price feeds
- **AI Trading Agent**: `MIKEY-AI/src/agents/TradingAgent.ts` - LangChain-based trading intelligence
- **Trading Interface**: `frontend/src/components/DexTradingInterface.tsx` - Main trading UI
- **Smart Contracts**: `contracts/programs/quantdesk-perp-dex/src/lib.rs` - Solana program with 50+ instructions

## Data Models and APIs

### Data Models

**Critical**: All database operations MUST use `databaseService` from `backend/src/services/supabaseDatabase.ts`

- **User Model**: See `backend/src/services/supabaseDatabase.ts` lines 28-46
- **Position Model**: See `backend/src/services/supabaseDatabase.ts` lines 72-92  
- **Order Model**: See `backend/src/services/supabaseDatabase.ts` lines 94-115
- **Market Model**: See `backend/src/services/supabaseDatabase.ts` lines 48-70

### API Specifications

- **OpenAPI Spec**: Available via `/api/docs/swagger` endpoint
- **AI Development Endpoints**: `/api/dev/*` - Special endpoints for AI assistance
- **Portfolio Analytics**: `/api/portfolio/*` - Portfolio management endpoints (EXISTS)
- **AI Integration**: `/api/ai/*` - AI service endpoints (EXISTS)

## Technical Debt and Known Issues

### Critical Technical Debt

1. **Package Manager Constraint**: MUST use pnpm, NEVER npm (enforced in package.json)
2. **Database Access Pattern**: All database operations MUST go through `databaseService` abstraction layer
3. **Oracle Integration**: Pyth prices fetched by backend, normalized and cached (backend-centric pattern)
4. **TypeScript Errors**: Backend and admin dashboard have non-blocking TypeScript errors
5. **Missing Functions**: `execute_sql` function missing in Supabase devnet (EXPECTED, non-critical)

### Workarounds and Gotchas

- **Environment Variables**: Must set `NODE_ENV=production` even for staging (historical reason)
- **Oracle Price Format**: Prices are in scientific notation, already normalized by backend
- **WebSocket Authentication**: Uses JWT tokens from session cookies for WebSocket connections
- **RPC Load Balancing**: Backend handles RPC load balancing for Solana connections
- **Rate Limiting**: Tiered rate limits via `backend/src/middleware/rateLimiting.ts`

## Integration Points and External Dependencies

### External Services

| Service  | Purpose  | Integration Type | Key Files                      |
| -------- | -------- | ---------------- | ------------------------------ |
| Pyth Network | Price Feeds | WebSocket + REST | `backend/src/services/pythOracleService.ts` |
| Supabase | Database | REST API | `backend/src/services/supabaseDatabase.ts` |
| Solana | Blockchain | RPC + Anchor | `contracts/programs/quantdesk-perp-dex/` |
| OpenAI/Anthropic | AI Models | REST API | `MIKEY-AI/src/services/OfficialLLMRouter.ts` |

### Internal Integration Points

- **Frontend ↔ Backend**: REST API on port 3002, WebSocket for real-time updates
- **MIKEY-AI ↔ Backend**: REST API integration via `MIKEY-AI/src/services/QuantDeskTools.ts`
- **Backend ↔ Smart Contracts**: Anchor integration for transaction execution
- **All Services ↔ Database**: Single `databaseService` abstraction layer

## Development and Deployment

### Local Development Setup

1. **Prerequisites**: Node.js 20+, pnpm, Solana CLI tools
2. **Installation**: `npm run install:all` (uses pnpm internally)
3. **Start Services**: `npm run dev` (starts all services)
4. **Individual Services**:
   - Backend: `cd backend && pnpm run start:dev`
   - Frontend: `cd frontend && pnpm run dev`
   - MIKEY-AI: `cd MIKEY-AI && pnpm run dev`

### Build and Deployment Process

- **Build Command**: `npm run build` (builds all components)
- **Backend Deployment**: Railway (configured in `backend/railway.json`)
- **Frontend Deployment**: Vercel (configured in `frontend/vercel.json`)
- **Smart Contracts**: Solana devnet/mainnet deployment

## Testing Reality

### Current Test Coverage

- **Backend Tests**: Vitest configured, coverage reporting available
- **Smart Contract Tests**: Anchor test framework
- **Frontend Tests**: Playwright E2E tests configured
- **Integration Tests**: Available via `/api/dev/*` endpoints

### Running Tests

```bash
npm run test:backend      # Backend unit tests
npm run test:contracts    # Smart contract tests  
npm run test:frontend     # Frontend E2E tests
```

## Enhancement Impact Analysis - AI Portfolio Rebalancing

### Files That Will Need Modification

Based on the AI Portfolio Rebalancing PRD, these files will be affected:

**Backend Changes:**
- `backend/src/routes/portfolioAnalytics.ts` - Add AI rebalancing endpoints
- `backend/src/services/portfolioAnalyticsService.ts` - Add AI analysis integration
- `backend/src/routes/ai.ts` - Extend AI service endpoints
- `backend/src/services/websocket.ts` - Real-time rebalancing notifications

**Frontend Changes:**
- `frontend/src/components/PortfolioDashboard.tsx` - Add AI recommendation display
- `frontend/src/components/DexTradingInterface.tsx` - Add rebalancing controls
- `frontend/src/contexts/MarketContext.tsx` - Real-time portfolio updates

**MIKEY-AI Changes:**
- `MIKEY-AI/src/agents/TradingAgent.ts` - Add portfolio analysis capabilities
- `MIKEY-AI/src/services/QuantDeskTools.ts` - Portfolio data integration
- `MIKEY-AI/src/services/PortfolioAnalysisService.ts` - NEW: Portfolio analysis service

**Smart Contract Changes:**
- `contracts/programs/quantdesk-perp-dex/src/instructions/order_management.rs` - Batch order execution
- `contracts/programs/quantdesk-perp-dex/src/instructions/position_management.rs` - Cross-collateral rebalancing

### New Files/Modules Needed

- `backend/src/services/aiPortfolioService.ts` - AI portfolio analysis service
- `backend/src/routes/aiRebalancing.ts` - AI rebalancing endpoints
- `frontend/src/components/AIRebalancingDashboard.tsx` - AI recommendation UI
- `MIKEY-AI/src/services/PortfolioAnalysisService.ts` - Portfolio analysis engine
- `contracts/programs/quantdesk-perp-dex/src/instructions/batch_execution.rs` - Batch order execution

### Integration Considerations

- **AI ↔ Database**: Portfolio data access via existing `databaseService`
- **Oracle ↔ Smart Contracts**: Real-time price feeds for rebalancing execution
- **Frontend ↔ Backend**: WebSocket for real-time AI recommendation updates
- **Smart Contracts ↔ Backend**: Transaction monitoring and execution results
- **All Departments**: Database for audit trail and analytics tracking

## Appendix - Useful Commands and Scripts

### Frequently Used Commands

```bash
npm run dev              # Start all services in development
npm run build            # Build all components
npm run test:all         # Run all tests
npm run lint             # Lint all code
pnpm --filter=* dev     # Start services via pnpm workspace
```

### Debugging and Troubleshooting

- **Backend Logs**: Check `backend/logs/` for application logs
- **AI Development**: Use `/api/dev/codebase-structure` endpoint for system info
- **Database Issues**: Use `databaseService.healthCheck()` method
- **Oracle Issues**: Check `pythOracleService` connection status

### AI Assistant Integration

The system provides special endpoints for AI development assistance:

- **System Architecture**: `GET /api/dev/codebase-structure`
- **Market Data**: `GET /api/dev/market-summary`  
- **User Portfolio**: `GET /api/dev/user-portfolio/:wallet`
- **API Documentation**: `GET /api/docs/swagger`

These endpoints return structured, machine-readable data about the system architecture, making it easier for AI assistants to understand the codebase and provide accurate suggestions for the AI Portfolio Rebalancing enhancement.
