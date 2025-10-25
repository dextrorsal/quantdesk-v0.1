# QuantDesk Architecture Overview

## System Architecture

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

## Data Flow

1. **User Request** → Frontend (React)
2. **API Call** → Backend (Express/TypeScript)
3. **Database Query** → Supabase (PostgreSQL)
4. **Oracle Price** → Pyth Network → Backend Gateway
5. **Smart Contract** → Solana Blockchain
6. **Response** → Frontend → User

## Key Design Decisions

### Backend-Centric Oracle Integration
- Pyth Network prices fetched by backend service
- Gateway API pattern for oracle data
- Prices normalized and cached in backend
- Frontend consumes oracle data via backend API

### Consolidated Database Service
- Single `databaseService` abstraction layer
- All database operations go through `supabaseDatabase.ts`
- Prevents direct Supabase client usage
- Consistent error handling and logging

### Multi-Service Architecture
- **Frontend**: React + Vite (Port 3001)
- **Backend**: Node.js + Express (Port 3002)
- **MIKEY-AI**: LangChain trading agent (Port 3000)
- **Data Ingestion**: Real-time pipeline (Port 3003)

## Technology Stack

### Backend
- **Runtime**: Node.js 20+
- **Framework**: Express.js
- **Language**: TypeScript
- **Package Manager**: pnpm
- **Database**: Supabase (PostgreSQL)
- **Oracle**: Pyth Network
- **Blockchain**: Solana

### Frontend
- **Framework**: React 18
- **Build Tool**: Vite
- **Styling**: Tailwind CSS
- **Language**: TypeScript

### Smart Contracts
- **Language**: Rust
- **Framework**: Anchor
- **Blockchain**: Solana
- **Oracle**: Pyth Network integration

### AI Agent
- **Framework**: LangChain
- **Language**: TypeScript
- **LLM Routing**: Multi-provider support
- **Tools**: QuantDesk API integration

## Development Workflow

1. **Local Development**
   - Start services: `npm run dev`
   - Backend: `cd backend && pnpm run start:dev`
   - Frontend: `cd frontend && pnpm run dev`
   - Smart contracts: `cd contracts && anchor build`

2. **Testing**
   - Backend: `cd backend && pnpm run test`
   - Smart contracts: `cd contracts && anchor test`
   - Integration: Use `/api/dev/*` endpoints

3. **Deployment**
   - Backend: Vercel/Railway
   - Frontend: Vercel
   - Smart contracts: Solana mainnet/devnet

## AI Assistant Integration

The system provides special endpoints for AI development assistance:

- **Codebase Structure**: `/api/dev/codebase-structure`
- **Market Summary**: `/api/dev/market-summary`
- **User Portfolio**: `/api/dev/user-portfolio/:wallet`
- **API Documentation**: `/api/docs/swagger`

These endpoints return structured, machine-readable data about the system architecture, making it easier for AI assistants to understand the codebase and provide accurate suggestions.
