# Source Tree and Module Organization

## Project Structure (Actual)

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

## Key Modules and Their Purpose

- **Backend Service**: `backend/src/server.ts` - Centralized API gateway with comprehensive trading functionality
- **Database Service**: `backend/src/services/supabaseDatabase.ts` - CRITICAL: Single abstraction layer, prevents direct Supabase usage
- **Oracle Service**: `backend/src/services/pythOracleService.ts` - Real-time Pyth Network price feeds with caching
- **Smart Contracts**: `contracts/programs/quantdesk-perp-dex/src/lib.rs` - Solana program with 50+ instructions
- **Frontend App**: `frontend/src/App.tsx` - React app with Solana wallet integration
- **AI Service**: `MIKEY-AI/src/api/index.ts` - LangChain-powered trading assistant
- **Data Pipeline**: `data-ingestion/src/collectors/` - Real-time data collection services
