# Quick Reference - Key Files and Entry Points

## Critical Files for Understanding the System

- **Main Entry**: `backend/src/server.ts` - Express server with comprehensive API
- **Configuration**: `backend/src/config/environment.ts`, `.env` files
- **Core Business Logic**: `backend/src/services/supabaseDatabase.ts`, `backend/src/services/pythOracleService.ts`
- **API Definitions**: `backend/src/routes/` - 50+ API endpoints
- **Database Models**: `database/schema.sql` - PostgreSQL schema with TimescaleDB
- **Key Algorithms**: `backend/src/services/pythOracleService.ts` - Oracle price processing
- **Smart Contracts**: `contracts/programs/quantdesk-perp-dex/src/lib.rs` - Solana program
- **Frontend Entry**: `frontend/src/main.tsx` - React application with wallet integration
- **AI Service**: `MIKEY-AI/src/api/index.ts` - LangChain-powered AI agent

## Enhancement Impact Areas

Based on the PRD requirements, these files/modules will be affected by the planned enhancement:

- **Backend API Routes**: New endpoints for social media, news, alpha channels
- **Database Schema**: New tables for news articles, social media posts, alpha channel messages
- **Frontend Components**: New unified dashboard components
- **AI Service**: Enhanced sentiment analysis and alpha channel processing
- **Data Ingestion**: New collectors for Twitter, Discord, Telegram, news feeds
