# Development and Deployment

## Local Development Setup

1. **Prerequisites**: Node.js 20+, pnpm, Solana CLI tools
2. **Installation**: `npm run install:all` (uses pnpm internally)
3. **Start Services**: `npm run dev` (starts all services)
4. **Individual Services**:
   - Backend: `cd backend && pnpm run dev`
   - Frontend: `cd frontend && pnpm run dev`
   - MIKEY-AI: `cd MIKEY-AI && pnpm run dev`
   - Data Ingestion: `cd data-ingestion && pnpm run dev`

## Build and Deployment Process

- **Build Command**: `npm run build` (builds all services)
- **Deployment**: 
  - Frontend: Vercel (automatic on push)
  - Backend: Vercel (automatic on push)
  - AI Service: Railway (configured)
  - Smart Contracts: Solana devnet/mainnet
- **Environments**: Development (local), Production (Vercel/Railway)
