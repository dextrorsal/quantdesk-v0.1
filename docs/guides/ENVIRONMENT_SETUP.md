# QuantDesk Environment Setup Guide

## Quick Answer: Do You Need Conda/Env Management?

**For this project: NO, not required** - but it can be helpful for organization.

## Current Setup (Recommended)

### What We Have Now
- ✅ **Frontend**: Uses `npm` (Node.js package manager)
- ✅ **Backend**: Will use `npm` (Node.js)
- ✅ **Smart Contracts**: Uses `cargo` (Rust package manager)

### What We Need to Install
```bash
# Run the setup script
./setup.sh

# Or install manually:
# 1. Node.js 18+ (for frontend/backend)
# 2. Rust (for smart contracts)
# 3. Solana CLI (for blockchain interaction)
# 4. Anchor (for Solana smart contract framework)
```

## Environment Management Options

### Option 1: **System-Wide (Current Approach)**
```bash
# Pros: Simple, fast, works everywhere
# Cons: Can have version conflicts

# Install tools globally
npm install -g @solana/web3.js
cargo install anchor-cli
```

### Option 2: **Conda Environment (If You Prefer)**
```bash
# Create isolated environment
conda create -n quantdesk python=3.11 nodejs=18 rust
conda activate quantdesk

# Install additional tools
conda install -c conda-forge solana-cli
```

### Option 3: **Docker (Most Robust)**
```bash
# Each service in its own container
# Frontend: Node.js container
# Backend: Node.js container  
# Smart Contracts: Rust container
```

## Recommended Approach

**Start with Option 1** (system-wide) because:
- ✅ Faster to get started
- ✅ Less complexity
- ✅ Easier debugging
- ✅ Can always add environment management later

## Environment Files Created

### Frontend (.env.local)
```env
VITE_SOLANA_NETWORK=devnet
VITE_RPC_URL=https://api.devnet.solana.com
VITE_WS_URL=wss://api.devnet.solana.com
```

### Backend (.env)
```env
NODE_ENV=development
PORT=3001
SOLANA_NETWORK=devnet
RPC_URL=https://api.devnet.solana.com
WS_URL=wss://api.devnet.solana.com
SUPABASE_URL=your_supabase_url_here
SUPABASE_ANON_KEY=your_supabase_anon_key_here
```

### Smart Contracts (.env)
```env
ANCHOR_PROVIDER_URL=https://api.devnet.solana.com
ANCHOR_WALLET=~/.config/solana/id.json
```

## Development Commands

```bash
# Frontend
./dev-frontend.sh
# or
cd frontend && npm run dev

# Backend (when ready)
./dev-backend.sh
# or  
cd backend && npm run dev

# Smart Contracts (when ready)
./dev-contracts.sh
# or
solana-test-validator --reset
```

## When You Might Want Conda/Docker

### Use Conda If:
- You work on multiple projects with different Node.js versions
- You want isolated environments
- You're on Windows and having path issues

### Use Docker If:
- You want identical environments across team members
- You're deploying to production
- You want to avoid "works on my machine" issues

## Current Status

**You're all set!** The frontend is working with just `npm install` and `npm run dev`. 

For the backend and smart contracts, we'll use the same simple approach unless you specifically want environment management.

## Next Steps

1. **Run the setup script**: `./setup.sh`
2. **Start frontend**: `cd frontend && npm run dev`
3. **Open browser**: `http://localhost:3001`

That's it! No conda or complex environment management needed for now. 🚀
