# QuantDesk Devnet Deployment Guide

## Overview
This guide will walk you through deploying QuantDesk to Solana devnet for testing and development.

## Prerequisites

### Required Tools
- Node.js >= 18.0.0
- pnpm package manager
- Solana CLI tools
- Anchor CLI v0.32.1
- Git

### Required Accounts
- Supabase project (free tier works)
- Solana devnet wallet with SOL

## Phase 1: Environment Setup

### 1.1 Clone and Install Dependencies

```bash
# Clone repository (if not already cloned)
git clone <your-repo-url>
cd quantdesk-1.0.6

# Install workspace dependencies
pnpm install

# Install backend dependencies
cd backend && pnpm install && cd ..

# Install frontend dependencies
cd frontend && pnpm install && cd ..

# Install sandbox dependencies
cd solana-sandbox && pnpm install && cd ..
```

### 1.2 Solana Wallet Setup

```bash
# Set Solana CLI to devnet
solana config set --url https://api.devnet.solana.com

# Create or use existing wallet
solana-keygen new --no-bip39-passphrase

# Get your wallet address
solana address

# Request devnet SOL
solana airdrop 2

# Verify balance
solana balance
```

### 1.3 Configure Backend Environment

Create `backend/.env` from the template:

```bash
cp backend/.env.example backend/.env
```

Edit `backend/.env` with your values:

```env
# Solana Configuration
SOLANA_NETWORK=devnet
RPC_URL=https://api.devnet.solana.com
SOLANA_RPC_URL=https://api.devnet.solana.com
PROGRAM_ID=GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
QUANTDESK_PROGRAM_ID=GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a

# Supabase Configuration
SUPABASE_URL=<your_supabase_project_url>
SUPABASE_ANON_KEY=<your_supabase_anon_key>
SUPABASE_ACCESS_TOKEN=<your_supabase_access_token>
SUPABASE_PROJECT_ID=<your_supabase_project_id>
DATABASE_URL=<your_database_connection_string>

# JWT Configuration
JWT_SECRET=<generate_random_secret>
JWT_EXPIRES_IN=7d

# Oracle Configuration (Devnet Pyth Feeds)
PYTH_PRICE_FEED_SOL=J83w4HKfqxwcq3BEMMkPFSppX3gqekLyLJBexebFVkix
PYTH_PRICE_FEED_BTC=HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J
PYTH_PRICE_FEED_ETH=EdVCmQ9FSPcVe5YySXDPCRmc8aDQLKJ9xvYBMZPie1Vw
```

## Phase 2: Smart Contract Deployment

### 2.1 Build Smart Contracts

```bash
cd contracts

# Build the program
anchor build

# Verify program ID matches
anchor keys list

# If Program ID doesn't match, update in:
# - Anchor.toml
# - lib.rs (declare_id!)
# - backend/.env
# - frontend/src/contexts/ProgramContext.tsx
```

### 2.2 Deploy to Devnet

```bash
# Deploy program
anchor deploy --provider.cluster devnet

# Verify deployment
solana program show GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
```

### 2.3 Copy IDL to Frontend

```bash
# From contracts/smart-contracts directory
cp target/idl/quantdesk_perp_dex.json ../../frontend/src/types/
```

## Phase 3: Database Setup

### 3.1 Run Supabase Migrations

```bash
# Apply production schema
psql <your_database_url> < database/production-schema.sql

# Apply security fixes
psql <your_database_url> < database/security-audit-fixes.sql

# Seed initial markets (optional)
psql <your_database_url> < database/seed-markets.sql
```

### 3.2 Initialize Devnet Markets

```bash
cd backend
pnpm run init:devnet
```

This will:
- Verify database connectivity
- Check Solana devnet connection
- Verify program deployment
- Seed SOL/USD, BTC/USD, ETH/USD markets

## Phase 4: Testing & Verification

### 4.1 Run Sandbox Tests

```bash
cd solana-sandbox

# Copy environment template
cp env.example .env

# Setup devnet environment
pnpm run setup

# Verify program deployment
pnpm run check-program

# Run complete trading flow test
pnpm run test:flow
```

Expected output:
```
✓ User Account PDA derivation
✓ Market PDA derivation
✓ Position PDA derivation
✓ Collateral PDA derivation
✓ Program verification
```

### 4.2 Validate IDL Consistency

```bash
cd solana-sandbox
pnpm run validate-idl
```

If mismatch detected, rebuild contracts and recopy IDL.

## Phase 5: Start Services

### 5.1 Start Backend

```bash
cd backend
pnpm run dev
```

Backend should start on port 3002. Verify:
- http://localhost:3002/api/health
- http://localhost:3002/api/markets

### 5.2 Start Frontend

```bash
cd frontend
pnpm run dev
```

Frontend should start on port 3001.

## Phase 6: Frontend Testing

### 6.1 Connect Wallet

1. Open http://localhost:3001
2. Click "Connect Wallet"
3. Select Phantom or Solflare
4. Ensure wallet is on devnet
5. Approve connection

### 6.2 Create User Account

Frontend will check if user account exists. If not:
1. Click "Create Account"
2. Approve transaction
3. Wait for confirmation
4. Verify account appears in UI

### 6.3 Test Trading Flow

1. **View Markets**: Check that SOL/USD, BTC/USD, ETH/USD appear
2. **Add Collateral**: (if implemented) deposit test tokens
3. **Open Position**: Try opening a small long/short position
4. **Close Position**: Close the test position
5. **Check Portfolio**: Verify position appears correctly

## Common Issues & Solutions

### Issue: Program ID Mismatch

**Symptoms**: Frontend shows "Program not found" or deserialization errors

**Solution**:
1. Check Program ID in `contracts/programs/src/lib.rs` (declare_id!)
2. Update in `Anchor.toml`
3. Rebuild: `anchor build`
4. Redeploy: `anchor deploy --provider.cluster devnet`
5. Update in `backend/.env` and `frontend/src/contexts/ProgramContext.tsx`

### Issue: AccountDidNotDeserialize Error

**Symptoms**: Error 3003 when fetching accounts

**Solution**:
1. Verify IDL is up-to-date: `pnpm run validate-idl` in sandbox
2. Rebuild contracts if IDL changed
3. Copy fresh IDL to frontend
4. Check PDA derivation matches Rust code

### Issue: Transaction Fails with "Account Not Found"

**Symptoms**: User account or market not initialized

**Solution**:
1. Run `pnpm run init:devnet` in backend to seed markets
2. Create user account in frontend before trading
3. Verify program is deployed: `solana program show <PROGRAM_ID>`

### Issue: Insufficient SOL Balance

**Symptoms**: "Insufficient funds" errors

**Solution**:
```bash
# Request more devnet SOL
solana airdrop 2

# Or use faucet
# https://faucet.solana.com
```

### Issue: RPC Rate Limiting

**Symptoms**: Frequent timeouts or "429 Too Many Requests"

**Solution**:
1. Use a paid RPC provider (Helius, QuickNode)
2. Update `RPC_URL` in backend/.env
3. Implement retry logic with exponential backoff

### Issue: Stale Oracle Prices

**Symptoms**: "Price stale" errors

**Solution**:
1. Verify Pyth price feeds are correct for devnet
2. Check oracle update logic in smart contract
3. Ensure price updates happen within staleness window (5 min)

## Debugging Tools

### Sandbox Utilities

```bash
cd solana-sandbox

# Inspect account data
pnpm run inspect -- <account_pubkey>

# Derive PDA
pnpm run derive -- --wallet <wallet_pubkey> --index 0

# Analyze transaction
pnpm run analyze -- <transaction_signature>

# Check program status
pnpm run check-program
```

### Browser DevTools

Check console for:
- PDA derivation logs
- Transaction signatures
- Account fetch errors
- WebSocket connection issues

### Solana Explorer

View transactions on devnet:
- https://explorer.solana.com/?cluster=devnet
- Paste transaction signature or account address
- Check program logs and instruction details

## Next Steps

After successful devnet deployment:

1. **Load Testing**: Test with multiple concurrent users
2. **Oracle Integration**: Verify real-time Pyth price updates
3. **Error Handling**: Test edge cases (insufficient collateral, liquidations)
4. **Performance**: Monitor transaction confirmation times
5. **Security Audit**: Review smart contract security
6. **Mainnet Prep**: Plan mainnet deployment strategy

## Monitoring & Maintenance

### Health Checks

```bash
# Backend health
curl http://localhost:3002/api/health

# Market data
curl http://localhost:3002/api/markets

# Oracle prices
curl http://localhost:3002/api/oracle/prices
```

### Logs

Check logs for errors:
```bash
# Backend logs
tail -f backend/logs/backend-dev.log

# Frontend logs
Check browser console

# Smart contract logs
solana logs <program_id>
```

### Regular Maintenance

- Monitor devnet SOL balance in program accounts
- Check for failed transactions
- Verify oracle price updates
- Monitor database growth
- Update dependencies regularly

## Support & Resources

- **Documentation**: `/docs`
- **Sandbox Guide**: `/solana-sandbox/README.md`
- **API Docs**: http://localhost:3002/api/docs/swagger
- **Solana Docs**: https://docs.solana.com
- **Anchor Docs**: https://www.anchor-lang.com

## Checklist

Before going live on devnet:

- [ ] Program deployed to devnet
- [ ] IDL synced to frontend
- [ ] Database schema applied
- [ ] Markets initialized
- [ ] Backend running and healthy
- [ ] Frontend connecting to backend
- [ ] Wallet connects successfully
- [ ] User account creation works
- [ ] Can view markets
- [ ] Oracle prices updating
- [ ] Position opening works
- [ ] Position closing works
- [ ] Error handling working
- [ ] Logs configured
- [ ] Monitoring set up

---

**Last Updated**: October 2025  
**Status**: Devnet Ready ✅  
**Version**: 1.0.6

