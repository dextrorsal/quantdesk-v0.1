# QuantDesk Devnet Quick Start

## 🚀 What We've Built

Your QuantDesk perpetual DEX is now ready for devnet testing! Here's what's been implemented:

### ✅ Completed

1. **Frontend Fixes**
   - ✅ Fixed PDA derivation in `ProgramContext.tsx` (proper u16 encoding)
   - ✅ Implemented `accountHelpers.ts` with all PDA derivation functions
   - ✅ Added `createUserAccount` method to ProgramContext
   - ✅ Integrated collateral account creation
   - ✅ Added Pyth price feed integration

2. **Backend Setup**
   - ✅ Updated `environment.ts` with correct Program ID
   - ✅ Created devnet initialization script (`init-devnet.ts`)
   - ✅ Added `pnpm run init:devnet` command

3. **Testing Infrastructure**
   - ✅ Created comprehensive E2E test in sandbox
   - ✅ Added `pnpm run test:flow` command
   - ✅ Sandbox tools ready for debugging

4. **Documentation**
   - ✅ Complete devnet deployment guide
   - ✅ Comprehensive troubleshooting runbook
   - ✅ MCP tool integration guidance

---

## 🎯 Next Steps (In Order)

### Step 1: Verify Your Environment (5 min)

```bash
# 1. Check Solana CLI
solana --version  # Should be >= 1.14
solana config get  # Should show devnet

# 2. Check wallet balance
solana balance  # Should have >= 2 SOL
# If not: solana airdrop 2

# 3. Verify Program ID
cd contracts
anchor keys list
# Should show: GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
```

### Step 2: Configure Backend (10 min)

```bash
# 1. Create backend .env
cd backend
cp .env.example .env

# 2. Edit .env with your values:
# - SUPABASE_URL
# - SUPABASE_ANON_KEY  
# - SUPABASE_ACCESS_TOKEN
# - JWT_SECRET (generate random string)
# - Verify PROGRAM_ID matches

# 3. Apply database schema
# In Supabase SQL Editor, run:
# - database/production-schema.sql
# - database/security-audit-fixes.sql
```

### Step 3: Deploy Smart Contract (if not deployed)

```bash
cd contracts

# Build
anchor build

# Deploy to devnet
anchor deploy --provider.cluster devnet

# Copy IDL to frontend
cp target/idl/quantdesk_perp_dex.json ../../frontend/src/types/
```

### Step 4: Initialize Backend (5 min)

```bash
cd backend

# This will:
# - Verify database connection
# - Check Solana devnet connection
# - Verify program deployment
# - Seed SOL/USD, BTC/USD, ETH/USD markets
pnpm run init:devnet
```

Expected output:
```
✅ Database connected
✅ Connected to Solana devnet
✅ Program deployed
✅ Created SOL/USD
✅ Created BTC/USD
✅ Created ETH/USD
✅ Devnet initialization complete!
```

### Step 5: Test with Sandbox (10 min)

```bash
cd solana-sandbox

# Setup sandbox
cp env.example .env
pnpm run setup

# Verify program deployment
pnpm run check-program

# Validate IDL consistency
pnpm run validate-idl

# Run end-to-end test
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

### Step 6: Start Services (2 min)

```bash
# Terminal 1: Start backend
cd backend
pnpm run dev
# Should show: Server listening on port 3002

# Terminal 2: Start frontend
cd frontend
pnpm run dev
# Should show: Local: http://localhost:3001
```

### Step 7: Test Frontend (15 min)

1. **Open Browser**
   - Navigate to http://localhost:3001
   - Open DevTools Console

2. **Connect Wallet**
   - Click "Connect Wallet"
   - Select Phantom or Solflare
   - Ensure wallet is on **devnet**
   - Approve connection

3. **Create User Account**
   - If prompt appears, click "Create Account"
   - Approve transaction in wallet
   - Wait for confirmation
   - Check console for: "User account created: <tx_signature>"

4. **Verify Markets**
   - Should see SOL/USD, BTC/USD, ETH/USD
   - Check prices are displayed
   - Verify oracle data loading

5. **Test Position (Optional)**
   - Try opening a small position
   - Check position appears in portfolio
   - Try closing position

---

## 🔍 Troubleshooting

### Issue: Program Not Found

**Check:**
```bash
solana program show GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
```

**Fix:**
```bash
cd contracts
anchor deploy --provider.cluster devnet
```

### Issue: Account Creation Fails

**Check Browser Console for:**
- "Invalid seeds for PDA" → PDA derivation issue
- "Insufficient funds" → Need more devnet SOL
- "Account already exists" → Already created!

**Quick Fix:**
```bash
# Get more SOL
solana airdrop 2

# Verify PDA derivation
cd solana-sandbox
pnpm run derive -- --wallet <YOUR_WALLET> --index 0
```

### Issue: Markets Not Loading

**Check:**
```bash
curl http://localhost:3002/api/markets
```

**Fix:**
```bash
cd backend
pnpm run init:devnet
```

### Issue: IDL Mismatch / Deserialization Error

**Fix:**
```bash
cd contracts
anchor build
cp target/idl/quantdesk_perp_dex.json ../../frontend/src/types/

# Restart frontend
cd frontend
pnpm run dev
```

### Full Troubleshooting Guide

For detailed troubleshooting, see:
- **Full Guide**: `/docs/debugging/devnet-troubleshooting.md`
- **Deployment Guide**: `/docs/deployment/devnet-deployment-guide.md`

---

## 🛠 Useful Commands

### Debugging

```bash
# Check program status
cd solana-sandbox && pnpm run check-program

# Validate IDL
cd solana-sandbox && pnpm run validate-idl

# Inspect account
cd solana-sandbox && pnpm run inspect -- <ACCOUNT_PUBKEY>

# View transaction
cd solana-sandbox && pnpm run analyze -- <TX_SIGNATURE>

# Backend health
curl http://localhost:3002/api/health

# Market data
curl http://localhost:3002/api/markets
```

### Logs

```bash
# Watch Solana program logs
solana logs GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a

# Backend logs
tail -f backend/logs/backend-dev.log

# Frontend logs
# Check browser DevTools console
```

---

## 📚 Documentation

- **Deployment Guide**: `/docs/deployment/devnet-deployment-guide.md`
- **Troubleshooting**: `/docs/debugging/devnet-troubleshooting.md`
- **User Monitoring**: `/docs/monitoring/USER_MONITORING_GUIDE.md`
- **Protocol Monitoring**: `/docs/monitoring/PROTOCOL_MONITORING_GUIDE.md`
- **Contract Docs**: `/contracts/docs/`
- **API Documentation**: http://localhost:3002/api/docs/swagger

---

## 🎓 Using MCP Expert Tools

You have Solana and Anchor experts available via MCP:

```javascript
// Ask for help
"How do I debug AccountDidNotDeserialize error?"
"Why is my PDA derivation failing?"
"How to handle transaction retries in Solana?"
```

**Available Tools:**
- Solana Expert: General Solana questions
- Anchor Framework Expert: Anchor-specific issues
- Supabase: Database queries and schema
- Documentation Search: Official docs lookup

---

## ✅ Pre-Flight Checklist

Before testing:
- [ ] Solana CLI configured for devnet
- [ ] Wallet has >= 2 SOL devnet balance
- [ ] Program deployed to devnet
- [ ] Backend `.env` configured with Supabase
- [ ] Database schema applied
- [ ] Markets initialized (`init:devnet` ran)
- [ ] IDL synced to frontend
- [ ] Backend running on port 3002
- [ ] Frontend running on port 3001

---

## 🚨 Known Issues

1. **Expected Warnings in Devnet**:
   - Missing `execute_sql` function → OK for devnet
   - Missing `auth_nonces` table → OK for devnet

2. **Common Errors**:
   - First transaction timeout → Retry after 30s
   - Occasional RPC rate limits → Wait and retry
   - Airdrop cooldown → Use faucet instead

---

## 📞 Support

If you encounter issues:

1. **Check documentation** (see above)
2. **Use sandbox diagnostic tools**
3. **Ask MCP experts** for Solana-specific questions
4. **Check browser/backend logs**
5. **Verify all checklist items**

---

## 🎉 Success Criteria

You're ready for devnet when:
- ✅ Wallet connects successfully
- ✅ User account creates without errors
- ✅ Markets load with prices
- ✅ Can view transaction on Solana Explorer
- ✅ All sandbox tests pass
- ✅ No critical errors in logs

---

**Version**: 1.0.6  
**Last Updated**: October 2025  
**Status**: Ready for Devnet Testing 🚀

Happy Testing! 🎊

