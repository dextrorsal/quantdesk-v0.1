# QuantDesk Devnet Troubleshooting Runbook

## Quick Diagnostic Commands

```bash
# Check all systems
cd solana-sandbox && pnpm run check-program  # Verify program
curl http://localhost:3002/api/health          # Backend health
curl http://localhost:3002/api/markets         # Markets data
solana balance                                  # Wallet balance
```

---

## Error Categories

### 🔴 Critical Errors (Service Down)
- Program not deployed
- Backend not responding
- Database unreachable

### 🟡 Transaction Errors (User Impact)
- Account creation fails
- Position opening fails
- Insufficient balance

### 🟢 Warning Errors (Degraded)
- Stale oracle prices
- RPC rate limits
- Slow confirmations

---

## Common Errors & Solutions

### Error: `AccountDidNotDeserialize` (Error 3003)

**Symptoms:**
```
Error: Account does not match expected type
Error Code: 3003 (AccountDidNotDeserialize)
```

**Cause:** IDL mismatch between frontend and deployed program

**Diagnosis:**
```bash
cd solana-sandbox
pnpm run validate-idl
```

**Solution:**
```bash
# Option 1: Rebuild and redeploy program
cd contracts
anchor build
anchor deploy --provider.cluster devnet
cp target/idl/quantdesk_perp_dex.json ../../frontend/src/types/

# Option 2: Use correct IDL from deployment
cd contracts
anchor idl fetch GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
cp .anchor/idl/quantdesk_perp_dex.json ../../frontend/src/types/
```

**Prevention:** Always sync IDL after redeployment

---

### Error: `InvalidAccountData` or Wrong PDA

**Symptoms:**
```
Error: Invalid seeds for PDA
Account not found at expected address
```

**Cause:** PDA derivation mismatch between frontend and smart contract

**Diagnosis:**
```bash
cd solana-sandbox
pnpm run derive -- --wallet <YOUR_WALLET> --index 0
```

**Compare with smart contract seeds:**
```rust
// In lib.rs - User Account PDA
seeds = [
    b"user_account",
    authority.key().as_ref(),
    &account_index.to_le_bytes()  // Must be u16 little-endian!
]
```

**Solution in Frontend:**
```typescript
// Correct PDA derivation
const accountIndex = 0;
const accountIndexBuffer = Buffer.alloc(2);
accountIndexBuffer.writeUInt16LE(accountIndex, 0);

const [userAccountPDA] = PublicKey.findProgramAddressSync(
    [
        Buffer.from("user_account"),
        wallet.publicKey.toBuffer(),
        accountIndexBuffer  // ✅ Correct: 2 bytes, little-endian
    ],
    program.programId
);

// ❌ WRONG:
Buffer.from([0, 0])  // Don't use this!
```

**Prevention:** Use helper functions from `frontend/src/utils/accountHelpers.ts`

---

### Error: `InsufficientFunds` / `0x1` (Custom Program Error)

**Symptoms:**
```
Error: Insufficient funds for transaction
Error: custom program error: 0x1
```

**Cause:** Not enough SOL for rent or transaction fees

**Diagnosis:**
```bash
solana balance
# Should show > 0.02 SOL
```

**Solution:**
```bash
# Request devnet airdrop
solana airdrop 2

# Or use web faucet
# https://faucet.solana.com
```

**Prevention:** Set up airdrop automation in test scripts

---

### Error: `Transaction Simulation Failed`

**Symptoms:**
```
Error: Transaction simulation failed: Error processing Instruction 0
Logs:
  Program GcpEy... invoke [1]
  Program log: AnchorError caused by account: user_account
  Program GcpEy... consumed 5000 compute units
  Program GcpEy... failed: invalid account data
```

**Diagnosis Steps:**

1. **Check account exists:**
```bash
cd solana-sandbox
pnpm run inspect -- <ACCOUNT_ADDRESS>
```

2. **Verify account ownership:**
```bash
solana account <ACCOUNT_ADDRESS>
# Owner should be: GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
```

3. **Check account constraints:**
- Review smart contract constraints
- Verify signer requirements
- Check mutability flags

**Common Fixes:**
- User account not initialized → Create account first
- Wrong account passed → Verify PDA derivation
- Missing signer → Add wallet.publicKey as signer
- Account not mutable → Add `mut` flag in context

---

### Error: `Program Not Found` / `ProgramAccountNotFound`

**Symptoms:**
```
Error: Program GcpEy...Wu3a not found
```

**Diagnosis:**
```bash
solana program show GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a
```

**If not found:**
```bash
cd contracts
anchor deploy --provider.cluster devnet
```

**If deployed but still error:**
1. Check Program ID matches everywhere:
   - `contracts/smart-contracts/lib.rs` (declare_id!)
   - `contracts/smart-contracts/Anchor.toml`
   - `backend/.env` (PROGRAM_ID)
   - `frontend/src/contexts/ProgramContext.tsx`

2. Rebuild with correct ID:
```bash
cd contracts
# Update lib.rs declare_id! first
anchor build
anchor deploy --provider.cluster devnet
```

---

### Error: `BlockhashNotFound` / Transaction Timeout

**Symptoms:**
```
Error: Transaction was not confirmed in 60 seconds
Error: Blockhash not found
```

**Cause:** Network congestion or outdated blockhash

**Solution:**
```typescript
// In frontend, increase confirmation timeout
const tx = await program.methods
    .createUserAccount(0, 10)
    .accounts({ /* ... */ })
    .rpc({ 
        skipPreflight: false,
        commitment: 'confirmed',
        maxRetries: 3
    });

// Wait with longer timeout
const confirmation = await connection.confirmTransaction(tx, {
    commitment: 'confirmed',
    maxRetries: 5
});
```

**Prevention:** Implement retry logic with exponential backoff

---

### Error: `429 Too Many Requests` (RPC Rate Limit)

**Symptoms:**
```
Error: 429 Too Many Requests
Failed to fetch
```

**Immediate Solution:**
```bash
# Wait 60 seconds and retry
# Or switch to paid RPC
```

**Long-term Solution:**
Update `backend/.env`:
```env
# Use Helius, QuickNode, or other paid RPC
RPC_URL=https://devnet.helius-rpc.com/?api-key=YOUR_KEY
SOLANA_RPC_URL=https://devnet.helius-rpc.com/?api-key=YOUR_KEY
```

**Prevention:** Implement request caching and batching

---

### Error: `MarketNotFound` / Empty Markets List

**Symptoms:**
- Frontend shows no markets
- API returns empty array

**Diagnosis:**
```bash
curl http://localhost:3002/api/markets
```

**If empty:**
```bash
cd backend
pnpm run init:devnet
```

**Check database:**
```sql
-- Connect to Supabase
SELECT * FROM markets WHERE is_active = true;
```

**If still empty:** Manually insert markets via Supabase dashboard or SQL

---

### Error: `PriceStale` / Oracle Update Failed

**Symptoms:**
```
Error: Oracle price is stale
Error: Price updated more than 5 minutes ago
```

**Diagnosis:**
```bash
# Check oracle service
curl http://localhost:3002/api/oracle/prices
```

**Solution:**
1. Verify Pyth price feeds are correct for devnet
2. Check backend oracle service is running
3. Manually update price (for testing):
```typescript
await program.methods
    .updateOraclePrice(new anchor.BN(100_000_000)) // $100
    .accounts({
        market: marketPDA,
        authority: wallet.publicKey,
        clock: SYSVAR_CLOCK_PUBKEY,
    })
    .rpc();
```

---

### Error: Frontend Can't Create User Account

**Symptoms:**
- "Create Account" button doesn't work
- Transaction fails silently
- No error message shown

**Diagnosis:**
1. **Check wallet connection:**
```javascript
console.log('Wallet:', wallet.publicKey?.toBase58());
console.log('Program:', program?.programId.toBase58());
```

2. **Check PDA derivation:**
```javascript
const [pda] = deriveUserAccountPDA(program, wallet.publicKey, 0);
console.log('User Account PDA:', pda.toBase58());
```

3. **Check transaction building:**
```javascript
const tx = await program.methods
    .createUserAccount(0, 10)
    .accounts({
        userAccount: userAccountPDA,
        authority: wallet.publicKey,
        systemProgram: SystemProgram.programId,
        rent: SYSVAR_RENT_PUBKEY,
    })
    .rpc();
console.log('Transaction:', tx);
```

**Common Issues:**
- ❌ Missing SystemProgram
- ❌ Missing Rent sysvar
- ❌ Wrong PDA seeds
- ❌ Insufficient SOL

---

## Debugging Workflow

### Step 1: Verify System Status
```bash
# Check all components
./scripts/health-check.sh  # (create this if needed)

# Or manually:
solana cluster-version              # Devnet is up
curl http://localhost:3002/health   # Backend is up
curl http://localhost:3001          # Frontend is up
```

### Step 2: Isolate the Problem

**Is it Frontend?**
- Check browser console
- Verify wallet connection
- Check network tab for failed API calls

**Is it Backend?**
- Check `backend/logs/backend-dev.log`
- Test API endpoints with curl
- Verify database connection

**Is it Smart Contract?**
- Run sandbox tests
- Check program logs with `solana logs`
- Verify program is deployed

**Is it Database?**
- Query Supabase directly
- Check table schemas
- Verify RLS policies

### Step 3: Use Sandbox Tools

```bash
cd solana-sandbox

# Full diagnostic
pnpm run test:flow

# Specific checks
pnpm run check-program     # Program status
pnpm run validate-idl      # IDL consistency
pnpm run inspect -- <addr> # Account data
```

---

## Performance Issues

### Slow Transaction Confirmations

**Symptoms:** Transactions take > 30 seconds

**Solutions:**
1. Use paid RPC with priority fees
2. Implement proper retry logic
3. Use `confirmed` instead of `finalized`
4. Add priority fees to transactions

```typescript
const tx = await program.methods
    .openPosition(/* args */)
    .accounts({ /* accounts */ })
    .preInstructions([
        ComputeBudgetProgram.setComputeUnitPrice({ microLamports: 1000 })
    ])
    .rpc();
```

### High Memory Usage

**Symptoms:** Browser/backend consumes excessive RAM

**Solutions:**
1. Limit WebSocket subscriptions
2. Implement pagination for markets/positions
3. Clear old data from state
4. Use React.memo and useMemo

---

## Emergency Procedures

### Critical: Program Upgrade Needed

```bash
# 1. Backup current state
solana program dump GcpEy...Wu3a backup.so

# 2. Build new version
anchor build

# 3. Deploy upgrade
anchor upgrade <BUFFER_ADDRESS> --program-id GcpEy...Wu3a

# 4. Update IDL everywhere
cp target/idl/quantdesk_perp_dex.json ../../frontend/src/types/

# 5. Restart services
cd backend && pnpm run dev
cd frontend && pnpm run dev
```

### Critical: Database Migration Required

```bash
# 1. Backup database
pg_dump <DATABASE_URL> > backup.sql

# 2. Run migration
psql <DATABASE_URL> < database/migration-file.sql

# 3. Verify
psql <DATABASE_URL> -c "SELECT * FROM markets LIMIT 1;"

# 4. Re-seed if needed
cd backend && pnpm run init:devnet
```

---

## Useful Commands Reference

```bash
# Solana
solana --version                           # Check CLI version
solana config get                          # Show config
solana address                             # Show wallet address
solana balance                             # Check balance
solana airdrop 2                          # Request SOL
solana program show <PROGRAM_ID>          # Program info
solana logs <PROGRAM_ID>                  # Watch logs
solana account <ACCOUNT_ADDRESS>          # Account details

# Anchor
anchor --version                           # Check version
anchor build                              # Build program
anchor deploy                             # Deploy program
anchor test                               # Run tests
anchor idl fetch <PROGRAM_ID>            # Fetch IDL

# Sandbox
cd solana-sandbox
pnpm run check-program                    # Verify deployment
pnpm run validate-idl                     # Check IDL sync
pnpm run inspect -- <ADDRESS>            # Inspect account
pnpm run derive -- --wallet <ADDR>       # Derive PDA
pnpm run test:flow                       # E2E test

# Backend
cd backend
pnpm run dev                             # Start server
pnpm run init:devnet                     # Initialize devnet
curl localhost:3002/api/health           # Health check
curl localhost:3002/api/markets          # Get markets

# Frontend
cd frontend
pnpm run dev                             # Start dev server
pnpm run build                           # Production build
```

---

## Getting Help

### Before Asking for Help

Collect this information:
1. Error message (full text)
2. Transaction signature (if available)
3. Account addresses involved
4. Browser console logs
5. Backend logs
6. Steps to reproduce

### MCP Expert Tools

Use the Solana MCP expert tools for complex issues:

```javascript
// Ask Solana Expert
"How do I debug AccountDidNotDeserialize error 3003 in Anchor?"

// Ask Anchor Expert  
"Why is my PDA derivation failing with invalid seeds?"

// Search Documentation
"How to handle transaction confirmation in web3.js?"
```

### External Resources

- **Solana Stack Exchange**: https://solana.stackexchange.com
- **Anchor Discord**: https://discord.gg/anchor
- **Solana Discord**: https://discord.gg/solana
- **Solana Cookbook**: https://solanacookbook.com

---

## Monitoring Checklist

Daily checks:
- [ ] Backend health endpoint responding
- [ ] Markets loading in frontend
- [ ] Oracle prices updating
- [ ] Transaction success rate > 95%
- [ ] No error spikes in logs
- [ ] Database within size limits
- [ ] Devnet SOL balance sufficient

---

**Last Updated**: October 2025  
**Maintainer**: QuantDesk Team  
**Emergency Contact**: Check project README

