# Devnet Smoke Test

CLI script to test SOL deposit functionality on devnet with **real** live data (no mocks).

## Quick Start

```bash
# From project root
npm run devnet:smoke
```

## Manual Run

```bash
# Set environment variables (optional - defaults provided)
export RPC_URL=https://api.devnet.solana.com
export PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw
export PYTH_SOL_FEED=H6ARHf6YXhGYeQfUzQNGk6rDN1aQfwbNgBEMwLf9f5vK  # devnet SOL/USD
export KEYPAIR=~/.config/solana/keys/id.json  # or ~/.config/solana/id.json
export DEPOSIT_SOL=0.001
export ACCOUNT_INDEX=0

# Run the test
npx ts-node scripts/devnet_smoke_test.ts
```

## What It Does

1. **Loads IDL** from `contracts/target/idl/` or `frontend/src/types/`
2. **Creates Anchor Provider** from your CLI keypair
3. **Derives PDAs:**
   - `user_account`: `["user_account", authority, [0u8, 0u8]]` (account_index=0)
   - `collateral`: `["collateral", authority, "SOL"]`
   - `protocol_vault`: `["protocol_sol_vault"]`
4. **Requests airdrop** if balance < 0.01 SOL
5. **Calls `deposit_native_sol`:**
   - Initializes `user_account` and `collateral_account` if needed (via `init_if_needed`)
   - Transfers SOL to protocol vault
   - Updates collateral values via Pyth oracle
6. **Verifies deposit** by fetching accounts

## Requirements

- Solana CLI configured for devnet
- Keypair with some SOL (script will airdrop if needed)
- Program deployed to devnet
- IDL file in `contracts/target/idl/` or `frontend/src/types/`

## Account Order (Critical!)

The script ensures accounts match IDL order exactly:
1. `user_account` (PDA, writable)
2. `user` (signer, writable) 
3. `protocol_vault` (PDA, writable)
4. `collateral_account` (PDA, writable)
5. `sol_usd_price_feed` (readonly)
6. `system_program` (SystemProgram.programId)
7. `rent` (SYSVAR_RENT_PUBKEY)

## Troubleshooting

### AccountNotSigner Error
- Verify `user` account matches the signer's public key
- Check account order matches IDL exactly

### Program Not Found
- Ensure program is deployed: `anchor deploy`
- Verify PROGRAM_ID matches `declare_id!()` in `lib.rs`

### IDL Not Found
- Build contracts: `cd contracts && anchor build`
- Or copy IDL to `frontend/src/types/quantdesk_perp_dex.json`

### Insufficient Funds
- Script auto-airdrops 0.02 SOL if balance < 0.01 SOL
- Or manually: `solana airdrop 2 -u devnet`

