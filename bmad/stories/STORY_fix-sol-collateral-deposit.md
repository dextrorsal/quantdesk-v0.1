# Story: Fix SOL collateral init and deposit flow with Wallet Standard adapters

## Problem
Users cannot initialize SOL collateral or deposit SOL. Anchor throws `AccountNotSigner (3010)` for `user` on both `InitializeCollateralAccount` and `DepositNativeSol`. This blocks the demo and end-to-end trading flow.

## Objective
Make on-chain SOL collateral initialization and deposit succeed using Wallet Standard/Phantom adapters, with confirmed signatures and UI state refreshed in `AccountContext`.

## Scope
- Use canonical wallet-adapter send path (no manual feePayer; set signers; single-instruction-per-tx).
- Align TS account order to Rust `#[derive(Accounts)]` exactly (see references).
- Derive PDAs with exact seeds per program.
- Improve Debug panel to run/check each step and show tx metas/logs.

## Acceptance Criteria
- Initializing SOL collateral succeeds; Debug panel row flips to OK.
- Depositing 0.01 SOL succeeds; confirmed sig returned; `AccountContext` shows collateral > 0.
- Works with Phantom and Wallet Standard adapters (manual test).
- Logs show no `AccountNotSigner`; printed AccountMetas show `user` isSigner=true.

## Tasks
1. Implement wallet.adapter.sendTransaction path with `tx.setSigners(userPubkey)`; single instruction per tx.
2. Align TS `.accounts({...})` order to Rust for both instructions.
3. Derive PDAs with exact seeds:
   - user_account: `["user_account", user, [0u8,0u8]]`
   - collateral (init): `["collateral", user, &[asset_type as u8]]` (SOL => 1)
   - collateral (deposit): `["collateral", user, b"SOL"]`
   - protocol_vault: `["protocol_sol_vault"]`
4. Add dev logging to print instruction metas (pubkey, isSigner, isWritable) before send (dev-only).
5. Extend Debug panel: show metas/logs, copy buttons; re-test on devnet.
6. Attach two confirmed tx signatures (init, deposit) in story and update result screenshots.

## References
- Rust: `contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs`
- Frontend services: `frontend/src/services/smartContractService.ts`
- Debug UI: `frontend/src/components/DebugPanel.tsx`

## Risks
- Wallet Standard adapter differences in signing/fee payer.
- PDA seeds mismatch between init vs deposit.
- Preflight failures in some wallets; may need `skipPreflight: false` and recent blockhash.

## Done Definition
- All AC met; code merged; Debug panel “Init SOL Collateral” and “Test Deposit 0.01 SOL” pass on devnet; story updated with signatures and brief test notes.


