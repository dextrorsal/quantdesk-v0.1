# Deposit Function Diagnostic Report

**Date:** 2025-10-29  
**Issue:** AccountNotSigner (3010) error persists despite correct setup

---

## 🔍 Current Status

### ✅ What's Working
1. **PDA Verification** - All seeds match Rust expectations perfectly
2. **Transaction Signing** - Transaction is successfully signed (1 signature confirmed)
3. **Instruction Structure** - User account is marked as `isSigner: true` in instruction metadata
4. **Account Order** - Matches Rust `#[derive(Accounts)]` struct exactly
5. **Provider Setup** - Using `provider.wallet.publicKey` correctly

### ❌ What's Failing
- **AccountNotSigner Error** - Even with correct setup, Anchor rejects with AccountNotSigner
- Error occurs **after** transaction is signed successfully
- Anchor validates the signer but doesn't recognize it

---

## 📊 Debug Output Analysis

From `quantdesk-debug-2025-10-29.json`:

```
✅ PDA VERIFICATION COMPLETE
  - All seeds verified: user_account, collateral, protocol_vault
  - Account index buffer matches Rust [0u8, 0u8]: true

✅ Transaction signed successfully
  - 1 signature confirmed

✅ Instruction structure OK
  - User account isSigner: true
  - User account isWritable: true

❌ Anchor error:
  "AnchorError caused by account: user. Error Code: AccountNotSigner. 
   Error Number: 3010. Error Message: The given account did not sign."
```

---

## 🎯 Key Observations

1. **The signature EXISTS** - Transaction has 1 signature
2. **The signer is marked correctly** - Instruction metadata shows `isSigner: true`
3. **Anchor still rejects it** - This suggests Anchor's internal validation is failing

This pattern indicates:
- ✅ Transaction building is correct
- ✅ Transaction signing is correct  
- ❌ Anchor's signer validation is failing

---

## 🔧 Hypothesis

Based on Anchor expert consultation and code analysis:

### Hypothesis 1: Object Reference Mismatch (Most Likely)
Anchor might be comparing signer by **object reference**, not just address equality. Even though `provider.wallet.publicKey` is used, if it's not the **exact same object** Anchor uses internally, validation fails.

**Status:** Being addressed with `signerPublicKey` reference storage

### Hypothesis 2: Fee Payer vs Signer Mismatch
Anchor sets `feePayer = provider.wallet.publicKey` automatically. If the `user` account in `.accounts()` doesn't match the feePayer by object reference, validation fails.

**Status:** Need to verify feePayer is correctly set and matches

### Hypothesis 3: Wallet Adapter Issue
The wallet adapter's `signTransaction` might be producing a signature that Anchor doesn't recognize, even though it's valid.

**Next Step:** Test with a basic `Keypair` wallet to rule out adapter issues

---

## 🛠️ Implementation Status

### ✅ Completed
- [x] Best-practice deposit function (`depositNativeSOLBestPractice`)
- [x] PDA seed verification utilities
- [x] Enhanced debug panel with verification tools
- [x] Account order alignment with Rust struct
- [x] Object reference tracking (`signerPublicKey`)
- [x] Pre-flight instruction structure validation
- [x] Enhanced error diagnostics

### 🔄 In Progress
- [ ] Testing with Keypair wallet (to rule out adapter issues)
- [ ] Verifying feePayer matches user account exactly
- [ ] Checking if `.signers([])` is needed despite `.rpc()` auto-signing

### 📋 Next Steps

1. **Test with Keypair** - Create a minimal test using a basic `Keypair` instead of wallet adapter
2. **Verify Fee Payer** - Ensure feePayer matches user account by object reference
3. **Check Anchor Version** - Verify `@coral-xyz/anchor` version compatibility
4. **Rust Program Check** - Add logging in Rust to see what the program receives

---

## 🧪 Recommended Test Sequence

### Test 1: Basic Keypair Test
```typescript
import { Keypair } from "@solana/web3.js";

// Generate keypair
const user = Keypair.generate();
// Fund it via airdrop
// Then test deposit with this keypair

// This will tell us if it's a wallet adapter issue
```

### Test 2: Explicit Fee Payer
```typescript
// Try explicitly setting feePayer before .rpc()
const tx = await program.methods
  .depositNativeSol(new BN(amount))
  .accounts({...})
  .transaction();

tx.feePayer = signerPublicKey;
// Then use provider.wallet.signTransaction(tx) and send
```

### Test 3: Rust Program Logging
Add to Rust program:
```rust
msg!("User account: {}, is_signer: {}", ctx.accounts.user.key(), ctx.accounts.user.is_signer);
```

---

## 📝 Code Locations

### TypeScript
- **Best Practice Function:** `frontend/src/services/smartContractService.ts:2009-2175`
- **PDA Utilities:** `frontend/src/services/smartContractService.ts:21-109`
- **Debug Panel:** `frontend/src/components/devnet-testing/DebugPanelComponent.tsx:84-205`

### Rust
- **DepositNativeSol:** `contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs:425-460`

---

## 🚨 Critical Finding

**The transaction IS signing correctly**, but Anchor's validation is rejecting it. This suggests:
1. Signature is valid (Solana accepts it)
2. Anchor's internal validation has stricter requirements
3. Likely an object reference or account metadata issue

---

## 💡 Expert Recommendations Applied

✅ Using `provider.wallet.publicKey` in `.accounts()`  
✅ Account order matches Rust struct exactly  
✅ PDA seeds verified before execution  
✅ Pre-flight instruction structure validation  
⏳ Testing with basic Keypair (next step)  
⏳ Verifying feePayer object reference (next step)

---

## 🔗 Related Files

- `DEPOSIT_FIX_INSIGHTS.md` - Comprehensive implementation guide
- `frontend/src/services/smartContractService.ts` - Main service file
- `frontend/src/components/devnet-testing/DebugPanelComponent.tsx` - Debug UI
- `contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs` - Rust program

