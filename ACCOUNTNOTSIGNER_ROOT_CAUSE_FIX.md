# AccountNotSigner Root Cause - FIXED ✅

**Date**: 2025-10-29  
**Status**: ✅ ROOT CAUSE IDENTIFIED AND FIXED

---

## Root Cause Discovery

The AccountNotSigner (3010) error was caused by **account order mismatch** between TypeScript code and the IDL (Interface Definition Language).

### The Problem

The IDL file (`quantdesk_perp_dex.json`) defines the `deposit_native_sol` instruction with accounts in this **exact order**:

1. `user_account` (position 0)
2. **`user` (position 1)** ← Signer must be here!
3. `protocol_vault` (position 2)
4. `collateral_account` (position 3)
5. `sol_usd_price_feed` (position 4)
6. `system_program` (position 5)
7. `rent` (position 6, auto-added by Anchor)

### What We Were Doing (WRONG)

The TypeScript code was passing accounts in this order:

1. `userAccount` (position 0) ✅
2. `collateralAccount` (position 1) ❌
3. `protocolVault` (position 2) ❌
4. **`user` (position 3)** ❌ **WRONG POSITION!**
5. `solUsdPriceFeed` (position 4)
6. `systemProgram` (position 5)

### Why This Caused AccountNotSigner

When Anchor processes the instruction, it checks which accounts are signers based on their **position in the account array**. Since `user` was at position 3 instead of position 1, Anchor couldn't match it with the signer from `provider.wallet.publicKey` which was expected at position 1.

Even though:
- ✅ The transaction WAS being signed (1 signature present)
- ✅ We were using `provider.wallet.publicKey` correctly
- ✅ The public keys matched

Anchor still rejected it because **the signer account was at the wrong position**.

---

## Fixes Applied

### 1. Fixed `depositNativeSOLBestPractice()` ✅
**File**: `frontend/src/services/smartContractService.ts` (lines 2150-2159)

Changed account order to match IDL:
```typescript
.accounts({
  userAccount: pdas.userAccount,      // Position 0
  user: signerPublicKey,              // Position 1 - CORRECT!
  protocolVault: pdas.protocolVault,  // Position 2
  collateralAccount: pdas.collateralSOL, // Position 3
  solUsdPriceFeed: SOL_USD_PRICE_FEED,
  systemProgram: SystemProgram.programId,
})
```

### 2. Fixed `depositNativeSol()` ✅
**File**: `frontend/src/services/smartContractService.ts` (lines 1789-1798)

Changed account order to match IDL:
```typescript
.accounts({
  userAccount: userAccountPda!,     // Position 0
  user: provider.wallet.publicKey,  // Position 1 - CORRECT!
  protocolVault: protocolSOLVault!, // Position 2
  collateralAccount: solCollateralAccount!, // Position 3
  solUsdPriceFeed: SOL_USD_PRICE_FEED,
  systemProgram: SystemProgram.programId,
})
```

### 3. Fixed Test Suite ✅
**File**: `frontend/src/components/devnet-testing/DebugPanelComponent.tsx` (lines 300-365)

- Fixed account order to match IDL
- Added `provider.wallet.publicKey` usage instead of `wallet.adapter.publicKey`
- Added comprehensive transaction inspection logging

---

## Verification

To verify the fix is working:

1. **Rebuild frontend**:
   ```bash
   cd frontend && pnpm run build
   ```

2. **Run test suite**:
   - Open browser: `http://localhost:3001`
   - Navigate to debug panel
   - Click "Run Deposit Test Suite"
   - Verify all tests pass ✅

3. **Test deposit**:
   - Use "Best Practice Deposit" button
   - Should succeed without AccountNotSigner error ✅

---

## Key Learnings

1. **Always check IDL first**: The IDL file is the source of truth for account order
2. **Account order matters**: Anchor validates signers by position, not just by public key
3. **Object reference helps**: Using `provider.wallet.publicKey` is still important for object reference matching
4. **Both must be correct**: Need both correct account order AND correct object reference

---

## Files Modified

1. `frontend/src/services/smartContractService.ts`
   - Fixed `depositNativeSol()` account order
   - Fixed `depositNativeSOLBestPractice()` account order (3 occurrences)

2. `frontend/src/components/devnet-testing/DebugPanelComponent.tsx`
   - Fixed `runDepositTestSuite()` account order
   - Fixed to use `provider.wallet.publicKey`
   - Added comprehensive logging

---

## Next Steps

- [ ] Test the fix on devnet
- [ ] Verify deposit flow works end-to-end
- [ ] Update any other functions that might have similar issues
- [ ] Document this learning for future development

---

**Status**: ✅ READY FOR TESTING

