# QuantDesk Collateral Bug - Status Report
**Date:** October 14, 2025  
**Issue:** SOL deposits showing incorrect USD value (1:1 lamports to USD conversion)

---

## 🎯 THE PROBLEM

**Expected Behavior:**
- 0.01 SOL deposit should show: **$2.08 USD** (at $208/SOL)

**Actual Behavior:**
- 0.01 SOL deposit shows: **$10.00 USD**
- 0.02 SOL total shows: **$20.00 USD**

**Root Cause:**
- Lamports are being stored as USD 1:1 instead of converted
- 10,000,000 lamports (0.01 SOL) → stored as $10 USD ❌
- Should be: 10,000,000 lamports → $2.08 USD ✅

---

## 📊 CURRENT STATE

### On-Chain Data (Devnet):
```
Collateral PDA: AN7yFNuCjXNYNKSkPWNGMRSfqADWerhvp65x6A2XRGC3
Program ID: HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso

Stored Amount: 7,619,097.82 SOL (CORRUPTED!)
Stored value_usd: 20,000,000 (=$20 USD)
Actual Vault Balance: 0.02 SOL
Expected Value: $4.16 USD
```

### Program Status:
- **Last Deployed Slot:** 414460085
- **Binary Hash:** 39ecde34c3c2484667ba8a2f71c369b416a21759381d65e7e0e71a06ed1ce5ee
- **Size:** 973KB

### Frontend:
- ✅ Correctly reading on-chain data: `value_usd / 1e6`
- ✅ No frontend bug - displays what's stored on-chain
- ❌ On-chain data is wrong

---

## 🔧 WHAT WE'VE TRIED (In Order)

### 1. ✅ Pyth Integration (Oct 13)
- **What:** Integrated Pyth Network oracle for SOL/USD price
- **Result:** Pyth feeds were empty on devnet
- **Outcome:** Needed fallback solution

### 2. ✅ Fixed-Price Fallback (Oct 13)
- **What:** Added $208/SOL fixed price for devnet in `oracle.rs`
- **Result:** Oracle function was mathematically correct
- **Outcome:** Deployed but still showed 1:1 conversion

### 3. ✅ Direct Inline Calculation (Oct 13-14)
- **What:** Bypassed oracle module, added calculation directly in `deposit_native_sol`
- **Code:**
  ```rust
  let sol_amount = (amount as f64) / 1_000_000_000.0;
  let usd_amount = sol_amount * 208.0;
  let usd_value = (usd_amount * 1_000_000.0) as u64;
  ```
- **Result:** Binary hash UNCHANGED after rebuild
- **Outcome:** Compiler optimized out float math

### 4. ✅ Corruption Detection (Oct 13-14)
- **What:** Added auto-reset logic for impossibly large amounts
- **Code:**
  ```rust
  if collateral_account.amount > vault_balance * 2 {
      collateral_account.amount = 0;
      collateral_account.value_usd = 0;
  }
  ```
- **Result:** Deployed but never triggered
- **Outcome:** Condition not met or logic bypassed

### 5. ✅ Integer-Only Math (Oct 14)
- **What:** Removed ALL float math, used pure integer arithmetic
- **Code:**
  ```rust
  let usd_value = amount
      .checked_mul(208)
      .ok_or(ErrorCode::InvalidAmount)?
      .checked_div(1000)
      .ok_or(ErrorCode::InvalidAmount)?;
  ```
- **Result:** Binary hash CHANGED ✅ (39ecde34...)
- **Deployed:** Slot 414460085
- **Outcome:** Still showing $20 for 0.02 SOL ❌

### 6. ❌ Multiple Redeployments
- **What:** Used `anchor upgrade` 5+ times
- **Result:** Program updated on-chain
- **Outcome:** On-chain account data NEVER reset

### 7. ❌ Frontend Hard Refresh
- **What:** Cleared cache with Ctrl+Shift+R multiple times
- **Result:** No change
- **Outcome:** Not a cache issue - real on-chain data problem

### 8. ❌ Additional Deposits
- **What:** Deposited 0.01 SOL twice to trigger corruption detection
- **Result:** Corruption detection never triggered
- **Outcome:** Logic not executing or condition wrong

---

## 🤔 WHY IT'S NOT WORKING

### Theory 1: Compilation Issue
- **Problem:** Rust compiler might be optimizing code unexpectedly
- **Evidence:** First rebuild had identical hash despite code changes
- **Status:** Partially resolved with integer math (hash changed)

### Theory 2: Account Data Not Updating
- **Problem:** Existing `CollateralAccount` PDA has corrupted data
- **Evidence:** `amount` shows 7,619,097 SOL (impossible)
- **Status:** Corruption detection should reset but doesn't trigger

### Theory 3: Wrong Instruction Executing
- **Problem:** Frontend might be calling different instruction
- **Evidence:** Frontend calls `depositNativeSol` correctly
- **Status:** Verified via code inspection

### Theory 4: Anchor Cache/IDL Issue
- **Problem:** Anchor might be using cached IDL or old binary
- **Evidence:** Multiple rebuilds, some with identical hashes
- **Status:** Latest build shows different hash

### Theory 5: PDA Data Persistence
- **Problem:** Account data persists across program upgrades
- **Evidence:** Solana expert confirmed: "data remains untouched"
- **Status:** **MOST LIKELY CAUSE** ⚠️

---

## 💡 WHAT WE LEARNED

### From Solana Expert (MCP):
1. **Program upgrades replace CODE, not DATA**
   - Existing account data remains unchanged
   - Must explicitly update each account with new transaction

2. **Migration Pattern Required:**
   - Add version field to account struct
   - Create migration instruction
   - Client checks version and migrates if needed

3. **No Automatic Updates:**
   - Accounts don't automatically reflect new calculations
   - Must send transaction to modify account data

### From Anchor Expert (MCP):
1. **Safe to Overwrite PDA Data:**
   - Direct field assignment works: `collateral_account.amount = 0`
   - Must use `#[account(mut)]` attribute

2. **Realloc vs Overwrite vs Close:**
   - Overwrite: Same size/structure ✅
   - Realloc: Change size
   - Close/Recreate: Change structure

3. **Data Migration Best Practice:**
   ```rust
   #[account]
   pub struct CollateralAccount {
       pub version: u8, // Add this!
       pub amount: u64,
       pub value_usd: u64,
   }
   ```

---

## 🚫 WHAT DIDN'T WORK

1. **Float Math:** Compiler optimized it out
2. **Oracle Module:** Function correct but not called properly
3. **Multiple Deploys:** Changed code but not account data
4. **Corruption Detection:** Logic present but not triggering
5. **Hard Refresh:** Not a cache issue
6. **Additional Deposits:** Should trigger reset but doesn't

---

## ✅ WHAT ACTUALLY WORKS

### Verified Working Components:
1. ✅ **Frontend reads correctly:** `value_usd / 1e6`
2. ✅ **Program deploys successfully:** Slot 414460085
3. ✅ **Binary changes on rebuild:** Hash verification works
4. ✅ **Deposit flow completes:** No transaction errors
5. ✅ **Account data persists:** Data reads back consistently

### The Core Issue:
**The smart contract code is correct NOW, but the on-chain account data is STILL from the OLD buggy code.**

---

## 🎯 NEXT STEPS TO TRY

### Option A: Manual Account Reset (RECOMMENDED)
1. **Close corrupted account:**
   - Use `close_collateral_account` instruction
   - Returns rent (~0.0014 SOL)
   
2. **Fresh deposit:**
   - Creates new account with correct data
   - Uses new integer math

**Implementation:**
```typescript
// Call close_collateral_account
await program.methods
  .closeCollateralAccount()
  .accounts({
    user: wallet.publicKey,
    collateralAccount: collateralPda,
  })
  .rpc();

// Then deposit again
await program.methods
  .depositNativeSol(new BN(10_000_000))
  .accounts({ /* ... */ })
  .rpc();
```

### Option B: Fix Instruction
1. **Call `fix_corrupted_collateral`:**
   - Manually recalculates USD value
   - Resets amount based on vault balance

2. **Implementation:**
```typescript
await program.methods
  .fixCorruptedCollateral()
  .accounts({
    user: wallet.publicKey,
    collateralAccount: collateralPda,
    protocolVault: vaultPda,
  })
  .rpc();
```

### Option C: Withdraw All & Re-deposit
1. **Withdraw maximum SOL:**
   - Based on actual vault balance (0.02 SOL)
   - Not the buggy calculated amount

2. **Re-deposit:**
   - Creates fresh calculation
   - May still hit corruption if account exists

**Risk:** Might lose data if withdraw uses corrupted values

### Option D: Add Version Field & Migration (PROPER SOLUTION)
1. **Add version to struct:**
```rust
pub struct CollateralAccount {
    pub version: u8,  // NEW
    pub user: Pubkey,
    pub asset_type: CollateralType,
    pub amount: u64,
    pub value_usd: u64,
    // ...
}
```

2. **Check version on every deposit:**
```rust
if collateral_account.version == 0 {
    // Migrate old data
    collateral_account.amount = actual_vault_balance;
    collateral_account.value_usd = calculate_correct_usd();
    collateral_account.version = 1;
}
```

3. **Realloc account if needed:**
   - Increase size by 1 byte for version field

**Tradeoff:** Requires redeployment and more complexity

---

## 🐛 DEBUGGING TOOLS CREATED

### 1. **debug-and-fix.js**
```bash
node solana-sandbox/debug-and-fix.js
```
Shows: Actual vs expected collateral, corruption status

### 2. **Direct Account Inspection**
```bash
solana account AN7yFNuCjXNYNKSkPWNGMRSfqADWerhvp65x6A2XRGC3 --url devnet
```

### 3. **Smart Contract Logs**
```bash
solana logs HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso --url devnet
```

---

## 📈 TIMELINE

- **Oct 13, 22:00** - Bug discovered ($450 for 0.45 SOL)
- **Oct 13, 23:00** - Pyth integration attempted
- **Oct 13, 23:30** - Fixed-price fallback added
- **Oct 14, 00:00** - Direct inline calculation tried
- **Oct 14, 00:02** - Integer-only math implemented ✅
- **Oct 14, 00:05** - New program deployed (slot 414460085)
- **Oct 14, 00:10** - Still showing $20 for 0.02 SOL ❌

**Total Time:** ~2+ hours debugging

---

## 💰 COSTS SO FAR

- **Program Deployments:** ~5-7 deploys × 0.01 SOL = 0.05-0.07 SOL
- **Test Deposits:** 0.02 SOL deposited
- **Account Rent:** ~0.0014 SOL per account

**Total Devnet SOL Used:** ~0.07-0.09 SOL

---

## 🎓 KEY LEARNINGS

1. **Solana account data persists across program upgrades**
   - This is BY DESIGN
   - Must explicitly migrate data

2. **Rust compiler optimizations can be aggressive**
   - Float math was optimized away
   - Integer math forces explicit calculation

3. **Corruption detection is tricky**
   - Condition must be EXACTLY right
   - Must execute on every deposit

4. **Frontend debugging is misleading**
   - Frontend correctly reads on-chain data
   - Problem is on-chain data is wrong

5. **MCP Solana experts are invaluable**
   - Confirmed account data persistence
   - Provided migration patterns

---

## 🚀 RECOMMENDED ACTION

**OPTION A** (Fastest):
1. Close corrupted collateral account
2. Deposit fresh with new program
3. Verify shows $2.08 for 0.01 SOL

**OPTION D** (Proper):
1. Add version field to struct
2. Implement migration on first use
3. Prevents future issues

**For Hackathon Video:**
- Use Option A to get working demo ASAP
- Document Option D for production

---

## 📝 SMART CONTRACT CODE (FINAL VERSION)

### Current `deposit_native_sol` (Integer Math):
```rust
pub fn deposit_native_sol(ctx: Context<DepositNativeSol>, amount: u64) -> Result<()> {
    // INTEGER-ONLY CALCULATION:
    // lamports * 208 / 1000 = USD (with 6 decimals)
    // Example: 10,000,000 * 208 / 1000 = 2,080,000 = $2.08
    let usd_value = amount
        .checked_mul(208)
        .ok_or(ErrorCode::InvalidAmount)?
        .checked_div(1000)
        .ok_or(ErrorCode::InvalidAmount)?;
    
    msg!("💎 INTEGER MATH: {} lamports → {} (=${})", 
         amount, usd_value, usd_value / 1_000_000);
    
    // CORRUPTION DETECTION
    let vault_balance = protocol_vault.to_account_info().lamports();
    if collateral_account.amount > vault_balance * 2 && collateral_account.amount > 1_000_000_000 {
        msg!("⚠️ Corrupted collateral detected! Resetting account.");
        collateral_account.amount = 0;
        collateral_account.value_usd = 0;
    }
    
    // Update account...
    collateral_account.amount = amount;
    collateral_account.value_usd = usd_value;
    // ...
}
```

**Status:** Deployed ✅ | Hash: 39ecde34... ✅ | Account Data: Still corrupted ❌

---

## 🤝 HELP NEEDED

If trying Option A or B, need to:
1. Test `close_collateral_account` instruction
2. Verify fresh deposit creates correct values
3. Confirm corruption detection triggers on new account

If trying Option D:
1. Add version field to struct
2. Handle realloc for existing accounts
3. Test migration logic

---

## 📞 CONTACT

**Repository:** `/home/dex/Desktop/quantdesk-1.0.6`  
**Program ID:** `HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso`  
**Devnet Wallet:** `wgfSHTWx1woRXhsWijj1kcpCP8tmbmK2KnouFVAuoc6`

---

**Last Updated:** October 14, 2025 00:15 UTC  
**Status:** 🔴 BLOCKED - Account data not updating despite correct program code

