# Refactoring Plan: Root-Level State Files

**Date:** 2025-10-29  
**Based On:** ANCHOR_PROGRAM_ORGANIZATION_BEST_PRACTICES.md  
**Status:** Verification Before Implementation

---

## Analysis of Current Root-Level Files

### File: `src/collateral.rs`
**Contents:**
- `CollateralType` enum (shared type, not an account)
- `CollateralAccount` struct `#[account]` ✅ **SHOULD BE IN STATE/**
- Helper functions (`calculate_initial_margin_contribution`, etc.)

**Current Usage:**
- Referenced as `crate::collateral::CollateralAccount` in instructions
- Used across multiple modules

**Best Practice Decision:**
- ✅ Move `CollateralAccount` struct to `state/collateral.rs` (merge with existing `CollateralConfig`)
- ✅ Keep `CollateralType` enum in `collateral.rs` or move to shared location
- ✅ Keep helper functions in `collateral.rs` (or move to `controller/collateral.rs`)

---

### File: `src/user_accounts.rs`
**Contents:**
- `UserAccount` struct `#[account]` ✅ **DUPLICATE** - same as `state/user_account.rs`
- Instruction Accounts structs:
  - `CreateUserAccount<'info>`
  - `UpdateUserAccount<'info>`
  - `CloseUserAccount<'info>`
- Helper functions

**Current Usage:**
- Referenced as `crate::user_accounts::UserAccount` in instructions
- Instruction Accounts structs might be used

**Best Practice Decision:**
- ✅ Remove duplicate `UserAccount` from `user_accounts.rs`
- ✅ Move instruction Accounts structs to `instructions/user_account_management.rs`
- ✅ Update all imports to use `crate::state::UserAccount`
- ❓ **Decision Needed:** Keep `user_accounts.rs` for helper functions or delete entirely?

---

### File: `src/markets.rs`
**Contents:**
- Just a re-export: `pub use crate::state::market::Market;`

**Best Practice Decision:**
- ✅ **DELETE** - unnecessary re-export. Instructions should import directly from `state::market`

---

### File: `src/margin.rs`
**Contents:**
- Pure utility functions (no account structs)
- Functions like `validate_leverage`, `calculate_margin_requirement`, etc.

**Best Practice Decision:**
- ✅ **KEEP AT ROOT** or move to `math/` or `utils/` folder (per best practices, pure functions can stay at root)
- No account structs, so not violating state organization rules

---

## Implementation Plan

### Step 1: Handle `collateral.rs`
1. Move `CollateralAccount` struct from `src/collateral.rs` to `src/state/collateral.rs`
2. Update `state/mod.rs` to include `pub use collateral::CollateralAccount;`
3. Keep `CollateralType` enum in `collateral.rs` (or create `types.rs` for shared enums)
4. Update imports: `crate::collateral::CollateralAccount` → `crate::state::CollateralAccount`

### Step 2: Handle `user_accounts.rs`
1. Verify `state/user_account.rs` is the same struct
2. Move instruction Accounts structs (`CreateUserAccount`, etc.) to `instructions/user_account_management.rs`
3. Delete `UserAccount` struct from `user_accounts.rs`
4. **Decision:** Delete entire `user_accounts.rs` OR keep only helper functions?
5. Update imports: `crate::user_accounts::UserAccount` → `crate::state::UserAccount`
6. Update `lib.rs` to remove `pub mod user_accounts;`

### Step 3: Handle `markets.rs`
1. Remove file `src/markets.rs`
2. Update `lib.rs` to remove `pub mod markets;`
3. Verify all imports use `crate::state::market::Market`

### Step 4: Keep `margin.rs`
- No changes needed (pure utility functions)

---

## Verification Checklist

- [ ] All `#[account]` structs are in `state/` folder
- [ ] No duplicate account structs
- [ ] All instruction Accounts structs are in `instructions/` folder
- [ ] All imports updated correctly
- [ ] `anchor build` succeeds
- [ ] No compilation errors
- [ ] Module structure follows best practices

---

**Next Action:** Review this plan, then execute if approved.

