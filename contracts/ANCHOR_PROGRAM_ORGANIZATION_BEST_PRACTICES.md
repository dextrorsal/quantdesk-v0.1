# Anchor Program Organization Best Practices

**Version:** 1.0.0  
**Date:** 2025-10-29  
**Based on:** Solana/Anchor Expert Recommendations + Drift Protocol Patterns  
**Status:** Reference Guide for QuantDesk Development

---

## Executive Summary

This document provides industry-standard best practices for organizing large Anchor programs to minimize bugs, prevent compilation conflicts, and maintain code quality. These practices are derived from Solana/Anchor framework documentation, expert recommendations, and analysis of production protocols like Drift.

---

## 1. Core Architecture Pattern: Instructions + State Separation

### 1.1 Recommended Folder Structure

**Anchor 0.29+ Multiple Files Template Pattern:**

```
programs/your-program/src/
├── lib.rs                    # Main entry point, declare modules
├── instructions/             # All instruction handlers
│   ├── mod.rs               # Module declarations
│   ├── deposit.rs           # Deposit instruction + its Accounts struct
│   ├── withdraw.rs          # Withdraw instruction + its Accounts struct
│   ├── open_position.rs     # Open position instruction + Accounts struct
│   └── ...
├── state/                    # All account state structs
│   ├── mod.rs               # Module declarations
│   ├── user_account.rs      # UserAccount struct
│   ├── position.rs          # Position struct
│   ├── market.rs            # Market struct
│   └── ...
├── error.rs                  # Error codes
├── controller/               # Business logic (optional, for large programs)
│   ├── mod.rs
│   ├── position.rs          # Position management logic
│   └── collateral.rs        # Collateral management logic
└── math/                     # Mathematical operations (optional)
    ├── mod.rs
    ├── amm.rs               # AMM calculations
    └── funding.rs            # Funding rate calculations
```

### 1.2 Why This Structure Works

✅ **Prevents Duplicate Struct Conflicts**
- Rust module system creates namespaces automatically
- `state::PriceCache` vs `oracle::PriceCache` are different types
- No compilation conflicts if both exist

✅ **Clear Separation of Concerns**
- Instructions → What users can do
- State → What data is stored
- Controller → Business logic
- Math → Pure calculations

✅ **Scalable Organization**
- Easy to find related code
- Supports large teams
- Clear dependencies

---

## 2. Preventing Duplicate Struct Definitions

### 2.1 The Problem

**DO NOT DO THIS:**
```rust
// src/price_cache.rs
#[account]
pub struct PriceCache { ... }

// src/state/price_cache.rs
#[account]
pub struct PriceCache { ... }  // ❌ COMPILATION ERROR if both imported
```

If both files are imported in the same scope, Rust will report a duplicate type definition.

### 2.2 Solution: Use Module Namespaces

**CORRECT APPROACH:**
```rust
// src/state/mod.rs
pub mod price_cache;
pub use price_cache::PriceCache;  // Re-export if needed

// src/lib.rs
pub mod state;
pub mod oracle;  // Different namespace = no conflict

// Instructions can use:
use crate::state::PriceCache;  // ✅ Clear namespace
```

### 2.3 Drift Protocol Pattern

Drift uses module separation to avoid conflicts:

```rust
// lib.rs
pub mod state;           // All account state structs
pub mod controller;      // Business logic
pub mod math;           // Calculations
pub mod instructions;    // Instruction handlers

// Each module has its own namespace
// No conflicts even if structs have similar purposes
```

---

## 3. Module Organization Rules

### 3.1 State Structs: Centralized Location

**Rule:** All `#[account]` structs should be in the `state/` folder.

**Rationale:**
- Single source of truth for account definitions
- Easier to manage account discriminators
- Clear separation from instruction logic

**Example:**
```rust
// state/mod.rs
pub mod user_account;
pub mod position;
pub mod market;
pub mod collateral;

// Each file contains ONE account struct
// state/user_account.rs:
#[account]
pub struct UserAccount {
    pub authority: Pubkey,
    pub positions: Vec<Position>,
}
```

### 3.2 Instructions: Co-locate with Accounts Structs

**Rule:** Each instruction file contains:
- The instruction handler function
- Its `#[derive(Accounts)]` struct
- Any instruction-specific helper functions

**Example:**
```rust
// instructions/deposit.rs
use crate::state::UserAccount;
use crate::state::CollateralAccount;

#[derive(Accounts)]
pub struct DepositNativeSol<'info> {
    #[account(mut)]
    pub user: Signer<'info>,
    #[account(mut)]
    pub user_account: Account<'info, UserAccount>,
    // ... other accounts
}

pub fn deposit_native_sol(ctx: Context<DepositNativeSol>, amount: u64) -> Result<()> {
    // Instruction logic
    Ok(())
}
```

**Benefit:** Related code stays together, easier to maintain.

### 3.3 Supporting Modules: Logical Grouping

**Pattern:** Organize by function, not by type.

**✅ GOOD:**
```
controller/
  ├── position.rs      # Position-related logic
  ├── collateral.rs     # Collateral-related logic
  └── market.rs         # Market-related logic

math/
  ├── amm.rs           # AMM calculations
  ├── funding.rs        # Funding calculations
  └── margin.rs         # Margin calculations
```

**❌ BAD:**
```
utils/
  ├── position_math.rs
  ├── position_state.rs
  ├── position_controller.rs  # Scattered!
```

---

## 4. Naming Conventions

### 4.1 File Naming

**Standard:** Use `snake_case` for all files.

**✅ GOOD:**
- `position_management.rs`
- `collateral_account.rs`
- `oracle_validation.rs`

**❌ BAD:**
- `PositionManagement.rs` (PascalCase)
- `collateralAccount.rs` (camelCase)
- `oracle-validation.rs` (hyphens)

### 4.2 Struct Naming

**Rule:** Use descriptive, unique names within their module.

**Pattern:** `{Action}{Entity}` for Accounts structs

**✅ GOOD:**
```rust
// instructions/collateral_management.rs
pub struct DepositNativeSol<'info> { ... }
pub struct WithdrawNativeSol<'info> { ... }

// state/
pub struct CollateralAccount { ... }      // Different namespace
pub struct UserCollateralAccount { ... }  // More specific if needed
```

**❌ BAD:**
```rust
pub struct Deposit<'info> { ... }        // Too generic
pub struct Deposit1<'info> { ... }       // Non-descriptive
pub struct DepositNativeSol<'info> { ... }  // In multiple files (duplicate)
```

---

## 5. Import Strategy

### 5.1 Module Declaration Order

**Recommended order in `lib.rs`:**

```rust
// 1. External dependencies (if needed at root)
use anchor_lang::prelude::*;

// 2. Core modules first
pub mod error;
pub mod state;
pub mod instructions;

// 3. Supporting modules
pub mod controller;
pub mod math;
pub mod oracle;

// 4. Re-exports for convenience (optional)
pub use state::*;
pub use error::ErrorCode;
```

### 5.2 Avoiding Circular Dependencies

**Rule:** Dependencies should flow in one direction:

```
lib.rs
  └── instructions/
        └── imports from state/, controller/, math/
              └── NO imports back to instructions/
```

**✅ GOOD:**
```rust
// instructions/deposit.rs
use crate::state::UserAccount;        // ✅ State → Instructions
use crate::controller::validate_deposit;  // ✅ Controller → Instructions
```

**❌ BAD:**
```rust
// state/user_account.rs
use crate::instructions::DepositNativeSol;  // ❌ Circular dependency!
```

---

## 6. Preventing Common Bugs

### 6.1 Account Order Mismatch

**Issue:** Account order in Rust struct must match IDL exactly.

**Solution:**
1. **Always verify IDL after building**
2. **Keep account order consistent**
3. **Document account order in comments**

**Example:**
```rust
#[derive(Accounts)]
pub struct DepositNativeSol<'info> {
    // CRITICAL: Account order MUST match IDL exactly
    // IDL order: user_account, user, protocol_vault, collateral_account, ...
    
    // Position 0
    #[account(mut)]
    pub user_account: Account<'info, UserAccount>,
    
    // Position 1 - MUST be signer
    #[account(mut)]
    pub user: Signer<'info>,
    
    // Position 2
    #[account(mut)]
    pub protocol_vault: Account<'info, ProtocolVault>,
    // ... rest in exact IDL order
}
```

### 6.2 Duplicate Import Conflicts

**Issue:** Importing the same struct from multiple locations.

**Solution:** Use explicit module paths or centralized re-exports.

**✅ GOOD:**
```rust
// Always use explicit paths
use crate::state::PriceCache;
use crate::state::UserAccount;

// OR use re-exports in mod.rs
use crate::state::*;  // If mod.rs re-exports everything
```

**❌ BAD:**
```rust
// Mixing root and module imports
use crate::PriceCache;          // From root?
use crate::state::PriceCache;   // From module?
// Which one is used? Confusing!
```

---

## 7. Large Program Organization (10+ Instructions)

### 7.1 Drift Protocol Pattern (Production Example)

Drift organizes a massive program (~100+ instructions) like this:

```
src/
├── lib.rs
├── instructions/        # Instruction handlers
│   ├── mod.rs
│   ├── deposit.rs
│   ├── withdraw.rs
│   └── ... (many more)
├── state/               # Account state structs
│   ├── mod.rs
│   ├── user.rs
│   ├── perp_market.rs
│   └── ... (many more)
├── controller/          # Business logic layer
│   ├── mod.rs
│   ├── position.rs
│   ├── amm.rs
│   └── ... (many more)
├── math/                # Mathematical operations
│   ├── mod.rs
│   ├── amm.rs
│   ├── funding.rs
│   └── ... (many more)
├── error.rs             # Error codes
└── ids.rs               # Program IDs for CPI
```

### 7.2 Benefits of This Pattern

1. **Clear Separation:** Instructions, state, and logic are separated
2. **No Duplicates:** Module namespaces prevent conflicts
3. **Maintainable:** Easy to locate code by function
4. **Scalable:** Can grow to 200+ instructions without chaos

### 7.3 QuantDesk Application

**Current Structure (Good Foundation):**
```
src/
├── lib.rs
├── instructions/        ✅ Correct
│   └── (14 modules)
├── state/               ✅ Correct
│   └── (10 modules)
├── oracle/              ⚠️ Needs consolidation
└── oracle_optimization/ ⚠️ Needs consolidation
```

**Recommended Improvements:**
1. ✅ Remove duplicate `price_cache.rs` (keep `state/price_cache.rs`)
2. ✅ Consolidate `oracle/` and `oracle_optimization/` folders
3. ✅ Move root-level state files to `state/` folder

---

## 8. Module Declaration Best Practices

### 8.1 Proper `mod.rs` Structure

**Example: `instructions/mod.rs`**

```rust
// Module declarations
pub mod position_management;
pub mod order_management;
pub mod collateral_management;
pub mod market_management;
// ... all instruction modules

// Re-exports (optional, for convenience)
pub use position_management::*;
pub use order_management::*;
// ... only if you want wildcard imports
```

### 8.2 When to Re-export

**✅ RE-EXPORT when:**
- Structs are commonly used across multiple modules
- You want a cleaner API surface
- Re-exports are organized and documented

**❌ DON'T RE-EXPORT when:**
- It creates ambiguity (which `PriceCache`?)
- You have duplicate names
- Re-exports would create circular dependencies

**Example:**
```rust
// state/mod.rs
pub mod price_cache;
pub mod user_account;

// ✅ GOOD: Selective re-exports
pub use price_cache::PriceCache;
pub use user_account::UserAccount;

// ❌ BAD: Wildcard re-export with potential conflicts
pub use price_cache::*;
pub use user_account::*;  // If both have similar names
```

---

## 9. Error Prevention Checklist

### Before Adding New Modules:

- [ ] Check if similar struct/function already exists
- [ ] Use descriptive, unique names
- [ ] Place in correct folder (`state/` vs `instructions/`)
- [ ] Declare in appropriate `mod.rs`
- [ ] Use explicit import paths
- [ ] Verify no circular dependencies

### Before Building:

- [ ] Run `anchor build` to check for conflicts
- [ ] Verify IDL matches struct account orders
- [ ] Check for unused imports (clippy warnings)
- [ ] Ensure all `pub mod` declarations are correct

### Code Review Checklist:

- [ ] No duplicate struct names across modules
- [ ] Account order matches IDL
- [ ] Imports use explicit paths
- [ ] File organization follows patterns
- [ ] Module namespaces are respected

---

## 10. Common Pitfalls and Solutions

### Pitfall 1: Root-Level State Files

**Problem:**
```
src/
├── collateral.rs          ❌ State struct at root
└── state/
    └── collateral.rs      ✅ Also exists here
```

**Solution:** Remove root-level file, use `state/collateral.rs` only.

### Pitfall 2: Mixed Organization

**Problem:**
```
src/
├── instructions/
│   └── deposit.rs
├── deposit.rs            ❌ Also at root
```

**Solution:** Pick one location and be consistent.

### Pitfall 3: Oracle Module Duplication

**Problem:**
```
src/
├── oracle/
│   └── consensus.rs
└── oracle_optimization/
    └── consensus.rs       ❌ Duplicate!
```

**Solution:** Consolidate into `oracle/` folder, use feature flags for optimizations.

---

## 11. QuantDesk-Specific Recommendations

### Current Issues Identified:

1. **Duplicate PriceCache:** `src/price_cache.rs` vs `src/state/price_cache.rs`
   - **Action:** Remove `src/price_cache.rs`, keep `state/price_cache.rs`

2. **Duplicate ConsensusResult:** `oracle/consensus.rs` vs `oracle_optimization/consensus.rs`
   - **Action:** Merge into `oracle/consensus.rs`, use feature flags for optimizations

3. **Duplicate DepositNativeSol:** `token_operations.rs` vs `instructions/collateral_management.rs`
   - **Action:** Remove from `token_operations.rs`, use `instructions/` version

4. **Mixed Organization:** State structs at root and in `state/` folder
   - **Action:** Move all state structs to `state/` folder

### Target Structure:

```
src/
├── lib.rs
├── error.rs
├── instructions/
│   ├── mod.rs
│   └── (14 instruction modules)
├── state/
│   ├── mod.rs
│   └── (ALL account state structs here)
├── oracle/              # Consolidated oracle
│   ├── mod.rs
│   ├── consensus.rs     # Merged from optimization
│   └── switchboard.rs
├── controller/          # Future: Business logic layer
└── math/                # Future: Mathematical operations
```

---

## 12. References and Resources

### Official Documentation:
- [Anchor Program Structure](https://www.anchor-lang.com/docs/basics/program-structure)
- [Anchor Multiple Files Template](https://github.com/coral-xyz/anchor/tree/master/tests/composite)
- [Solana StackExchange: Program Organization](https://solana.stackexchange.com/questions/7670/how-to-organize-programs-anchor-code-properly)

### Production Examples:
- **Drift Protocol:** `instructions/`, `state/`, `controller/`, `math/` separation
- **Serum/DEX:** Similar patterns with module organization

### Expert Recommendations:
- Use Rust module system for namespacing
- Keep state structs centralized in `state/` folder
- Co-locate instruction handlers with their Accounts structs
- Avoid duplicates by using unique module paths

---

## 13. Quick Reference: Decision Tree

**Where should this code go?**

```
Is it an #[account] struct?
├─ Yes → state/{name}.rs
└─ No → Is it an instruction handler?
    ├─ Yes → instructions/{name}.rs (include Accounts struct)
    └─ No → Is it business logic?
        ├─ Yes → controller/{name}.rs
        └─ No → Is it pure math?
            ├─ Yes → math/{name}.rs
            └─ No → root level utility file
```

**Does this struct name already exist?**
```
├─ Yes → Is it in a different namespace/module?
│   ├─ Yes → ✅ OK (use explicit module path)
│   └─ No → ❌ RENAME (avoid confusion)
└─ No → ✅ OK
```

---

**Document Status:** Active Reference  
**Next Action:** Apply these patterns during Phase 1 refactoring

---

**Maintained By:** QuantDesk Architecture Team  
**Last Updated:** 2025-10-29

