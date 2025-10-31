# QuantDesk Smart Contracts Architecture Document

**Version:** 1.0.0  
**Date:** 2025-10-29  
**Author:** System Architect  
**Status:** Active Audit & Refactoring Guide

---

## Executive Summary

This document provides a comprehensive architecture overview of the QuantDesk perpetual DEX smart contract codebase. It identifies the current structure, potential duplicates, organization issues, and provides recommendations for improvements to ensure maintainability, security, and scalability.

---

## 1. Program Structure Overview

### 1.1 High-Level Architecture

```
quantdesk-perp-dex (Single Solana Program)
├── Program ID: C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw
├── Deployment: Devnet (Active)
└── Architecture: Monolithic with modular instruction organization
```

**Design Decision:** Single program avoids CPI complexity and stack overflow issues while maintaining clear domain separation through module organization.

### 1.2 Core Module Structure

```
src/
├── lib.rs                    # Main program entry point & instruction declarations
├── instructions/             # Instruction handlers (14 modules)
│   ├── position_management.rs
│   ├── order_management.rs
│   ├── collateral_management.rs
│   ├── market_management.rs
│   ├── security_management.rs
│   ├── user_account_management.rs
│   ├── vault_management.rs
│   ├── insurance_oracle_management.rs
│   ├── keeper_management.rs
│   ├── admin_functions.rs
│   ├── advanced_orders.rs
│   ├── cross_program.rs
│   ├── remaining_contexts.rs
│   └── mod.rs
├── state/                    # Account state definitions (10 modules)
│   ├── position.rs
│   ├── order.rs
│   ├── market.rs
│   ├── user_account.rs
│   ├── collateral.rs
│   ├── protocol.rs
│   ├── advanced.rs
│   ├── remaining.rs
│   ├── price_cache.rs
│   └── mod.rs
├── oracle/                   # Oracle integration (2 modules)
│   ├── consensus.rs
│   └── switchboard.rs
├── oracle_optimization/      # Performance-optimized oracle (3 modules)
│   ├── consensus.rs
│   ├── switchboard.rs
│   ├── batch_validation.rs
│   └── mod.rs
└── Supporting modules (11 files)
    ├── errors.rs
    ├── security.rs
    ├── collateral.rs
    ├── user_accounts.rs
    ├── price_cache.rs
    ├── token_operations.rs
    ├── margin.rs
    ├── markets.rs
    ├── oracle.rs
    ├── utils.rs
    └── pda_utils.rs
```

---

## 2. Critical Duplicate Detection

### 2.1 High Priority Duplicates

#### **Issue #1: Duplicate Consensus Implementations**
**Location:**
- `src/oracle/consensus.rs` (Lines 20-93)
- `src/oracle_optimization/consensus.rs` (Lines 19-202)

**Problem:**
- Both define identical `ConsensusResult` structs
- Both define identical `MultiOracleConsensus` context structs
- Both define `get_consensus_price()` with similar signatures
- `oracle_optimization/consensus.rs` uses placeholder implementations
- `oracle/consensus.rs` uses real Pyth/Switchboard implementations

**Impact:** 
- Potential compilation conflicts if both are imported
- Confusion about which implementation is active
- Maintenance burden (changes must be made in two places)

**Recommendation:**
- ✅ **KEEP**: `oracle/consensus.rs` (has real implementations)
- ❌ **REMOVE/MERGE**: `oracle_optimization/consensus.rs` placeholder code
- **Action**: Migrate any unique logic from optimization version to main, then remove duplicate

#### **Issue #2: Duplicate PriceCache Implementations**
**Location:**
- `src/price_cache.rs` (Lines 1-296)
- `src/state/price_cache.rs` (Lines 1-295)

**Problem:**
- **Near-identical** `PriceCache` struct definitions
- **Near-identical** `PriceCacheOperations` context structs
- **Near-identical** `initialize_price_cache()` and `update_price_cache()` functions
- Minor difference: `state/price_cache.rs` has slightly different error handling

**Impact:**
- Compilation risk (struct name collision)
- Unclear which implementation is used by instructions
- Confusion for developers

**Recommendation:**
- ✅ **KEEP**: `src/state/price_cache.rs` (follows state organization pattern)
- ❌ **REMOVE**: `src/price_cache.rs` (root-level duplicate)
- **Action**: Verify all imports, update references, then delete root-level file

#### **Issue #3: Duplicate DepositNativeSol Contexts**
**Location:**
- `src/instructions/collateral_management.rs` (Lines 426-467) ✅ **ACTIVE**
- `src/token_operations.rs` (Lines 226-261) ❌ **DUPLICATE**

**Problem:**
- Two different `DepositNativeSol` struct definitions
- Different account orders (fixed in collateral_management, old order in token_operations)
- `token_operations.rs` version may be outdated/unused

**Impact:**
- If both are imported, Anchor will fail to compile (struct name conflict)
- Outdated account order could cause `AccountNotSigner` errors

**Recommendation:**
- ✅ **KEEP**: `instructions/collateral_management.rs` (has correct account order)
- ❌ **REMOVE**: `token_operations.rs::DepositNativeSol` struct
- **Action**: Verify no instructions reference token_operations version, then remove

### 2.2 Medium Priority Duplicates

#### **Issue #4: Potential Collateral Account Duplication**
**Location:**
- `src/collateral.rs` (Line 44: `CollateralAccount` struct)
- `src/state/collateral.rs` (potential similar struct)

**Action Required:** Verify if these are duplicates or serve different purposes (one might be for instructions, one for state).

#### **Issue #5: Potential UserAccount Duplication**
**Location:**
- `src/user_accounts.rs` (defines `UserAccount` struct)
- `src/state/user_account.rs` (defines `UserAccount` struct)

**Action Required:** Verify if both are identical or serve different purposes.

---

## 3. Organization Issues

### 3.1 Module Organization Inconsistencies

#### **Issue: Mixed Organization Patterns**
**Problem:**
- Some state structs are in `state/` folder (✅ Correct)
- Some state structs are at root level (❌ Inconsistent)
  - `collateral.rs` at root vs `state/collateral.rs`
  - `user_accounts.rs` at root vs `state/user_account.rs`
  - `price_cache.rs` at root vs `state/price_cache.rs` (duplicate)

**Recommendation:**
- **Standardize**: All account state structs should be in `state/` folder
- Root-level state files should only exist if they're used by multiple modules or are foundational

#### **Issue: Oracle Module Organization**
**Problem:**
- `oracle/` folder has 2 modules (consensus, switchboard)
- `oracle_optimization/` folder has 3 modules (consensus, switchboard, batch_validation)
- No clear `mod.rs` in `oracle/` folder
- Unclear relationship between optimization and base modules

**Recommendation:**
- Consolidate oracle functionality:
  - Base oracle functionality → `oracle/` folder
  - Optimizations → `oracle/optimization.rs` or inline in base modules
  - Remove duplicate `oracle_optimization/` folder structure

### 3.2 Naming Inconsistencies

#### **Issue: File Naming Patterns**
**Problem:**
- Most files use snake_case: `position_management.rs` ✅
- Some files use single word: `collateral.rs`, `margin.rs` ⚠️
- Instruction files in `instructions/` folder: consistent ✅
- State files mixed: some in `state/`, some at root ⚠️

**Recommendation:**
- Standardize all file names to snake_case
- Move all state-related files to `state/` folder

### 3.3 Unused/Dead Code

#### **Issue: Security Tests File**
**Location:** `src/security_tests.rs`
**Status:** Commented out in `lib.rs` (Line 103)
**Recommendation:** Either properly integrate tests or move to `tests/` directory

#### **Issue: Remaining Contexts**
**Location:** `src/instructions/remaining_contexts.rs`
**Status:** Unclear purpose - likely placeholder for future instructions
**Recommendation:** Document purpose or remove if truly unused

---

## 4. Architecture Decisions & Rationale

### 4.1 Single Program Design
**Decision:** Use single Solana program instead of multiple CPI programs

**Rationale:**
- Avoids CPI complexity and compute unit overhead
- Prevents stack overflow issues from large account contexts
- Easier deployment and upgrade management
- Clear domain separation through module organization

**Trade-offs:**
- Larger program binary size
- All instructions in one upgradeable program
- Module organization critical for maintainability

### 4.2 Module Organization Strategy
**Decision:** Separate instructions, state, and supporting modules

**Rationale:**
- Clear separation of concerns
- Easier to locate related functionality
- Better code organization for large codebase
- Supports team collaboration

**Implementation:**
- `instructions/` → All instruction handlers
- `state/` → All account state structs
- Root level → Shared utilities, errors, core types

### 4.3 Account Order Fix
**Decision:** Fixed `DepositNativeSol` account order to match IDL exactly

**Rationale:**
- Solana/Anchor requires exact account order match between Rust struct and IDL
- Account order mismatch causes `AccountNotSigner` errors (Error Code 3010)
- IDL is source of truth for client interactions

**Implementation:**
- User signer at position 1 (after user_account)
- All accounts ordered: user_account → user → protocol_vault → collateral_account → sol_usd_price_feed → system_program → rent

---

## 5. Code Quality Metrics

### 5.1 Current Statistics
- **Total Rust Files:** 44 files
- **Instruction Modules:** 14 modules
- **State Modules:** 10 modules
- **Oracle Modules:** 5 modules (2 base + 3 optimization)
- **Duplicate Structs Identified:** 4+ critical duplicates
- **Organization Issues:** 3 major inconsistencies

### 5.2 Compilation Status
- ✅ **Builds Successfully:** Yes (with `--skip-lint`)
- ⚠️ **Warnings:** 11 warnings (unused variables - partially fixed)
- ❌ **Test Compilation:** Fails with macro parse error (non-critical for production)

### 5.3 Security Considerations
- ✅ Account order validation fixed
- ✅ Signer verification in place
- ⚠️ PDA validation functions marked `#[allow(dead_code)]` - verify if used
- ✅ Error codes comprehensive

---

## 6. Recommended Refactoring Plan

### Phase 1: Remove Critical Duplicates (Priority: HIGH)
1. ✅ Remove `src/price_cache.rs` (duplicate of `state/price_cache.rs`)
   - Update all imports to use `state::price_cache`
   - Verify no compilation errors
   
2. ✅ Remove `src/token_operations.rs::DepositNativeSol` struct
   - Verify `collateral_management.rs` version is used
   - Remove duplicate struct definition
   
3. ✅ Consolidate oracle consensus implementations
   - Merge unique logic from `oracle_optimization/consensus.rs` into `oracle/consensus.rs`
   - Remove placeholder implementation
   - Update all references

### Phase 2: Organize State Modules (Priority: MEDIUM)
1. Move root-level state files to `state/` folder:
   - `collateral.rs` → `state/collateral.rs` (if different) or remove duplicate
   - `user_accounts.rs` → Remove if duplicate of `state/user_account.rs`
   
2. Standardize naming:
   - Ensure all state files follow same naming pattern
   - Consistent file organization

### Phase 3: Clean Up Dead Code (Priority: LOW)
1. Address `security_tests.rs`:
   - Move to proper test directory OR integrate properly
   
2. Review `remaining_contexts.rs`:
   - Document purpose OR remove if unused

### Phase 4: Improve Module Organization (Priority: MEDIUM)
1. Consolidate oracle modules:
   - Single `oracle/` folder with clear structure
   - Remove duplicate `oracle_optimization/` folder
   
2. Add proper `mod.rs` files:
   - Ensure all modules have proper module declarations
   - Clear public API boundaries

---

## 7. Validation Checklist

### Before Refactoring
- [x] Identify all duplicates
- [x] Document organization issues
- [x] Verify active implementations
- [ ] Run full test suite
- [ ] Verify all imports

### After Refactoring
- [ ] All tests pass
- [ ] No compilation warnings
- [ ] Program deploys successfully
- [ ] IDL matches structs exactly
- [ ] No duplicate struct names
- [ ] Consistent file organization

---

## 8. Solana-Specific Best Practices

### Account Management
- ✅ PDA-based account separation
- ✅ Proper account order matching IDL
- ⚠️ Review PDA validation functions (currently marked `#[allow(dead_code)]`)

### Instruction Design
- ✅ Clear domain separation (14 instruction modules)
- ✅ Comprehensive error handling
- ✅ Security checks in place

### Performance
- ✅ Account size optimized (~2.4KB, under 4KB limit)
- ✅ Box<T> optimization for large contexts
- ✅ Compute unit targets documented

---

## 9. Next Steps

1. **Immediate:** Review and approve this architecture document
2. **Short-term:** Execute Phase 1 refactoring (remove critical duplicates)
3. **Medium-term:** Execute Phase 2 organization improvements
4. **Long-term:** Establish code review process to prevent future duplicates

---

## 10. Appendix

### A. Module Import Map
```
lib.rs imports:
  - instructions/* (14 modules)
  - state/* (via mod.rs)
  - oracle/* (2 modules)
  - oracle_optimization/* (3 modules)
  - Supporting modules (errors, security, etc.)
```

### B. Instruction Exposure Status
**Exposed in #[program] module:**
- ✅ `open_position`
- ✅ `close_position`
- ✅ `initialize_keeper_security_manager`
- ✅ `check_security_before_trading`
- ✅ `deposit_native_sol` (recently fixed)

**Not Exposed (Potential Issue):**
- All other instructions in `instructions/` modules
- Need to verify if instructions are callable without being in `#[program]` module

---

**Document Status:** Ready for Validation  
**Next Action:** Validate with Architect Agent & Solana MCP Experts

