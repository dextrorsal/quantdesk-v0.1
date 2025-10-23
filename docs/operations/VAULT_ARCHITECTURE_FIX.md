# Vault Architecture Fix - Drift-Inspired Pattern

## 🎯 **Expert Recommendations Summary**

Based on consultation with Solana and Anchor experts, here's how successful DEXs like Drift structure their vaults:

### **Key Insights:**

1. **Separate Metadata from Treasury**: Drift uses separate PDAs for:
   - **Metadata Account**: Tracks deposits, withdrawals, fees (program-owned, stores data)
   - **Treasury Account**: Holds native SOL (program-owned, stores lamports)

2. **Account Ownership**:
   - Your `protocol_vault` is owned by YOUR PROGRAM, not the System Program
   - This means you **CANNOT** use `system_program::transfer` (that's for System Program-owned accounts)
   - You **MUST** use direct lamport manipulation

3. **Security Considerations**:
   - Always maintain rent-exemption
   - Validate PDA derivation strictly
   - Check for overflow/underflow
   - Guard against reentrancy

---

## 🔧 **Current Architecture Issues**

### **Problem 1: Mixed Responsibilities**
```rust
// ❌ CURRENT: One PDA does both metadata AND treasury
#[account]
pub struct ProtocolSolVault {
    pub total_deposits: u64,      // Metadata
    pub total_withdrawals: u64,   // Metadata
    pub is_active: bool,          // Metadata
    pub bump: u8,                 // Metadata
}
// AND this account also holds SOL lamports!
```

**Why this causes issues:**
- Program-owned accounts (Account<'info, T>) cannot use `system_program::transfer`
- Mixing concerns makes the code harder to reason about
- Direct lamport manipulation works BUT is less clear

### **Problem 2: Incorrect Transfer Method**
```rust
// ❌ WRONG: Using system_program::transfer on program-owned account
system_program::transfer(
    CpiContext::new_with_signer(...),
    amount,
)?;
// This triggers: "Cross-program invocation with unauthorized signer"
```

---

## ✅ **Solution Options**

### **Option A: Direct Lamport Manipulation (Quick Fix)**

**Use this if you want minimal changes:**

```rust
pub fn withdraw_native_sol(ctx: Context<WithdrawNativeSol>, amount: u64) -> Result<()> {
    let protocol_vault = &mut ctx.accounts.protocol_vault;
    let user = &ctx.accounts.user;
    
    // Validation
    require!(amount > 0, ErrorCode::InvalidAmount);
    require!(protocol_vault.is_active, ErrorCode::VaultInactive);
    
    // Get current lamports
    let vault_lamports = protocol_vault.to_account_info().lamports();
    let min_rent = Rent::get()?.minimum_balance(protocol_vault.to_account_info().data_len());
    
    // Ensure we maintain rent exemption
    require!(
        vault_lamports.checked_sub(amount).unwrap() >= min_rent,
        ErrorCode::InsufficientFunds
    );
    
    // Direct lamport manipulation (program-owned account)
    **protocol_vault.to_account_info().try_borrow_mut_lamports()? -= amount;
    **user.to_account_info().try_borrow_mut_lamports()? += amount;
    
    // Update metadata
    protocol_vault.total_withdrawals = protocol_vault.total_withdrawals
        .checked_add(amount)
        .ok_or(ErrorCode::Overflow)?;
    
    msg!("✅ Withdrawal complete: {} lamports", amount);
    Ok(())
}
```

**Pros:**
- ✅ Minimal code changes
- ✅ Works with current architecture
- ✅ Direct and efficient

**Cons:**
- ⚠️ Less intuitive (mixing metadata and treasury in one account)
- ⚠️ Harder to audit

---

### **Option B: Drift-Style Separation (Recommended for Production)**

**Use this for a production-grade architecture:**

```rust
// Metadata Account - stores statistics
#[account]
pub struct ProtocolVaultMetadata {
    pub total_deposits: u64,
    pub total_withdrawals: u64,
    pub is_active: bool,
    pub treasury_bump: u8,
    pub metadata_bump: u8,
}

// Treasury Account - holds SOL (no data structure needed)
// Just use AccountInfo<'info> or SystemAccount<'info>

#[derive(Accounts)]
pub struct WithdrawNativeSol<'info> {
    // Metadata PDA
    #[account(
        mut,
        seeds = [b"vault_metadata"],
        bump = metadata.metadata_bump,
    )]
    pub metadata: Account<'info, ProtocolVaultMetadata>,
    
    // Treasury PDA (holds SOL)
    /// CHECK: Safe - treasury PDA for SOL storage
    #[account(
        mut,
        seeds = [b"vault_treasury"],
        bump = metadata.treasury_bump,
    )]
    pub treasury: AccountInfo<'info>,
    
    #[account(mut)]
    pub user: Signer<'info>,
    
    pub system_program: Program<'info, System>,
}

pub fn withdraw_native_sol(ctx: Context<WithdrawNativeSol>, amount: u64) -> Result<()> {
    let metadata = &mut ctx.accounts.metadata;
    
    // Validation
    require!(metadata.is_active, ErrorCode::VaultInactive);
    require!(amount > 0, ErrorCode::InvalidAmount);
    
    // Prepare signer seeds for treasury
    let treasury_bump = metadata.treasury_bump;
    let signer_seeds: &[&[&[u8]]] = &[&[
        b"vault_treasury",
        &[treasury_bump],
    ]];
    
    // CPI to System Program (treasury signs via program)
    let transfer_ix = system_program::Transfer {
        from: ctx.accounts.treasury.to_account_info(),
        to: ctx.accounts.user.to_account_info(),
    };
    
    let cpi_ctx = CpiContext::new_with_signer(
        ctx.accounts.system_program.to_account_info(),
        transfer_ix,
        signer_seeds,
    );
    
    system_program::transfer(cpi_ctx, amount)?;
    
    // Update metadata
    metadata.total_withdrawals = metadata.total_withdrawals
        .checked_add(amount)
        .ok_or(ErrorCode::Overflow)?;
    
    msg!("✅ Withdrawal complete: {} lamports", amount);
    Ok(())
}
```

**Pros:**
- ✅ Clean separation of concerns
- ✅ Follows Drift/production patterns
- ✅ More auditable
- ✅ Treasury can be a simple SystemAccount

**Cons:**
- ⚠️ Requires refactoring
- ⚠️ Two PDAs to manage

---

## 📊 **Comparison: Drift vs Your Current Setup**

| Aspect | Drift Pattern | Your Current Setup |
|--------|---------------|-------------------|
| **Metadata Storage** | Separate PDA | Combined with treasury |
| **Treasury** | Separate SystemAccount PDA | Same as metadata |
| **Transfer Method** | `system_program::transfer` with signer | Should use direct lamports |
| **Clarity** | Very clear | Mixed responsibilities |
| **Auditability** | Easy | Harder |

---

## 🚀 **Recommended Action Plan**

### **Phase 1: Quick Fix (For Hackathon)**
1. Use **Option A** (direct lamport manipulation)
2. Add rent-exemption checks
3. Deploy and test withdrawals
4. Document the architecture for future refactor

### **Phase 2: Production Refactor (Post-Hackathon)**
1. Implement **Option B** (Drift-style separation)
2. Migrate existing data
3. Add comprehensive tests
4. Audit the new architecture

---

## 🔒 **Security Checklist**

- [ ] Maintain rent-exemption on all PDAs
- [ ] Validate all PDA derivations
- [ ] Use checked arithmetic (no overflows)
- [ ] Implement proper access controls
- [ ] Test edge cases (withdraw all, withdraw more than balance)
- [ ] Guard against reentrancy
- [ ] Add rate limiting for withdrawals
- [ ] Implement emergency pause mechanism

---

## 📝 **Implementation Status**

**Current State:**
- ✅ Metadata account exists (`ProtocolSolVault`)
- ✅ Treasury holds SOL (same account)
- ❌ Withdrawal failing (using wrong transfer method)

**Next Steps:**
1. Update `withdraw_native_sol` to use direct lamport manipulation
2. Add rent-exemption validation
3. Deploy and test
4. Plan Phase 2 refactor for production

---

## 🎓 **Learning Resources**

- [Drift Protocol Architecture](https://docs.drift.trade)
- [Anchor PDA Best Practices](https://www.anchor-lang.com/docs/basics/pda)
- [Solana Program Derived Addresses](https://docs.solana.com/developing/programming-model/calling-between-programs#program-derived-addresses)
- [Direct Lamport Manipulation Example](https://solana.stackexchange.com/questions/1158/how-to-withdraw-sol-from-a-pda)

