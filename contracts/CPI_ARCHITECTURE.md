# Cross-Program Invocation (CPI) Architecture

## Overview

The QuantDesk protocol uses Cross-Program Invocations (CPI) to enable communication between specialized programs while maintaining modularity and reducing stack overflow issues.

## Program Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Core Program  │    │ Trading Program  │    │Collateral Program│
│                 │    │                 │    │                 │
│ • User Accounts │◄──►│ • Positions     │◄──►│ • Deposits     │
│ • Markets      │    │ • Orders        │    │ • Withdrawals  │
│ • Basic Ops    │    │ • Advanced Ops  │    │ • Cross-Collat │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         ▲                       ▲                       ▲
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│ Security Program│    │ Oracle Program  │    │                 │
│                 │    │                 │    │                 │
│ • Circuit Breaks│◄──►│ • Price Feeds  │    │                 │
│ • Keeper Mgmt   │    │ • Insurance    │    │                 │
│ • Risk Mgmt     │    │ • Emergency     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## CPI Interfaces

### 1. Trading Program CPI Calls

**To Collateral Program:**
- `deposit_native_sol()` - Deposit SOL as collateral
- `withdraw_native_sol()` - Withdraw SOL from collateral

**To Security Program:**
- `check_security_before_trading()` - Security checks before trading
- `check_keeper_authorization()` - Verify keeper permissions

**To Oracle Program:**
- `update_oracle_price()` - Update market prices

**To Core Program:**
- `update_user_account()` - Update user account state

### 2. Collateral Program CPI Calls

**To Core Program:**
- `update_user_account()` - Update user account state

**To Security Program:**
- `check_security_before_trading()` - Security checks before collateral ops

**To Oracle Program:**
- `get_current_price()` - Get current asset prices

### 3. Security Program CPI Calls

**To Trading Program:**
- `liquidate_position()` - Execute liquidations

**To Core Program:**
- `update_user_account()` - Update user account state

**To Oracle Program:**
- `get_current_price()` - Get current prices for risk assessment

### 4. Oracle Program CPI Calls

**To Core Program:**
- `update_user_account()` - Update user account state

**To Security Program:**
- `check_security_before_trading()` - Security checks before oracle updates

### 5. Core Program CPI Calls

**To Collateral Program:**
- `initialize_collateral_account()` - Initialize collateral accounts

**To Security Program:**
- `check_security_before_trading()` - Security checks before user ops

**To Oracle Program:**
- `get_current_price()` - Get current prices

## CPI Implementation Example

```rust
// Trading Program calling Collateral Program
pub fn deposit_collateral_via_cpi(
    ctx: Context<DepositCollateralViaCpi>,
    amount: u64,
) -> Result<()> {
    // CPI call to Collateral Program
    collateral::deposit_native_sol(
        CpiContext::new(
            ctx.accounts.collateral_program.to_account_info(),
            collateral::DepositNativeSolCpi {
                user: ctx.accounts.user.to_account_info(),
                user_account: ctx.accounts.user_account.to_account_info(),
                collateral_account: ctx.accounts.collateral_account.to_account_info(),
                protocol_vault: ctx.accounts.protocol_vault.to_account_info(),
                system_program: ctx.accounts.system_program.to_account_info(),
            },
        ),
        amount,
    )?;
    
    Ok(())
}
```

## CPI Account Contexts

Each CPI call requires specific account contexts:

```rust
#[derive(Accounts)]
pub struct DepositNativeSolCpi<'info> {
    pub user: AccountInfo<'info>,
    pub user_account: AccountInfo<'info>,
    pub collateral_account: AccountInfo<'info>,
    pub protocol_vault: AccountInfo<'info>,
    pub system_program: AccountInfo<'info>,
}
```

## Benefits of CPI Architecture

1. **Modularity**: Each program handles specific functionality
2. **Stack Optimization**: Reduces stack usage per program
3. **Maintainability**: Easier to update individual programs
4. **Security**: Isolated security checks and risk management
5. **Scalability**: Programs can be deployed independently

## CPI Flow Examples

### Opening a Position
1. Trading Program calls Security Program for risk checks
2. Trading Program calls Oracle Program for price validation
3. Trading Program calls Core Program to update user account
4. Position is created in Trading Program

### Depositing Collateral
1. Collateral Program calls Security Program for security checks
2. Collateral Program calls Oracle Program for price updates
3. Collateral Program calls Core Program to update user account
4. Collateral is deposited in Collateral Program

### Liquidating a Position
1. Security Program calls Trading Program to execute liquidation
2. Security Program calls Core Program to update user account
3. Security Program records liquidation attempt
4. Position is closed in Trading Program

## Error Handling

CPI calls use Anchor's error propagation:
- Errors from CPI calls are propagated to the calling program
- Each program can handle errors appropriately
- Security checks can halt operations if conditions aren't met

## Future Enhancements

1. **Program Upgrades**: Individual programs can be upgraded independently
2. **New Programs**: Additional specialized programs can be added
3. **Cross-Chain**: CPI can be extended for cross-chain operations
4. **Governance**: CPI can include governance program calls
