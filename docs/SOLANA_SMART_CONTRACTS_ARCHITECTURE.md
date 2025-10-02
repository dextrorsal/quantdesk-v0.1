# Solana Smart Contracts Architecture for Perpetual DEX

## Overview
This document outlines the critical Solana smart contract architecture that powers perpetual DEX platforms like Drift Protocol. The key components are **Solana Programs** (smart contracts) and **Program Derived Addresses (PDAs)**.

## Core Concepts

### 1. Solana Programs (Smart Contracts)
Solana programs are the equivalent of smart contracts on other blockchains. They:
- Execute on-chain logic
- Manage state and data
- Handle user interactions
- Enforce business rules
- Manage account creation and permissions

### 2. Program Derived Addresses (PDAs)
PDAs are unique addresses generated deterministically by programs that:
- **Have no private keys** - cannot sign transactions themselves
- **Are derived from seeds** - deterministic based on input data
- **Are controlled by programs** - only the deriving program can sign for them
- **Enable account management** - store user data and assets securely

## Drift Protocol Architecture

### Account Creation Flow with PDAs

```
1. User connects wallet (e.g., Phantom)
2. Frontend checks if Drift account exists
3. If no account exists:
   - Program derives PDA using user's wallet address + seed
   - Creates account at PDA address
   - Initializes account state
4. User can now deposit and trade
```

### PDA Derivation Pattern
```rust
// Example PDA derivation for user accounts
let (user_account_pda, bump) = Pubkey::find_program_address(
    &[
        b"user_account",           // Seed: "user_account"
        user_wallet.as_ref(),      // Seed: User's wallet address
    ],
    program_id,                   // Program ID
);
```

## Smart Contract Components

### 1. User Account Program
**Purpose**: Manages individual user accounts and their state

**Key Functions**:
- `create_user_account()` - Initialize new user account
- `update_user_account()` - Modify account settings
- `close_user_account()` - Close account and reclaim rent

**Account Structure**:
```rust
pub struct UserAccount {
    pub authority: Pubkey,        // User's wallet address
    pub account_index: u16,       // Account number (for sub-accounts)
    pub collateral: u64,          // Total collateral deposited
    pub positions: Vec<Position>,  // Open positions
    pub orders: Vec<Order>,        // Active orders
    pub last_funding_payment: i64, // Last funding payment timestamp
    pub bump: u8,                 // PDA bump seed
}
```

### 2. Collateral Vault Program
**Purpose**: Manages deposits, withdrawals, and collateral calculations

**Key Functions**:
- `deposit_collateral()` - Deposit assets as collateral
- `withdraw_collateral()` - Withdraw collateral
- `calculate_margin()` - Calculate available margin
- `liquidate_position()` - Execute liquidation

**Account Structure**:
```rust
pub struct CollateralVault {
    pub user_account: Pubkey,     // Associated user account
    pub asset_mint: Pubkey,      // Token mint (USDC, SOL, etc.)
    pub amount: u64,             // Amount deposited
    pub locked_amount: u64,       // Amount locked in positions
    pub available_amount: u64,     // Available for trading
    pub bump: u8,                // PDA bump seed
}
```

### 3. Trading Program
**Purpose**: Handles order matching, position management, and trade execution

**Key Functions**:
- `place_order()` - Submit new order
- `cancel_order()` - Cancel existing order
- `execute_trade()` - Execute matched orders
- `update_position()` - Update position after trade

**Account Structure**:
```rust
pub struct Order {
    pub user_account: Pubkey,     // User who placed order
    pub market: Pubkey,          // Market being traded
    pub side: OrderSide,         // Buy or Sell
    pub order_type: OrderType,    // Market, Limit, Stop, etc.
    pub size: u64,               // Order size
    pub price: u64,              // Order price (for limit orders)
    pub leverage: u8,            // Leverage multiplier
    pub status: OrderStatus,     // Active, Filled, Cancelled
    pub created_at: i64,         // Timestamp
}
```

### 4. Oracle Program
**Purpose**: Manages price feeds and market data

**Key Functions**:
- `update_price()` - Update asset prices
- `get_price()` - Retrieve current price
- `validate_price()` - Ensure price is within acceptable range

**Account Structure**:
```rust
pub struct OracleAccount {
    pub asset: String,           // Asset symbol (BTC, ETH, etc.)
    pub price: u64,             // Current price
    pub confidence: u64,        // Price confidence interval
    pub last_updated: i64,       // Last update timestamp
    pub source: OracleSource,    // Price source (Pyth, Switchboard, etc.)
}
```

## Implementation Architecture

### Program Structure
```
quantdesk_programs/
├── user_accounts/          # User account management
│   ├── lib.rs
│   ├── instructions/
│   │   ├── create_account.rs
│   │   ├── update_account.rs
│   │   └── close_account.rs
│   └── state/
│       └── user_account.rs
├── collateral_vault/        # Collateral management
│   ├── lib.rs
│   ├── instructions/
│   │   ├── deposit.rs
│   │   ├── withdraw.rs
│   │   └── liquidate.rs
│   └── state/
│       └── vault.rs
├── trading/               # Trading engine
│   ├── lib.rs
│   ├── instructions/
│   │   ├── place_order.rs
│   │   ├── cancel_order.rs
│   │   └── execute_trade.rs
│   └── state/
│       ├── order.rs
│       └── position.rs
└── oracle/               # Price feeds
    ├── lib.rs
    ├── instructions/
    │   ├── update_price.rs
    │   └── get_price.rs
    └── state/
        └── oracle.rs
```

### PDA Seeds Strategy
```rust
// User Account PDA
let (user_account_pda, bump) = Pubkey::find_program_address(
    &[b"user_account", user_wallet.as_ref()],
    program_id,
);

// Collateral Vault PDA
let (vault_pda, bump) = Pubkey::find_program_address(
    &[b"vault", user_wallet.as_ref(), asset_mint.as_ref()],
    program_id,
);

// Order PDA
let (order_pda, bump) = Pubkey::find_program_address(
    &[b"order", user_wallet.as_ref(), &order_id.to_le_bytes()],
    program_id,
);

// Position PDA
let (position_pda, bump) = Pubkey::find_program_address(
    &[b"position", user_wallet.as_ref(), market.as_ref()],
    program_id,
);
```

## Account State Management

### State Transitions
```
1. Wallet Connected (No Drift Account)
   ├── Check if PDA exists
   ├── If not, show "Create Account" button
   └── If exists, load account state

2. Account Created (No Deposits)
   ├── Show account info
   ├── Show "Deposit" button
   └── Display $0.00 balance

3. Account with Deposits (Ready to Trade)
   ├── Show balances
   ├── Show trading interface
   └── Enable order placement

4. Active Trading
   ├── Show positions
   ├── Show orders
   ├── Show P&L
   └── Risk management
```

### Frontend Integration
```typescript
// Check if user has Drift account
const checkUserAccount = async (walletAddress: string) => {
  const [userAccountPda] = await PublicKey.findProgramAddress(
    [Buffer.from("user_account"), new PublicKey(walletAddress).toBuffer()],
    programId
  );
  
  const accountInfo = await connection.getAccountInfo(userAccountPda);
  return accountInfo !== null;
};

// Create user account
const createUserAccount = async (wallet: Wallet) => {
  const [userAccountPda] = await PublicKey.findProgramAddress(
    [Buffer.from("user_account"), wallet.publicKey.toBuffer()],
    programId
  );
  
  const transaction = new Transaction().add(
    new TransactionInstruction({
      keys: [
        { pubkey: wallet.publicKey, isSigner: true, isWritable: false },
        { pubkey: userAccountPda, isSigner: false, isWritable: true },
        { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
      ],
      programId,
      data: Buffer.from([0]), // Create account instruction
    })
  );
  
  await wallet.sendTransaction(transaction, connection);
};
```

## Security Considerations

### PDA Security
- **Deterministic**: Same inputs always produce same PDA
- **Program-controlled**: Only the deriving program can sign
- **No private keys**: Eliminates key management risks
- **Collision-resistant**: Extremely unlikely to have collisions

### Account Security
- **Authority checks**: Verify user owns the account
- **State validation**: Ensure account state is valid
- **Permission checks**: Verify user can perform action
- **Reentrancy protection**: Prevent recursive calls

## Performance Optimizations

### Account Size Management
- **Minimize account size**: Only store necessary data
- **Use discriminators**: Distinguish account types
- **Optimize data layout**: Pack data efficiently

### Transaction Optimization
- **Batch operations**: Combine multiple operations
- **Use lookup tables**: Reduce transaction size
- **Optimize instruction data**: Minimize data transfer

## Integration with Backend

### Hybrid Architecture
```
Frontend (React) ↔ Backend API ↔ Solana Programs
     ↓                    ↓              ↓
   UI State          Database        On-chain State
   Wallet Conn.      User Data       Account Data
   Trading UI        Order History   Positions
   Portfolio         Analytics       Balances
```

### Data Synchronization
- **On-chain**: Account state, positions, orders, balances
- **Off-chain**: User preferences, analytics, order history
- **Real-time**: Price feeds, position updates, notifications

## Development Workflow

### 1. Smart Contract Development
- Write Solana programs using Anchor framework
- Implement PDA derivation and account management
- Add instruction handlers and state management
- Test with Solana test validator

### 2. Frontend Integration
- Connect to Solana programs
- Implement wallet connection
- Handle account creation and management
- Integrate with backend API

### 3. Backend Integration
- Sync on-chain data with database
- Provide API endpoints for frontend
- Handle off-chain operations
- Manage user sessions and preferences

## Next Steps

1. **Implement Solana Programs**
   - User account program
   - Collateral vault program
   - Trading program
   - Oracle integration

2. **Frontend Integration**
   - Wallet connection
   - Account creation flow
   - Deposit/withdrawal
   - Trading interface

3. **Backend Synchronization**
   - On-chain data sync
   - API endpoints
   - Real-time updates
   - Analytics and reporting

---

*This architecture ensures a secure, efficient, and scalable perpetual DEX platform on Solana.*
