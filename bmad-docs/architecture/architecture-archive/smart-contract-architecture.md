# Smart Contract Architecture

## Solana Program Structure
- **Program ID**: `HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso`
- **Anchor Framework**: Rust-based smart contracts
- **Modular Design**: Organized instruction modules by domain

## Contract Components
- **Market Management**: `initialize_market()`, `update_oracle_price()`, `settle_funding()`
- **Position Management**: `open_position()`, `close_position()` - Critical trading functions
- **Order Management**: `place_order()`, `cancel_order()`, `execute_order()` - Advanced order types
- **Collateral Management**: `deposit_native_sol()`, `withdraw_native_sol()` - SOL operations
- **Vault Management**: `initialize_token_vault()`, `deposit_tokens()`, `withdraw_tokens()` - Token operations
- **User Account Management**: `create_user_account()`, `update_user_account()` - Account lifecycle
- **Security Management**: Enterprise-grade security controls and validation

## Account Structures
- **UserAccount**: User state management with sub-accounts
- **TokenVault**: Token vault with authority and deposit tracking
- **ProtocolSolVault**: Protocol-level SOL vault management
- **Market**: Market configuration and state
- **Position**: Position data with health factors
- **Order**: Order management with advanced types
