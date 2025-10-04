# IDL Space Setup for QuantDesk Perpetual DEX

## 🚀 Quick Start

IDL Space is now properly configured for your QuantDesk Solana program! Here's how to use it:

### 1. Quick Setup (Recommended)
```bash
# Run the setup script - it will open IDL Space and copy your IDL file path
./setup-idl-space.sh
```

### 2. Manual Setup
```bash
# Option 1: Open the setup guide
open idl-space-setup-guide.html

# Option 2: Run the analysis script
node idl-space-setup.js

# Option 3: Direct browser access
# https://idl.space
```

### 2. Program Information
- **Program ID**: `GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a`
- **Network**: Devnet (for testing)
- **Instructions**: 26 available
- **Account Types**: 7 different account structures
- **Custom Types**: 13 enums and structs

## 🛠️ What You Can Do with IDL Space

### Build Transactions
- Create complex multi-instruction transactions
- Test instruction parameters
- Validate account requirements
- Simulate transaction execution

### Find PDAs (Program Derived Addresses)
- **User Account**: `["user_account", authority, account_index]`
- **Market**: `["market", base_asset, quote_asset]`
- **Position**: `["position", user, market]`
- **Order**: `["order", user, market]`
- **Collateral Account**: `["collateral", user, asset_type]`
- **Protocol SOL Vault**: `["protocol_sol_vault"]`

### Inspect Accounts
- View account states
- Decode account data
- Track account changes
- Debug account issues

## 📋 Key Instructions Available

### Market Management
- `initialize_market` - Create new trading markets
- `update_oracle_price` - Update price feeds
- `settle_funding` - Settle funding payments

### Position Management
- `open_position` - Open trading positions
- `close_position` - Close positions
- `liquidate_position` - Liquidate unhealthy positions
- `open_position_cross_collateral` - Cross-collateralized positions

### Order Management
- `place_order` - Place various order types
- `cancel_order` - Cancel pending orders
- `execute_conditional_order` - Execute stop/take profit orders

### Collateral Management
- `initialize_collateral_account` - Create collateral accounts
- `add_collateral` - Add collateral
- `remove_collateral` - Remove collateral
- `update_collateral_value` - Update USD values

### Token Operations
- `initialize_token_vault` - Create token vaults
- `deposit_tokens` - Deposit tokens
- `withdraw_tokens` - Withdraw tokens
- `create_user_token_account` - Create ATA

### SOL Operations
- `initialize_protocol_sol_vault` - Initialize SOL vault
- `deposit_native_sol` - Deposit SOL
- `withdraw_native_sol` - Withdraw SOL

### User Account Management
- `create_user_account` - Create user accounts
- `update_user_account` - Update account state
- `close_user_account` - Close accounts
- `check_user_permissions` - Verify permissions

## 🎯 Example Workflows

### 1. Create a User Account
1. Go to IDL Space → Build Transaction
2. Select `create_user_account`
3. Fill parameters:
   - `authority`: Your wallet address
   - `account_index`: 0 (first account)
4. Build and test the transaction

### 2. Initialize a Market
1. Select `initialize_market`
2. Fill parameters:
   - `base_asset`: "BTC"
   - `quote_asset`: "USDT"
   - `initial_price`: 50000000000 (50000 USDT with 6 decimals)
   - `max_leverage`: 10
   - `initial_margin_ratio`: 1000 (10%)
   - `maintenance_margin_ratio`: 500 (5%)
3. Build transaction

### 3. Deposit SOL
1. Select `deposit_native_sol`
2. Fill parameters:
   - `amount`: 1000000000 (1 SOL in lamports)
3. Build and execute

### 4. Open a Position
1. Select `open_position`
2. Fill parameters:
   - `size`: 1000000 (position size)
   - `side`: "Long" or "Short"
   - `leverage`: 5
3. Build transaction

## 🔧 Advanced Features

### Multi-Instruction Transactions
- Combine multiple instructions in one transaction
- Test complex workflows
- Validate dependencies

### Account State Inspection
- View current account data
- Decode binary account data
- Track state changes

### PDA Derivation
- Derive any PDA address
- Test seed combinations
- Validate PDA calculations

## 💡 Pro Tips

1. **Use Devnet First**: Always test on devnet before mainnet
2. **Connect Wallet**: Use Phantom or other Solana wallets
3. **Start Simple**: Begin with basic instructions
4. **Test Parameters**: Validate all parameter types
5. **Check Accounts**: Ensure all required accounts exist
6. **Monitor State**: Watch account state changes

## 🚨 Important Notes

- **Program ID**: `GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a`
- **Network**: Currently configured for devnet
- **RPC**: Uses public devnet RPC (consider paid RPC for production)
- **Wallet**: Connect Phantom or other Solana wallet for signing

## 📁 Files Created

- `idl-space-setup.js` - Setup script with program analysis
- `idl-space-quick-access.html` - Quick access web interface
- `contracts/smart-contracts/target/idl/quantdesk_perp_dex.json` - IDL file

## 🔗 Useful Links

- [IDL Space Main Site](https://idl.space)
- [Your Program on IDL Space](https://idl.space/program/GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a)
- [Solana Cookbook](https://solanacookbook.com)
- [Anchor Documentation](https://anchor-lang.com)

---

**Happy Building! 🚀**

Your QuantDesk perpetual DEX is now ready for interactive development with IDL Space!
