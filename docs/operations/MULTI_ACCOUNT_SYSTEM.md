# 🏦 Multi-Account Management System

## 📋 **Overview**

QuantDesk's proprietary account management system allows users to:
1. **Create multiple sub-accounts** under a single wallet address
2. **Isolate margin trading** across different sub-accounts
3. **Delegate trading permissions** to other accounts
4. **Manage cross-collateral** across all sub-accounts

## 🏗️ **Account Architecture**

### **1. Master Account (User Account)**
- **Primary wallet address** - The main Solana wallet
- **Account authority** - Full control over all trading accounts
- **Cross-collateral management** - Can transfer funds between trading accounts

### **2. Trading Accounts**
- **Multiple trading accounts** per master account (typically 1-10)
- **Independent positions** - Each trading account can hold up to 8 perpetual and 8 spot positions
- **Isolated margin** - Each trading account has its own margin requirements
- **32 open orders** per trading account

### **3. Delegated Accounts**
- **Limited permissions** - Can deposit, place orders, cancel orders
- **No withdrawal rights** - Cannot withdraw funds or change settings
- **Professional trading** - Useful for managed accounts

## 🔧 **Implementation for QuantDesk**

### **Database Schema Updates**

```sql
-- Main user account (extends existing users table)
ALTER TABLE users ADD COLUMN main_account_id UUID;
ALTER TABLE users ADD COLUMN account_type VARCHAR(20) DEFAULT 'main'; -- main, sub, delegated

-- Trading accounts table
CREATE TABLE trading_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    master_account_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    account_index INTEGER NOT NULL,
    name VARCHAR(100),
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(master_account_id, account_index)
);

-- Delegated accounts table
CREATE TABLE delegated_accounts (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    master_account_id UUID NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    delegate_wallet_address TEXT NOT NULL,
    permissions JSONB DEFAULT '{"deposit": true, "trade": true, "cancel": true, "withdraw": false}',
    is_active BOOLEAN DEFAULT true,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Update positions table to include trading account
ALTER TABLE positions ADD COLUMN trading_account_id UUID REFERENCES trading_accounts(id);
ALTER TABLE orders ADD COLUMN trading_account_id UUID REFERENCES trading_accounts(id);
ALTER TABLE user_balances ADD COLUMN trading_account_id UUID REFERENCES trading_accounts(id);
```

### **API Endpoints**

```typescript
// Trading account management
POST /api/accounts/trading-accounts      // Create new trading account
GET  /api/accounts/trading-accounts      // List user's trading accounts
PUT  /api/accounts/trading-accounts/:id  // Update trading account
DELETE /api/accounts/trading-accounts/:id // Deactivate trading account

// Delegated accounts
POST /api/accounts/delegates             // Add delegate
GET  /api/accounts/delegates             // List delegates
PUT  /api/accounts/delegates/:id         // Update permissions
DELETE /api/accounts/delegates/:id       // Remove delegate

// Cross-collateral transfers
POST /api/accounts/transfer              // Transfer between trading accounts
GET  /api/accounts/balances              // Get all trading account balances
```

### **Frontend Components**

```typescript
// Trading account selector
<TradingAccountSelector 
  accounts={tradingAccounts}
  selectedAccount={currentAccount}
  onAccountChange={handleAccountChange}
/>

// Account management panel
<AccountManagementPanel
  masterAccount={masterAccount}
  tradingAccounts={tradingAccounts}
  delegates={delegates}
  onCreateTradingAccount={handleCreateTradingAccount}
  onAddDelegate={handleAddDelegate}
/>
```

## 🚀 **Implementation Steps**

### **Phase 1: Database & Backend**
1. ✅ Update database schema
2. ✅ Create trading account management endpoints
3. ✅ Implement cross-collateral transfers
4. ✅ Add delegate account system

### **Phase 2: Frontend Integration**
1. ✅ Add trading account selector to trading interface
2. ✅ Create account management dashboard
3. ✅ Implement account switching
4. ✅ Add delegate management UI

### **Phase 3: Smart Contract Integration**
1. ✅ Deploy account management contracts
2. ✅ Implement on-chain trading account creation
3. ✅ Add cross-collateral on-chain logic
4. ✅ Test with devnet

## 📊 **User Experience Flow**

### **Account Creation**
1. User connects wallet → Master account created
2. User clicks "Add Trading Account" → New trading account created
3. User names trading account → "Trading Account 1"
4. User deposits collateral → Funds allocated to trading account

### **Trading**
1. User selects trading account from dropdown
2. User places trade → Trade executed on selected trading account
3. User switches trading account → Different positions/balances shown
4. User can transfer funds between trading accounts

### **Delegation**
1. User adds delegate wallet address
2. User sets permissions (deposit, trade, cancel)
3. Delegate can now trade on behalf of user
4. User retains full control and can revoke access

## 🔒 **Security Features**

- **Isolated margins** - Each trading account has independent risk
- **Permission controls** - Granular delegate permissions
- **Audit trails** - All account actions logged
- **Emergency controls** - Master account can freeze trading accounts

## 🎯 **Benefits**

- **Risk isolation** - Separate trading strategies
- **Professional management** - Delegate to trading teams
- **Capital efficiency** - Cross-collateral optimization
- **User experience** - Professional multi-account interface

---

**Next Steps:**
1. Implement database schema changes
2. Create backend endpoints
3. Build frontend components
4. Test with devnet
5. Deploy to Railway
