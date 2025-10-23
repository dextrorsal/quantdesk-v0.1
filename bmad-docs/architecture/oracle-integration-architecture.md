# Oracle Integration Architecture - QuantDesk Perpetual DEX

## 🏗️ Unified Oracle Data Flow Architecture

### **Data Flow: Pyth → Smart Contracts → User Balances**

```
┌─────────────────┐    WebSocket     ┌──────────────────┐    REST API    ┌─────────────────┐
│   Pyth Network  │ ────────────────► │   Backend        │ ─────────────► │   Frontend      │
│   (Price Feeds) │                   │   (Port 3002)    │               │   (Port 3001)   │
└─────────────────┘                   └──────────────────┘               └─────────────────┘
         │                                      │                                   │
         │ Real-time Price Updates              │ Oracle Price Storage              │ User Balance
         │                                      │ & Market Data                    │ Calculations
         ▼                                      ▼                                   ▼
┌─────────────────┐                   ┌──────────────────┐               ┌─────────────────┐
│   WebSocket     │                   │   Supabase       │               │   Portfolio     │
│   Connection    │                   │   Database       │               │   Dashboard     │
│   (Hermes API)  │                   │   (oracle_prices)│               │   Real-time     │
└─────────────────┘                   └──────────────────┘               └─────────────────┘
         │                                      │                                   │
         │ Price Validation &                   │ Market State Updates              │ Balance Updates
         │ Confidence Checks                    │ & Position Tracking               │ & PnL Display
         ▼                                      ▼                                   ▼
┌─────────────────┐                   ┌──────────────────┐               ┌─────────────────┐
│   Smart         │                   │   Market         │               │   Trading       │
│   Contracts     │◄─────────────────►│   Management     │               │   Interface     │
│   (Oracle.rs)   │   Price Updates   │   System         │               │   Real-time     │
└─────────────────┘                   └──────────────────┘               └─────────────────┘
```

## 🔧 Integration Points

### **1. Backend Oracle Service (Working ✅)**
- **File**: `backend/src/services/pythOracleService.ts`
- **Status**: ✅ Fully functional WebSocket connection to Pyth Network
- **Features**:
  - Real-time price feeds via Hermes WebSocket
  - Fallback to CoinGecko API
  - Price validation and confidence checks
  - Database storage in `oracle_prices` table

### **2. Smart Contract Oracle Integration (Failing ❌)**
- **File**: `contracts/programs/quantdesk-perp-dex/src/oracle.rs`
- **Status**: ❌ Compilation errors due to incomplete implementation
- **Issues**:
  - Missing error handling in `get_price_from_pyth()`
  - Incomplete price validation logic
  - No integration with backend oracle service

### **3. Frontend Balance Service (Type Errors ❌)**
- **File**: `frontend/src/services/balanceService.ts`
- **Status**: ❌ Type errors in oracle price integration
- **Issues**:
  - Oracle price fetching returns inconsistent types
  - Missing error handling for oracle failures
  - No real-time price updates integration

## 🎯 Architectural Resolution Plan

### **Phase 1: Smart Contract Oracle Fixes**
1. **Complete Oracle Implementation**:
   ```rust
   // Fix incomplete get_price_from_pyth function
   fn get_price_from_pyth(price_feed: &AccountInfo) -> Result<OraclePrice> {
       let price_account_data = price_feed.try_borrow_data()
           .map_err(|_| ErrorCode::OracleFeedNotFound)?;
       
       // Complete implementation with proper error handling
       // Add price validation and confidence checks
   }
   ```

2. **Add Oracle Price Update Instructions**:
   ```rust
   pub fn update_oracle_price(
       ctx: Context<UpdateOraclePrice>,
       price: u64,
       confidence: u64,
   ) -> Result<()> {
       // Update market price from backend oracle service
   }
   ```

### **Phase 2: Backend-Smart Contract Integration**
1. **Oracle Price Sync Service**:
   ```typescript
   // New service to sync backend oracle prices to smart contracts
   class OracleSyncService {
     async syncPricesToSmartContract(): Promise<void> {
       // Fetch prices from pythOracleService
       // Call smart contract update_oracle_price instruction
     }
   }
   ```

2. **Real-time Price Broadcasting**:
   ```typescript
   // WebSocket service to broadcast oracle updates
   class OracleWebSocketService {
     broadcastPriceUpdate(symbol: string, price: number): void {
       // Broadcast to all connected clients
     }
   }
   ```

### **Phase 3: Frontend Integration**
1. **Real-time Price Context**:
   ```typescript
   // Context provider for real-time oracle prices
   const OraclePriceContext = createContext<{
     prices: Map<string, number>;
     subscribe: (symbol: string) => void;
   }>();
   ```

2. **Balance Service Integration**:
   ```typescript
   // Updated balance service with real-time oracle prices
   class BalanceService {
     async getUserBalances(walletAddress: PublicKey): Promise<UserBalances> {
       // Use real-time oracle prices from context
       // Calculate accurate USD values
     }
   }
   ```

## 🔄 Data Flow Implementation

### **Real-time Price Flow**:
1. **Pyth Network** → WebSocket → **Backend Oracle Service**
2. **Backend Oracle Service** → Database → **Smart Contract Updates**
3. **Smart Contract Updates** → WebSocket → **Frontend Real-time Updates**
4. **Frontend Updates** → **Portfolio Dashboard** → **User Balance Display**

### **Error Handling & Fallbacks**:
1. **Pyth WebSocket Fails** → Fallback to CoinGecko API
2. **Backend Oracle Fails** → Use cached prices
3. **Smart Contract Oracle Fails** → Use backend oracle prices
4. **Frontend Oracle Fails** → Use mock prices for development

## 🚀 Implementation Priority

### **Critical (Blocking Compilation)**:
1. Fix smart contract enum conflicts
2. Implement missing keeper network methods
3. Complete oracle.rs implementation

### **High Priority (Trading Functionality)**:
1. Backend-smart contract oracle sync
2. Real-time price broadcasting
3. Frontend balance service integration

### **Medium Priority (User Experience)**:
1. Real-time portfolio updates
2. Advanced oracle features (confidence, staleness)
3. Oracle price history and charts

## 📊 Success Metrics

### **Technical Metrics**:
- ✅ Smart contracts compile without errors
- ✅ Oracle prices update in real-time (< 1 second latency)
- ✅ User balances calculate accurately with oracle prices
- ✅ Keeper network liquidations execute successfully

### **User Experience Metrics**:
- ✅ Portfolio values update in real-time
- ✅ Trading interface shows accurate prices
- ✅ Position PnL calculations are precise
- ✅ Liquidation alerts trigger correctly

---

**Next Steps**: Implement Phase 1 fixes to resolve compilation errors and establish the foundation for real-time oracle integration.
