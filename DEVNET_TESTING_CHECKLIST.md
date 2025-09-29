# 🧪 QuantDesk Devnet Testing Checklist

## 🎯 **Pre-Railway Deployment Testing**

This checklist ensures your QuantDesk app can perform the **BARE MINIMUM** functionality in a devnet environment before deploying to Railway.

---

## 📋 **Core Functionality Tests**

### 1. 🔐 **Wallet Connection & Authentication**
- [ ] **Connect Wallet**: User can connect Solana wallet (Phantom, Solflare, etc.)
- [ ] **Sign Message**: User can sign a message for authentication/login
- [ ] **JWT Token**: Backend generates and validates JWT tokens
- [ ] **Session Persistence**: User stays logged in across page refreshes
- [ ] **Logout**: User can disconnect wallet and logout

**Test Steps:**
1. Open app in browser
2. Click "Connect Wallet" 
3. Select wallet (Phantom/Solflare)
4. Approve connection
5. Sign authentication message
6. Verify user is logged in
7. Refresh page - should stay logged in
8. Test logout functionality

**Expected Result:** ✅ User successfully connects wallet and authenticates

---

### 2. 👤 **User Account Creation (Drift-style)**
- [ ] **Account Creation**: User can create trading account after wallet connection
- [ ] **Account Management**: User can view account details and settings
- [ ] **Account Switching**: User can switch between multiple accounts (if applicable)
- [ ] **Account Permissions**: Proper account permissions and access control

**Test Steps:**
1. After wallet connection, check for "Create Account" prompt
2. Create new trading account
3. Verify account creation in database
4. View account details page
5. Test account settings and preferences

**Expected Result:** ✅ User successfully creates and manages trading account

---

### 3. 💰 **Token Deposits**
- [ ] **USDT Deposit**: User can deposit USDT from wallet to platform
- [ ] **USDC Deposit**: User can deposit USDC from wallet to platform  
- [ ] **BTC Deposit**: User can deposit BTC from wallet to platform
- [ ] **ETH Deposit**: User can deposit ETH from wallet to platform
- [ ] **SOL Deposit**: User can deposit SOL from wallet to platform
- [ ] **Balance Display**: Deposited tokens show in user balance
- [ ] **Transaction History**: Deposit transactions are recorded

**Test Steps:**
1. Navigate to deposit page
2. Select token (USDT, USDC, BTC, ETH, SOL)
3. Enter deposit amount
4. Approve transaction in wallet
5. Wait for confirmation
6. Verify balance updates
7. Check transaction history

**Expected Result:** ✅ User can deposit all supported tokens successfully

---

### 4. 📈 **Perpetual Trading with Leverage**
- [ ] **Market Selection**: User can select perpetual markets (BTC-PERP, ETH-PERP, etc.)
- [ ] **Leverage Selection**: User can set leverage (1x-100x)
- [ ] **Long Position**: User can open long positions
- [ ] **Short Position**: User can open short positions
- [ ] **Position Sizing**: User can set position size
- [ ] **Margin Calculation**: Proper margin requirements displayed
- [ ] **Order Execution**: Orders execute successfully on-chain

**Test Steps:**
1. Select a perpetual market (e.g., BTC-PERP)
2. Choose leverage (start with 5x for testing)
3. Set position size
4. Choose long or short
5. Place market order
6. Approve transaction in wallet
7. Wait for execution
8. Verify position opens

**Expected Result:** ✅ User can execute leveraged perpetual trades

---

### 5. 📊 **Position Visualization**
- [ ] **Position Display**: Current positions are displayed clearly
- [ ] **PnL Calculation**: Real-time PnL (profit/loss) is calculated and shown
- [ ] **Entry Price**: Entry price is displayed for each position
- [ ] **Current Price**: Current market price is shown
- [ ] **Liquidation Price**: Liquidation price is calculated and displayed
- [ ] **Margin Info**: Margin used and available margin shown
- [ ] **Health Factor**: Position health factor displayed
- [ ] **Unrealized PnL**: Unrealized PnL updates in real-time

**Test Steps:**
1. Open a position (from previous test)
2. Navigate to positions page
3. Verify all position details are displayed
4. Check PnL calculations are correct
5. Verify liquidation price calculation
6. Test real-time updates

**Expected Result:** ✅ All position data is accurately displayed and updated

---

### 6. 📋 **Limit Orders**
- [ ] **Limit Order Placement**: User can place limit orders
- [ ] **Price Setting**: User can set custom entry/exit prices
- [ ] **Order Management**: User can view pending limit orders
- [ ] **Order Cancellation**: User can cancel pending orders
- [ ] **Order Execution**: Limit orders execute when price is reached
- [ ] **Partial Fills**: Orders can be partially filled
- [ ] **Order History**: Completed orders are recorded

**Test Steps:**
1. Navigate to trading interface
2. Select "Limit" order type
3. Set custom price (different from market price)
4. Set order size and leverage
5. Place limit order
6. Verify order appears in "Open Orders"
7. Test order cancellation
8. Wait for price to reach limit (or manually trigger)

**Expected Result:** ✅ Limit orders work correctly with proper execution

---

## 🔧 **Technical Infrastructure Tests**

### 7. 🗄️ **Database Connectivity**
- [ ] **Supabase Connection**: Backend connects to Supabase database
- [ ] **Data Persistence**: User data, positions, orders are saved
- [ ] **Real-time Updates**: Database changes reflect in UI
- [ ] **Data Integrity**: No data corruption or loss

### 8. 🌐 **API Endpoints**
- [ ] **Authentication API**: `/api/auth/*` endpoints work
- [ ] **Trading API**: `/api/orders/*`, `/api/positions/*` work
- [ ] **Market Data API**: `/api/markets/*` returns data
- [ ] **Portfolio API**: `/api/portfolio/*` returns user data
- [ ] **Error Handling**: Proper error responses

### 9. ⚡ **Real-time Features**
- [ ] **WebSocket Connection**: Real-time price updates work
- [ ] **Position Updates**: Position PnL updates in real-time
- [ ] **Order Status**: Order status updates in real-time
- [ ] **Market Data**: Live market data feeds work

### 10. 🔒 **Security & Authentication**
- [ ] **JWT Validation**: Tokens are properly validated
- [ ] **Rate Limiting**: API rate limits are enforced
- [ ] **CORS**: Cross-origin requests are handled
- [ ] **Input Validation**: User inputs are validated

---

## 🚀 **Deployment Readiness Checklist**

### Environment Configuration
- [ ] **Devnet RPC**: Using Solana devnet RPC endpoints
- [ ] **Environment Variables**: All required env vars set
- [ ] **Database**: Supabase project configured for devnet
- [ ] **Smart Contracts**: Contracts deployed to devnet
- [ ] **Oracle Feeds**: Pyth oracle feeds configured for devnet

### Performance & Reliability
- [ ] **Load Testing**: App handles multiple concurrent users
- [ ] **Error Recovery**: App recovers from network errors
- [ ] **Transaction Retry**: Failed transactions can be retried
- [ ] **Data Consistency**: No data inconsistencies under load

### User Experience
- [ ] **Loading States**: Proper loading indicators
- [ ] **Error Messages**: Clear error messages for users
- [ ] **Mobile Responsive**: App works on mobile devices
- [ ] **Browser Compatibility**: Works on major browsers

---

## 📝 **Testing Notes & Issues**

### Test Environment Setup
```bash
# Backend
cd backend
npm install
npm run dev

# Frontend  
cd frontend
npm install
npm run dev
```

### Devnet Configuration
- **Solana Network**: Devnet
- **RPC URL**: https://api.devnet.solana.com
- **Wallet Network**: Devnet
- **Test Tokens**: Use devnet faucets for testing

### Common Issues to Watch For
1. **Wallet Connection**: Ensure wallet is on devnet
2. **Transaction Fees**: Have enough SOL for transaction fees
3. **Oracle Prices**: Verify Pyth oracle feeds are working
4. **Smart Contracts**: Ensure contracts are deployed to devnet
5. **Database**: Check Supabase connection and permissions

---

## ✅ **Sign-off Checklist**

- [ ] All core functionality tests pass
- [ ] No critical bugs or errors
- [ ] Performance is acceptable
- [ ] Security measures are in place
- [ ] Documentation is updated
- [ ] Team has reviewed and approved

**Ready for Railway Deployment:** [ ] YES / [ ] NO

---

## 🎯 **Next Steps After Testing**

1. **Fix any issues** found during testing
2. **Update documentation** with any changes
3. **Prepare Railway deployment** configuration
4. **Set up production environment** variables
5. **Deploy to Railway** and test in production
6. **Monitor performance** and user feedback

---

*Last Updated: [Current Date]*
*Testing Environment: Solana Devnet*
*Target Deployment: Railway*
