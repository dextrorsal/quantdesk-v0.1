# QuantDesk Devnet Testing Guide

## 🎯 Overview

This guide will help you test the QuantDesk perpetual DEX platform on Solana devnet. The devnet environment allows you to trade with test SOL and experience all platform features without financial risk.

## 🚀 Getting Started

### Prerequisites
- Solana wallet (Phantom, Solflare, or Backpack)
- Test SOL for devnet (free from faucets)
- Modern web browser

### Access the Platform
1. Start local development server: `pnpm run dev`
2. Visit `http://localhost:3001`
3. Connect your Solana wallet
4. Switch to Devnet network in your wallet
5. Request test SOL if needed

## 💰 Getting Test SOL

### Method 1: Solana Faucet
```bash
# Using Solana CLI
solana airdrop 2

# Or visit: https://faucet.solana.com
```

### Method 2: Phantom Wallet
1. Open Phantom wallet
2. Switch to Devnet
3. Click "Request Airdrop"
4. Wait for confirmation

### Method 3: Solflare Wallet
1. Open Solflare wallet
2. Switch to Devnet
3. Go to "Receive" tab
4. Click "Request Airdrop"

## 🔧 Wallet Setup

### Connecting Your Wallet
1. Click "Connect Wallet" on the platform
2. Select your preferred wallet
3. Approve the connection
4. Verify you're on Devnet network

### Network Configuration
- **Network**: Solana Devnet
- **RPC URL**: https://api.devnet.solana.com
- **Program ID**: C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

## 📊 Trading Interface

### Market Overview
The platform displays available trading pairs:
- **SOL/USDC** - Solana vs USD Coin
- **BTC/USDC** - Bitcoin vs USD Coin
- **ETH/USDC** - Ethereum vs USD Coin

### Price Information
Each market shows:
- Current price
- 24h change
- 24h volume
- Open interest
- Funding rate

## 📈 Placing Orders

### Order Types
1. **Market Orders**
   - Execute immediately at current market price
   - Best for quick entry/exit
   - Higher slippage risk

2. **Limit Orders**
   - Execute only at specified price or better
   - Better price control
   - May not fill immediately

3. **Stop Orders**
   - Trigger when price reaches stop level
   - Risk management tool
   - Automatic execution

### Order Parameters
- **Symbol**: Trading pair (e.g., SOL/USDC)
- **Side**: Buy (Long) or Sell (Short)
- **Size**: Order amount in base asset
- **Price**: Limit price (for limit orders)
- **Leverage**: Multiplier (1x to 20x)

### Example: Opening a Long Position
1. Select SOL/USDC market
2. Choose "Buy" (Long)
3. Set size: 1 SOL
4. Set leverage: 5x
5. Click "Place Order"
6. Confirm in wallet

## 📋 Managing Positions

### Viewing Positions
- **Dashboard**: Overview of all positions
- **P&L**: Real-time profit/loss calculation
- **Margin**: Required collateral
- **Liquidation Price**: Risk level indicator

### Position Actions
1. **Close Position**
   - Click "Close" on position
   - Confirm transaction
   - Realize P&L

2. **Add Margin**
   - Increase collateral
   - Lower liquidation risk
   - Improve position safety

3. **Partial Close**
   - Close portion of position
   - Reduce exposure
   - Maintain remaining position

## 🔄 Order Management

### Order Status
- **Pending**: Waiting for execution
- **Filled**: Successfully executed
- **Cancelled**: Order cancelled
- **Rejected**: Order failed

### Order Actions
1. **Cancel Order**
   - Click "Cancel" on pending order
   - Confirm cancellation
   - Funds returned to balance

2. **Modify Order**
   - Change price or size
   - Update order parameters
   - Maintain order priority

## 💡 Trading Strategies

### Basic Strategies
1. **Long Position**
   - Buy asset expecting price increase
   - Profit when price goes up
   - Risk when price goes down

2. **Short Position**
   - Sell asset expecting price decrease
   - Profit when price goes down
   - Risk when price goes up

3. **Hedging**
   - Offset risk with opposite positions
   - Reduce overall portfolio risk
   - Maintain market exposure

### Risk Management
1. **Position Sizing**
   - Never risk more than you can afford
   - Use appropriate leverage
   - Diversify across markets

2. **Stop Losses**
   - Set automatic exit points
   - Limit potential losses
   - Protect capital

3. **Take Profits**
   - Lock in gains at target levels
   - Secure profits
   - Reduce position risk

## 🛡️ Security Features

### Circuit Breakers
- **Price Deviation**: Triggers on 5% price movements
- **Volume Spikes**: Limits on unusual volume
- **Oracle Protection**: Staleness detection

### Keeper System
- **Liquidation Protection**: Automated risk management
- **Performance Monitoring**: Keeper scoring
- **Multi-signature**: Enhanced security

## 📱 Mobile Testing

### Responsive Design
- Optimized for mobile devices
- Touch-friendly interface
- Responsive charts and tables

### Mobile Features
- Wallet connection via mobile wallets
- Touch trading interface
- Mobile-optimized charts

## 🔍 Testing Scenarios

### Scenario 1: Basic Trading
1. Connect wallet
2. Get test SOL
3. Open long position
4. Monitor P&L
5. Close position

### Scenario 2: Leverage Testing
1. Open position with 5x leverage
2. Monitor liquidation price
3. Add margin if needed
4. Test liquidation scenarios

### Scenario 3: Order Management
1. Place limit order
2. Cancel order
3. Modify order
4. Test different order types

### Scenario 4: Risk Management
1. Open multiple positions
2. Test stop losses
3. Monitor portfolio risk
4. Test liquidation scenarios

## 📊 Performance Testing

### Load Testing
- Multiple concurrent users
- High-frequency trading
- Large position sizes
- Stress test scenarios

### Latency Testing
- Order execution speed
- Price update frequency
- Transaction confirmation time
- Oracle feed latency

## 🐛 Common Issues

### Wallet Connection Issues
1. **Wallet Not Detected**
   - Refresh page
   - Check wallet extension
   - Try different wallet

2. **Network Mismatch**
   - Switch to Devnet in wallet
   - Clear browser cache
   - Reconnect wallet

### Trading Issues
1. **Order Rejection**
   - Check sufficient balance
   - Verify order parameters
   - Check market status

2. **Transaction Failures**
   - Check network congestion
   - Increase priority fee
   - Retry transaction

### Performance Issues
1. **Slow Loading**
   - Check internet connection
   - Clear browser cache
   - Try different browser

2. **Price Updates**
   - Refresh page
   - Check oracle status
   - Verify network connection

## 📞 Support

### Getting Help
- **Documentation**: Check platform docs
- **Discord**: Join QuantDesk community
- **GitHub**: Report issues
- **Email**: support@quantdesk.com

### Reporting Issues
When reporting issues, include:
- Browser and version
- Wallet type and version
- Network (Devnet)
- Steps to reproduce
- Error messages
- Screenshots if helpful

## 🎓 Learning Resources

### Educational Content
- **Trading Basics**: Learn perpetual trading
- **Risk Management**: Understand leverage risks
- **Technical Analysis**: Chart reading skills
- **DeFi Concepts**: Decentralized finance

### Community
- **Discord**: Real-time support
- **Twitter**: Updates and announcements
- **Medium**: Educational articles
- **YouTube**: Video tutorials

## 🔄 Environment Switching

### Devnet to Testnet
1. Update wallet network
2. Get testnet SOL
3. Update program ID
4. Test functionality

### Testnet to Mainnet
1. **WARNING**: Real money involved
2. Update to mainnet
3. Use real SOL
4. Start with small amounts

## 📈 Success Metrics

### Testing Goals
- [ ] Successfully connect wallet
- [ ] Place and execute orders
- [ ] Manage positions
- [ ] Test all order types
- [ ] Experience liquidation scenarios
- [ ] Test mobile interface
- [ ] Verify security features

### Performance Benchmarks
- Order execution < 2 seconds
- Price updates < 1 second
- Transaction confirmation < 30 seconds
- 99.9% uptime target

---

**Happy Testing!** 🚀

Remember: Devnet is for testing only. Never use real funds on devnet, and always verify you're on the correct network before trading.
