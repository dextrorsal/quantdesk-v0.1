# QuantDesk Protocol SDK

Official TypeScript/JavaScript SDK for QuantDesk Protocol.

## Installation

```bash
npm install @quantdesk/sdk
```

## Quick Start

```typescript
import { QuantDeskClient } from '@quantdesk/sdk'
import { PhantomWalletAdapter } from '@solana/wallet-adapter-wallets'

// Initialize client
const client = new QuantDeskClient(
  'https://api.quantdesk.app',
  'https://api.mainnet-beta.solana.com',
  new PhantomWalletAdapter()
)

// Get markets
const markets = await client.getMarkets()
console.log('Available markets:', markets)

// Place an order
const order = await client.placeOrder({
  symbol: 'BTC-PERP',
  side: 'buy',
  type: 'limit',
  amount: 0.1,
  price: 50000,
  timeInForce: 'GTC'
})

console.log('Order placed:', order)
```

## Features

- **Trading**: Place, cancel, and manage orders
- **Portfolio**: Get positions and portfolio analytics
- **Risk Management**: Access risk metrics and alerts
- **Cross-Collateral**: Manage multi-asset collateral
- **JIT Liquidity**: Participate in liquidity auctions
- **Market Data**: Real-time prices and market information

## API Reference

### Client Methods

#### Market Data
- `getMarkets()` - Get all available markets
- `getMarket(symbol)` - Get specific market data
- `getMarketPrice(symbol)` - Get current market price

#### Trading
- `placeOrder(order)` - Place a new order
- `getOrders(symbol?)` - Get user orders
- `getOrder(id)` - Get specific order
- `cancelOrder(id)` - Cancel an order

#### Positions
- `getPositions()` - Get all positions
- `getPosition(id)` - Get specific position

#### Portfolio
- `getPortfolioSummary()` - Get portfolio overview
- `getPortfolioAnalytics()` - Get detailed analytics

#### Risk Management
- `getRiskMetrics()` - Get risk metrics
- `getRiskAlerts()` - Get active risk alerts

#### Cross-Collateral
- `getCollateralAccounts()` - Get collateral accounts
- `addCollateral(accountId, symbol, amount)` - Add collateral

#### JIT Liquidity
- `getLiquidityAuctions()` - Get active auctions
- `createLiquidityAuction(auction)` - Create new auction

## Error Handling

```typescript
import { QuantDeskError, InsufficientBalanceError } from '@quantdesk/sdk'

try {
  await client.placeOrder(order)
} catch (error) {
  if (error instanceof InsufficientBalanceError) {
    console.error('Not enough balance:', error.message)
  } else if (error instanceof QuantDeskError) {
    console.error('QuantDesk error:', error.message)
  } else {
    console.error('Unknown error:', error)
  }
}
```

## Utilities

```typescript
import { 
  formatPrice, 
  formatPercentage, 
  calculatePnLPercent,
  calculateLiquidationPrice 
} from '@quantdesk/sdk'

// Format numbers
const price = formatPrice(1234.5678, 2) // "1234.57"
const percent = formatPercentage(5.6789, 1) // "5.7%"

// Calculate PnL
const pnlPercent = calculatePnLPercent(50000, 51000, 'long') // 2.0

// Calculate liquidation price
const liqPrice = calculateLiquidationPrice(50000, 'long', 10) // 45000
```

## License

MIT License - see [LICENSE](../../LICENSE) for details.
