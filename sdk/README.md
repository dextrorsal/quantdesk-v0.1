# QuantDesk SDK

## 🚀 **Complete TypeScript SDK for QuantDesk Perpetual DEX**

The QuantDesk SDK provides comprehensive TypeScript integration for the QuantDesk perpetual DEX platform, featuring advanced trading capabilities, AI integration, and professional-grade tools.

## 📊 **SDK Architecture**

```
sdk/
├── typescript/
│   ├── src/
│   │   ├── client.ts          # Main SDK client
│   │   ├── types.ts           # TypeScript type definitions
│   │   ├── errors.ts          # Error handling
│   │   ├── utils.ts           # Utility functions
│   │   └── index.ts           # SDK exports
│   ├── examples/              # Integration examples
│   │   ├── basic-trading-realistic.ts # Basic trading operations
│   │   ├── portfolio-tracking.ts     # Portfolio monitoring
│   │   ├── market-data-monitoring.ts # Market data tracking
│   │   ├── api-client.ts            # API client wrapper
│   │   └── integration-examples.ts  # Complete integration
│   ├── bots/                  # Trading bot templates
│   │   ├── market-maker.ts    # Market making bot
│   │   ├── liquidator.ts      # Liquidation bot
│   │   ├── arbitrage.ts       # Arbitrage bot
│   │   └── portfolio-mgr.ts   # Portfolio manager bot
│   └── README.md              # SDK documentation
```

## 🛠️ **Installation**

```bash
# Install via npm
npm install @quantdesk/sdk

# Or via yarn
yarn add @quantdesk/sdk

# Or via pnpm
pnpm add @quantdesk/sdk
```

## 🚀 **Quick Start**

### **Basic Setup**
```typescript
import { QuantDeskClient } from '@quantdesk/sdk';

const client = new QuantDeskClient({
  rpcUrl: 'https://api.devnet.solana.com',
  programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
  wallet: wallet, // Solana wallet instance
});

// Initialize client
await client.initialize();
```

### **Basic Trading**
```typescript
// Get available markets
const markets = await client.getMarkets();
console.log('Available markets:', markets);

// Get market data
const marketData = await client.getMarketData('SOL-PERP');
console.log('SOL-PERP data:', marketData);

// Open a position
const position = await client.openPosition({
  market: 'SOL-PERP',
  side: 'long',
  size: 1.0,
  leverage: 10,
  entryPrice: 100.0
});

console.log('Position opened:', position);
```

## 📚 **Core Features**

### **Trading Operations**
- **Position Management**: Open, close, and manage positions
- **Order Management**: Place, modify, and cancel orders
- **Portfolio Tracking**: Real-time portfolio monitoring
- **Risk Management**: Position sizing and risk controls

### **Market Data**
- **Real-time Prices**: Live price feeds and market data
- **Order Book**: Real-time order book data
- **Historical Data**: Historical price and volume data
- **Market Statistics**: Market metrics and analytics

### **AI Integration**
- **MIKEY-AI**: AI-powered trading insights
- **Market Analysis**: AI-driven market analysis
- **Risk Assessment**: AI risk evaluation
- **Trading Signals**: AI-generated trading signals

## 🎯 **Realistic Examples**

### **Basic Trading Example**
```typescript
import { BasicTradingExample } from '@quantdesk/sdk/examples/basic-trading-realistic';

const example = new BasicTradingExample(client);
await example.runExample();
```

### **Portfolio Tracking**
```typescript
import { PortfolioTrackingExample } from '@quantdesk/sdk/examples/portfolio-tracking';

const tracker = new PortfolioTrackingExample(client, 30000); // 30 second updates
await tracker.startTracking();
```

### **Market Data Monitoring**
```typescript
import { MarketDataMonitoringExample } from '@quantdesk/sdk/examples/market-data-monitoring';

const monitor = new MarketDataMonitoringExample(client, 10000); // 10 second updates
await monitor.startMonitoring(['SOL-PERP', 'ETH-PERP']);
```

### **API Client Wrapper**
```typescript
import { QuantDeskAPIClient } from '@quantdesk/sdk/examples/api-client';

const apiClient = new QuantDeskAPIClient(client);
await apiClient.initialize();

const markets = await apiClient.getMarkets();
const portfolio = await apiClient.getPortfolio();
```

### **Security Integration**
```typescript
import { QuantDeskSecurity } from '@quantdesk/sdk/utils/security';

const security = new QuantDeskSecurity(client);
security.validateMarketSymbol('SOL-PERP');
security.validateOrderData(orderData);
```

## 🎯 **Advanced Usage**

### **Portfolio Management**
```typescript
// Get portfolio overview
const portfolio = await client.getPortfolio();
console.log('Portfolio:', portfolio);

// Get positions
const positions = await client.getPositions();
console.log('Positions:', positions);

// Calculate PnL
const pnl = await client.calculatePnL();
console.log('Total PnL:', pnl);
```

### **Advanced Orders**
```typescript
// Place limit order
const limitOrder = await client.placeOrder({
  market: 'SOL-PERP',
  side: 'buy',
  size: 1.0,
  price: 95.0,
  orderType: 'limit'
});

// Place stop-loss order
const stopLoss = await client.placeOrder({
  market: 'SOL-PERP',
  side: 'sell',
  size: 1.0,
  triggerPrice: 90.0,
  orderType: 'stop-loss'
});
```

### **AI Integration**
```typescript
// Get AI market analysis
const analysis = await client.getAIAnalysis('SOL-PERP');
console.log('AI Analysis:', analysis);

// Get trading signals
const signals = await client.getTradingSignals();
console.log('Trading Signals:', signals);

// Get risk assessment
const riskAssessment = await client.getRiskAssessment();
console.log('Risk Assessment:', riskAssessment);
```

## 🤖 **Trading Bot Templates**

### **Market Maker Bot**
```typescript
import { MarketMakerBot } from '@quantdesk/sdk/bots/market-maker';

const bot = new MarketMakerBot({
  client: client,
  market: 'SOL-PERP',
  spread: 0.001, // 0.1% spread
  size: 0.1,     // 0.1 SOL per order
  maxPositions: 5
});

// Start market making
await bot.start();
```

### **Liquidation Bot**
```typescript
import { LiquidationBot } from '@quantdesk/sdk/bots/liquidator';

const bot = new LiquidationBot({
  client: client,
  markets: ['SOL-PERP', 'ETH-PERP'],
  liquidationThreshold: 0.8, // 80% liquidation threshold
  maxGasPrice: 0.001 // Max gas price in SOL
});

// Start liquidation monitoring
await bot.start();
```

### **Arbitrage Bot**
```typescript
import { ArbitrageBot } from '@quantdesk/sdk/bots/arbitrage';

const bot = new ArbitrageBot({
  client: client,
  markets: ['SOL-PERP', 'ETH-PERP'],
  minProfit: 0.001, // 0.1% minimum profit
  maxSize: 1.0      // Max 1 SOL per trade
});

// Start arbitrage monitoring
await bot.start();
```

### **Portfolio Manager Bot**
```typescript
import { PortfolioManagerBot } from '@quantdesk/sdk/bots/portfolio-mgr';

const bot = new PortfolioManagerBot({
  client: client,
  rebalanceThreshold: 0.05, // 5% rebalance threshold
  maxLeverage: 10,
  riskLimit: 0.1 // 10% max risk per position
});

// Start portfolio management
await bot.start();
```

## 🔧 **Configuration**

### **Client Configuration**
```typescript
interface QuantDeskClientConfig {
  rpcUrl: string;           // Solana RPC URL
  programId: string;        // QuantDesk program ID
  wallet: Wallet;           // Solana wallet instance
  commitment?: Commitment;  // Transaction commitment level
  timeout?: number;         // Request timeout in ms
  retries?: number;         // Number of retries
}
```

### **Environment Variables**
```bash
# Required
QUANTDESK_RPC_URL=https://api.devnet.solana.com
QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

# Optional
QUANTDESK_COMMITMENT=confirmed
QUANTDESK_TIMEOUT=30000
QUANTDESK_RETRIES=3
```

## 📊 **Error Handling**

### **Error Types**
```typescript
import { QuantDeskError, ErrorCode } from '@quantdesk/sdk';

try {
  await client.openPosition(positionData);
} catch (error) {
  if (error instanceof QuantDeskError) {
    switch (error.code) {
      case ErrorCode.INSUFFICIENT_COLLATERAL:
        console.log('Insufficient collateral');
        break;
      case ErrorCode.POSITION_SIZE_TOO_LARGE:
        console.log('Position size too large');
        break;
      case ErrorCode.MARKET_CLOSED:
        console.log('Market is closed');
        break;
      default:
        console.log('Unknown error:', error.message);
    }
  }
}
```

### **Retry Logic**
```typescript
// Automatic retry with exponential backoff
const client = new QuantDeskClient({
  ...config,
  retries: 3,
  retryDelay: 1000 // 1 second base delay
});
```

## 🧪 **Testing**

### **Unit Tests**
```typescript
import { QuantDeskClient } from '@quantdesk/sdk';

describe('QuantDeskClient', () => {
  let client: QuantDeskClient;

  beforeEach(async () => {
    client = new QuantDeskClient(testConfig);
    await client.initialize();
  });

  it('should get markets', async () => {
    const markets = await client.getMarkets();
    expect(markets).toBeDefined();
    expect(markets.length).toBeGreaterThan(0);
  });

  it('should open position', async () => {
    const position = await client.openPosition({
      market: 'SOL-PERP',
      side: 'long',
      size: 1.0,
      leverage: 10,
      entryPrice: 100.0
    });
    expect(position).toBeDefined();
  });
});
```

### **Integration Tests**
```typescript
// Test with real devnet
const integrationClient = new QuantDeskClient({
  rpcUrl: 'https://api.devnet.solana.com',
  programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
  wallet: testWallet
});
```

## 📖 **API Reference**

### **Core Methods**
- `initialize()`: Initialize the client
- `getMarkets()`: Get available markets
- `getMarketData(market)`: Get market data
- `openPosition(data)`: Open a new position
- `closePosition(positionId)`: Close a position
- `placeOrder(data)`: Place an order
- `cancelOrder(orderId)`: Cancel an order
- `getPortfolio()`: Get portfolio overview
- `getPositions()`: Get all positions
- `calculatePnL()`: Calculate PnL

### **AI Methods**
- `getAIAnalysis(market)`: Get AI market analysis
- `getTradingSignals()`: Get trading signals
- `getRiskAssessment()`: Get risk assessment
- `getMarketSentiment(market)`: Get market sentiment

### **Utility Methods**
- `getAccountInfo(address)`: Get account information
- `getTransactionHistory()`: Get transaction history
- `estimateGas(transaction)`: Estimate gas costs
- `validateOrder(order)`: Validate order data

## 🚀 **Deployment**

### **Production Setup**
```typescript
const productionClient = new QuantDeskClient({
  rpcUrl: 'https://api.mainnet-beta.solana.com',
  programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
  wallet: productionWallet,
  commitment: 'confirmed',
  timeout: 30000,
  retries: 5
});
```

### **Monitoring**
```typescript
// Enable monitoring
client.enableMonitoring({
  metrics: true,
  logging: true,
  alerts: true
});

// Get performance metrics
const metrics = await client.getMetrics();
console.log('Performance metrics:', metrics);
```

## 🤝 **Contributing**

### **Development Setup**
```bash
# Clone repository
git clone https://github.com/dextrorsal/quantdesk-v0.1.git
cd quantdesk-v0.1/sdk/typescript

# Install dependencies
npm install

# Build SDK
npm run build

# Run tests
npm test
```

### **Code Style**
- **TypeScript**: Strict mode enabled
- **ESLint**: Configured for TypeScript
- **Prettier**: Code formatting
- **Jest**: Testing framework

## 📞 **Support**

### **Resources**
- **Documentation**: Complete API documentation
- **Examples**: Integration examples and patterns
- **Community**: Discord community support
- **Issues**: GitHub issue tracking

### **Contact**
- **GitHub**: [QuantDesk SDK Repository](https://github.com/dextrorsal/quantdesk-v0.1)
- **Discord**: Community support channel
- **Email**: Technical support contact

---

**QuantDesk SDK: Professional-grade TypeScript integration for the QuantDesk perpetual DEX platform with AI-powered trading capabilities.**