# Examples Directory

This directory contains comprehensive examples and demos for using QuantDesk components in your own projects.

## 📁 Structure

```
examples/
├── frontend-ui-components.tsx    # React UI component examples
├── backend-api-services.ts      # Node.js API service examples
├── mikey-ai-agents.ts          # LangChain AI agent examples
├── data-ingestion-processors.ts # Real-time data processing examples
├── smart-contract-interactions.ts # Solana program interaction examples
└── typescript/                 # TypeScript-specific examples
```

## 🚀 Quick Start

Each example file is self-contained and can be used as a reference for implementing similar functionality in your projects.

### Frontend Components
```typescript
import { PriceDisplay, TradingButton, OrderForm } from './frontend-ui-components';

// Use in your React components
<PriceDisplay 
  symbol="SOL" 
  price={100.50} 
  change24h={5.2} 
/>
```

### Backend Services
```typescript
import { MarketDataService, OrderService } from './backend-api-services';

// Create service instances
const marketService = new MarketDataService(database, oracle);
const orderService = new OrderService(database);
```

### AI Agents
```typescript
import { MarketAnalysisAgent, TradingStrategyAgent } from './mikey-ai-agents';

// Create agent instances
const analysisAgent = new MarketAnalysisAgent(apiKey);
const strategyAgent = new TradingStrategyAgent(apiKey);
```

### Data Processing
```typescript
import { MarketDataProcessor, PriceAggregator } from './data-ingestion-processors';

// Create processor instances
const processor = new MarketDataProcessor(config);
const aggregator = new PriceAggregator(config);
```

### Smart Contract Interactions
```typescript
import { QuantDeskProgramClient, WalletHelper } from './smart-contract-interactions';

// Create client instances
const client = new QuantDeskProgramClient(connection, wallet, programId, idl);
const walletHelper = new WalletHelper(connection);
```

## 📚 Example Categories

### 1. Frontend UI Components
- **PriceDisplay** - Market price display component
- **TradingButton** - Buy/sell action buttons
- **OrderForm** - Order placement form
- **PortfolioSummary** - Portfolio overview component
- **useMarketData** - Market data React hook

### 2. Backend API Services
- **MarketDataService** - Market data operations
- **OrderService** - Order management
- **PortfolioService** - Portfolio operations
- **createApiRoutes** - Express route creation
- **errorHandler** - Error handling middleware

### 3. AI Agents
- **MarketAnalysisAgent** - Market sentiment analysis
- **TradingStrategyAgent** - Strategy generation
- **RiskManagementAgent** - Risk assessment
- **AgentFactory** - Agent creation utilities
- **AgentOrchestrator** - Multi-agent coordination

### 4. Data Processing
- **MarketDataProcessor** - WebSocket data processing
- **PriceAggregator** - Multi-source price aggregation
- **DataValidator** - Data quality validation
- **DataPipeline** - Complete data pipeline
- **RestApiDataSource** - REST API data source

### 5. Smart Contract Interactions
- **QuantDeskProgramClient** - Solana program client
- **MarketDataClient** - Market data client
- **TransactionBuilder** - Transaction construction
- **WalletHelper** - Wallet utilities
- **EventListener** - Event monitoring
- **ContractUtils** - Utility functions

## 🔧 Usage Examples

### Creating a Trading Bot
```typescript
import { MarketAnalysisAgent, OrderService } from './examples';

// Initialize services
const analysisAgent = new MarketAnalysisAgent(apiKey);
const orderService = new OrderService(database);

// Analyze market and place orders
const analysis = await analysisAgent.analyzeMarket('SOL', marketData);
if (analysis.sentiment === 'bullish') {
  await orderService.createOrder(userId, {
    symbol: 'SOL',
    side: 'buy',
    amount: 1.0,
    type: 'market'
  });
}
```

### Building a Data Dashboard
```typescript
import { MarketDataProcessor, PriceAggregator } from './examples';

// Set up data processing
const processor = new MarketDataProcessor(config);
const aggregator = new PriceAggregator(config);

processor.on('marketData', (data) => {
  aggregator.addPriceData(data);
});

const prices = aggregator.getAllPrices();
// Display prices in your dashboard
```

### Implementing Wallet Integration
```typescript
import { WalletHelper, QuantDeskProgramClient } from './examples';

// Create wallet and fund it
const wallet = WalletHelper.createWallet();
await WalletHelper.fundWallet(wallet, 1.0); // 1 SOL

// Create program client
const client = new QuantDeskProgramClient(connection, wallet, programId, idl);

// Create user account
const tx = await client.createUserAccount(wallet);
```

## 🧪 Testing Examples

Each example includes test patterns:

```typescript
// Example test structure
describe('MarketDataService', () => {
  let service: MarketDataService;
  
  beforeEach(() => {
    service = new MarketDataService(mockDatabase, mockOracle);
  });
  
  it('should get price for symbol', async () => {
    const price = await service.getPrice('SOL');
    expect(price).toBeGreaterThan(0);
  });
});
```

## 📖 Documentation

Each example file includes:
- **Comprehensive Comments** - Detailed explanations
- **Type Definitions** - TypeScript interfaces
- **Error Handling** - Proper error management
- **Best Practices** - Recommended patterns
- **Usage Examples** - Practical implementations

## 🔒 Security Considerations

Examples include security best practices:
- **Input Validation** - All inputs validated
- **Error Handling** - Secure error management
- **API Key Protection** - Secure credential handling
- **Rate Limiting** - Prevent API abuse
- **Data Sanitization** - Clean user inputs

## 🚀 Deployment

Examples are designed to be:
- **Production Ready** - Can be used in production
- **Scalable** - Handle high loads
- **Maintainable** - Easy to modify and extend
- **Testable** - Comprehensive test coverage

## 📄 License

All examples are part of QuantDesk and are licensed under Apache License 2.0.

## 🤝 Contributing

We welcome contributions to improve examples:
- **New Examples** - Additional use cases
- **Bug Fixes** - Fix issues in existing examples
- **Documentation** - Improve example documentation
- **Tests** - Add test coverage

## 📞 Support

For questions about examples:
- **GitHub Issues** - Report bugs or ask questions
- **Documentation** - Check service-specific READMEs
- **Community** - Join our Discord community
