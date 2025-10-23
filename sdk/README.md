# SDK Directory

This directory contains the QuantDesk Software Development Kit (SDK) - a collection of reusable libraries and utilities for building applications on top of QuantDesk.

## 📁 Structure

```
sdk/
├── typescript/        # TypeScript SDK
│   ├── src/         # Source code
│   ├── dist/        # Compiled output
│   ├── tests/       # Test files
│   └── package.json # Package configuration
└── README.md        # This file
```

## 🚀 Quick Start

### Installation
```bash
# Install the TypeScript SDK
cd sdk/typescript
pnpm install
pnpm run build
```

### Usage
```typescript
import { QuantDeskSDK } from '@quantdesk/sdk';

// Initialize the SDK
const sdk = new QuantDeskSDK({
  apiUrl: 'https://api.quantdesk.com',
  network: 'mainnet-beta'
});

// Use SDK methods
const marketData = await sdk.market.getPrice('SOL');
const portfolio = await sdk.portfolio.getBalance();
```

## 🏗️ SDK Architecture

### Core Components
- **Client** - Main SDK client
- **Services** - Feature-specific services
- **Types** - TypeScript type definitions
- **Utils** - Utility functions
- **Constants** - SDK constants

### Service Modules
- **Market** - Market data operations
- **Portfolio** - Portfolio management
- **Orders** - Order management
- **Wallet** - Wallet operations
- **AI** - AI agent interactions

## 📚 API Reference

### Market Service
```typescript
// Get current price
const price = await sdk.market.getPrice('SOL');

// Get price history
const history = await sdk.market.getHistory('SOL', '1h');

// Get market summary
const summary = await sdk.market.getSummary();
```

### Portfolio Service
```typescript
// Get portfolio
const portfolio = await sdk.portfolio.get();

// Get balance for specific asset
const balance = await sdk.portfolio.getBalance('SOL');

// Get positions
const positions = await sdk.portfolio.getPositions();
```

### Order Service
```typescript
// Create order
const order = await sdk.orders.create({
  symbol: 'SOL',
  side: 'buy',
  amount: 1.0,
  type: 'market'
});

// Get orders
const orders = await sdk.orders.get();

// Cancel order
await sdk.orders.cancel(orderId);
```

### Wallet Service
```typescript
// Connect wallet
await sdk.wallet.connect();

// Get wallet address
const address = sdk.wallet.getAddress();

// Sign transaction
const signature = await sdk.wallet.signTransaction(tx);
```

### AI Service
```typescript
// Analyze market
const analysis = await sdk.ai.analyzeMarket('SOL');

// Generate strategy
const strategy = await sdk.ai.generateStrategy({
  riskTolerance: 'moderate',
  timeHorizon: 'medium'
});

// Assess risk
const risk = await sdk.ai.assessRisk(portfolio);
```

## 🔧 Configuration

### SDK Options
```typescript
interface SDKOptions {
  apiUrl: string;           // Backend API URL
  network: 'devnet' | 'mainnet-beta'; // Solana network
  apiKey?: string;         // Optional API key
  timeout?: number;        // Request timeout
  retries?: number;        // Retry attempts
}
```

### Environment Setup
```typescript
// Development
const sdk = new QuantDeskSDK({
  apiUrl: 'http://localhost:3002',
  network: 'devnet'
});

// Production
const sdk = new QuantDeskSDK({
  apiUrl: 'https://api.quantdesk.com',
  network: 'mainnet-beta',
  apiKey: 'your-api-key'
});
```

## 🧪 Testing

### Unit Tests
```bash
cd sdk/typescript
pnpm run test
```

### Integration Tests
```bash
pnpm run test:integration
```

### Coverage
```bash
pnpm run coverage
```

## 📦 Building

### Development Build
```bash
pnpm run build:dev
```

### Production Build
```bash
pnpm run build:prod
```

### Type Definitions
```bash
pnpm run build:types
```

## 🚀 Publishing

### Version Management
```bash
# Bump version
pnpm version patch  # 1.0.0 -> 1.0.1
pnpm version minor  # 1.0.0 -> 1.1.0
pnpm version major  # 1.0.0 -> 2.0.0
```

### Publishing
```bash
pnpm publish
```

## 📖 Documentation

### API Documentation
- **Market API** - Market data endpoints
- **Portfolio API** - Portfolio operations
- **Order API** - Order management
- **Wallet API** - Wallet operations
- **AI API** - AI agent interactions

### Type Definitions
```typescript
// Market data types
interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  timestamp: Date;
}

// Order types
interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  amount: number;
  price?: number;
  type: 'market' | 'limit';
  status: 'pending' | 'filled' | 'cancelled';
}

// Portfolio types
interface Portfolio {
  totalValue: number;
  positions: Position[];
}

interface Position {
  symbol: string;
  amount: number;
  value: number;
  pnl: number;
}
```

## 🔒 Security

### Authentication
```typescript
// JWT token authentication
sdk.setAuthToken('your-jwt-token');

// API key authentication
sdk.setApiKey('your-api-key');
```

### Request Security
- **HTTPS Only** - All requests use HTTPS
- **Token Validation** - JWT tokens validated
- **Rate Limiting** - Built-in rate limiting
- **Input Validation** - All inputs validated

## 🐛 Error Handling

### Error Types
```typescript
// SDK errors
class SDKError extends Error {
  code: string;
  status?: number;
}

// Network errors
class NetworkError extends SDKError {}

// Validation errors
class ValidationError extends SDKError {}
```

### Error Handling
```typescript
try {
  const price = await sdk.market.getPrice('SOL');
} catch (error) {
  if (error instanceof NetworkError) {
    console.error('Network error:', error.message);
  } else if (error instanceof ValidationError) {
    console.error('Validation error:', error.message);
  }
}
```

## 📊 Monitoring

### Metrics
- **Request Rate** - API request frequency
- **Response Time** - API response latency
- **Error Rate** - API error frequency
- **Cache Hit Rate** - Cache performance

### Logging
```typescript
// Enable debug logging
sdk.setLogLevel('debug');

// Custom logger
sdk.setLogger({
  info: (message) => console.log(message),
  warn: (message) => console.warn(message),
  error: (message) => console.error(message)
});
```

## 🔄 Caching

### Built-in Caching
```typescript
// Enable caching
sdk.enableCache({
  ttl: 60000, // 1 minute
  maxSize: 1000
});

// Cache market data
const price = await sdk.market.getPrice('SOL'); // Cached
```

### Custom Cache
```typescript
// Custom cache implementation
sdk.setCache({
  get: (key) => customCache.get(key),
  set: (key, value, ttl) => customCache.set(key, value, ttl),
  delete: (key) => customCache.delete(key)
});
```

## 📄 License

The QuantDesk SDK is licensed under Apache License 2.0.

## 🤝 Contributing

We welcome contributions to the SDK:
- **New Features** - Additional SDK functionality
- **Bug Fixes** - Fix issues in existing code
- **Documentation** - Improve SDK documentation
- **Tests** - Add test coverage

## 📞 Support

For SDK support:
- **GitHub Issues** - Report bugs or ask questions
- **Documentation** - Check API documentation
- **Community** - Join our Discord community
- **Email** - Contact support@quantdesk.com
