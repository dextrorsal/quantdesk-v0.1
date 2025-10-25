# QuantDesk SDK Setup Guide

## 🚀 **Complete Setup Guide for QuantDesk SDK**

This guide provides step-by-step instructions for setting up and using the QuantDesk SDK with proper security measures and realistic examples.

## 📋 **Prerequisites**

### **Required Software**
- Node.js 20+ (LTS recommended)
- pnpm (package manager)
- TypeScript 5+
- Solana CLI tools
- Git

### **Required Accounts**
- Solana wallet (Phantom, Solflare, etc.)
- QuantDesk account (for API access)
- Devnet SOL (for testing)

## 🛠️ **Installation**

### **Step 1: Install Node.js and pnpm**
```bash
# Install Node.js (using nvm recommended)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.0/install.sh | bash
nvm install 20
nvm use 20

# Install pnpm
npm install -g pnpm
```

### **Step 2: Install Solana CLI**
```bash
# Install Solana CLI
sh -c "$(curl -sSfL https://release.solana.com/v1.16.0/install)"
export PATH="$HOME/.local/share/solana/install/active_release/bin:$PATH"

# Verify installation
solana --version
```

### **Step 3: Install QuantDesk SDK**
```bash
# Create project directory
mkdir quantdesk-sdk-project
cd quantdesk-sdk-project

# Initialize project
pnpm init

# Install QuantDesk SDK
pnpm add @quantdesk/sdk

# Install additional dependencies
pnpm add @solana/web3.js @solana/spl-token
pnpm add -D typescript @types/node ts-node
```

## 🔧 **Project Setup**

### **Step 1: Create TypeScript Configuration**
```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2022",
    "module": "commonjs",
    "lib": ["ES2022"],
    "outDir": "./dist",
    "rootDir": "./src",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "resolveJsonModule": true,
    "declaration": true,
    "declarationMap": true,
    "sourceMap": true
  },
  "include": ["src/**/*"],
  "exclude": ["node_modules", "dist"]
}
```

### **Step 2: Create Environment Configuration**
```bash
# Create .env file
touch .env
```

```bash
# .env file content
# QuantDesk Configuration
QUANTDESK_RPC_URL=https://api.devnet.solana.com
QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw

# Wallet Configuration (NEVER commit private keys)
SOLANA_PRIVATE_KEY=your_base58_private_key_here
SOLANA_WALLET_ADDRESS=your_wallet_address_here

# API Configuration
QUANTDESK_API_KEY=your_api_key_here
QUANTDESK_API_SECRET=your_api_secret_here

# Environment
NODE_ENV=development
```

### **Step 3: Create .gitignore**
```gitignore
# Dependencies
node_modules/
pnpm-lock.yaml

# Build output
dist/
build/

# Environment variables
.env
.env.local
.env.production

# Logs
logs/
*.log

# Runtime data
pids/
*.pid
*.seed
*.pid.lock

# Coverage directory used by tools like istanbul
coverage/

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db
```

## 🔒 **Security Setup**

### **Step 1: Environment Variable Security**
```typescript
// src/config/environment.ts
import { config } from 'dotenv';

// Load environment variables
config();

// Validate required environment variables
const requiredEnvVars = [
  'QUANTDESK_RPC_URL',
  'QUANTDESK_PROGRAM_ID',
  'SOLANA_PRIVATE_KEY',
  'SOLANA_WALLET_ADDRESS'
];

for (const envVar of requiredEnvVars) {
  if (!process.env[envVar]) {
    throw new Error(`Missing required environment variable: ${envVar}`);
  }
}

// Export configuration
export const config = {
  quantdesk: {
    rpcUrl: process.env.QUANTDESK_RPC_URL!,
    programId: process.env.QUANTDESK_PROGRAM_ID!,
    apiKey: process.env.QUANTDESK_API_KEY,
    apiSecret: process.env.QUANTDESK_API_SECRET
  },
  solana: {
    privateKey: process.env.SOLANA_PRIVATE_KEY!,
    walletAddress: process.env.SOLANA_WALLET_ADDRESS!
  },
  environment: process.env.NODE_ENV || 'development'
};
```

### **Step 2: Input Validation**
```typescript
// src/utils/validation.ts
export class ValidationError extends Error {
  constructor(message: string) {
    super(message);
    this.name = 'ValidationError';
  }
}

export function validateMarketSymbol(symbol: string): void {
  if (!symbol || typeof symbol !== 'string') {
    throw new ValidationError('Market symbol must be a non-empty string');
  }
  
  if (!symbol.includes('-PERP')) {
    throw new ValidationError('Market symbol must end with -PERP');
  }
}

export function validateOrderSize(size: number): void {
  if (typeof size !== 'number' || size <= 0) {
    throw new ValidationError('Order size must be a positive number');
  }
  
  if (size < 0.001) {
    throw new ValidationError('Order size must be at least 0.001');
  }
  
  if (size > 1000) {
    throw new ValidationError('Order size must not exceed 1000');
  }
}

export function validatePrice(price: number): void {
  if (typeof price !== 'number' || price <= 0) {
    throw new ValidationError('Price must be a positive number');
  }
  
  if (price < 0.01) {
    throw new ValidationError('Price must be at least 0.01');
  }
}
```

### **Step 3: Error Handling**
```typescript
// src/utils/error-handling.ts
export class QuantDeskError extends Error {
  public readonly code: string;
  public readonly details?: any;

  constructor(message: string, code: string, details?: any) {
    super(message);
    this.name = 'QuantDeskError';
    this.code = code;
    this.details = details;
  }
}

export function handleError(error: any): QuantDeskError {
  if (error instanceof QuantDeskError) {
    return error;
  }
  
  // Handle common error types
  if (error.message?.includes('insufficient funds')) {
    return new QuantDeskError('Insufficient funds', 'INSUFFICIENT_FUNDS', error);
  }
  
  if (error.message?.includes('market closed')) {
    return new QuantDeskError('Market is closed', 'MARKET_CLOSED', error);
  }
  
  if (error.message?.includes('rate limit')) {
    return new QuantDeskError('Rate limit exceeded', 'RATE_LIMIT_EXCEEDED', error);
  }
  
  // Generic error
  return new QuantDeskError('Unknown error occurred', 'UNKNOWN_ERROR', error);
}
```

## 📚 **Basic Usage Examples**

### **Step 1: Create Basic Client**
```typescript
// src/examples/basic-client.ts
import { QuantDeskClient } from '@quantdesk/sdk';
import { config } from '../config/environment';
import { handleError } from '../utils/error-handling';

export class BasicClient {
  private client: QuantDeskClient;

  constructor() {
    this.client = new QuantDeskClient({
      rpcUrl: config.quantdesk.rpcUrl,
      programId: config.quantdesk.programId,
      wallet: this.createWallet()
    });
  }

  private createWallet() {
    // Create wallet from private key
    // Note: In production, use proper wallet management
    const { Keypair } = require('@solana/web3.js');
    const bs58 = require('bs58');
    
    const privateKeyBytes = bs58.decode(config.solana.privateKey);
    return Keypair.fromSecretKey(privateKeyBytes);
  }

  async initialize(): Promise<void> {
    try {
      await this.client.initialize();
      console.log('✅ Client initialized successfully');
    } catch (error) {
      throw handleError(error);
    }
  }

  async getMarkets(): Promise<any[]> {
    try {
      return await this.client.getMarkets();
    } catch (error) {
      throw handleError(error);
    }
  }

  async getMarketData(market: string): Promise<any> {
    try {
      return await this.client.getMarketData(market);
    } catch (error) {
      throw handleError(error);
    }
  }
}
```

### **Step 2: Create Portfolio Tracker**
```typescript
// src/examples/portfolio-tracker.ts
import { BasicClient } from './basic-client';
import { validateMarketSymbol } from '../utils/validation';

export class PortfolioTracker {
  private client: BasicClient;

  constructor(client: BasicClient) {
    this.client = client;
  }

  async trackPortfolio(): Promise<void> {
    try {
      console.log('📊 Portfolio Tracking Started');
      
      // Get portfolio data
      const portfolio = await this.client.getPortfolio();
      
      console.log(`💰 Total Value: $${portfolio.totalValue.toFixed(2)}`);
      console.log(`📈 Total PnL: $${portfolio.totalPnL.toFixed(2)}`);
      console.log(`📊 PnL %: ${((portfolio.totalPnL / portfolio.totalValue) * 100).toFixed(2)}%`);
      
    } catch (error) {
      console.error('❌ Portfolio tracking failed:', error);
    }
  }

  async trackMarket(market: string): Promise<void> {
    try {
      validateMarketSymbol(market);
      
      const marketData = await this.client.getMarketData(market);
      
      console.log(`📊 ${market} Market Data:`);
      console.log(`  Price: $${marketData.price.toFixed(2)}`);
      console.log(`  Volume: ${marketData.volume.toFixed(2)}`);
      console.log(`  Change 24h: ${(marketData.change24h * 100).toFixed(2)}%`);
      
    } catch (error) {
      console.error(`❌ Market tracking failed for ${market}:`, error);
    }
  }
}
```

### **Step 3: Create Main Application**
```typescript
// src/index.ts
import { BasicClient } from './examples/basic-client';
import { PortfolioTracker } from './examples/portfolio-tracker';

async function main() {
  try {
    console.log('🚀 Starting QuantDesk SDK Application');
    
    // Initialize client
    const client = new BasicClient();
    await client.initialize();
    
    // Create portfolio tracker
    const tracker = new PortfolioTracker(client);
    
    // Track portfolio
    await tracker.trackPortfolio();
    
    // Track specific markets
    await tracker.trackMarket('SOL-PERP');
    await tracker.trackMarket('ETH-PERP');
    
    console.log('✅ Application completed successfully');
    
  } catch (error) {
    console.error('❌ Application failed:', error);
    process.exit(1);
  }
}

// Run the application
main();
```

## 🧪 **Testing Setup**

### **Step 1: Install Testing Dependencies**
```bash
pnpm add -D jest @types/jest ts-jest
```

### **Step 2: Create Jest Configuration**
```javascript
// jest.config.js
module.exports = {
  preset: 'ts-jest',
  testEnvironment: 'node',
  roots: ['<rootDir>/src'],
  testMatch: ['**/__tests__/**/*.ts', '**/?(*.)+(spec|test).ts'],
  collectCoverageFrom: [
    'src/**/*.ts',
    '!src/**/*.d.ts',
    '!src/index.ts'
  ],
  coverageDirectory: 'coverage',
  coverageReporters: ['text', 'lcov', 'html']
};
```

### **Step 3: Create Test Files**
```typescript
// src/__tests__/validation.test.ts
import { validateMarketSymbol, validateOrderSize, validatePrice } from '../utils/validation';

describe('Validation', () => {
  describe('validateMarketSymbol', () => {
    it('should validate correct market symbol', () => {
      expect(() => validateMarketSymbol('SOL-PERP')).not.toThrow();
    });

    it('should throw error for invalid market symbol', () => {
      expect(() => validateMarketSymbol('INVALID')).toThrow('Market symbol must end with -PERP');
    });
  });

  describe('validateOrderSize', () => {
    it('should validate correct order size', () => {
      expect(() => validateOrderSize(1.0)).not.toThrow();
    });

    it('should throw error for negative order size', () => {
      expect(() => validateOrderSize(-1.0)).toThrow('Order size must be a positive number');
    });
  });
});
```

## 🚀 **Running the Application**

### **Step 1: Build the Project**
```bash
# Build TypeScript
pnpm run build

# Or run directly with ts-node
pnpm add -D ts-node
```

### **Step 2: Run the Application**
```bash
# Run with ts-node
npx ts-node src/index.ts

# Or run compiled JavaScript
node dist/index.js
```

### **Step 3: Run Tests**
```bash
# Run all tests
pnpm test

# Run tests with coverage
pnpm test -- --coverage
```

## 📖 **Package.json Scripts**

```json
{
  "scripts": {
    "build": "tsc",
    "start": "node dist/index.js",
    "dev": "ts-node src/index.ts",
    "test": "jest",
    "test:watch": "jest --watch",
    "test:coverage": "jest --coverage",
    "lint": "eslint src/**/*.ts",
    "lint:fix": "eslint src/**/*.ts --fix",
    "clean": "rm -rf dist"
  }
}
```

## 🔒 **Security Best Practices**

### **1. Environment Variables**
- Never commit `.env` files
- Use different environments for dev/staging/production
- Rotate API keys regularly
- Use environment-specific configurations

### **2. Input Validation**
- Validate all user inputs
- Sanitize data before processing
- Use TypeScript strict mode
- Implement proper error handling

### **3. Error Handling**
- Don't expose sensitive information in errors
- Log errors securely
- Implement proper error recovery
- Use structured error responses

### **4. Rate Limiting**
- Implement client-side rate limiting
- Respect API rate limits
- Use exponential backoff for retries
- Monitor API usage

## 📞 **Support and Resources**

### **Documentation**
- [QuantDesk SDK Documentation](https://github.com/dextrorsal/quantdesk-v0.1/tree/main/sdk)
- [API Reference](https://github.com/dextrorsal/quantdesk-v0.1/tree/main/docs/api)
- [Examples](https://github.com/dextrorsal/quantdesk-v0.1/tree/main/examples)

### **Community**
- [GitHub Issues](https://github.com/dextrorsal/quantdesk-v0.1/issues)
- [Discord Community](https://discord.gg/quantdesk)
- [Documentation](https://docs.quantdesk.com)

---

**QuantDesk SDK Setup Guide: Complete setup instructions with security measures and realistic examples for practical implementation.**
