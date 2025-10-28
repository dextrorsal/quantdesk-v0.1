# QuantDesk Developer API Guide

## 🚀 Quick Start

### Prerequisites
- Node.js 20+
- Solana wallet (Phantom, Solflare, or Backpack)
- Test SOL (for devnet testing)
- pnpm package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/dextrorsal/quantdesk.git
cd quantdesk

# Install dependencies
pnpm install

# Start all services
pnpm run dev
```

### Service Overview
| Service | Port | Description | Public APIs |
|---------|------|-------------|-------------|
| **Frontend** | 3001 | React trading interface | `/devnet-testing` |
| **Backend** | 3002 | API gateway and services | `/api/health`, `/api/dev/*` |
| **Data Ingestion** | 3003 | Real-time market data | `/health`, `/api/prices/*`, `/api/whales/*` |

---

## 🧪 Devnet Testing Interface

### Access the Testing Interface
- **URL**: http://localhost:3001/devnet-testing
- **Purpose**: Test Solana contract interactions and service integrations
- **Features**: Real-time service monitoring, wallet testing, transaction debugging

### Testing Components
- **Wallet Testing**: Connect and test wallet functionality
- **Account Testing**: Test account creation and management
- **Deposit/Withdraw Testing**: Test transaction flows
- **Service Health Monitoring**: Real-time status of all services
- **Debug Panel**: Comprehensive debugging information

### QuantDesk Program Integration
```typescript
import { Connection, PublicKey } from '@solana/web3.js';

// QuantDesk Program Configuration
const QUANTDESK_PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';
const DEVNET_RPC = 'https://api.devnet.solana.com';

// Initialize connection
const connection = new Connection(DEVNET_RPC, 'confirmed');
const programId = new PublicKey(QUANTDESK_PROGRAM_ID);

// Check program status
async function checkProgramStatus() {
  const accountInfo = await connection.getAccountInfo(programId);
  
  if (accountInfo) {
    console.log('✅ QuantDesk program found on devnet');
    console.log(`📊 Program data length: ${accountInfo.data.length} bytes`);
    console.log(`💰 Program balance: ${accountInfo.lamports / 1e9} SOL`);
    console.log(`🔧 Executable: ${accountInfo.executable}`);
  } else {
    console.log('❌ QuantDesk program not found on devnet');
  }
}
```

---

## 📊 Data Ingestion API Reference

### Health Check
```bash
GET /health
```

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2025-01-XX...",
  "version": "1.0.0",
  "environment": "development",
  "services": {
    "redis": "connected",
    "solana": "connected",
    "wallet": "loaded"
  },
  "wallet": {
    "address": "wallet_address",
    "loaded": true,
    "path": "/path/to/wallet.json"
  }
}
```

### Price Data
```bash
GET /api/prices/latest
```

**Response:**
```json
{
  "success": true,
  "data": {
    "prices": {
      "SOL": 100.5,
      "BTC": 45000,
      "ETH": 3000,
      "USDC": 1
    },
    "timestamp": "2025-01-XX...",
    "source": "pyth-network"
  }
}
```

### Whale Monitoring
```bash
GET /api/whales/recent?limit=10
```

**Response:**
```json
{
  "success": true,
  "data": {
    "transactions": [
      {
        "signature": "transaction_signature",
        "amount": 1000,
        "token": "SOL",
        "timestamp": "2025-01-XX..."
      }
    ],
    "count": 10,
    "timestamp": "2025-01-XX..."
  }
}
```

### Wallet Balance
```bash
GET /api/wallet/balance
```

**Response:**
```json
{
  "success": true,
  "data": {
    "address": "wallet_address",
    "balance": 1.5,
    "lamports": 1500000000,
    "timestamp": "2025-01-XX..."
  }
}
```

### Market Summary
```bash
GET /api/market/summary
```

**Response:**
```json
{
  "success": true,
  "data": {
    "totalVolume": 1000000,
    "activeMarkets": 3,
    "topGainers": ["SOL", "BTC"],
    "topLosers": ["ETH"],
    "timestamp": "2025-01-XX..."
  }
}
```

---

## 🔗 Solana Integration Examples

### Basic Wallet Integration
```typescript
import { Connection, Keypair, PublicKey } from '@solana/web3.js';

async function testWalletIntegration() {
  const connection = new Connection('https://api.devnet.solana.com', 'confirmed');
  const wallet = Keypair.generate();
  
  try {
    // Fund wallet with devnet SOL
    const signature = await connection.requestAirdrop(wallet.publicKey, 2 * 1e9);
    await connection.confirmTransaction(signature);
    
    // Check balance
    const balance = await connection.getBalance(wallet.publicKey);
    console.log(`✅ Wallet funded: ${balance / 1e9} SOL`);
    
    return wallet;
  } catch (error) {
    console.error('❌ Wallet funding failed:', error);
    throw error;
  }
}
```

### Program Interaction
```typescript
import { Connection, PublicKey, Transaction } from '@solana/web3.js';

async function interactWithQuantDeskProgram() {
  const connection = new Connection('https://api.devnet.solana.com');
  const programId = new PublicKey('C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw');
  
  // Check if program exists
  const accountInfo = await connection.getAccountInfo(programId);
  
  if (!accountInfo) {
    throw new Error('QuantDesk program not found on devnet');
  }
  
  console.log('✅ Program found:', {
    executable: accountInfo.executable,
    dataLength: accountInfo.data.length,
    balance: accountInfo.lamports / 1e9
  });
  
  return accountInfo;
}
```

---

## 🧪 Testing Examples

### Service Health Testing
```typescript
async function testServiceHealth() {
  const services = [
    { name: 'Backend', url: 'http://localhost:3002/api/health' },
    { name: 'Data Ingestion', url: 'http://localhost:3003/health' }
  ];
  
  const results = await Promise.allSettled(
    services.map(async (service) => {
      const response = await fetch(service.url);
      const data = await response.json();
      return { name: service.name, status: data.status, healthy: response.ok };
    })
  );
  
  results.forEach((result, index) => {
    if (result.status === 'fulfilled') {
      console.log(`✅ ${result.value.name}: ${result.value.status}`);
    } else {
      console.log(`❌ ${services[index].name}: ${result.reason.message}`);
    }
  });
}
```

### Integration Testing
```typescript
import { describe, it, expect, beforeAll } from 'vitest';
import { Connection, Keypair } from '@solana/web3.js';

describe('QuantDesk Integration Tests', () => {
  let connection: Connection;
  let testWallet: Keypair;

  beforeAll(async () => {
    connection = new Connection('https://api.devnet.solana.com', 'confirmed');
    testWallet = Keypair.generate();
    
    // Fund test wallet
    try {
      const signature = await connection.requestAirdrop(testWallet.publicKey, 2 * 1e9);
      await connection.confirmTransaction(signature);
      console.log('✅ Test wallet funded');
    } catch (error) {
      console.warn('⚠️ Could not fund test wallet:', error);
    }
  }, 30000);

  it('should connect to all services', async () => {
    const services = [
      'http://localhost:3002/api/health',
      'http://localhost:3003/health'
    ];

    for (const service of services) {
      const response = await fetch(service);
      expect(response.ok).toBe(true);
    }
  });

  it('should interact with QuantDesk program', async () => {
    const programId = new PublicKey('C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw');
    const accountInfo = await connection.getAccountInfo(programId);
    
    expect(accountInfo).toBeDefined();
    expect(accountInfo!.executable).toBe(true);
    expect(accountInfo!.owner.toString()).toBe('BPFLoaderUpgradeab1e11111111111111111111111');
  });

  it('should fetch real-time data', async () => {
    const response = await fetch('http://localhost:3003/api/prices/latest');
    const data = await response.json();
    
    expect(data.success).toBe(true);
    expect(data.data.prices).toHaveProperty('SOL');
    expect(data.data.prices).toHaveProperty('BTC');
    expect(data.data.prices).toHaveProperty('ETH');
  });
});
```

---

## 🚀 Deployment

### Environment Setup
```bash
# Copy environment template
cp env.template .env

# Configure services
SOLANA_RPC_URL=https://api.devnet.solana.com
SOLANA_WALLET=/path/to/your/wallet.json
DATA_INGESTION_PORT=3003
CORS_ORIGIN=http://localhost:3001
```

### Production Deployment
```bash
# Build all services
pnpm run build

# Start production
pnpm run start:prod
```

### Docker Deployment
```bash
# Build Docker images
docker-compose build

# Start services
docker-compose up -d
```

---

## 🔧 Development Tools

### Testing Commands
```bash
# Run integration tests
cd frontend && pnpm test src/tests/integration/real-services.test.ts

# Run all tests
pnpm run test

# Run tests with coverage
pnpm run test:coverage
```

### Debugging
```bash
# Check service logs
docker-compose logs -f data-ingestion
docker-compose logs -f backend

# Monitor service health
curl http://localhost:3003/health | jq
```

---

## 📚 Additional Resources

- [Solana Documentation](https://docs.solana.com/)
- [Anchor Framework](https://www.anchor-lang.com/)
- [React Solana Wallet Adapter](https://github.com/solana-labs/wallet-adapter)
- [Pyth Network](https://pyth.network/)

---

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details on how to:
- Submit bug reports
- Propose new features
- Submit pull requests
- Follow our coding standards

---

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.
