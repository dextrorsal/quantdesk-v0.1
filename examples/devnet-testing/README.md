# Devnet Testing Examples

This directory contains working examples for testing QuantDesk's Solana devnet integration.

## 🚀 Quick Start

```bash
# Install dependencies
npm install

# Run all examples
npm run test-all

# Run specific examples
npm run basic-service-test
npm run wallet-integration
npm run program-interaction
```

## 📁 Examples

### 1. Basic Service Testing (`basic-service-test.js`)
Tests the health of all QuantDesk services.

```javascript
const services = [
  { name: 'Backend', url: 'http://localhost:3002/api/health' },
  { name: 'Data Ingestion', url: 'http://localhost:3003/health' }
];

async function testServices() {
  for (const service of services) {
    try {
      const response = await fetch(service.url);
      const data = await response.json();
      console.log(`✅ ${service.name}: ${data.status || 'healthy'}`);
    } catch (error) {
      console.log(`❌ ${service.name}: ${error.message}`);
    }
  }
}
```

### 2. Wallet Integration (`wallet-integration.js`)
Demonstrates wallet creation, funding, and balance checking.

```javascript
import { Connection, Keypair } from '@solana/web3.js';

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

### 3. Program Interaction (`program-interaction.js`)
Tests interaction with the deployed QuantDesk program.

```javascript
import { Connection, PublicKey } from '@solana/web3.js';

async function testQuantDeskProgram() {
  const connection = new Connection('https://api.devnet.solana.com');
  const programId = new PublicKey('C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw');
  
  try {
    // Check if program exists
    const accountInfo = await connection.getAccountInfo(programId);
    
    if (!accountInfo) {
      throw new Error('QuantDesk program not found on devnet');
    }
    
    console.log('✅ QuantDesk Program Status:', {
      executable: accountInfo.executable,
      dataLength: accountInfo.data.length,
      balance: `${accountInfo.lamports / 1e9} SOL`,
      owner: accountInfo.owner.toString()
    });
    
    return accountInfo;
  } catch (error) {
    console.error('❌ Program check failed:', error);
    throw error;
  }
}
```

### 4. Real-Time Data Testing (`realtime-data.js`)
Tests fetching real-time market data from the data ingestion service.

```javascript
async function testRealTimeData() {
  const endpoints = [
    { name: 'Latest Prices', url: 'http://localhost:3003/api/prices/latest' },
    { name: 'Whale Transactions', url: 'http://localhost:3003/api/whales/recent?limit=5' },
    { name: 'Market Summary', url: 'http://localhost:3003/api/market/summary' },
    { name: 'Wallet Balance', url: 'http://localhost:3003/api/wallet/balance' }
  ];
  
  for (const endpoint of endpoints) {
    try {
      const response = await fetch(endpoint.url);
      const data = await response.json();
      
      console.log(`✅ ${endpoint.name}:`, {
        success: data.success,
        dataKeys: Object.keys(data.data || {}),
        timestamp: data.data?.timestamp || data.timestamp
      });
    } catch (error) {
      console.log(`❌ ${endpoint.name}: ${error.message}`);
    }
  }
}
```

## 🧪 Integration Testing

### Complete Integration Test (`integration-test.js`)
Comprehensive test that validates all components working together.

```javascript
import { describe, it, expect, beforeAll } from 'vitest';
import { Connection, Keypair, PublicKey } from '@solana/web3.js';

describe('QuantDesk Devnet Integration', () => {
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

## 🔧 Configuration

### Environment Variables
```bash
# .env file
SOLANA_RPC_URL=https://api.devnet.solana.com
QUANTDESK_PROGRAM_ID=C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw
BACKEND_URL=http://localhost:3002
DATA_INGESTION_URL=http://localhost:3003
```

### Package.json Scripts
```json
{
  "scripts": {
    "test-all": "node basic-service-test.js && node wallet-integration.js && node program-interaction.js",
    "basic-service-test": "node basic-service-test.js",
    "wallet-integration": "node wallet-integration.js",
    "program-interaction": "node program-interaction.js",
    "realtime-data": "node realtime-data.js",
    "integration-test": "vitest integration-test.js"
  }
}
```

## 📊 Expected Output

### Successful Service Test
```
✅ Backend: healthy
✅ Data Ingestion: healthy
```

### Successful Wallet Integration
```
✅ Wallet funded: 2.0000 SOL
```

### Successful Program Interaction
```
✅ QuantDesk Program Status: {
  executable: true,
  dataLength: 36,
  balance: '0.00114144 SOL',
  owner: 'BPFLoaderUpgradeab1e11111111111111111111111'
}
```

### Successful Real-Time Data
```
✅ Latest Prices: { success: true, dataKeys: ['prices', 'timestamp', 'source'] }
✅ Whale Transactions: { success: true, dataKeys: ['transactions', 'count', 'timestamp'] }
✅ Market Summary: { success: true, dataKeys: ['totalVolume', 'activeMarkets'] }
✅ Wallet Balance: { success: true, dataKeys: ['address', 'balance', 'lamports'] }
```

## 🚨 Troubleshooting

### Common Issues

1. **Services not running**
   ```bash
   # Start all services
   cd /path/to/quantdesk
   pnpm run dev
   ```

2. **Wallet funding fails**
   ```bash
   # Use Solana faucet
   solana airdrop 2 --url https://api.devnet.solana.com
   ```

3. **Program not found**
   - Ensure you're connected to devnet
   - Check program ID is correct: `C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw`

## 📚 Additional Resources

- [Solana Devnet Documentation](https://docs.solana.com/developing/test-validator)
- [QuantDesk Developer Guide](../docs/DEVELOPER_API_GUIDE.md)
- [Solana Web3.js Documentation](https://solana-labs.github.io/solana-web3.js/)
