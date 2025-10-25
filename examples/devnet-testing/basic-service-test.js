import { Connection, Keypair, PublicKey } from '@solana/web3.js';

// QuantDesk Program Configuration
const QUANTDESK_PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';
const DEVNET_RPC = 'https://api.devnet.solana.com';

/**
 * Basic Service Health Testing
 * Tests the health of all QuantDesk services
 */
async function testServiceHealth() {
  console.log('🧪 Testing QuantDesk Service Health...\n');
  
  const services = [
    { name: 'Backend API', url: 'http://localhost:3002/api/health' },
    { name: 'Data Ingestion', url: 'http://localhost:3003/health' }
  ];
  
  const results = await Promise.allSettled(
    services.map(async (service) => {
      const response = await fetch(service.url);
      const data = await response.json();
      return { 
        name: service.name, 
        status: data.status || data.data?.status || 'unknown',
        healthy: response.ok 
      };
    })
  );
  
  results.forEach((result, index) => {
    if (result.status === 'fulfilled') {
      const { name, status, healthy } = result.value;
      console.log(`${healthy ? '✅' : '❌'} ${name}: ${status}`);
    } else {
      console.log(`❌ ${services[index].name}: ${result.reason.message}`);
    }
  });
  
  console.log('');
}

/**
 * Wallet Integration Testing
 * Demonstrates wallet creation, funding, and balance checking
 */
async function testWalletIntegration() {
  console.log('💰 Testing Wallet Integration...\n');
  
  const connection = new Connection(DEVNET_RPC, 'confirmed');
  const wallet = Keypair.generate();
  
  try {
    console.log(`📍 Generated wallet: ${wallet.publicKey.toString()}`);
    
    // Fund wallet with devnet SOL
    console.log('🔄 Requesting devnet SOL airdrop...');
    const signature = await connection.requestAirdrop(wallet.publicKey, 2 * 1e9);
    await connection.confirmTransaction(signature);
    
    // Check balance
    const balance = await connection.getBalance(wallet.publicKey);
    console.log(`✅ Wallet funded: ${balance / 1e9} SOL`);
    
    return wallet;
  } catch (error) {
    console.error('❌ Wallet funding failed:', error.message);
    console.log('💡 Try using the Solana faucet: https://faucet.solana.com');
    throw error;
  }
}

/**
 * QuantDesk Program Interaction
 * Tests interaction with the deployed QuantDesk program
 */
async function testQuantDeskProgram() {
  console.log('🔗 Testing QuantDesk Program Integration...\n');
  
  const connection = new Connection(DEVNET_RPC);
  const programId = new PublicKey(QUANTDESK_PROGRAM_ID);
  
  try {
    // Check if program exists
    const accountInfo = await connection.getAccountInfo(programId);
    
    if (!accountInfo) {
      throw new Error('QuantDesk program not found on devnet');
    }
    
    console.log('✅ QuantDesk Program Status:', {
      programId: QUANTDESK_PROGRAM_ID,
      executable: accountInfo.executable,
      dataLength: `${accountInfo.data.length} bytes`,
      balance: `${accountInfo.lamports / 1e9} SOL`,
      owner: accountInfo.owner.toString()
    });
    
    return accountInfo;
  } catch (error) {
    console.error('❌ Program check failed:', error.message);
    throw error;
  }
}

/**
 * Real-Time Data Testing
 * Tests fetching real-time market data from the data ingestion service
 */
async function testRealTimeData() {
  console.log('📊 Testing Real-Time Data APIs...\n');
  
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
      
      if (data.success) {
        console.log(`✅ ${endpoint.name}:`, {
          success: data.success,
          dataKeys: Object.keys(data.data || {}),
          timestamp: data.data?.timestamp || data.timestamp
        });
      } else {
        console.log(`⚠️ ${endpoint.name}: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.log(`❌ ${endpoint.name}: ${error.message}`);
    }
  }
  
  console.log('');
}

/**
 * Complete Integration Test
 * Runs all tests in sequence
 */
async function runCompleteTest() {
  console.log('🚀 Starting QuantDesk Devnet Integration Test\n');
  console.log('=' .repeat(50));
  
  try {
    // Test service health
    await testServiceHealth();
    
    // Test wallet integration
    const wallet = await testWalletIntegration();
    
    // Test program interaction
    await testQuantDeskProgram();
    
    // Test real-time data
    await testRealTimeData();
    
    console.log('=' .repeat(50));
    console.log('🎉 All tests completed successfully!');
    console.log('✅ QuantDesk devnet integration is working properly');
    
  } catch (error) {
    console.log('=' .repeat(50));
    console.error('❌ Integration test failed:', error.message);
    console.log('💡 Make sure all services are running: pnpm run dev');
  }
}

// Export functions for individual testing
export {
  testServiceHealth,
  testWalletIntegration,
  testQuantDeskProgram,
  testRealTimeData,
  runCompleteTest
};

// Run complete test if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runCompleteTest();
}
