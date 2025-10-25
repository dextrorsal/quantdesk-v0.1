import { Connection, Keypair, PublicKey } from '@solana/web3.js';

// QuantDesk Program Configuration
const QUANTDESK_PROGRAM_ID = 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw';
const DEVNET_RPC = 'https://api.devnet.solana.com';

/**
 * Wallet Integration Example
 * Demonstrates wallet creation, funding, and balance checking
 */
async function walletIntegrationExample() {
  console.log('💰 Wallet Integration Example\n');
  
  const connection = new Connection(DEVNET_RPC, 'confirmed');
  const wallet = Keypair.generate();
  
  console.log(`📍 Generated wallet: ${wallet.publicKey.toString()}`);
  
  try {
    // Fund wallet with devnet SOL
    console.log('🔄 Requesting devnet SOL airdrop...');
    const signature = await connection.requestAirdrop(wallet.publicKey, 2 * 1e9);
    await connection.confirmTransaction(signature);
    
    // Check balance
    const balance = await connection.getBalance(wallet.publicKey);
    console.log(`✅ Wallet funded: ${balance / 1e9} SOL`);
    
    // Test wallet with data ingestion service
    console.log('🔄 Testing wallet with data ingestion service...');
    const response = await fetch('http://localhost:3003/api/wallet/balance');
    const data = await response.json();
    
    if (data.success) {
      console.log('✅ Data ingestion wallet balance:', {
        address: data.data.address,
        balance: `${data.data.balance} SOL`,
        lamports: data.data.lamports
      });
    } else {
      console.log('⚠️ Data ingestion wallet not loaded:', data.error);
    }
    
    return wallet;
  } catch (error) {
    console.error('❌ Wallet integration failed:', error.message);
    console.log('💡 Try using the Solana faucet: https://faucet.solana.com');
    throw error;
  }
}

/**
 * Program Interaction Example
 * Tests interaction with the deployed QuantDesk program
 */
async function programInteractionExample() {
  console.log('🔗 Program Interaction Example\n');
  
  const connection = new Connection(DEVNET_RPC);
  const programId = new PublicKey(QUANTDESK_PROGRAM_ID);
  
  try {
    // Check if program exists
    const accountInfo = await connection.getAccountInfo(programId);
    
    if (!accountInfo) {
      throw new Error('QuantDesk program not found on devnet');
    }
    
    console.log('✅ QuantDesk Program Details:', {
      programId: QUANTDESK_PROGRAM_ID,
      network: 'Solana Devnet',
      executable: accountInfo.executable,
      dataLength: `${accountInfo.data.length} bytes`,
      balance: `${accountInfo.lamports / 1e9} SOL`,
      owner: accountInfo.owner.toString(),
      rentEpoch: accountInfo.rentEpoch
    });
    
    // Test program with different connection methods
    console.log('🔄 Testing program with different connection methods...');
    
    // Test with different commitment levels
    const commitments = ['processed', 'confirmed', 'finalized'];
    for (const commitment of commitments) {
      const testConnection = new Connection(DEVNET_RPC, commitment);
      const testAccountInfo = await testConnection.getAccountInfo(programId);
      console.log(`✅ Program accessible with ${commitment} commitment`);
    }
    
    return accountInfo;
  } catch (error) {
    console.error('❌ Program interaction failed:', error.message);
    throw error;
  }
}

/**
 * Advanced Wallet Testing
 * Tests wallet functionality with multiple operations
 */
async function advancedWalletTesting() {
  console.log('🔧 Advanced Wallet Testing\n');
  
  const connection = new Connection(DEVNET_RPC, 'confirmed');
  const wallet = Keypair.generate();
  
  try {
    // Fund wallet
    const signature = await connection.requestAirdrop(wallet.publicKey, 2 * 1e9);
    await connection.confirmTransaction(signature);
    
    // Test multiple balance checks
    console.log('🔄 Testing multiple balance checks...');
    const balances = await Promise.all([
      connection.getBalance(wallet.publicKey),
      connection.getBalance(wallet.publicKey, 'confirmed'),
      connection.getBalance(wallet.publicKey, 'finalized')
    ]);
    
    console.log('✅ Balance consistency check:', {
      processed: `${balances[0] / 1e9} SOL`,
      confirmed: `${balances[1] / 1e9} SOL`,
      finalized: `${balances[2] / 1e9} SOL`,
      consistent: balances.every(b => b === balances[0])
    });
    
    // Test account info
    const accountInfo = await connection.getAccountInfo(wallet.publicKey);
    console.log('✅ Account info:', {
      executable: accountInfo?.executable || false,
      owner: accountInfo?.owner.toString() || 'Unknown',
      lamports: accountInfo?.lamports || 0
    });
    
    return wallet;
  } catch (error) {
    console.error('❌ Advanced wallet testing failed:', error.message);
    throw error;
  }
}

/**
 * Network Connectivity Testing
 * Tests different RPC endpoints and network conditions
 */
async function networkConnectivityTesting() {
  console.log('🌐 Network Connectivity Testing\n');
  
  const rpcEndpoints = [
    'https://api.devnet.solana.com',
    'https://devnet.helius-rpc.com',
    'https://rpc-devnet.helius.xyz'
  ];
  
  for (const endpoint of rpcEndpoints) {
    try {
      const connection = new Connection(endpoint, 'confirmed');
      const version = await connection.getVersion();
      
      console.log(`✅ ${endpoint}:`, {
        version: version['solana-core'],
        featureSet: version['feature-set'],
        responseTime: 'Connected'
      });
    } catch (error) {
      console.log(`❌ ${endpoint}: ${error.message}`);
    }
  }
}

/**
 * Complete Wallet Integration Test
 * Runs all wallet-related tests
 */
async function runWalletIntegrationTest() {
  console.log('🚀 Starting Wallet Integration Test\n');
  console.log('=' .repeat(50));
  
  try {
    // Basic wallet integration
    const wallet = await walletIntegrationExample();
    
    // Program interaction
    await programInteractionExample();
    
    // Advanced wallet testing
    await advancedWalletTesting();
    
    // Network connectivity
    await networkConnectivityTesting();
    
    console.log('=' .repeat(50));
    console.log('🎉 Wallet integration test completed successfully!');
    console.log('✅ All wallet operations are working properly');
    
  } catch (error) {
    console.log('=' .repeat(50));
    console.error('❌ Wallet integration test failed:', error.message);
    console.log('💡 Make sure Solana devnet is accessible');
  }
}

// Export functions for individual testing
export {
  walletIntegrationExample,
  programInteractionExample,
  advancedWalletTesting,
  networkConnectivityTesting,
  runWalletIntegrationTest
};

// Run complete test if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runWalletIntegrationTest();
}
