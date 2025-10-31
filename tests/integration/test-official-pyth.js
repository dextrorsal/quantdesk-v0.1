#!/usr/bin/env node

/**
 * Test Official Pyth Network Client WebSocket Connection
 * Using @pythnetwork/client package
 */

const { PythConnection, getPythProgramKeyForCluster } = require('@pythnetwork/client');
const { Connection, clusterApiUrl } = require('@solana/web3.js');

console.log('🚀 Testing Official Pyth Network Client WebSocket...\n');

async function testPythWebSocket() {
  try {
    console.log('📡 Setting up Solana connection...');
    const connection = new Connection(clusterApiUrl('mainnet-beta'));
    
    console.log('🔑 Getting Pyth program key...');
    const pythProgramKey = getPythProgramKeyForCluster('mainnet-beta');
    
    console.log('🔌 Creating Pyth connection...');
    const pythConnection = new PythConnection(connection, pythProgramKey);
    
    console.log('📊 Setting up price change listener...');
    pythConnection.onPriceChange((product, price) => {
      console.log(`💰 ${product.symbol}: $${price.price} ±$${price.confidence} Status: ${price.status}`);
    });
    
    console.log('▶️ Starting Pyth connection...');
    await pythConnection.start();
    
    console.log('✅ Pyth WebSocket connection established successfully!');
    console.log('📈 Listening for price updates...\n');
    
    // Keep the connection alive for 30 seconds
    setTimeout(() => {
      console.log('\n🛑 Stopping connection after 30 seconds...');
      pythConnection.close();
      process.exit(0);
    }, 30000);
    
  } catch (error) {
    console.error('❌ Error testing Pyth WebSocket:', error.message);
    console.error('Stack:', error.stack);
    process.exit(1);
  }
}

// Handle process termination
process.on('SIGINT', () => {
  console.log('\n🛑 Received SIGINT, shutting down gracefully...');
  process.exit(0);
});

process.on('SIGTERM', () => {
  console.log('\n🛑 Received SIGTERM, shutting down gracefully...');
  process.exit(0);
});

// Start the test
testPythWebSocket();
