#!/usr/bin/env node

/**
 * Test Pyth Hermes Client (EventSource/SSE)
 * Using @pythnetwork/hermes-client package
 */

const { HermesClient } = require('@pythnetwork/hermes-client');

console.log('🚀 Testing Pyth Hermes Client (EventSource)...\n');

async function testHermesClient() {
  try {
    console.log('🔌 Creating Hermes client...');
    const hermesClient = new HermesClient();
    
    console.log('📊 Setting up price update listener...');
    hermesClient.onPriceUpdate((priceUpdate) => {
      console.log(`💰 Price Update:`, {
        id: priceUpdate.id,
        price: priceUpdate.price?.price,
        confidence: priceUpdate.price?.conf,
        exponent: priceUpdate.price?.expo,
        publishTime: priceUpdate.price?.publish_time
      });
    });
    
    console.log('▶️ Starting Hermes connection...');
    await hermesClient.start();
    
    console.log('✅ Hermes connection established successfully!');
    console.log('📈 Listening for price updates...\n');
    
    // Keep the connection alive for 30 seconds
    setTimeout(() => {
      console.log('\n🛑 Stopping connection after 30 seconds...');
      hermesClient.close();
      process.exit(0);
    }, 30000);
    
  } catch (error) {
    console.error('❌ Error testing Hermes client:', error.message);
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
testHermesClient();
