#!/usr/bin/env node

/**
 * Test Pyth Hermes Client - Price Updates Stream
 * Using @pythnetwork/hermes-client package
 */

const { HermesClient } = require('@pythnetwork/hermes-client');

console.log('🚀 Testing Pyth Hermes Client - Price Updates Stream...\n');

async function testHermesStream() {
  try {
    console.log('🔌 Creating Hermes client...');
    const hermesClient = new HermesClient();
    
    console.log('📊 Getting price updates stream...');
    
    // Get the stream
    const stream = await hermesClient.getPriceUpdatesStream({
      ids: [
        'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43', // BTC/USD
        'ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace', // ETH/USD
        'ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d'  // SOL/USD
      ]
    });
    
    console.log('✅ Stream created successfully!');
    console.log('📈 Listening for price updates...\n');
    
    // Listen for price updates
    stream.on('data', (data) => {
      console.log('💰 Price Update Received:', {
        timestamp: new Date().toISOString(),
        dataLength: data.length,
        firstUpdate: data[0] ? {
          id: data[0].id,
          price: data[0].price?.price,
          confidence: data[0].price?.conf,
          exponent: data[0].price?.expo
        } : 'No data'
      });
    });
    
    stream.on('error', (error) => {
      console.error('❌ Stream error:', error.message);
    });
    
    stream.on('end', () => {
      console.log('🔚 Stream ended');
    });
    
    // Keep the connection alive for 30 seconds
    setTimeout(() => {
      console.log('\n🛑 Stopping stream after 30 seconds...');
      stream.destroy();
      process.exit(0);
    }, 30000);
    
  } catch (error) {
    console.error('❌ Error testing Hermes stream:', error.message);
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
testHermesStream();
