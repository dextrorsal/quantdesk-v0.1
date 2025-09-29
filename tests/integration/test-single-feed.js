#!/usr/bin/env node

/**
 * Test Hermes Client with Single Feed ID
 */

const { HermesClient } = require('@pythnetwork/hermes-client');

console.log('🚀 Testing Hermes Client with Single Feed ID...\n');

async function testSingleFeed() {
  try {
    console.log('🔌 Creating Hermes client...');
    const hermesClient = new HermesClient('https://hermes.pyth.network');
    
    // Test with just BTC first
    const btcFeedId = 'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43';
    
    console.log('📊 Fetching BTC price update...');
    const response = await hermesClient.getLatestPriceUpdates([btcFeedId]);
    
    console.log('✅ REST API call successful!');
    console.log('📈 Response:', JSON.stringify(response, null, 2));
    
  } catch (error) {
    console.error('❌ Error testing single feed:', error.message);
    console.error('Stack:', error.stack);
    process.exit(1);
  }
}

// Start the test
testSingleFeed();
