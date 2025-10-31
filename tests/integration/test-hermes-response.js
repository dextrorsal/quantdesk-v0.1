#!/usr/bin/env node

/**
 * Test Pyth Hermes Client - Check response format
 * Using @pythnetwork/hermes-client package
 */

const { HermesClient } = require('@pythnetwork/hermes-client');

console.log('🚀 Testing Pyth Hermes Client - Check response format...\n');

async function testHermesResponse() {
  try {
    console.log('🔌 Creating Hermes client with base URL...');
    const hermesClient = new HermesClient('https://hermes.pyth.network');
    
    console.log('📊 Getting latest price updates...');
    
    // Test the REST API with correct parameter format
    const ids = [
      'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43', // BTC/USD
      'ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace', // ETH/USD
      'ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d'  // SOL/USD
    ];
    
    const response = await hermesClient.getLatestPriceUpdates(ids);
    
    console.log('✅ REST API call successful!');
    console.log('📈 Response type:', typeof response);
    console.log('📈 Response:', JSON.stringify(response, null, 2));
    
    console.log('\n🎉 Hermes REST API is working!');
    
  } catch (error) {
    console.error('❌ Error testing Hermes REST API:', error.message);
    console.error('Stack:', error.stack);
    process.exit(1);
  }
}

// Start the test
testHermesResponse();
