#!/usr/bin/env node

/**
 * Pyth Network Connection Debug Script
 * Tests all possible ways to connect to Pyth Network and identifies issues
 */

const axios = require('axios');
const WebSocket = require('ws');

console.log('🔍 Pyth Network Connection Debug Script');
console.log('=====================================\n');

// Test configurations
const PYTH_ENDPOINTS = {
  hermes_rest: 'https://hermes.pyth.network',
  hermes_ws: 'wss://hermes.pyth.network/ws',
  hermes_v2: 'https://hermes.pyth.network/v2',
  pyth_api: 'https://api.pyth.network',
  pyth_ws: 'wss://api.pyth.network'
};

const TEST_FEED_IDS = {
  BTC: '0xe62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43',
  ETH: '0xff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace',
  SOL: '0xef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d'
};

async function testRestEndpoints() {
  console.log('📡 Testing REST API Endpoints...');
  console.log('--------------------------------');
  
  const endpoints = [
    { name: 'Hermes v2 Latest Prices', url: `${PYTH_ENDPOINTS.hermes_v2}/latest_price_feeds` },
    { name: 'Hermes v2 Price Feeds', url: `${PYTH_ENDPOINTS.hermes_v2}/price_feeds` },
    { name: 'Hermes v2 Updates', url: `${PYTH_ENDPOINTS.hermes_v2}/updates/price/latest` },
    { name: 'Hermes Root', url: `${PYTH_ENDPOINTS.hermes_rest}` },
    { name: 'Pyth API', url: `${PYTH_ENDPOINTS.pyth_api}` }
  ];

  for (const endpoint of endpoints) {
    try {
      console.log(`\n🔗 Testing: ${endpoint.name}`);
      console.log(`   URL: ${endpoint.url}`);
      
      const response = await axios.get(endpoint.url, {
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-Debug/1.0'
        }
      });
      
      console.log(`   ✅ Status: ${response.status}`);
      console.log(`   📊 Data Type: ${typeof response.data}`);
      console.log(`   📏 Data Size: ${JSON.stringify(response.data).length} chars`);
      
      if (Array.isArray(response.data)) {
        console.log(`   📋 Array Length: ${response.data.length}`);
        if (response.data.length > 0) {
          console.log(`   🔍 First Item Keys: ${Object.keys(response.data[0]).join(', ')}`);
        }
      } else if (typeof response.data === 'object') {
        console.log(`   🔍 Object Keys: ${Object.keys(response.data).join(', ')}`);
      }
      
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
      if (error.response) {
        console.log(`   📊 Status: ${error.response.status}`);
        console.log(`   📝 Response: ${error.response.data}`);
      }
    }
  }
}

async function testFeedIdFormats() {
  console.log('\n\n🔑 Testing Feed ID Formats...');
  console.log('-----------------------------');
  
  const formats = [
    { name: 'Array Format', params: { ids: Object.values(TEST_FEED_IDS) } },
    { name: 'Comma Separated', params: { ids: Object.values(TEST_FEED_IDS).join(',') } },
    { name: 'Single BTC', params: { ids: TEST_FEED_IDS.BTC } },
    { name: 'Multiple IDs Param', params: Object.values(TEST_FEED_IDS).map(id => ({ ids: id })) }
  ];

  for (const format of formats) {
    try {
      console.log(`\n🔗 Testing: ${format.name}`);
      console.log(`   Params: ${JSON.stringify(format.params)}`);
      
      const response = await axios.get(`${PYTH_ENDPOINTS.hermes_v2}/latest_price_feeds`, {
        params: format.params,
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-Debug/1.0'
        }
      });
      
      console.log(`   ✅ Status: ${response.status}`);
      console.log(`   📊 Response Type: ${typeof response.data}`);
      
      if (Array.isArray(response.data)) {
        console.log(`   📋 Array Length: ${response.data.length}`);
        if (response.data.length > 0) {
          const firstItem = response.data[0];
          console.log(`   🔍 First Item: ${JSON.stringify(firstItem, null, 2)}`);
        }
      }
      
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
      if (error.response) {
        console.log(`   📊 Status: ${error.response.status}`);
        console.log(`   📝 Response: ${error.response.data}`);
      }
    }
  }
}

async function testWebSocketConnections() {
  console.log('\n\n🔌 Testing WebSocket Connections...');
  console.log('-----------------------------------');
  
  const wsEndpoints = [
    { name: 'Hermes WebSocket', url: PYTH_ENDPOINTS.hermes_ws },
    { name: 'Pyth WebSocket', url: PYTH_ENDPOINTS.pyth_ws }
  ];

  for (const endpoint of wsEndpoints) {
    try {
      console.log(`\n🔗 Testing: ${endpoint.name}`);
      console.log(`   URL: ${endpoint.url}`);
      
      const ws = new WebSocket(endpoint.url);
      
      await new Promise((resolve, reject) => {
        const timeout = setTimeout(() => {
          ws.close();
          reject(new Error('Connection timeout'));
        }, 10000);
        
        ws.on('open', () => {
          console.log(`   ✅ Connected successfully`);
          clearTimeout(timeout);
          ws.close();
          resolve();
        });
        
        ws.on('error', (error) => {
          console.log(`   ❌ Connection error: ${error.message}`);
          clearTimeout(timeout);
          reject(error);
        });
        
        ws.on('message', (data) => {
          console.log(`   📨 Received message: ${data.toString().substring(0, 100)}...`);
        });
      });
      
    } catch (error) {
      console.log(`   ❌ Error: ${error.message}`);
    }
  }
}

async function testOurBackend() {
  console.log('\n\n🏠 Testing Our Backend...');
  console.log('------------------------');
  
  try {
    console.log('\n🔗 Testing: Backend API Endpoint');
    console.log('   URL: http://localhost:3002/api/prices');
    
    const response = await axios.get('http://localhost:3002/api/prices', {
      timeout: 10000
    });
    
    console.log(`   ✅ Status: ${response.status}`);
    console.log(`   📊 Success: ${response.data.success}`);
    console.log(`   📋 Data Length: ${response.data.data?.length || 0}`);
    
    if (response.data.data && response.data.data.length > 0) {
      console.log(`   🔍 First Price: ${JSON.stringify(response.data.data[0], null, 2)}`);
    }
    
  } catch (error) {
    console.log(`   ❌ Error: ${error.message}`);
    if (error.code === 'ECONNREFUSED') {
      console.log(`   💡 Backend is not running on port 3002`);
    }
  }
}

async function testHermesClient() {
  console.log('\n\n📦 Testing Hermes Client...');
  console.log('---------------------------');
  
  try {
    // Try to import and use Hermes client
    const { HermesClient } = require('@pythnetwork/hermes-client');
    
    console.log('   ✅ Hermes client imported successfully');
    
    const client = new HermesClient('https://hermes.pyth.network', {});
    console.log('   ✅ Hermes client created');
    
    // Try to get price feeds
    const feeds = await client.getPriceFeeds({
      query: 'crypto',
      assetType: 'crypto'
    });
    
    console.log(`   ✅ Got ${feeds?.length || 0} price feeds`);
    
    if (feeds && feeds.length > 0) {
      console.log(`   🔍 First feed: ${JSON.stringify(feeds[0], null, 2)}`);
    }
    
  } catch (error) {
    console.log(`   ❌ Hermes client error: ${error.message}`);
    console.log(`   📝 Stack: ${error.stack}`);
  }
}

async function runAllTests() {
  try {
    await testRestEndpoints();
    await testFeedIdFormats();
    await testWebSocketConnections();
    await testOurBackend();
    await testHermesClient();
    
    console.log('\n\n🎯 Debug Summary');
    console.log('================');
    console.log('✅ All tests completed');
    console.log('📋 Check the output above to identify connection issues');
    console.log('💡 Common issues:');
    console.log('   - Wrong API endpoint format');
    console.log('   - Incorrect feed ID format');
    console.log('   - Network/firewall blocking connections');
    console.log('   - Backend not running');
    console.log('   - Hermes client version incompatibility');
    
  } catch (error) {
    console.error('❌ Test suite failed:', error.message);
  }
}

// Run the debug script
runAllTests();
