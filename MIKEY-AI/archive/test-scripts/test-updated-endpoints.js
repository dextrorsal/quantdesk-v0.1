#!/usr/bin/env node

/**
 * Test Updated QuantDesk API Endpoints
 * Based on Postman collection findings
 */

const axios = require('axios');

const BACKEND_URL = 'http://localhost:3002';
const AI_URL = 'http://localhost:3003';

console.log('🧪 Testing Updated QuantDesk API Endpoints');
console.log('📡 Backend URL:', BACKEND_URL);
console.log('🤖 AI URL:', AI_URL);
console.log('');

async function testEndpoint(url, description) {
  try {
    console.log(`🔍 Testing: ${description}`);
    console.log(`   URL: ${url}`);
    
    const response = await axios.get(url, { timeout: 5000 });
    
    console.log(`   ✅ Status: ${response.status}`);
    console.log(`   📊 Response: ${JSON.stringify(response.data).substring(0, 150)}...`);
    return true;
  } catch (error) {
    console.log(`   ❌ Error: ${error.response?.status || error.message}`);
    return false;
  }
}

async function main() {
  const endpoints = [
    // Backend endpoints (port 3002)
    { url: `${BACKEND_URL}/health`, description: 'Backend Health' },
    { url: `${BACKEND_URL}/api/oracle/prices`, description: 'Pyth Oracle Prices' },
    { url: `${BACKEND_URL}/api/real-supabase-markets`, description: 'Supabase Markets' },
    
    // AI endpoints (port 3003) - these might not exist yet
    { url: `${AI_URL}/api/v1/trading/whales?threshold=100000&timeframe=24h`, description: 'Whale Data' },
    { url: `${AI_URL}/api/v1/market/sentiment?symbol=SOL`, description: 'Market Sentiment' },
    { url: `${AI_URL}/api/v1/market/prices?symbols=BTC,ETH,SOL`, description: 'Market Prices' }
  ];

  let successCount = 0;
  
  for (const endpoint of endpoints) {
    console.log('');
    const success = await testEndpoint(endpoint.url, endpoint.description);
    if (success) successCount++;
  }

  console.log('\n📊 Test Results:');
  console.log(`   ✅ Working: ${successCount}/${endpoints.length}`);
  console.log(`   ❌ Failed: ${endpoints.length - successCount}/${endpoints.length}`);
  
  if (successCount === 0) {
    console.log('\n🔧 No endpoints are working!');
    console.log('   Make sure both servers are running:');
    console.log('   - Backend: cd ../backend && npm start');
    console.log('   - AI: cd MIKEY-AI && PORT=3003 npm start');
  } else if (successCount < endpoints.length) {
    console.log('\n⚠️  Some endpoints are not working');
    console.log('   Backend endpoints should work if backend is running');
    console.log('   AI endpoints might not be implemented yet');
  } else {
    console.log('\n🎉 All endpoints are working!');
    console.log('   Mikey AI should now be able to fetch real data');
  }
  
  console.log('\n🔧 Next Steps:');
  console.log('   1. Test Mikey AI with: node test-tool-integration.js');
  console.log('   2. Check server logs for debug output');
  console.log('   3. Verify tools are calling correct endpoints');
}

main().catch(console.error);
