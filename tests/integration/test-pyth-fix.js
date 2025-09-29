#!/usr/bin/env node

/**
 * Test script to verify our Pyth Network fix
 */

const axios = require('axios');

console.log('🧪 Testing Pyth Network Fix');
console.log('==========================\n');

async function testPythAPI() {
  console.log('🔗 Testing Hermes API with correct format...');
  
  const feedIds = [
    'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43', // BTC
    'ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace', // ETH
    'ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d'  // SOL
  ];
  
  // Build query string with array format
  const queryParams = feedIds.map(id => `ids[]=${id}`).join('&');
  const url = `https://hermes.pyth.network/v2/updates/price/latest?${queryParams}`;
  
  console.log(`📡 URL: ${url}`);
  
  try {
    const response = await axios.get(url, {
      timeout: 10000,
      headers: {
        'Accept': 'application/json',
        'User-Agent': 'QuantDesk-Test/1.0'
      }
    });
    
    console.log(`✅ Status: ${response.status}`);
    console.log(`📊 Response Type: ${typeof response.data}`);
    
    if (response.data && response.data.parsed && Array.isArray(response.data.parsed)) {
      console.log(`📋 Parsed Array Length: ${response.data.parsed.length}`);
      
      response.data.parsed.forEach((priceFeed, index) => {
        if (priceFeed && priceFeed.price) {
          const price = parseFloat(priceFeed.price.price);
          const exponent = parseInt(priceFeed.price.expo || '0');
          const actualPrice = price * Math.pow(10, exponent);
          
          console.log(`💰 ${index + 1}. ${priceFeed.id}: $${actualPrice.toFixed(2)}`);
        }
      });
      
      console.log('\n🎉 SUCCESS! Pyth Network API is working correctly!');
      console.log('✅ Our backend should now be able to fetch real prices');
      
    } else {
      console.log('❌ No parsed data found in response');
      console.log('📝 Response structure:', Object.keys(response.data));
    }
    
  } catch (error) {
    console.log(`❌ Error: ${error.message}`);
    if (error.response) {
      console.log(`📊 Status: ${error.response.status}`);
      console.log(`📝 Response: ${error.response.data}`);
    }
  }
}

async function testBackendAPI() {
  console.log('\n🏠 Testing our backend API...');
  
  try {
    const response = await axios.get('http://localhost:3002/api/prices', {
      timeout: 10000
    });
    
    console.log(`✅ Backend Status: ${response.status}`);
    console.log(`📊 Success: ${response.data.success}`);
    console.log(`📋 Data Length: ${response.data.data?.length || 0}`);
    
    if (response.data.data && response.data.data.length > 0) {
      console.log('🔍 Sample prices:');
      response.data.data.slice(0, 3).forEach(price => {
        console.log(`   ${price.symbol}: $${price.price}`);
      });
    }
    
  } catch (error) {
    console.log(`❌ Backend Error: ${error.message}`);
    if (error.code === 'ECONNREFUSED') {
      console.log('💡 Backend is not running on port 3002');
      console.log('💡 Start it with: cd backend && npm run dev');
    }
  }
}

async function runTests() {
  await testPythAPI();
  await testBackendAPI();
  
  console.log('\n🎯 Test Summary');
  console.log('===============');
  console.log('✅ Pyth Network API test completed');
  console.log('✅ Backend API test completed');
  console.log('\n💡 Next steps:');
  console.log('   1. Start backend: cd backend && npm run dev');
  console.log('   2. Start frontend: cd frontend && npm run dev');
  console.log('   3. Check browser for live prices!');
}

runTests();
