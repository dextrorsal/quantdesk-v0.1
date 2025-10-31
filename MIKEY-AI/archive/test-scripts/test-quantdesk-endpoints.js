#!/usr/bin/env node

/**
 * Test QuantDesk API Endpoints
 */

const axios = require('axios');

const QUANTDESK_URL = process.env.QUANTDESK_URL || 'http://localhost:3002';

console.log('🧪 Testing QuantDesk API Endpoints');
console.log('📡 Base URL:', QUANTDESK_URL);
console.log('');

async function testEndpoint(endpoint, description) {
  try {
    console.log(`🔍 Testing: ${description}`);
    console.log(`   URL: ${QUANTDESK_URL}${endpoint}`);
    
    const response = await axios.get(`${QUANTDESK_URL}${endpoint}`, {
      timeout: 5000
    });
    
    console.log(`   ✅ Status: ${response.status}`);
    console.log(`   📊 Response: ${JSON.stringify(response.data).substring(0, 100)}...`);
    return true;
  } catch (error) {
    console.log(`   ❌ Error: ${error.response?.status || error.message}`);
    return false;
  }
}

async function testPostEndpoint(endpoint, data, description) {
  try {
    console.log(`🔍 Testing: ${description}`);
    console.log(`   URL: ${QUANTDESK_URL}${endpoint}`);
    
    const response = await axios.post(`${QUANTDESK_URL}${endpoint}`, data, {
      timeout: 5000,
      headers: { 'Content-Type': 'application/json' }
    });
    
    console.log(`   ✅ Status: ${response.status}`);
    console.log(`   📊 Response: ${JSON.stringify(response.data).substring(0, 100)}...`);
    return true;
  } catch (error) {
    console.log(`   ❌ Error: ${error.response?.status || error.message}`);
    return false;
  }
}

async function main() {
  const endpoints = [
    { path: '/health', method: 'GET', description: 'Health Check' },
    { path: '/api/health', method: 'GET', description: 'API Health Check' },
    { path: '/api/markets', method: 'GET', description: 'Markets Endpoint' },
    { path: '/api/prices', method: 'GET', description: 'Prices Endpoint' },
    { path: '/api/whales/recent', method: 'GET', description: 'Whale Data' },
    { path: '/api/news', method: 'GET', description: 'News Data' },
    { path: '/api/sentiment', method: 'GET', description: 'Sentiment Data' }
  ];

  let successCount = 0;
  
  for (const endpoint of endpoints) {
    console.log('');
    const success = await testEndpoint(endpoint.path, endpoint.description);
    if (success) successCount++;
  }

  console.log('\n📊 Test Results:');
  console.log(`   ✅ Working: ${successCount}/${endpoints.length}`);
  console.log(`   ❌ Failed: ${endpoints.length - successCount}/${endpoints.length}`);
  
  if (successCount === 0) {
    console.log('\n🔧 QuantDesk API Server is not running!');
    console.log('   Start it with: cd ../backend && npm start');
    console.log('   Or run: node start-quantdesk-api.js');
  } else if (successCount < endpoints.length) {
    console.log('\n⚠️  Some endpoints are not working');
    console.log('   Check server logs for errors');
  } else {
    console.log('\n🎉 All endpoints are working!');
    console.log('   Mikey AI should now be able to fetch real data');
  }
}

main().catch(console.error);