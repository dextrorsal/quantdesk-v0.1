#!/usr/bin/env node

/**
 * Frontend Price System Test Script
 * Tests our frontend price system implementation
 */

const axios = require('axios');

console.log('⚛️ Frontend Price System Test Script');
console.log('===================================\n');

async function testFrontendAPI() {
  console.log('🌐 Testing Frontend API Proxy...');
  console.log('--------------------------------');
  
  try {
    console.log('🔗 Testing: GET http://localhost:3001/api/prices');
    const response = await axios.get('http://localhost:3001/api/prices', {
      timeout: 10000
    });
    
    console.log(`✅ Status: ${response.status}`);
    console.log(`📊 Response Type: ${typeof response.data}`);
    
    if (response.data.success) {
      console.log(`📋 Data Length: ${response.data.data?.length || 0}`);
      if (response.data.data && response.data.data.length > 0) {
        console.log('🔍 Sample prices:');
        response.data.data.slice(0, 3).forEach(price => {
          console.log(`   ${price.symbol}: $${price.price}`);
        });
      }
    } else {
      console.log(`❌ API returned success: false`);
      console.log(`📝 Error: ${response.data.error}`);
    }
    
  } catch (error) {
    console.log(`❌ Frontend API Error: ${error.message}`);
    if (error.code === 'ECONNREFUSED') {
      console.log('💡 Frontend is not running on port 3001');
    }
  }
}

async function testFrontendPage() {
  console.log('\n📄 Testing Frontend Page...');
  console.log('---------------------------');
  
  try {
    console.log('🔗 Testing: GET http://localhost:3001');
    const response = await axios.get('http://localhost:3001', {
      timeout: 10000
    });
    
    console.log(`✅ Status: ${response.status}`);
    console.log(`📊 Content Type: ${response.headers['content-type']}`);
    console.log(`📏 Content Length: ${response.data.length} chars`);
    
    // Check if it's HTML
    if (response.data.includes('<!doctype html')) {
      console.log('✅ Frontend is serving HTML');
      
      // Check for React app
      if (response.data.includes('react')) {
        console.log('✅ React app detected');
      }
      
      // Check for our price components
      if (response.data.includes('PriceDisplay') || response.data.includes('price-display')) {
        console.log('✅ Price components detected');
      }
      
    } else {
      console.log('❌ Frontend is not serving HTML');
      console.log(`📝 Response: ${response.data.substring(0, 200)}...`);
    }
    
  } catch (error) {
    console.log(`❌ Frontend Page Error: ${error.message}`);
  }
}

async function testWebSocketConnection() {
  console.log('\n🔌 Testing Frontend WebSocket Connection...');
  console.log('-------------------------------------------');
  
  try {
    const { io } = require('socket.io-client');
    
    console.log('🔗 Connecting to frontend WebSocket...');
    const socket = io('http://localhost:3001', {
      transports: ['websocket', 'polling'],
      timeout: 10000
    });

    return new Promise((resolve, reject) => {
      const timeout = setTimeout(() => {
        console.log('❌ WebSocket connection timeout');
        socket.disconnect();
        reject(new Error('WebSocket timeout'));
      }, 10000);

      socket.on('connect', () => {
        console.log('✅ Connected to frontend WebSocket');
        clearTimeout(timeout);
        
        // Test if frontend forwards to backend
        socket.emit('subscribe_market_data');
        
        setTimeout(() => {
          socket.disconnect();
          resolve();
        }, 3000);
      });

      socket.on('connect_error', (error) => {
        console.log('❌ Frontend WebSocket error:', error.message);
        clearTimeout(timeout);
        reject(error);
      });
    });
    
  } catch (error) {
    console.log(`❌ WebSocket test error: ${error.message}`);
  }
}

async function runTests() {
  try {
    await testFrontendAPI();
    await testFrontendPage();
    await testWebSocketConnection();
    
    console.log('\n🎯 Frontend Test Summary');
    console.log('========================');
    console.log('✅ All frontend tests completed');
    console.log('💡 If tests pass, the frontend should be working');
    console.log('💡 If tests fail, check:');
    console.log('   - Frontend is running on port 3001');
    console.log('   - Vite proxy is configured correctly');
    console.log('   - Backend is running on port 3002');
    
  } catch (error) {
    console.error('❌ Frontend test failed:', error.message);
  }
}

runTests();
