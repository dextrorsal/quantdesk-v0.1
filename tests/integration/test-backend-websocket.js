#!/usr/bin/env node

/**
 * Backend WebSocket Test Script
 * Tests our backend WebSocket implementation specifically
 */

const { io } = require('socket.io-client');

console.log('🔌 Backend WebSocket Test Script');
console.log('===============================\n');

async function testBackendWebSocket() {
  console.log('🔗 Connecting to backend WebSocket...');
  console.log('   URL: http://localhost:3002');
  
  const socket = io('http://localhost:3002', {
    transports: ['websocket', 'polling'],
    timeout: 10000,
    forceNew: true
  });

  return new Promise((resolve, reject) => {
    const timeout = setTimeout(() => {
      console.log('❌ Connection timeout after 10 seconds');
      socket.disconnect();
      reject(new Error('Connection timeout'));
    }, 10000);

    socket.on('connect', () => {
      console.log('✅ Connected to backend WebSocket');
      clearTimeout(timeout);
      
      // Test subscribing to market data
      console.log('📡 Subscribing to market data...');
      socket.emit('subscribe_market_data');
      
      // Listen for price updates
      socket.on('price_update', (data) => {
        console.log('💰 Received price update:', JSON.stringify(data, null, 2));
      });
      
      socket.on('market_data', (data) => {
        console.log('📊 Received market data:', JSON.stringify(data, null, 2));
      });
      
      // Wait for some data, then disconnect
      setTimeout(() => {
        console.log('🔌 Disconnecting...');
        socket.disconnect();
        resolve();
      }, 5000);
    });

    socket.on('connect_error', (error) => {
      console.log('❌ Connection error:', error.message);
      clearTimeout(timeout);
      reject(error);
    });

    socket.on('disconnect', (reason) => {
      console.log('🔌 Disconnected:', reason);
    });

    socket.on('error', (error) => {
      console.log('❌ Socket error:', error.message);
    });
  });
}

async function testBackendAPI() {
  console.log('\n🌐 Testing Backend API...');
  console.log('------------------------');
  
  try {
    const axios = require('axios');
    
    console.log('🔗 Testing: GET /api/prices');
    const response = await axios.get('http://localhost:3002/api/prices', {
      timeout: 10000
    });
    
    console.log(`✅ Status: ${response.status}`);
    console.log(`📊 Success: ${response.data.success}`);
    console.log(`📋 Data Length: ${response.data.data?.length || 0}`);
    
    if (response.data.data && response.data.data.length > 0) {
      console.log('🔍 Sample prices:');
      response.data.data.slice(0, 3).forEach(price => {
        console.log(`   ${price.symbol}: $${price.price}`);
      });
    }
    
  } catch (error) {
    console.log(`❌ API Error: ${error.message}`);
    if (error.code === 'ECONNREFUSED') {
      console.log('💡 Backend is not running on port 3002');
    }
  }
}

async function runTests() {
  try {
    await testBackendAPI();
    await testBackendWebSocket();
    
    console.log('\n🎯 Backend Test Summary');
    console.log('======================');
    console.log('✅ All backend tests completed');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

runTests();
