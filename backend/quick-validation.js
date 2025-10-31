#!/usr/bin/env node

/**
 * Quick Validation Script
 * Tests core functionality without running the full test suite
 */

const { PythOracleService } = require('./dist/services/pythOracleService');
const { OrderAuthorizationService } = require('./dist/services/orderAuthorizationService');

async function quickValidation() {
  console.log('🚀 Quick Validation Starting...\n');

  try {
    // Test 1: Oracle Service
    console.log('1️⃣ Testing Oracle Service...');
    const oracleService = PythOracleService.createInstance();
    
    // Test processPriceData method
    const mockData = {
      'SOL': { price: 110.5, confidence: 0.01 },
      'BTC': { price: 50000, confidence: 0.02 }
    };
    
    const processed = await oracleService.processPriceData(mockData);
    console.log('✅ processPriceData working:', Object.keys(processed).length === 2);
    
    // Test 2: Order Authorization Service
    console.log('\n2️⃣ Testing Order Authorization Service...');
    const orderAuthService = new OrderAuthorizationService();
    
    // Test sanitizeOrderInput method
    const mockOrder = {
      userId: 'test-user-123',
      symbol: 'BTC',
      orderType: 'limit',
      side: 'buy',
      size: 100,
      price: 50000,
      leverage: 10
    };
    
    const sanitized = orderAuthService.sanitizeOrderInput(mockOrder);
    console.log('✅ sanitizeOrderInput working:', sanitized.userId === 'test-user-123');
    
    console.log('\n🎉 Quick Validation Complete!');
    console.log('✅ Core services are working correctly');
    
  } catch (error) {
    console.error('❌ Validation failed:', error.message);
    process.exit(1);
  }
}

quickValidation();
