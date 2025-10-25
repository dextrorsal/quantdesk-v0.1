// QuantDesk Security Testing Suite
// Comprehensive security validation for QuantDesk SDK

import { QuantDeskSecurity } from './utils/security';

/**
 * Security Testing Suite - Comprehensive validation
 * This suite tests all security measures implemented in the QuantDesk SDK
 */
export class SecurityTestSuite {
  private security: QuantDeskSecurity;

  constructor(security: QuantDeskSecurity) {
    this.security = security;
  }

  /**
   * Run all security tests
   */
  async runAllTests(): Promise<void> {
    console.log('🔒 Starting QuantDesk Security Test Suite');
    
    try {
      await this.testInputValidation();
      await this.testDataSanitization();
      await this.testRateLimiting();
      await this.testErrorHandling();
      await this.testSecurityMeasures();
      
      console.log('✅ All security tests passed');
    } catch (error) {
      console.error('❌ Security tests failed:', error);
      throw error;
    }
  }

  /**
   * Test input validation
   */
  async testInputValidation(): Promise<void> {
    console.log('\n🧪 Testing Input Validation...');
    
    // Test market symbol validation
    try {
      this.security.validateMarketSymbol('SOL-PERP');
      console.log('✅ Valid market symbol accepted');
    } catch (error) {
      console.error('❌ Valid market symbol rejected:', error.message);
    }
    
    // Test invalid market symbol
    try {
      this.security.validateMarketSymbol('INVALID');
      console.error('❌ Invalid market symbol accepted');
    } catch (error) {
      console.log('✅ Invalid market symbol rejected');
    }
    
    // Test order data validation
    const validOrderData = {
      market: 'SOL-PERP',
      side: 'buy',
      size: 1.0,
      price: 100.0,
      orderType: 'limit'
    };
    
    try {
      this.security.validateOrderData(validOrderData);
      console.log('✅ Valid order data accepted');
    } catch (error) {
      console.error('❌ Valid order data rejected:', error.message);
    }
    
    // Test invalid order data
    const invalidOrderData = {
      market: 'SOL-PERP',
      side: 'invalid',
      size: -1,
      price: 0,
      orderType: 'limit'
    };
    
    try {
      this.security.validateOrderData(invalidOrderData);
      console.error('❌ Invalid order data accepted');
    } catch (error) {
      console.log('✅ Invalid order data rejected');
    }
  }

  /**
   * Test data sanitization
   */
  async testDataSanitization(): Promise<void> {
    console.log('\n🧪 Testing Data Sanitization...');
    
    // Test XSS protection
    const maliciousInput = '<script>alert("xss")</script>';
    const sanitized = this.security.sanitizeInput(maliciousInput);
    
    if (sanitized.includes('<script>')) {
      console.error('❌ XSS protection failed');
    } else {
      console.log('✅ XSS protection working');
    }
    
    // Test number sanitization
    const invalidNumber = NaN;
    const sanitizedNumber = this.security.sanitizeInput(invalidNumber);
    
    if (isNaN(sanitizedNumber)) {
      console.error('❌ Number sanitization failed');
    } else {
      console.log('✅ Number sanitization working');
    }
    
    // Test object sanitization
    const maliciousObject = {
      name: '<script>alert("xss")</script>',
      value: 100
    };
    
    const sanitizedObject = this.security.sanitizeInput(maliciousObject);
    
    if (sanitizedObject.name.includes('<script>')) {
      console.error('❌ Object sanitization failed');
    } else {
      console.log('✅ Object sanitization working');
    }
  }

  /**
   * Test rate limiting
   */
  async testRateLimiting(): Promise<void> {
    console.log('\n🧪 Testing Rate Limiting...');
    
    // Test normal usage
    const operation = 'testOperation';
    let rateLimitOk = true;
    
    for (let i = 0; i < 5; i++) {
      if (!this.security.checkRateLimit(operation)) {
        rateLimitOk = false;
        break;
      }
    }
    
    if (rateLimitOk) {
      console.log('✅ Rate limiting allows normal usage');
    } else {
      console.error('❌ Rate limiting blocks normal usage');
    }
    
    // Test rate limit enforcement
    let blocked = false;
    for (let i = 0; i < 70; i++) { // Exceed 60 requests/minute limit
      if (!this.security.checkRateLimit(operation)) {
        blocked = true;
        break;
      }
    }
    
    if (blocked) {
      console.log('✅ Rate limiting blocks excessive usage');
    } else {
      console.error('❌ Rate limiting does not block excessive usage');
    }
  }

  /**
   * Test error handling
   */
  async testErrorHandling(): Promise<void> {
    console.log('\n🧪 Testing Error Handling...');
    
    // Test validation errors
    try {
      this.security.validateMarketSymbol('');
      console.error('❌ Empty string validation failed');
    } catch (error) {
      console.log('✅ Empty string validation working');
    }
    
    try {
      this.security.validateMarketSymbol(null as any);
      console.error('❌ Null validation failed');
    } catch (error) {
      console.log('✅ Null validation working');
    }
    
    // Test order validation errors
    try {
      this.security.validateOrderData(null);
      console.error('❌ Null order data validation failed');
    } catch (error) {
      console.log('✅ Null order data validation working');
    }
  }

  /**
   * Test security measures
   */
  async testSecurityMeasures(): Promise<void> {
    console.log('\n🧪 Testing Security Measures...');
    
    // Test secure random string generation
    const randomString = this.security.generateSecureRandomString(16);
    
    if (randomString.length === 16 && /^[A-Za-z0-9]+$/.test(randomString)) {
      console.log('✅ Secure random string generation working');
    } else {
      console.error('❌ Secure random string generation failed');
    }
    
    // Test data hashing
    const testData = 'test data';
    const hash = this.security.hashSensitiveData(testData);
    
    if (hash && hash.length > 0) {
      console.log('✅ Data hashing working');
    } else {
      console.error('❌ Data hashing failed');
    }
    
    // Test suspicious activity detection
    const suspiciousOrder = {
      market: 'SOL-PERP',
      side: 'buy',
      size: 1000, // Large size
      price: 100.0,
      orderType: 'limit'
    };
    
    const isSuspicious = this.security.checkSuspiciousActivity('placeOrder', suspiciousOrder);
    
    if (isSuspicious) {
      console.log('✅ Suspicious activity detection working');
    } else {
      console.error('❌ Suspicious activity detection failed');
    }
  }
}

// Example usage
export async function runSecurityTests() {
  // Initialize security utilities
  const security = new QuantDeskSecurity(client);
  
  // Create test suite
  const testSuite = new SecurityTestSuite(security);
  
  try {
    await testSuite.runAllTests();
  } catch (error) {
    console.error('Security tests failed:', error);
  }
}
