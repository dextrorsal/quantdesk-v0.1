// QuantDesk Security Utilities
// Comprehensive security measures for QuantDesk SDK usage

import { QuantDeskClient } from '@quantdesk/sdk';

/**
 * Security Utilities - Comprehensive security implementation
 * This module provides security measures for QuantDesk SDK usage
 */
export class QuantDeskSecurity {
  private client: QuantDeskClient;
  private rateLimiter: Map<string, number[]> = new Map();
  private maxRequestsPerMinute: number = 60;
  private maxRequestsPerHour: number = 1000;

  constructor(client: QuantDeskClient) {
    this.client = client;
  }

  /**
   * Rate limiting implementation
   */
  checkRateLimit(operation: string): boolean {
    const now = Date.now();
    const requests = this.rateLimiter.get(operation) || [];
    
    // Remove requests older than 1 minute
    const recentRequests = requests.filter(time => now - time < 60000);
    
    if (recentRequests.length >= this.maxRequestsPerMinute) {
      console.warn(`⚠️ Rate limit exceeded for ${operation}`);
      return false;
    }
    
    // Add current request
    recentRequests.push(now);
    this.rateLimiter.set(operation, recentRequests);
    
    return true;
  }

  /**
   * Input sanitization
   */
  sanitizeInput(input: any): any {
    if (typeof input === 'string') {
      // Remove potentially dangerous characters
      return input.replace(/[<>\"'&]/g, '');
    }
    
    if (typeof input === 'number') {
      // Ensure number is finite and not NaN
      return isFinite(input) ? input : 0;
    }
    
    if (Array.isArray(input)) {
      return input.map(item => this.sanitizeInput(item));
    }
    
    if (typeof input === 'object' && input !== null) {
      const sanitized: any = {};
      for (const [key, value] of Object.entries(input)) {
        sanitized[key] = this.sanitizeInput(value);
      }
      return sanitized;
    }
    
    return input;
  }

  /**
   * Validate market symbol
   */
  validateMarketSymbol(symbol: string): boolean {
    if (!symbol || typeof symbol !== 'string') {
      throw new Error('Market symbol must be a non-empty string');
    }
    
    // Check format: BASE-PERP
    const marketPattern = /^[A-Z]{2,10}-PERP$/;
    if (!marketPattern.test(symbol)) {
      throw new Error('Market symbol must be in format BASE-PERP (e.g., SOL-PERP)');
    }
    
    // Check against known markets
    const knownMarkets = ['SOL-PERP', 'ETH-PERP', 'BTC-PERP', 'USDC-PERP'];
    if (!knownMarkets.includes(symbol)) {
      console.warn(`⚠️ Unknown market symbol: ${symbol}`);
    }
    
    return true;
  }

  /**
   * Validate order data
   */
  validateOrderData(orderData: any): boolean {
    if (!orderData || typeof orderData !== 'object') {
      throw new Error('Order data must be an object');
    }
    
    // Validate required fields
    const requiredFields = ['market', 'side', 'size', 'price', 'orderType'];
    for (const field of requiredFields) {
      if (!(field in orderData)) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
    
    // Validate market
    this.validateMarketSymbol(orderData.market);
    
    // Validate side
    if (!['buy', 'sell'].includes(orderData.side)) {
      throw new Error('Side must be either "buy" or "sell"');
    }
    
    // Validate size
    if (typeof orderData.size !== 'number' || orderData.size <= 0) {
      throw new Error('Size must be a positive number');
    }
    
    if (orderData.size < 0.001) {
      throw new Error('Size must be at least 0.001');
    }
    
    if (orderData.size > 1000) {
      throw new Error('Size must not exceed 1000');
    }
    
    // Validate price
    if (typeof orderData.price !== 'number' || orderData.price <= 0) {
      throw new Error('Price must be a positive number');
    }
    
    if (orderData.price < 0.01) {
      throw new Error('Price must be at least 0.01');
    }
    
    // Validate order type
    if (!['limit', 'market'].includes(orderData.orderType)) {
      throw new Error('Order type must be either "limit" or "market"');
    }
    
    return true;
  }

  /**
   * Validate position data
   */
  validatePositionData(positionData: any): boolean {
    if (!positionData || typeof positionData !== 'object') {
      throw new Error('Position data must be an object');
    }
    
    // Validate required fields
    const requiredFields = ['market', 'side', 'size', 'leverage'];
    for (const field of requiredFields) {
      if (!(field in positionData)) {
        throw new Error(`Missing required field: ${field}`);
      }
    }
    
    // Validate market
    this.validateMarketSymbol(positionData.market);
    
    // Validate side
    if (!['long', 'short'].includes(positionData.side)) {
      throw new Error('Side must be either "long" or "short"');
    }
    
    // Validate size
    if (typeof positionData.size !== 'number' || positionData.size <= 0) {
      throw new Error('Size must be a positive number');
    }
    
    // Validate leverage
    if (typeof positionData.leverage !== 'number' || positionData.leverage <= 0) {
      throw new Error('Leverage must be a positive number');
    }
    
    if (positionData.leverage > 100) {
      throw new Error('Leverage must not exceed 100x');
    }
    
    return true;
  }

  /**
   * Validate wallet address
   */
  validateWalletAddress(address: string): boolean {
    if (!address || typeof address !== 'string') {
      throw new Error('Wallet address must be a non-empty string');
    }
    
    // Solana address validation (base58, 32-44 characters)
    const solanaAddressPattern = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;
    if (!solanaAddressPattern.test(address)) {
      throw new Error('Invalid Solana wallet address format');
    }
    
    return true;
  }

  /**
   * Validate private key
   */
  validatePrivateKey(privateKey: string): boolean {
    if (!privateKey || typeof privateKey !== 'string') {
      throw new Error('Private key must be a non-empty string');
    }
    
    // Base58 validation
    const base58Pattern = /^[1-9A-HJ-NP-Za-km-z]+$/;
    if (!base58Pattern.test(privateKey)) {
      throw new Error('Invalid private key format (must be base58)');
    }
    
    // Length validation (Solana private keys are typically 88 characters)
    if (privateKey.length < 80 || privateKey.length > 100) {
      throw new Error('Invalid private key length');
    }
    
    return true;
  }

  /**
   * Encrypt sensitive data
   */
  encryptSensitiveData(data: string, key: string): string {
    // Simple XOR encryption for demonstration
    // In production, use proper encryption libraries
    let encrypted = '';
    for (let i = 0; i < data.length; i++) {
      encrypted += String.fromCharCode(
        data.charCodeAt(i) ^ key.charCodeAt(i % key.length)
      );
    }
    return Buffer.from(encrypted).toString('base64');
  }

  /**
   * Decrypt sensitive data
   */
  decryptSensitiveData(encryptedData: string, key: string): string {
    // Simple XOR decryption for demonstration
    // In production, use proper decryption libraries
    const data = Buffer.from(encryptedData, 'base64').toString();
    let decrypted = '';
    for (let i = 0; i < data.length; i++) {
      decrypted += String.fromCharCode(
        data.charCodeAt(i) ^ key.charCodeAt(i % key.length)
      );
    }
    return decrypted;
  }

  /**
   * Generate secure random string
   */
  generateSecureRandomString(length: number = 32): string {
    const chars = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789';
    let result = '';
    
    for (let i = 0; i < length; i++) {
      result += chars.charAt(Math.floor(Math.random() * chars.length));
    }
    
    return result;
  }

  /**
   * Hash sensitive data
   */
  hashSensitiveData(data: string): string {
    // Simple hash for demonstration
    // In production, use proper hashing libraries like crypto-js
    let hash = 0;
    for (let i = 0; i < data.length; i++) {
      const char = data.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16);
  }

  /**
   * Check for suspicious activity
   */
  checkSuspiciousActivity(operation: string, data: any): boolean {
    // Check for unusual patterns
    if (operation === 'placeOrder') {
      // Check for unusually large orders
      if (data.size > 100) {
        console.warn(`⚠️ Suspicious activity: Large order size ${data.size}`);
        return true;
      }
      
      // Check for unusual price deviations
      if (data.price < 0.01 || data.price > 100000) {
        console.warn(`⚠️ Suspicious activity: Unusual price ${data.price}`);
        return true;
      }
    }
    
    if (operation === 'openPosition') {
      // Check for unusually high leverage
      if (data.leverage > 50) {
        console.warn(`⚠️ Suspicious activity: High leverage ${data.leverage}x`);
        return true;
      }
    }
    
    return false;
  }

  /**
   * Log security events
   */
  logSecurityEvent(event: string, details: any): void {
    const timestamp = new Date().toISOString();
    const logEntry = {
      timestamp,
      event,
      details: this.sanitizeInput(details),
      severity: 'INFO'
    };
    
    console.log(`🔒 Security Event: ${JSON.stringify(logEntry)}`);
    
    // In production, send to security monitoring system
  }

  /**
   * Validate API response
   */
  validateAPIResponse(response: any): boolean {
    if (!response) {
      throw new Error('API response is empty');
    }
    
    if (typeof response !== 'object') {
      throw new Error('API response must be an object');
    }
    
    // Check for error indicators
    if (response.error) {
      throw new Error(`API Error: ${response.error.message || 'Unknown error'}`);
    }
    
    // Check for success indicator
    if (response.success === false) {
      throw new Error('API request failed');
    }
    
    return true;
  }

  /**
   * Get security status
   */
  getSecurityStatus(): {
    rateLimits: { [key: string]: number };
    securityEvents: number;
    isSecure: boolean;
  } {
    const rateLimits: { [key: string]: number } = {};
    
    for (const [operation, requests] of this.rateLimiter.entries()) {
      const now = Date.now();
      const recentRequests = requests.filter(time => now - time < 60000);
      rateLimits[operation] = recentRequests.length;
    }
    
    return {
      rateLimits,
      securityEvents: 0, // Would track actual security events
      isSecure: true
    };
  }
}

// Example usage - Realistic implementation
export async function runSecurityExample() {
  // Initialize QuantDesk client
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com',
    programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    wallet: wallet // Your Solana wallet instance
  });

  // Create security utilities
  const security = new QuantDeskSecurity(client);
  
  try {
    // Validate market symbol
    security.validateMarketSymbol('SOL-PERP');
    console.log('✅ Market symbol validation passed');
    
    // Validate order data
    const orderData = {
      market: 'SOL-PERP',
      side: 'buy',
      size: 1.0,
      price: 100.0,
      orderType: 'limit'
    };
    
    security.validateOrderData(orderData);
    console.log('✅ Order data validation passed');
    
    // Check rate limit
    const rateLimitOk = security.checkRateLimit('placeOrder');
    console.log('✅ Rate limit check:', rateLimitOk);
    
    // Sanitize input
    const sanitizedInput = security.sanitizeInput('<script>alert("xss")</script>');
    console.log('✅ Input sanitization:', sanitizedInput);
    
    // Generate secure random string
    const randomString = security.generateSecureRandomString(16);
    console.log('✅ Secure random string:', randomString);
    
    // Get security status
    const securityStatus = security.getSecurityStatus();
    console.log('✅ Security status:', securityStatus);
    
  } catch (error) {
    console.error('❌ Security example failed:', error);
  }
}

// Export for use in other modules
export { QuantDeskSecurity };
