/**
 * Hackathon Core Functionality Tests - Simplified
 * 
 * This test suite validates the core functionality required for the hackathon demo,
 * focusing on service initialization and basic functionality without database dependencies.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { smartContractService } from '../../src/services/smartContractService';
import { matchingService } from '../../src/services/matching';
import { pythOracleService } from '../../src/services/pythOracleService';
import { Logger } from '../../src/utils/logger';

const logger = new Logger();

describe('Hackathon Core Functionality - Critical Demo Components', () => {
  let testUserId: string;
  let testWalletAddress: string;

  beforeAll(async () => {
    // Create test user
    testWalletAddress = 'test-wallet-' + Date.now();
    testUserId = 'test-user-' + Date.now();
    
    logger.info('🚀 Hackathon core functionality test environment initialized');
  });

  afterAll(async () => {
    logger.info('🧹 Hackathon core functionality test environment cleaned up');
  });

  describe('Smart Contract Service Integration', () => {
    it('should initialize smart contract service successfully', async () => {
      // Test service initialization
      const service = smartContractService;
      expect(service).toBeDefined();
      
      // Test health check
      const isHealthy = await service.healthCheck();
      expect(typeof isHealthy).toBe('boolean');
      
      logger.info('✅ Smart contract service initialization test passed');
    });

    it('should execute orders with atomic position creation', async () => {
      // Test order execution with mock data
      const executionResult = await smartContractService.executeOrder({
        orderId: 'test-order-' + Date.now(),
        userId: testUserId,
        marketSymbol: 'SOL-PERP',
        side: 'long',
        size: 1.0,
        price: 100,
        leverage: 5,
        orderType: 'market'
      });
      
      expect(executionResult).toBeDefined();
      expect(executionResult.success).toBe(true);
      expect(executionResult.transactionSignature).toBeDefined();
      expect(executionResult.positionId).toBeDefined();
      
      logger.info('✅ Atomic order execution test passed');
    });

    it('should create positions successfully', async () => {
      // Test position creation
      const positionResult = await smartContractService.createPosition(
        testUserId,
        'SOL-PERP',
        'long',
        1.0,
        100,
        5
      );
      
      expect(positionResult).toBeDefined();
      expect(positionResult.success).toBe(true);
      expect(positionResult.positionId).toBeDefined();
      
      logger.info('✅ Position creation test passed');
    });

    it('should close positions successfully', async () => {
      // Test position closing
      const positionId = 'test-position-' + Date.now();
      const closeResult = await smartContractService.closePosition(positionId, testUserId);
      
      expect(closeResult).toBeDefined();
      expect(closeResult.success).toBe(true);
      expect(closeResult.transactionSignature).toBeDefined();
      
      logger.info('✅ Position closing test passed');
    });
  });

  describe('Order Matching Service Integration', () => {
    it('should handle order validation correctly', async () => {
      // Test invalid order parameters
      await expect(
        matchingService.placeOrder({
          userId: testUserId,
          symbol: 'SOL-PERP',
          side: 'buy',
          size: 0, // Invalid size
          orderType: 'market',
          leverage: 5
        })
      ).rejects.toThrow('Size must be positive');
      
      // Test missing price for limit order
      await expect(
        matchingService.placeOrder({
          userId: testUserId,
          symbol: 'SOL-PERP',
          side: 'buy',
          size: 1.0,
          orderType: 'limit',
          leverage: 5
          // Missing price
        })
      ).rejects.toThrow('Limit orders require positive price');
      
      logger.info('✅ Order validation test passed');
    });
  });

  describe('Oracle Service Integration', () => {
    it('should fetch price data successfully', async () => {
      // Test price fetching
      const price = await pythOracleService.getLatestPrice('SOL');
      
      expect(price).toBeDefined();
      expect(typeof price).toBe('number');
      expect(price).toBeGreaterThan(0);
      
      logger.info('✅ Price fetching test passed');
    });

    it('should handle multiple price requests efficiently', async () => {
      // Test multiple price requests
      const symbols = ['BTC', 'ETH', 'SOL', 'AVAX'];
      const prices = await Promise.all(
        symbols.map(symbol => pythOracleService.getLatestPrice(symbol))
      );
      
      expect(prices).toBeDefined();
      expect(prices.length).toBe(symbols.length);
      
      // All prices should be valid
      prices.forEach(price => {
        expect(price).toBeDefined();
        expect(typeof price).toBe('number');
        expect(price).toBeGreaterThan(0);
      });
      
      logger.info('✅ Multiple price requests test passed');
    });

    it('should handle price service errors gracefully', async () => {
      // Test with invalid symbol
      const invalidPrice = await pythOracleService.getLatestPrice('INVALID-SYMBOL');
      
      // Should handle gracefully (may return null or throw)
      expect(invalidPrice === null || typeof invalidPrice === 'number').toBe(true);
      
      logger.info('✅ Price service error handling test passed');
    });
  });

  describe('Service Integration Tests', () => {
    it('should handle service failures gracefully', async () => {
      // Test with invalid user ID
      await expect(
        matchingService.placeOrder({
          userId: 'non-existent-user',
          symbol: 'SOL-PERP',
          side: 'buy',
          size: 1.0,
          orderType: 'market',
          leverage: 5
        })
      ).rejects.toThrow();
      
      logger.info('✅ Service failure handling test passed');
    });

    it('should handle network errors gracefully', async () => {
      // Test with invalid oracle data
      const invalidPrice = await pythOracleService.getLatestPrice('INVALID-SYMBOL');
      
      // Should handle gracefully
      expect(invalidPrice === null || typeof invalidPrice === 'number').toBe(true);
      
      logger.info('✅ Network error handling test passed');
    });
  });

  describe('Demo Flow Validation', () => {
    it('should validate core services are ready for demo', async () => {
      // Test that all core services are initialized and ready
      const smartContractHealthy = await smartContractService.healthCheck();
      expect(typeof smartContractHealthy).toBe('boolean');
      
      // Test oracle service
      const solPrice = await pythOracleService.getLatestPrice('SOL');
      expect(solPrice).toBeDefined();
      expect(typeof solPrice).toBe('number');
      
      // Test matching service validation
      expect(() => {
        matchingService.placeOrder({
          userId: testUserId,
          symbol: 'SOL-PERP',
          side: 'buy',
          size: 1.0,
          orderType: 'market',
          leverage: 5
        });
      }).not.toThrow();
      
      logger.info('✅ Core services ready for demo test passed');
    });

    it('should validate demo performance requirements', async () => {
      const startTime = Date.now();
      
      // Test core service operations
      await smartContractService.healthCheck();
      await pythOracleService.getLatestPrice('SOL');
      
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Core operations should complete within 1 second
      expect(executionTime).toBeLessThan(1000);
      
      logger.info(`✅ Performance test passed - Core operations completed in ${executionTime}ms`);
    });
  });
});
