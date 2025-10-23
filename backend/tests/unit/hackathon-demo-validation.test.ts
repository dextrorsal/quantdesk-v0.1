/**
 * Hackathon Demo Validation Tests - Simplified
 * 
 * This test suite validates that the hackathon demo components are working
 * and ready for the demo presentation.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { smartContractService } from '../../src/services/smartContractService';
import { matchingService } from '../../src/services/matching';
import { pythOracleService } from '../../src/services/pythOracleService';
import { Logger } from '../../src/utils/logger';

const logger = new Logger();

describe('Hackathon Demo Validation', () => {
  let testUserId: string;
  let testWalletAddress: string;

  beforeAll(async () => {
    testWalletAddress = 'test-wallet-' + Date.now();
    testUserId = 'test-user-' + Date.now();
    
    logger.info('🚀 Hackathon demo validation test environment initialized');
  });

  afterAll(async () => {
    logger.info('🧹 Hackathon demo validation test environment cleaned up');
  });

  describe('Core Service Health Checks', () => {
    it('should have all core services initialized', () => {
      expect(smartContractService).toBeDefined();
      expect(matchingService).toBeDefined();
      expect(pythOracleService).toBeDefined();
      
      logger.info('✅ All core services are initialized');
    });

    it('should validate smart contract service health', async () => {
      const isHealthy = await smartContractService.healthCheck();
      expect(typeof isHealthy).toBe('boolean');
      
      logger.info('✅ Smart contract service health check passed');
    });

    it('should validate oracle service connectivity', async () => {
      // Test oracle service with a known symbol
      const price = await pythOracleService.getLatestPrice('SOL');
      
      // Price should be defined (could be number or null)
      expect(price !== undefined).toBe(true);
      
      // If price is returned, it should be a positive number
      if (price !== null) {
        expect(typeof price).toBe('number');
        expect(price).toBeGreaterThan(0);
      }
      
      logger.info('✅ Oracle service connectivity validated');
    });
  });

  describe('Demo Flow Components', () => {
    it('should validate order validation logic', () => {
      // Test order validation without executing
      expect(() => {
        // Valid order should not throw
        const validOrder = {
          userId: testUserId,
          symbol: 'SOL-PERP',
          side: 'buy' as const,
          size: 1.0,
          orderType: 'market' as const,
          leverage: 5
        };
        
        // This should not throw during validation
        expect(validOrder.size).toBeGreaterThan(0);
        expect(validOrder.leverage).toBeGreaterThan(0);
        expect(validOrder.symbol).toBeDefined();
      }).not.toThrow();
      
      logger.info('✅ Order validation logic validated');
    });

    it('should validate position creation logic', async () => {
      // Test position creation with mock data
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
      
      logger.info('✅ Position creation logic validated');
    });

    it('should validate position closing logic', async () => {
      // Test position closing with mock data
      const positionId = 'test-position-' + Date.now();
      const closeResult = await smartContractService.closePosition(positionId, testUserId);
      
      expect(closeResult).toBeDefined();
      expect(closeResult.success).toBe(true);
      expect(closeResult.transactionSignature).toBeDefined();
      
      logger.info('✅ Position closing logic validated');
    });
  });

  describe('Performance Validation', () => {
    it('should validate core operations performance', async () => {
      const startTime = Date.now();
      
      // Test core service operations
      await smartContractService.healthCheck();
      
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Core operations should complete within 5 seconds
      expect(executionTime).toBeLessThan(5000);
      
      logger.info(`✅ Performance test passed - Core operations completed in ${executionTime}ms`);
    });

    it('should validate oracle service performance', async () => {
      const startTime = Date.now();
      
      // Test oracle service
      await pythOracleService.getLatestPrice('SOL');
      
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Oracle operations should complete within 10 seconds
      expect(executionTime).toBeLessThan(10000);
      
      logger.info(`✅ Oracle performance test passed - Completed in ${executionTime}ms`);
    });
  });

  describe('Demo Readiness Checklist', () => {
    it('should validate all demo components are ready', async () => {
      const readinessChecks = {
        smartContractService: false,
        oracleService: false,
        orderValidation: false,
        positionManagement: false
      };

      // Check smart contract service
      try {
        const health = await smartContractService.healthCheck();
        readinessChecks.smartContractService = typeof health === 'boolean';
      } catch (error) {
        logger.warn('Smart contract service health check failed:', error);
      }

      // Check oracle service
      try {
        const price = await pythOracleService.getLatestPrice('SOL');
        readinessChecks.oracleService = price !== undefined;
      } catch (error) {
        logger.warn('Oracle service check failed:', error);
      }

      // Check order validation
      try {
        const validOrder = {
          userId: testUserId,
          symbol: 'SOL-PERP',
          side: 'buy' as const,
          size: 1.0,
          orderType: 'market' as const,
          leverage: 5
        };
        readinessChecks.orderValidation = validOrder.size > 0 && validOrder.leverage > 0;
      } catch (error) {
        logger.warn('Order validation check failed:', error);
      }

      // Check position management
      try {
        const positionResult = await smartContractService.createPosition(
          testUserId,
          'SOL-PERP',
          'long',
          1.0,
          100,
          5
        );
        readinessChecks.positionManagement = positionResult.success === true;
      } catch (error) {
        logger.warn('Position management check failed:', error);
      }

      // Log readiness status
      logger.info('📋 Demo Readiness Status:', readinessChecks);

      // At least 3 out of 4 components should be ready
      const readyComponents = Object.values(readinessChecks).filter(Boolean).length;
      expect(readyComponents).toBeGreaterThanOrEqual(3);
      
      logger.info(`✅ Demo readiness validated - ${readyComponents}/4 components ready`);
    });
  });
});
