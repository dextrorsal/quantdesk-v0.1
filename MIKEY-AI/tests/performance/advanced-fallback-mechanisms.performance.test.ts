/**
 * Performance tests for advanced fallback mechanisms
 */

import { ProviderHealthMonitor } from '../services/ProviderHealthMonitor';
import { IntelligentFallbackManager } from '../services/IntelligentFallbackManager';

// Mock dependencies
jest.mock('../config/fallback-config', () => {
  return {
    FallbackConfiguration: jest.fn().mockImplementation(() => ({
      getInstance: jest.fn().mockReturnThis(),
      getFallbackConfig: jest.fn().mockReturnValue({
        maxRetries: 3,
        retryDelay: 1000,
        circuitBreakerThreshold: 5,
        circuitBreakerTimeout: 60000,
        healthCheckInterval: 30000,
        fallbackOrder: ['openai', 'google', 'mistral', 'cohere'],
        enableCircuitBreaker: true,
        enableHealthMonitoring: true,
        retryBackoffMultiplier: 2,
        maxRetryDelay: 10000
      }),
      getCircuitBreakerConfig: jest.fn().mockReturnValue({
        failureThreshold: 5,
        timeout: 60000,
        halfOpenMaxCalls: 3,
        enableAutoRecovery: true,
        recoveryCheckInterval: 30000
      }),
      getRetryConfig: jest.fn().mockReturnValue({
        maxRetries: 3,
        baseDelay: 1000,
        maxDelay: 10000,
        backoffMultiplier: 2,
        jitter: true,
        retryableErrors: ['timeout', 'network', 'rate_limit']
      }),
      getFallbackStrategy: jest.fn().mockReturnValue({
        strategy: 'balanced',
        maxFallbackDepth: 3,
        enableCascadingFallback: true,
        qualityThreshold: 0.7,
        costThreshold: 0.1
      }),
      getHealthMonitorConfig: jest.fn().mockReturnValue({
        checkInterval: 30000,
        timeout: 5000,
        enableContinuousMonitoring: true,
        alertThresholds: {
          errorRate: 0.1,
          responseTime: 5000,
          availability: 0.95
        }
      }),
      isHealthMonitoringEnabled: jest.fn().mockReturnValue(true),
      isCircuitBreakerEnabled: jest.fn().mockReturnValue(true),
      getHealthCheckInterval: jest.fn().mockReturnValue(30000),
      getCircuitBreakerThreshold: jest.fn().mockReturnValue(5),
      getCircuitBreakerTimeout: jest.fn().mockReturnValue(60000),
      getFallbackOrder: jest.fn().mockReturnValue(['openai', 'google', 'mistral', 'cohere']),
      getMaxRetries: jest.fn().mockReturnValue(3),
      getRetryDelay: jest.fn().mockReturnValue(1000),
      getRetryBackoffMultiplier: jest.fn().mockReturnValue(2),
      getMaxRetryDelay: jest.fn().mockReturnValue(10000)
    }))
  };
});

jest.mock('../services/CostOptimizationEngine', () => {
  return {
    CostOptimizationEngine: jest.fn().mockImplementation(() => ({
      getCostStatistics: jest.fn().mockReturnValue({ totalCost: 0.1, averageCost: 0.05 })
    }))
  };
});

jest.mock('../services/QualityThresholdManager', () => {
  return {
    QualityThresholdManager: jest.fn().mockImplementation(() => ({
      getQualityStats: jest.fn().mockReturnValue({
        totalEvaluations: 10,
        averageQualityScore: 0.8,
        escalationCount: 2
      })
    }))
  };
});

jest.mock('../utils/logger', () => ({
  systemLogger: {
    startup: jest.fn()
  },
  errorLogger: {
    aiError: jest.fn()
  }
}));

describe('Advanced Fallback Performance Tests', () => {
  let healthMonitor: ProviderHealthMonitor;
  let fallbackManager: IntelligentFallbackManager;

  beforeEach(() => {
    healthMonitor = new ProviderHealthMonitor();
    fallbackManager = new IntelligentFallbackManager();
  });

  afterEach(() => {
    healthMonitor.destroy();
    fallbackManager.destroy();
    jest.clearAllMocks();
  });

  describe('Service Availability Tests', () => {
    test('should maintain 99.9% service availability', async () => {
      const totalRequests = 1000;
      let successfulRequests = 0;
      let failedRequests = 0;

      // Simulate multiple requests
      for (let i = 0; i < totalRequests; i++) {
        try {
          const result = await healthMonitor.checkProviderHealth('openai');
          if (result.isHealthy) {
            successfulRequests++;
          } else {
            failedRequests++;
          }
        } catch (error) {
          failedRequests++;
        }
      }

      const availability = successfulRequests / totalRequests;
      expect(availability).toBeGreaterThanOrEqual(0.999); // 99.9%
    });

    test('should handle high concurrent load', async () => {
      const concurrentRequests = 100;
      const promises: Promise<any>[] = [];

      // Create concurrent health checks
      for (let i = 0; i < concurrentRequests; i++) {
        promises.push(healthMonitor.checkProviderHealth('openai'));
      }

      const results = await Promise.allSettled(promises);
      const successfulResults = results.filter(result => 
        result.status === 'fulfilled' && result.value.isHealthy
      );

      // Should handle concurrent load without significant degradation
      expect(successfulResults.length).toBeGreaterThan(concurrentRequests * 0.95);
    });

    test('should recover quickly from failures', async () => {
      const provider = 'openai';
      
      // Simulate failures
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, 'Test error');
      }

      // Check circuit breaker state
      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');

      // Simulate recovery
      await healthMonitor.updateProviderStatus(provider, true, 1500);
      
      // Should move to half-open state
      const updatedStatus = healthMonitor.getProviderStatus(provider);
      expect(updatedStatus!.circuitBreakerState).toBe('HALF_OPEN');
    });
  });

  describe('Fallback Performance Tests', () => {
    test('should make fallback decisions quickly', async () => {
      const startTime = Date.now();
      
      const decision = await fallbackManager.makeFallbackDecision(
        'openai',
        new Error('Network timeout'),
        0,
        'analysis'
      );
      
      const endTime = Date.now();
      const decisionTime = endTime - startTime;
      
      expect(decisionTime).toBeLessThan(200); // <200ms requirement
      expect(decision).toBeDefined();
    });

    test('should handle multiple fallback decisions concurrently', async () => {
      const concurrentDecisions = 50;
      const promises: Promise<any>[] = [];

      const startTime = Date.now();

      // Create concurrent fallback decisions
      for (let i = 0; i < concurrentDecisions; i++) {
        promises.push(
          fallbackManager.makeFallbackDecision(
            'openai',
            new Error(`Test error ${i}`),
            0,
            'analysis'
          )
        );
      }

      const results = await Promise.allSettled(promises);
      const endTime = Date.now();
      const totalTime = endTime - startTime;

      // Should handle concurrent decisions efficiently
      expect(results.length).toBe(concurrentDecisions);
      expect(totalTime).toBeLessThan(1000); // Should complete within 1 second
    });

    test('should maintain performance under load', async () => {
      const iterations = 100;
      const times: number[] = [];

      for (let i = 0; i < iterations; i++) {
        const startTime = Date.now();
        
        await fallbackManager.makeFallbackDecision(
          'openai',
          new Error('Test error'),
          0,
          'analysis'
        );
        
        const endTime = Date.now();
        times.push(endTime - startTime);
      }

      const averageTime = times.reduce((sum, time) => sum + time, 0) / times.length;
      const maxTime = Math.max(...times);

      expect(averageTime).toBeLessThan(100); // Average <100ms
      expect(maxTime).toBeLessThan(200); // Max <200ms
    });
  });

  describe('Circuit Breaker Performance Tests', () => {
    test('should detect failures quickly', async () => {
      const provider = 'openai';
      const startTime = Date.now();

      // Simulate rapid failures
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      const endTime = Date.now();
      const detectionTime = endTime - startTime;

      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');
      expect(detectionTime).toBeLessThan(100); // <100ms detection
    });

    test('should recover efficiently', async () => {
      const provider = 'openai';
      
      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');

      // Simulate timeout and recovery
      if (status) {
        status.lastFailure = new Date(Date.now() - 70000); // 70 seconds ago
      }

      const startTime = Date.now();
      await healthMonitor.updateProviderStatus(provider, true, 1500);
      const endTime = Date.now();

      const recoveryTime = endTime - startTime;
      expect(recoveryTime).toBeLessThan(100); // <100ms recovery
    });

    test('should handle circuit breaker state transitions efficiently', async () => {
      const provider = 'openai';
      const stateTransitions: string[] = [];

      // Monitor state transitions
      const originalUpdateStatus = healthMonitor.updateProviderStatus.bind(healthMonitor);
      jest.spyOn(healthMonitor, 'updateProviderStatus').mockImplementation(async (...args) => {
        const result = await originalUpdateStatus(...args);
        const status = healthMonitor.getProviderStatus(provider);
        if (status) {
          stateTransitions.push(status.circuitBreakerState);
        }
        return result;
      });

      // Simulate state transitions
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      // Move to half-open
      const status = healthMonitor.getProviderStatus(provider);
      if (status) {
        status.lastFailure = new Date(Date.now() - 70000);
      }
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Still failing');

      // Close circuit breaker
      await healthMonitor.updateProviderStatus(provider, true, 1500);

      expect(stateTransitions).toContain('CLOSED');
      expect(stateTransitions).toContain('OPEN');
      expect(stateTransitions).toContain('HALF_OPEN');
    });
  });

  describe('Memory and Resource Usage Tests', () => {
    test('should not leak memory with repeated operations', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Perform many operations
      for (let i = 0; i < 1000; i++) {
        await healthMonitor.checkProviderHealth('openai');
        await fallbackManager.makeFallbackDecision(
          'openai',
          new Error('Test error'),
          0,
          'analysis'
        );
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;

      // Memory increase should be reasonable (less than 10MB)
      expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024);
    });

    test('should handle large numbers of fallback events', () => {
      const eventCount = 10000;
      
      for (let i = 0; i < eventCount; i++) {
        healthMonitor.recordFallbackEvent({
          originalProvider: 'openai',
          fallbackProvider: 'google',
          reason: `Test event ${i}`,
          timestamp: new Date(),
          success: true,
          responseTime: 1500,
          retryCount: 0
        });
      }

      const events = healthMonitor.getFallbackEvents();
      expect(events.length).toBeLessThanOrEqual(1000); // Should be limited to 1000
    });

    test('should handle large numbers of health alerts', async () => {
      const alertCount = 1000;
      
      for (let i = 0; i < alertCount; i++) {
        await healthMonitor.updateProviderStatus('openai', false, 5000, `Alert ${i}`);
      }

      const alerts = healthMonitor.getHealthAlerts();
      expect(alerts.length).toBeLessThanOrEqual(1000); // Should be limited to 1000
    });
  });

  describe('Stress Testing', () => {
    test('should handle stress test scenarios', async () => {
      const stressTestDuration = 5000; // 5 seconds
      const startTime = Date.now();
      let operationsCompleted = 0;

      while (Date.now() - startTime < stressTestDuration) {
        try {
          await Promise.all([
            healthMonitor.checkProviderHealth('openai'),
            healthMonitor.checkProviderHealth('google'),
            fallbackManager.makeFallbackDecision('openai', new Error('Stress test'), 0, 'analysis'),
            fallbackManager.makeFallbackDecision('google', new Error('Stress test'), 0, 'analysis')
          ]);
          operationsCompleted += 4;
        } catch (error) {
          // Expected some errors in stress test
        }
      }

      // Should complete reasonable number of operations
      expect(operationsCompleted).toBeGreaterThan(100);
    });

    test('should maintain stability under high error rates', async () => {
      const errorRate = 0.8; // 80% error rate
      const totalOperations = 1000;
      let successfulOperations = 0;

      for (let i = 0; i < totalOperations; i++) {
        try {
          const shouldFail = Math.random() < errorRate;
          
          if (shouldFail) {
            await healthMonitor.updateProviderStatus('openai', false, 5000, 'High error rate test');
          } else {
            await healthMonitor.updateProviderStatus('openai', true, 1500);
            successfulOperations++;
          }
        } catch (error) {
          // Expected errors
        }
      }

      // Should handle high error rates gracefully
      expect(successfulOperations).toBeLessThan(totalOperations * 0.3); // <30% success rate
    });
  });

  describe('Performance Benchmarks', () => {
    test('should meet performance benchmarks', async () => {
      const benchmarks = {
        healthCheckTime: 0,
        fallbackDecisionTime: 0,
        circuitBreakerDetectionTime: 0,
        stateTransitionTime: 0
      };

      // Benchmark health check time
      const healthCheckStart = Date.now();
      await healthMonitor.checkProviderHealth('openai');
      benchmarks.healthCheckTime = Date.now() - healthCheckStart;

      // Benchmark fallback decision time
      const fallbackStart = Date.now();
      await fallbackManager.makeFallbackDecision('openai', new Error('Benchmark'), 0, 'analysis');
      benchmarks.fallbackDecisionTime = Date.now() - fallbackStart;

      // Benchmark circuit breaker detection time
      const detectionStart = Date.now();
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus('openai', false, 5000, `Benchmark ${i}`);
      }
      benchmarks.circuitBreakerDetectionTime = Date.now() - detectionStart;

      // Benchmark state transition time
      const transitionStart = Date.now();
      await healthMonitor.updateProviderStatus('openai', true, 1500);
      benchmarks.stateTransitionTime = Date.now() - transitionStart;

      // Verify benchmarks
      expect(benchmarks.healthCheckTime).toBeLessThan(100);
      expect(benchmarks.fallbackDecisionTime).toBeLessThan(200);
      expect(benchmarks.circuitBreakerDetectionTime).toBeLessThan(100);
      expect(benchmarks.stateTransitionTime).toBeLessThan(100);
    });
  });
});
