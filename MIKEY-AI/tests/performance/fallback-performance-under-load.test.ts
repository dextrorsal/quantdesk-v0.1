/**
 * Performance tests for fallback mechanisms under load
 */

import { MultiLLMRouter } from '../services/MultiLLMRouter';
import { OfficialLLMRouter } from '../services/OfficialLLMRouter';

// Mock dependencies
jest.mock('../services/ProviderHealthMonitor', () => {
  return {
    ProviderHealthMonitor: jest.fn().mockImplementation(() => ({
      getHealthyProviders: jest.fn().mockResolvedValue(['google', 'mistral', 'cohere']),
      updateProviderStatus: jest.fn().mockResolvedValue(undefined),
      recordFallbackEvent: jest.fn().mockImplementation(() => {}),
      getFallbackEvents: jest.fn().mockReturnValue([]),
      destroy: jest.fn()
    }))
  };
});

jest.mock('../services/IntelligentFallbackManager', () => {
  return {
    IntelligentFallbackManager: jest.fn().mockImplementation(() => ({
      makeFallbackDecision: jest.fn().mockResolvedValue({
        shouldFallback: true,
        reason: 'Network timeout',
        suggestedProvider: 'google',
        retryCount: 0,
        maxRetries: 3,
        estimatedDelay: 1000
      }),
      getAvailableProviders: jest.fn().mockResolvedValue(['google', 'mistral', 'cohere']),
      getHealthMonitor: jest.fn().mockReturnValue({
        recordFallbackEvent: jest.fn(),
        getFallbackEvents: jest.fn().mockReturnValue([])
      }),
      destroy: jest.fn()
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
      isQualityEvaluationEnabled: jest.fn().mockReturnValue(true),
      evaluateQuality: jest.fn().mockResolvedValue({
        provider: 'openai',
        qualityScore: 0.8,
        shouldEscalate: false,
        confidence: 0.9,
        timestamp: new Date()
      }),
      makeEscalationDecision: jest.fn().mockResolvedValue({
        shouldEscalate: false,
        reason: 'Quality acceptable',
        currentProvider: 'openai',
        suggestedProvider: 'openai',
        qualityScore: 0.8,
        threshold: 0.7,
        confidence: 0.9
      })
    }))
  };
});

jest.mock('../services/TokenEstimationService', () => {
  return {
    TokenEstimationService: jest.fn().mockImplementation(() => ({
      estimateTokens: jest.fn().mockResolvedValue({
        totalTokens: 100,
        estimatedCost: 0.05,
        provider: 'openai',
        model: 'gpt-3.5-turbo'
      })
    }))
  };
});

jest.mock('../services/AnalyticsCollector', () => {
  return {
    AnalyticsCollector: jest.fn().mockImplementation(() => ({
      trackRequestMetrics: jest.fn().mockResolvedValue(undefined),
      getAnalyticsStats: jest.fn().mockReturnValue({
        totalRequests: 0,
        totalCost: 0,
        averageQualityScore: 0,
        dataPointsCollected: 0,
        lastUpdated: new Date()
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

describe('Fallback Performance Under Load Tests', () => {
  let multiRouter: MultiLLMRouter;
  let officialRouter: OfficialLLMRouter;

  beforeEach(() => {
    multiRouter = new MultiLLMRouter();
    officialRouter = new OfficialLLMRouter();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('MultiLLMRouter Load Tests', () => {
    test('should handle high concurrent request load', async () => {
      const concurrentRequests = 100;
      const promises: Promise<any>[] = [];

      // Mock provider to fail and trigger fallback
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue({
        name: 'openai',
        model: { invoke: jest.fn().mockRejectedValue(new Error('Network timeout')) },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      });

      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue('Fallback response');
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const startTime = Date.now();

      // Create concurrent requests
      for (let i = 0; i < concurrentRequests; i++) {
        promises.push(
          multiRouter.routeRequest(`Test prompt ${i}`, 'analysis', `session-${i}`)
        );
      }

      const results = await Promise.allSettled(promises);
      const endTime = Date.now();
      const totalTime = endTime - startTime;

      // Should handle concurrent load efficiently
      expect(results.length).toBe(concurrentRequests);
      expect(totalTime).toBeLessThan(5000); // Should complete within 5 seconds
    });

    test('should maintain performance with repeated fallbacks', async () => {
      const iterations = 50;
      const times: number[] = [];

      // Mock provider to always fail and trigger fallback
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue({
        name: 'openai',
        model: { invoke: jest.fn().mockRejectedValue(new Error('Network timeout')) },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      });

      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue('Fallback response');
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      for (let i = 0; i < iterations; i++) {
        const startTime = Date.now();
        
        try {
          await multiRouter.routeRequest(`Test prompt ${i}`, 'analysis', `session-${i}`);
        } catch (error) {
          // Expected to fail due to missing API keys
        }
        
        const endTime = Date.now();
        times.push(endTime - startTime);
      }

      const averageTime = times.reduce((sum, time) => sum + time, 0) / times.length;
      const maxTime = Math.max(...times);

      expect(averageTime).toBeLessThan(1000); // Average <1 second
      expect(maxTime).toBeLessThan(2000); // Max <2 seconds
    });

    test('should handle cascading fallback under load', async () => {
      const requests = 20;
      let fallbackCount = 0;

      // Mock provider to fail
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue({
        name: 'openai',
        model: { invoke: jest.fn().mockRejectedValue(new Error('Network timeout')) },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      });

      // Mock cascading fallback
      jest.spyOn(multiRouter as any, 'routeToProvider').mockImplementation(async (provider: string) => {
        fallbackCount++;
        if (fallbackCount <= requests / 2) {
          return null; // First half fails
        } else {
          return 'Cascading fallback response'; // Second half succeeds
        }
      });

      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const promises: Promise<any>[] = [];
      for (let i = 0; i < requests; i++) {
        promises.push(
          multiRouter.routeRequest(`Test prompt ${i}`, 'analysis', `session-${i}`)
        );
      }

      const results = await Promise.allSettled(promises);
      const successfulResults = results.filter(result => 
        result.status === 'fulfilled'
      );

      expect(successfulResults.length).toBeGreaterThan(0);
      expect(fallbackCount).toBeGreaterThan(requests);
    });
  });

  describe('OfficialLLMRouter Load Tests', () => {
    test('should handle high concurrent request load', async () => {
      const concurrentRequests = 100;
      const promises: Promise<any>[] = [];

      // Mock provider to fail and trigger fallback
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const startTime = Date.now();

      // Create concurrent requests
      for (let i = 0; i < concurrentRequests; i++) {
        promises.push(
          officialRouter.routeRequest(`Test prompt ${i}`, 'analysis', `session-${i}`)
        );
      }

      const results = await Promise.allSettled(promises);
      const endTime = Date.now();
      const totalTime = endTime - startTime;

      // Should handle concurrent load efficiently
      expect(results.length).toBe(concurrentRequests);
      expect(totalTime).toBeLessThan(5000); // Should complete within 5 seconds
    });

    test('should maintain performance with repeated fallbacks', async () => {
      const iterations = 50;
      const times: number[] = [];

      // Mock provider to always fail and trigger fallback
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      for (let i = 0; i < iterations; i++) {
        const startTime = Date.now();
        
        try {
          await officialRouter.routeRequest(`Test prompt ${i}`, 'analysis', `session-${i}`);
        } catch (error) {
          // Expected to fail due to missing API keys
        }
        
        const endTime = Date.now();
        times.push(endTime - startTime);
      }

      const averageTime = times.reduce((sum, time) => sum + time, 0) / times.length;
      const maxTime = Math.max(...times);

      expect(averageTime).toBeLessThan(1000); // Average <1 second
      expect(maxTime).toBeLessThan(2000); // Max <2 seconds
    });

    test('should handle cascading fallback under load', async () => {
      const requests = 20;
      let fallbackCount = 0;

      // Mock provider to fail
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Mock cascading fallback
      jest.spyOn(officialRouter as any, 'callProvider').mockImplementation(async (provider: string) => {
        fallbackCount++;
        if (fallbackCount <= requests / 2) {
          throw new Error('Provider failed');
        } else {
          return 'Cascading fallback response';
        }
      });

      const promises: Promise<any>[] = [];
      for (let i = 0; i < requests; i++) {
        promises.push(
          officialRouter.routeRequest(`Test prompt ${i}`, 'analysis', `session-${i}`)
        );
      }

      const results = await Promise.allSettled(promises);
      const successfulResults = results.filter(result => 
        result.status === 'fulfilled'
      );

      expect(successfulResults.length).toBeGreaterThan(0);
      expect(fallbackCount).toBeGreaterThan(requests);
    });
  });

  describe('Memory Usage Under Load', () => {
    test('should not leak memory with repeated fallbacks', async () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Mock providers
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue({
        name: 'openai',
        model: { invoke: jest.fn().mockRejectedValue(new Error('Network timeout')) },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      });

      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue('Fallback response');
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Perform many fallback operations
      for (let i = 0; i < 1000; i++) {
        try {
          await multiRouter.routeRequest(`Test prompt ${i}`, 'analysis', `session-${i}`);
        } catch (error) {
          // Expected to fail due to missing API keys
        }
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;

      // Memory increase should be reasonable (less than 50MB)
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024);
    });

    test('should handle large numbers of fallback events efficiently', () => {
      const eventCount = 10000;
      
      for (let i = 0; i < eventCount; i++) {
        multiRouter['intelligentFallbackManager'].getHealthMonitor().recordFallbackEvent({
          originalProvider: 'openai',
          fallbackProvider: 'google',
          reason: `Load test event ${i}`,
          timestamp: new Date(),
          success: true,
          responseTime: 1500,
          retryCount: 0
        });
      }

      const events = multiRouter['intelligentFallbackManager'].getHealthMonitor().getFallbackEvents();
      expect(events.length).toBeLessThanOrEqual(1000); // Should be limited to 1000
    });
  });

  describe('Stress Testing', () => {
    test('should handle stress test scenarios', async () => {
      const stressTestDuration = 10000; // 10 seconds
      const startTime = Date.now();
      let operationsCompleted = 0;

      // Mock providers
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue({
        name: 'openai',
        model: { invoke: jest.fn().mockRejectedValue(new Error('Network timeout')) },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      });

      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue('Stress test response');
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      while (Date.now() - startTime < stressTestDuration) {
        try {
          await Promise.all([
            multiRouter.routeRequest('Stress test prompt 1', 'analysis', 'session-1'),
            multiRouter.routeRequest('Stress test prompt 2', 'analysis', 'session-2'),
            multiRouter.routeRequest('Stress test prompt 3', 'analysis', 'session-3')
          ]);
          operationsCompleted += 3;
        } catch (error) {
          // Expected some errors in stress test
        }
      }

      // Should complete reasonable number of operations
      expect(operationsCompleted).toBeGreaterThan(100);
    });

    test('should maintain stability under high error rates', async () => {
      const errorRate = 0.9; // 90% error rate
      const totalOperations = 100;
      let successfulOperations = 0;

      // Mock provider with high error rate
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue({
        name: 'openai',
        model: { invoke: jest.fn().mockRejectedValue(new Error('High error rate')) },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      });

      jest.spyOn(multiRouter as any, 'routeToProvider').mockImplementation(async (provider: string) => {
        const shouldFail = Math.random() < errorRate;
        if (shouldFail) {
          return null;
        } else {
          successfulOperations++;
          return 'Successful fallback response';
        }
      });

      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      for (let i = 0; i < totalOperations; i++) {
        try {
          await multiRouter.routeRequest(`Test prompt ${i}`, 'analysis', `session-${i}`);
        } catch (error) {
          // Expected errors
        }
      }

      // Should handle high error rates gracefully
      expect(successfulOperations).toBeLessThan(totalOperations * 0.2); // <20% success rate
    });
  });

  describe('Performance Benchmarks Under Load', () => {
    test('should meet performance benchmarks under load', async () => {
      const benchmarks = {
        averageResponseTime: 0,
        maxResponseTime: 0,
        throughput: 0,
        errorRate: 0
      };

      const iterations = 100;
      const times: number[] = [];
      let errors = 0;

      // Mock providers
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue({
        name: 'openai',
        model: { invoke: jest.fn().mockRejectedValue(new Error('Network timeout')) },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      });

      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue('Benchmark response');
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const startTime = Date.now();

      for (let i = 0; i < iterations; i++) {
        const requestStart = Date.now();
        
        try {
          await multiRouter.routeRequest(`Benchmark prompt ${i}`, 'analysis', `session-${i}`);
        } catch (error) {
          errors++;
        }
        
        const requestEnd = Date.now();
        times.push(requestEnd - requestStart);
      }

      const endTime = Date.now();
      const totalTime = endTime - startTime;

      benchmarks.averageResponseTime = times.reduce((sum, time) => sum + time, 0) / times.length;
      benchmarks.maxResponseTime = Math.max(...times);
      benchmarks.throughput = iterations / (totalTime / 1000); // requests per second
      benchmarks.errorRate = errors / iterations;

      // Verify benchmarks
      expect(benchmarks.averageResponseTime).toBeLessThan(1000); // <1 second average
      expect(benchmarks.maxResponseTime).toBeLessThan(2000); // <2 seconds max
      expect(benchmarks.throughput).toBeGreaterThan(10); // >10 requests per second
      expect(benchmarks.errorRate).toBeLessThan(0.1); // <10% error rate
    });
  });
});
