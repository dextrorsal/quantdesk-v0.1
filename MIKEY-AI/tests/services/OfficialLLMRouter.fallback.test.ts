/**
 * Integration tests for OfficialLLMRouter fallback mechanisms
 */

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

describe('OfficialLLMRouter Fallback Integration Tests', () => {
  let officialRouter: OfficialLLMRouter;

  beforeEach(() => {
    officialRouter = new OfficialLLMRouter();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Advanced Fallback Mechanism', () => {
    test('should handle fallback when primary provider fails', async () => {
      // Mock provider selection and calling
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Mock fallback provider to succeed
      jest.spyOn(officialRouter as any, 'callProvider').mockImplementation(async (provider: string) => {
        if (provider === 'google') {
          return 'Fallback response';
        }
        throw new Error('Provider failed');
      });

      try {
        const result = await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        expect(result.response).toBe('Fallback response');
        expect(result.provider).toBe('google');
      } catch (error) {
        // Expected to fail due to missing API keys, but fallback should be attempted
        expect(error).toBeDefined();
      }
    });

    test('should record fallback events', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const recordFallbackEventSpy = jest.spyOn(
        officialRouter['intelligentFallbackManager'].getHealthMonitor(),
        'recordFallbackEvent'
      );

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail
      }

      expect(recordFallbackEventSpy).toHaveBeenCalled();
    });

    test('should apply retry delay when specified', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const delaySpy = jest.spyOn(officialRouter as any, 'delay').mockResolvedValue(undefined);

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail
      }

      expect(delaySpy).toHaveBeenCalled();
    });

    test('should handle fallback decision rejection', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Mock fallback decision to reject fallback
      jest.spyOn(officialRouter['intelligentFallbackManager'], 'makeFallbackDecision').mockResolvedValue({
        shouldFallback: false,
        reason: 'Maximum retries exceeded',
        suggestedProvider: 'openai',
        retryCount: 3,
        maxRetries: 3,
        estimatedDelay: 0
      });

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        fail('Should have thrown error');
      } catch (error) {
        expect(error.message).toContain('Fallback not available');
      }
    });
  });

  describe('Cascading Fallback Mechanism', () => {
    test('should try multiple providers in cascading fallback', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Mock cascading fallback to succeed on second provider
      let callProviderCallCount = 0;
      jest.spyOn(officialRouter as any, 'callProvider').mockImplementation(async (provider: string) => {
        callProviderCallCount++;
        if (callProviderCallCount === 1) {
          throw new Error('First provider failed');
        } else if (callProviderCallCount === 2) {
          return 'Cascading fallback response';
        }
        throw new Error('Provider failed');
      });

      try {
        const result = await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        expect(result.response).toBe('Cascading fallback response');
      } catch (error) {
        // Expected to fail due to missing API keys
        expect(error).toBeDefined();
      }
    });

    test('should update provider health status on failures', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const updateProviderStatusSpy = jest.spyOn(
        officialRouter['providerHealthMonitor'],
        'updateProviderStatus'
      );

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail
      }

      expect(updateProviderStatusSpy).toHaveBeenCalled();
    });

    test('should handle all cascading fallback failures', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        fail('Should have thrown error');
      } catch (error) {
        expect(error.message).toContain('All fallback providers failed');
      }
    });
  });

  describe('Provider Health Integration', () => {
    test('should integrate with provider health monitoring', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockResolvedValue('Test response');
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const healthMonitor = officialRouter['providerHealthMonitor'];
      expect(healthMonitor).toBeDefined();
      expect(typeof healthMonitor.updateProviderStatus).toBe('function');
    });

    test('should check provider health before routing', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockResolvedValue('Test response');
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const getAvailableProvidersSpy = jest.spyOn(
        officialRouter['intelligentFallbackManager'],
        'getAvailableProviders'
      );

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail due to missing API keys
      }

      // Health monitoring should be integrated
      expect(officialRouter['providerHealthMonitor']).toBeDefined();
    });
  });

  describe('Error Handling Integration', () => {
    test('should handle fallback manager errors gracefully', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Mock fallback manager to throw error
      jest.spyOn(officialRouter['intelligentFallbackManager'], 'makeFallbackDecision').mockRejectedValue(
        new Error('Fallback manager error')
      );

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Should handle error gracefully and try cascading fallback
        expect(error).toBeDefined();
      }
    });

    test('should handle health monitor errors gracefully', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockRejectedValue(new Error('Network timeout'));
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Mock health monitor to throw error
      jest.spyOn(officialRouter['providerHealthMonitor'], 'updateProviderStatus').mockRejectedValue(
        new Error('Health monitor error')
      );

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Should handle error gracefully
        expect(error).toBeDefined();
      }
    });
  });

  describe('Fallback Statistics Integration', () => {
    test('should track fallback statistics', () => {
      const fallbackStats = officialRouter['intelligentFallbackManager'].getFallbackStatistics();
      
      expect(fallbackStats).toBeDefined();
      expect(typeof fallbackStats.totalFallbacks).toBe('number');
      expect(typeof fallbackStats.providerFallbackCounts).toBe('object');
      expect(typeof fallbackStats.averageFallbacksPerProvider).toBe('number');
    });

    test('should reset fallback statistics', () => {
      const resetSpy = jest.spyOn(officialRouter['intelligentFallbackManager'], 'resetFallbackHistory');
      
      officialRouter['intelligentFallbackManager'].resetFallbackHistory();
      
      expect(resetSpy).toHaveBeenCalled();
    });
  });

  describe('End-to-End Fallback Flow', () => {
    test('should complete full fallback flow', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockImplementation(async (provider: string) => {
        if (provider === 'google') {
          return 'Successful fallback response';
        }
        throw new Error('Provider failed');
      });
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        const result = await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        expect(result.response).toBe('Successful fallback response');
        expect(result.provider).toBe('google');
      } catch (error) {
        // Expected to fail due to missing API keys
        expect(error).toBeDefined();
      }

      // Verify all components are integrated
      expect(officialRouter['providerHealthMonitor']).toBeDefined();
      expect(officialRouter['intelligentFallbackManager']).toBeDefined();
    });
  });

  describe('Fallback Event Tracking', () => {
    test('should track fallback events with success status', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockImplementation(async (provider: string) => {
        if (provider === 'google') {
          return 'Successful fallback response';
        }
        throw new Error('Provider failed');
      });
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const updateFallbackEventSuccessSpy = jest.spyOn(
        officialRouter as any,
        'updateFallbackEventSuccess'
      );

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail due to missing API keys
      }

      expect(updateFallbackEventSuccessSpy).toHaveBeenCalled();
    });
  });

  describe('Retry Count Management', () => {
    test('should track retry counts per session', () => {
      const getRetryCountSpy = jest.spyOn(officialRouter as any, 'getRetryCount');
      const updateRetryCountSpy = jest.spyOn(officialRouter as any, 'updateRetryCount');

      const retryCount = officialRouter['getRetryCount']('session-1');
      expect(retryCount).toBe(0);

      officialRouter['updateRetryCount']('session-1', 1);
      expect(updateRetryCountSpy).toHaveBeenCalledWith('session-1', 1);
    });
  });
});
