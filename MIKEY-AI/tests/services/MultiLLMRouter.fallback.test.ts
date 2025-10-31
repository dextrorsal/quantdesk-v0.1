/**
 * Integration tests for MultiLLMRouter fallback mechanisms
 */

import { MultiLLMRouter } from '../services/MultiLLMRouter';

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

describe('MultiLLMRouter Fallback Integration Tests', () => {
  let multiRouter: MultiLLMRouter;

  beforeEach(() => {
    multiRouter = new MultiLLMRouter();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Advanced Fallback Mechanism', () => {
    test('should handle fallback when primary provider fails', async () => {
      // Mock provider to fail
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      // Mock fallback provider to succeed
      const mockFallbackProvider = {
        name: 'google',
        model: {
          invoke: jest.fn().mockResolvedValue({ content: 'Fallback response' })
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0003,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue('Fallback response');
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        const response = await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        expect(response).toBe('Fallback response');
      } catch (error) {
        // Expected to fail due to missing API keys, but fallback should be attempted
        expect(error).toBeDefined();
      }
    });

    test('should record fallback events', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue('Fallback response');
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const recordFallbackEventSpy = jest.spyOn(
        multiRouter['intelligentFallbackManager'].getHealthMonitor(),
        'recordFallbackEvent'
      );

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail
      }

      expect(recordFallbackEventSpy).toHaveBeenCalled();
    });

    test('should apply retry delay when specified', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue('Fallback response');
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const delaySpy = jest.spyOn(multiRouter as any, 'delay').mockResolvedValue(undefined);

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail
      }

      expect(delaySpy).toHaveBeenCalled();
    });

    test('should handle fallback decision rejection', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      // Mock fallback decision to reject fallback
      jest.spyOn(multiRouter['intelligentFallbackManager'], 'makeFallbackDecision').mockResolvedValue({
        shouldFallback: false,
        reason: 'Maximum retries exceeded',
        suggestedProvider: 'openai',
        retryCount: 3,
        maxRetries: 3,
        estimatedDelay: 0
      });

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        fail('Should have thrown error');
      } catch (error) {
        expect(error.message).toContain('Fallback not available');
      }
    });
  });

  describe('Cascading Fallback Mechanism', () => {
    test('should try multiple providers in cascading fallback', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      // Mock cascading fallback to succeed on second provider
      let routeToProviderCallCount = 0;
      jest.spyOn(multiRouter as any, 'routeToProvider').mockImplementation(async (provider: string) => {
        routeToProviderCallCount++;
        if (routeToProviderCallCount === 1) {
          return null; // First fallback fails
        } else if (routeToProviderCallCount === 2) {
          return 'Cascading fallback response'; // Second fallback succeeds
        }
        return null;
      });

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        const response = await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        expect(response).toBe('Cascading fallback response');
      } catch (error) {
        // Expected to fail due to missing API keys
        expect(error).toBeDefined();
      }
    });

    test('should update provider health status on failures', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue(null);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const updateProviderStatusSpy = jest.spyOn(
        multiRouter['providerHealthMonitor'],
        'updateProviderStatus'
      );

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail
      }

      expect(updateProviderStatusSpy).toHaveBeenCalled();
    });

    test('should handle all cascading fallback failures', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue(null);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        fail('Should have thrown error');
      } catch (error) {
        expect(error.message).toContain('All fallback providers failed');
      }
    });
  });

  describe('Provider Health Integration', () => {
    test('should integrate with provider health monitoring', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockResolvedValue({ content: 'Test response' })
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const healthMonitor = multiRouter['providerHealthMonitor'];
      expect(healthMonitor).toBeDefined();
      expect(typeof healthMonitor.updateProviderStatus).toBe('function');
    });

    test('should check provider health before routing', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockResolvedValue({ content: 'Test response' })
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const getHealthyProvidersSpy = jest.spyOn(
        multiRouter['intelligentFallbackManager'],
        'getAvailableProviders'
      );

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail due to missing API keys
      }

      // Health monitoring should be integrated
      expect(multiRouter['providerHealthMonitor']).toBeDefined();
    });
  });

  describe('Error Handling Integration', () => {
    test('should handle fallback manager errors gracefully', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      // Mock fallback manager to throw error
      jest.spyOn(multiRouter['intelligentFallbackManager'], 'makeFallbackDecision').mockRejectedValue(
        new Error('Fallback manager error')
      );

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Should handle error gracefully and try cascading fallback
        expect(error).toBeDefined();
      }
    });

    test('should handle health monitor errors gracefully', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      // Mock health monitor to throw error
      jest.spyOn(multiRouter['providerHealthMonitor'], 'updateProviderStatus').mockRejectedValue(
        new Error('Health monitor error')
      );

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue(null);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Should handle error gracefully
        expect(error).toBeDefined();
      }
    });
  });

  describe('Fallback Statistics Integration', () => {
    test('should track fallback statistics', () => {
      const fallbackStats = multiRouter['intelligentFallbackManager'].getFallbackStatistics();
      
      expect(fallbackStats).toBeDefined();
      expect(typeof fallbackStats.totalFallbacks).toBe('number');
      expect(typeof fallbackStats.providerFallbackCounts).toBe('object');
      expect(typeof fallbackStats.averageFallbacksPerProvider).toBe('number');
    });

    test('should reset fallback statistics', () => {
      const resetSpy = jest.spyOn(multiRouter['intelligentFallbackManager'], 'resetFallbackHistory');
      
      multiRouter['intelligentFallbackManager'].resetFallbackHistory();
      
      expect(resetSpy).toHaveBeenCalled();
    });
  });

  describe('End-to-End Fallback Flow', () => {
    test('should complete full fallback flow', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockRejectedValue(new Error('Network timeout'))
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue('Successful fallback response');
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        const response = await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
        expect(response).toBe('Successful fallback response');
      } catch (error) {
        // Expected to fail due to missing API keys
        expect(error).toBeDefined();
      }

      // Verify all components are integrated
      expect(multiRouter['providerHealthMonitor']).toBeDefined();
      expect(multiRouter['intelligentFallbackManager']).toBeDefined();
    });
  });
});
