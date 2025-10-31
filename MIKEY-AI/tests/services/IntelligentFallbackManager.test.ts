/**
 * Unit tests for IntelligentFallbackManager
 */

import { IntelligentFallbackManager } from '../services/IntelligentFallbackManager';
import { FallbackStrategy } from '../types/provider-health';

// Mock dependencies
jest.mock('../services/ProviderHealthMonitor', () => {
  return {
    ProviderHealthMonitor: jest.fn().mockImplementation(() => ({
      getHealthyProviders: jest.fn().mockResolvedValue(['openai', 'google', 'mistral', 'cohere']),
      getProviderHealthMetrics: jest.fn().mockReturnValue({
        provider: 'openai',
        availability: 0.95,
        averageResponseTime: 1500,
        errorRate: 0.05,
        successRate: 0.95,
        circuitBreakerState: 'CLOSED',
        lastUpdated: new Date()
      }),
      destroy: jest.fn()
    }))
  };
});

jest.mock('../services/CostOptimizationEngine', () => {
  return {
    CostOptimizationEngine: jest.fn().mockImplementation(() => ({
      getCostStatistics: jest.fn().mockReturnValue({
        totalCost: 0.1,
        averageCost: 0.05
      })
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

jest.mock('../config/fallback-config', () => {
  return {
    FallbackConfiguration: jest.fn().mockImplementation(() => ({
      getInstance: jest.fn().mockReturnThis(),
      getFallbackStrategy: jest.fn().mockReturnValue({
        strategy: 'balanced',
        maxFallbackDepth: 3,
        enableCascadingFallback: true,
        qualityThreshold: 0.7,
        costThreshold: 0.1
      }),
      getMaxRetries: jest.fn().mockReturnValue(3),
      getRetryDelay: jest.fn().mockReturnValue(1000),
      getRetryBackoffMultiplier: jest.fn().mockReturnValue(2),
      getMaxRetryDelay: jest.fn().mockReturnValue(10000),
      getFallbackOrder: jest.fn().mockReturnValue(['openai', 'google', 'mistral', 'cohere'])
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

describe('IntelligentFallbackManager', () => {
  let fallbackManager: IntelligentFallbackManager;

  beforeEach(() => {
    fallbackManager = new IntelligentFallbackManager();
  });

  afterEach(() => {
    fallbackManager.destroy();
    jest.clearAllMocks();
  });

  describe('Fallback Decision Making', () => {
    test('should make fallback decision successfully', async () => {
      const error = new Error('Network timeout');
      const decision = await fallbackManager.makeFallbackDecision('openai', error, 0, 'analysis');

      expect(decision).toBeDefined();
      expect(decision.shouldFallback).toBe(true);
      expect(decision.suggestedProvider).toBeDefined();
      expect(decision.reason).toBeDefined();
      expect(decision.retryCount).toBe(0);
      expect(decision.maxRetries).toBe(3);
      expect(decision.estimatedDelay).toBeGreaterThan(0);
    });

    test('should not fallback when max retries exceeded', async () => {
      const error = new Error('Network timeout');
      const decision = await fallbackManager.makeFallbackDecision('openai', error, 3, 'analysis');

      expect(decision.shouldFallback).toBe(false);
      expect(decision.reason).toBe('Maximum retries exceeded');
      expect(decision.suggestedProvider).toBe('openai');
    });

    test('should not fallback when no available providers', async () => {
      // Mock no healthy providers
      const mockHealthMonitor = fallbackManager.getHealthMonitor();
      jest.spyOn(mockHealthMonitor, 'getHealthyProviders').mockResolvedValue([]);

      const error = new Error('Network timeout');
      const decision = await fallbackManager.makeFallbackDecision('openai', error, 0, 'analysis');

      expect(decision.shouldFallback).toBe(false);
      expect(decision.reason).toBe('No available providers for fallback');
    });

    test('should handle fallback decision errors gracefully', async () => {
      // Mock error in fallback decision
      jest.spyOn(fallbackManager as any, 'getAvailableProviders').mockRejectedValue(new Error('Provider error'));

      const error = new Error('Network timeout');
      const decision = await fallbackManager.makeFallbackDecision('openai', error, 0, 'analysis');

      expect(decision.shouldFallback).toBe(false);
      expect(decision.reason).toBe('Fallback decision error');
    });
  });

  describe('Available Providers', () => {
    test('should get available providers excluding original', async () => {
      const availableProviders = await fallbackManager.getAvailableProviders('openai');

      expect(availableProviders).toBeDefined();
      expect(Array.isArray(availableProviders)).toBe(true);
      expect(availableProviders).not.toContain('openai');
      expect(availableProviders.length).toBeGreaterThan(0);
    });

    test('should sort providers by fallback order', async () => {
      const availableProviders = await fallbackManager.getAvailableProviders('openai');

      // Should be sorted according to fallback order
      expect(availableProviders[0]).toBe('google'); // Next in fallback order
      expect(availableProviders[1]).toBe('mistral');
      expect(availableProviders[2]).toBe('cohere');
    });

    test('should handle errors when getting available providers', async () => {
      // Mock error
      const mockHealthMonitor = fallbackManager.getHealthMonitor();
      jest.spyOn(mockHealthMonitor, 'getHealthyProviders').mockRejectedValue(new Error('Health check error'));

      const availableProviders = await fallbackManager.getAvailableProviders('openai');

      expect(availableProviders).toEqual([]);
    });
  });

  describe('Provider Selection Strategies', () => {
    test('should select cost-first provider', async () => {
      const mockConfig = require('../config/fallback-config').FallbackConfiguration;
      mockConfig.mockImplementation(() => ({
        getInstance: jest.fn().mockReturnThis(),
        getFallbackStrategy: jest.fn().mockReturnValue({
          strategy: 'cost-first',
          maxFallbackDepth: 3,
          enableCascadingFallback: true,
          qualityThreshold: 0.7,
          costThreshold: 0.1
        }),
        getMaxRetries: jest.fn().mockReturnValue(3),
        getRetryDelay: jest.fn().mockReturnValue(1000),
        getRetryBackoffMultiplier: jest.fn().mockReturnValue(2),
        getMaxRetryDelay: jest.fn().mockReturnValue(10000),
        getFallbackOrder: jest.fn().mockReturnValue(['openai', 'google', 'mistral', 'cohere'])
      }));

      const manager = new IntelligentFallbackManager();
      const availableProviders = ['google', 'mistral', 'cohere'];
      
      const selectedProvider = await (manager as any).selectBestFallbackProvider(
        availableProviders,
        { strategy: 'cost-first', costThreshold: 0.1 },
        'analysis',
        new Error('Test error')
      );

      expect(selectedProvider).toBeDefined();
      expect(availableProviders).toContain(selectedProvider);
      
      manager.destroy();
    });

    test('should select quality-first provider', async () => {
      const availableProviders = ['google', 'mistral', 'cohere'];
      
      const selectedProvider = await (fallbackManager as any).selectBestFallbackProvider(
        availableProviders,
        { strategy: 'quality-first', qualityThreshold: 0.7 },
        'analysis',
        new Error('Test error')
      );

      expect(selectedProvider).toBeDefined();
      expect(availableProviders).toContain(selectedProvider);
    });

    test('should select availability-first provider', async () => {
      const availableProviders = ['google', 'mistral', 'cohere'];
      
      const selectedProvider = await (fallbackManager as any).selectBestFallbackProvider(
        availableProviders,
        { strategy: 'availability-first' },
        'analysis',
        new Error('Test error')
      );

      expect(selectedProvider).toBeDefined();
      expect(availableProviders).toContain(selectedProvider);
    });

    test('should select balanced provider', async () => {
      const availableProviders = ['google', 'mistral', 'cohere'];
      
      const selectedProvider = await (fallbackManager as any).selectBestFallbackProvider(
        availableProviders,
        { strategy: 'balanced', qualityThreshold: 0.7, costThreshold: 0.1 },
        'analysis',
        new Error('Test error')
      );

      expect(selectedProvider).toBeDefined();
      expect(availableProviders).toContain(selectedProvider);
    });

    test('should handle unknown strategy gracefully', async () => {
      const availableProviders = ['google', 'mistral', 'cohere'];
      
      const selectedProvider = await (fallbackManager as any).selectBestFallbackProvider(
        availableProviders,
        { strategy: 'unknown-strategy' as any },
        'analysis',
        new Error('Test error')
      );

      expect(selectedProvider).toBeDefined();
      expect(availableProviders).toContain(selectedProvider);
    });
  });

  describe('Retry Delay Calculation', () => {
    test('should calculate retry delay with exponential backoff', () => {
      const delay1 = (fallbackManager as any).calculateRetryDelay(0);
      const delay2 = (fallbackManager as any).calculateRetryDelay(1);
      const delay3 = (fallbackManager as any).calculateRetryDelay(2);

      expect(delay1).toBe(1000); // Base delay
      expect(delay2).toBe(2000); // Base delay * 2
      expect(delay3).toBe(4000); // Base delay * 4
    });

    test('should respect max retry delay', () => {
      const delay = (fallbackManager as any).calculateRetryDelay(10); // Very high retry count

      expect(delay).toBeLessThanOrEqual(10000); // Max delay
    });
  });

  describe('Provider Cost and Quality', () => {
    test('should get provider cost', async () => {
      const cost = await (fallbackManager as any).getProviderCost('openai');

      expect(cost).toBeGreaterThan(0);
      expect(typeof cost).toBe('number');
    });

    test('should get provider quality', async () => {
      const quality = await (fallbackManager as any).getProviderQuality('openai', 'analysis');

      expect(quality).toBeGreaterThan(0);
      expect(quality).toBeLessThanOrEqual(1);
      expect(typeof quality).toBe('number');
    });

    test('should get provider availability', () => {
      const availability = (fallbackManager as any).getProviderAvailability('openai');

      expect(availability).toBeGreaterThan(0);
      expect(availability).toBeLessThanOrEqual(1);
      expect(typeof availability).toBe('number');
    });

    test('should handle errors in cost calculation', async () => {
      // Mock error
      jest.spyOn(fallbackManager as any, 'getProviderCost').mockRejectedValue(new Error('Cost error'));

      const cost = await (fallbackManager as any).getProviderCost('openai');

      expect(cost).toBe(0.0005); // Default cost
    });

    test('should handle errors in quality calculation', async () => {
      // Mock error
      jest.spyOn(fallbackManager as any, 'getProviderQuality').mockRejectedValue(new Error('Quality error'));

      const quality = await (fallbackManager as any).getProviderQuality('openai', 'analysis');

      expect(quality).toBe(0.8); // Default quality
    });

    test('should handle errors in availability calculation', () => {
      // Mock error
      jest.spyOn(fallbackManager as any, 'getProviderAvailability').mockImplementation(() => {
        throw new Error('Availability error');
      });

      const availability = (fallbackManager as any).getProviderAvailability('openai');

      expect(availability).toBe(0.5); // Default availability
    });
  });

  describe('Fallback History', () => {
    test('should update fallback history', () => {
      (fallbackManager as any).updateFallbackHistory('google');

      const stats = fallbackManager.getFallbackStatistics();
      expect(stats.providerFallbackCounts['google']).toBe(1);
    });

    test('should get fallback statistics', () => {
      (fallbackManager as any).updateFallbackHistory('google');
      (fallbackManager as any).updateFallbackHistory('google');
      (fallbackManager as any).updateFallbackHistory('mistral');

      const stats = fallbackManager.getFallbackStatistics();
      expect(stats.totalFallbacks).toBe(3);
      expect(stats.providerFallbackCounts['google']).toBe(2);
      expect(stats.providerFallbackCounts['mistral']).toBe(1);
      expect(stats.averageFallbacksPerProvider).toBe(1.5);
    });

    test('should reset fallback history', () => {
      (fallbackManager as any).updateFallbackHistory('google');
      (fallbackManager as any).updateFallbackHistory('mistral');

      fallbackManager.resetFallbackHistory();

      const stats = fallbackManager.getFallbackStatistics();
      expect(stats.totalFallbacks).toBe(0);
      expect(stats.providerFallbackCounts).toEqual({});
    });
  });

  describe('Fallback Reasons', () => {
    test('should generate appropriate fallback reasons', () => {
      const timeoutError = new Error('Request timeout');
      const rateLimitError = new Error('Rate limit exceeded');
      const networkError = new Error('Network connection failed');
      const circuitBreakerError = new Error('Circuit breaker open');
      const genericError = new Error('Unknown error');

      const timeoutReason = (fallbackManager as any).getFallbackReason(timeoutError, 'openai', 'google');
      const rateLimitReason = (fallbackManager as any).getFallbackReason(rateLimitError, 'openai', 'google');
      const networkReason = (fallbackManager as any).getFallbackReason(networkError, 'openai', 'google');
      const circuitBreakerReason = (fallbackManager as any).getFallbackReason(circuitBreakerError, 'openai', 'google');
      const genericReason = (fallbackManager as any).getFallbackReason(genericError, 'openai', 'google');

      expect(timeoutReason).toContain('Timeout');
      expect(rateLimitReason).toContain('Rate limit');
      expect(networkReason).toContain('Network error');
      expect(circuitBreakerReason).toContain('Circuit breaker');
      expect(genericReason).toContain('Error');
    });
  });

  describe('Health Monitor Integration', () => {
    test('should provide access to health monitor', () => {
      const healthMonitor = fallbackManager.getHealthMonitor();

      expect(healthMonitor).toBeDefined();
      expect(typeof healthMonitor.getHealthyProviders).toBe('function');
    });
  });

  describe('Error Handling', () => {
    test('should handle provider selection errors', async () => {
      // Mock error in provider selection
      jest.spyOn(fallbackManager as any, 'selectBestFallbackProvider').mockRejectedValue(new Error('Selection error'));

      const error = new Error('Network timeout');
      const decision = await fallbackManager.makeFallbackDecision('openai', error, 0, 'analysis');

      expect(decision.shouldFallback).toBe(false);
      expect(decision.reason).toBe('No suitable fallback provider found');
    });

    test('should handle cost engine errors', async () => {
      // Mock error in cost engine
      const mockCostEngine = require('../services/CostOptimizationEngine').CostOptimizationEngine;
      mockCostEngine.mockImplementation(() => ({
        getCostStatistics: jest.fn().mockImplementation(() => {
          throw new Error('Cost engine error');
        })
      }));

      const manager = new IntelligentFallbackManager();
      const availableProviders = ['google', 'mistral', 'cohere'];
      
      const selectedProvider = await (manager as any).selectCostFirstProvider(availableProviders, { costThreshold: 0.1 });

      expect(selectedProvider).toBeNull();
      
      manager.destroy();
    });

    test('should handle quality manager errors', async () => {
      // Mock error in quality manager
      const mockQualityManager = require('../services/QualityThresholdManager').QualityThresholdManager;
      mockQualityManager.mockImplementation(() => ({
        getQualityStats: jest.fn().mockImplementation(() => {
          throw new Error('Quality manager error');
        })
      }));

      const manager = new IntelligentFallbackManager();
      const availableProviders = ['google', 'mistral', 'cohere'];
      
      const selectedProvider = await (manager as any).selectQualityFirstProvider(availableProviders, { qualityThreshold: 0.7 }, 'analysis');

      expect(selectedProvider).toBeNull();
      
      manager.destroy();
    });
  });
});
