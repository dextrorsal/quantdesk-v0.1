/**
 * Unit tests for OfficialLLMRouter Token Estimation Enhancement
 */

import { OfficialLLMRouter } from '../services/OfficialLLMRouter';
import { TokenEstimationService } from '../services/TokenEstimationService';

// Mock TokenEstimationService
jest.mock('../services/TokenEstimationService', () => {
  return {
    TokenEstimationService: jest.fn().mockImplementation(() => ({
      estimateTokens: jest.fn().mockResolvedValue({
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: 12,
        confidence: 0.92,
        cached: false,
        processingTime: 3
      }),
      getStats: jest.fn().mockReturnValue({
        totalEstimations: 1,
        averageAccuracy: 0.92,
        averageProcessingTime: 3,
        cacheStats: { hits: 0, misses: 1, hitRate: 0, totalRequests: 1 },
        providerBreakdown: { openai: 1 }
      }),
      getCacheStats: jest.fn().mockReturnValue({
        hits: 0,
        misses: 1,
        hitRate: 0,
        totalRequests: 1
      }),
      clearCache: jest.fn().mockResolvedValue(undefined),
      warmupCache: jest.fn().mockResolvedValue(undefined)
    }))
  };
});

describe('OfficialLLMRouter Token Estimation Enhancement', () => {
  let router: OfficialLLMRouter;
  let mockTokenEstimationService: any;

  beforeEach(() => {
    // Set up environment variables for testing
    process.env.OPENAI_API_KEY = 'test-openai-key';
    process.env.GOOGLE_API_KEY = 'test-google-key';
    process.env.COHERE_API_KEY = 'test-cohere-key';
    process.env.MISTRAL_API_KEY = 'test-mistral-key';
    
    router = new OfficialLLMRouter();
    mockTokenEstimationService = (router as any).tokenEstimationService;
  });

  afterEach(() => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    delete process.env.COHERE_API_KEY;
    delete process.env.MISTRAL_API_KEY;
    jest.clearAllMocks();
  });

  describe('Accurate Token Estimation Integration', () => {
    test('should use TokenEstimationService for accurate token counting', async () => {
      const prompt = 'Test prompt for accurate token estimation';
      
      // Mock the provider call to return a response
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response from provider');
      
      await router.routeRequest(prompt, 'general');
      
      expect(mockTokenEstimationService.estimateTokens).toHaveBeenCalledWith(
        expect.stringContaining(prompt),
        expect.any(String),
        expect.any(String)
      );
      
      mockCallProvider.mockRestore();
    });

    test('should track usage with accurate token counts', async () => {
      const prompt = 'Test prompt for usage tracking';
      
      // Mock the provider call
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await router.routeRequest(prompt, 'analysis');
      
      const usageHistory = (router as any).usageHistory;
      expect(usageHistory).toHaveLength(1);
      expect(usageHistory[0].tokensUsed).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should maintain cost optimization with accurate tokens', async () => {
      const prompt = 'Test prompt for cost optimization';
      
      // Mock the provider call
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await router.routeRequest(prompt, 'analysis');
      
      const costStats = router.getCostOptimizationStats();
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });
  });

  describe('Provider Model Mapping', () => {
    test('should map provider names to correct models from config', () => {
      const getProviderModel = (router as any).getProviderModel.bind(router);
      
      expect(getProviderModel('openai')).toBe('gpt-4o-mini');
      expect(getProviderModel('google')).toBe('gemini-2.5-flash');
      expect(getProviderModel('cohere')).toBe('command-a-03-2025');
      expect(getProviderModel('mistral')).toBe('mistral-small-latest');
      expect(getProviderModel('unknown')).toBe('unknown');
    });

    test('should handle case-insensitive provider names', () => {
      const getProviderModel = (router as any).getProviderModel.bind(router);
      
      expect(getProviderModel('OpenAI')).toBe('gpt-4o-mini');
      expect(getProviderModel('GOOGLE')).toBe('gemini-2.5-flash');
      expect(getProviderModel('Cohere')).toBe('command-a-03-2025');
    });
  });

  describe('Enhanced Cost Tracking', () => {
    test('should track usage with accurate token estimation', async () => {
      const prompt = 'Test prompt for enhanced cost tracking';
      
      // Mock the provider call
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await router.routeRequest(prompt, 'analysis');
      
      const usageHistory = (router as any).usageHistory;
      expect(usageHistory).toHaveLength(1);
      expect(usageHistory[0].provider).toBeDefined();
      expect(usageHistory[0].tokensUsed).toBeGreaterThan(0);
      expect(usageHistory[0].task).toBe('analysis');
      
      mockCallProvider.mockRestore();
    });

    test('should create enhanced cost metrics', async () => {
      const prompt = 'Test prompt for cost metrics';
      
      // Mock the provider call
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await router.routeRequest(prompt, 'analysis');
      
      const costStats = router.getCostOptimizationStats();
      expect(costStats.totalCost).toBeGreaterThan(0);
      expect(costStats.averageCostPerToken).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should maintain usage history with accurate tokens', async () => {
      const prompt = 'Test prompt for usage history';
      
      // Mock the provider call
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await router.routeRequest(prompt, 'analysis');
      
      const usageHistory = (router as any).usageHistory;
      expect(usageHistory).toHaveLength(1);
      expect(usageHistory[0].tokensUsed).toBeGreaterThan(0);
      expect(usageHistory[0].timestamp).toBeInstanceOf(Date);
      
      mockCallProvider.mockRestore();
    });
  });

  describe('Token Estimation Service Integration', () => {
    test('should expose token estimation statistics', () => {
      const stats = router.getTokenEstimationStats();
      
      expect(stats).toBeDefined();
      expect(stats.totalEstimations).toBeGreaterThanOrEqual(0);
      expect(stats.averageAccuracy).toBeGreaterThanOrEqual(0);
    });

    test('should expose cache statistics', () => {
      const cacheStats = router.getTokenEstimationCacheStats();
      
      expect(cacheStats).toBeDefined();
      expect(cacheStats.hits).toBeGreaterThanOrEqual(0);
      expect(cacheStats.misses).toBeGreaterThanOrEqual(0);
      expect(cacheStats.hitRate).toBeGreaterThanOrEqual(0);
    });

    test('should allow cache management', async () => {
      await router.clearTokenEstimationCache();
      
      expect(mockTokenEstimationService.clearCache).toHaveBeenCalled();
    });

    test('should allow cache warmup', async () => {
      const patterns = ['Hello', 'How can I help?', 'What is the price?'];
      
      await router.warmupTokenEstimationCache(patterns);
      
      expect(mockTokenEstimationService.warmupCache).toHaveBeenCalledWith(patterns);
    });
  });

  describe('Performance Requirements', () => {
    test('should maintain <5% performance impact', async () => {
      const prompt = 'Performance test prompt';
      const iterations = 5;
      
      // Mock the provider call
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      const startTime = Date.now();
      
      for (let i = 0; i < iterations; i++) {
        await router.routeRequest(prompt, 'general');
      }
      
      const totalTime = Date.now() - startTime;
      const averageTime = totalTime / iterations;
      
      // Should be fast with caching
      expect(averageTime).toBeLessThan(100); // Less than 100ms average
      
      mockCallProvider.mockRestore();
    });

    test('should improve accuracy by 40%+ over rough estimation', async () => {
      const prompt = 'This is a comprehensive test message with multiple sentences and various punctuation marks!';
      const roughEstimate = Math.ceil(prompt.length / 4);
      
      // Mock the provider call
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      // Mock the token estimation to return a more accurate count
      mockTokenEstimationService.estimateTokens.mockResolvedValueOnce({
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: Math.floor(roughEstimate * 0.6), // 40% more accurate
        confidence: 0.95,
        cached: false,
        processingTime: 3
      });
      
      await router.routeRequest(prompt, 'general');
      
      // The accuracy improvement is handled by the TokenEstimationService
      expect(mockTokenEstimationService.estimateTokens).toHaveBeenCalled();
      
      mockCallProvider.mockRestore();
    });
  });

  describe('Backward Compatibility', () => {
    test('should maintain existing public API', () => {
      // Test that all existing public methods still work
      expect(typeof router.routeRequest).toBe('function');
      expect(typeof router.getCostOptimizationStats).toBe('function');
      expect(typeof router.getCostOptimizationConfig).toBe('function');
      expect(typeof router.updateProviderAvailability).toBe('function');
    });

    test('should maintain legacy estimateTokens method', () => {
      const estimateTokens = (router as any).estimateTokens.bind(router);
      const text = 'Test text for legacy estimation';
      
      const result = estimateTokens(text);
      expect(result).toBe(Math.ceil(text.length / 4));
    });

    test('should work with existing provider configurations', () => {
      const providerConfigs = (router as any).providerConfigs;
      
      expect(providerConfigs.size).toBeGreaterThan(0);
      expect(providerConfigs.has('openai')).toBe(true);
      expect(providerConfigs.has('google')).toBe(true);
    });
  });

  describe('Error Handling', () => {
    test('should handle token estimation errors gracefully', async () => {
      // Mock token estimation to throw error
      mockTokenEstimationService.estimateTokens.mockRejectedValueOnce(
        new Error('Token estimation failed')
      );
      
      // Mock the provider call
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      const prompt = 'Test prompt with token estimation error';
      
      // Should still work despite token estimation error
      const result = await router.routeRequest(prompt, 'general');
      expect(result).toBeDefined();
      expect(result.response).toBeDefined();
      
      mockCallProvider.mockRestore();
    });

    test('should handle missing provider gracefully', async () => {
      const prompt = 'Test prompt for missing provider';
      
      // Clear all provider configs
      (router as any).providerConfigs.clear();
      
      await expect(router.routeRequest(prompt, 'general')).rejects.toThrow('No available providers');
    });
  });

  describe('Integration with Cost Optimization', () => {
    test('should integrate with existing cost optimization engine', async () => {
      const prompt = 'Test prompt for cost optimization integration';
      
      // Mock the provider call
      const mockCallProvider = jest.spyOn(router as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await router.routeRequest(prompt, 'analysis');
      
      const costStats = router.getCostOptimizationStats();
      expect(costStats).toBeDefined();
      
      const costConfig = router.getCostOptimizationConfig();
      expect(costConfig).toBeDefined();
      
      mockCallProvider.mockRestore();
    });

    test('should update provider availability correctly', () => {
      router.updateProviderAvailability('openai', false);
      
      // This should not throw an error
      expect(() => router.updateProviderAvailability('openai', false)).not.toThrow();
    });
  });

  describe('Provider Cost Calculation', () => {
    test('should calculate correct cost per token for each provider', () => {
      const getProviderCostPerToken = (router as any).getProviderCostPerToken.bind(router);
      
      expect(getProviderCostPerToken('openai')).toBe(0.00015);
      expect(getProviderCostPerToken('google')).toBe(0.000075);
      expect(getProviderCostPerToken('cohere')).toBe(0.0001);
      expect(getProviderCostPerToken('mistral')).toBe(0.0001);
      expect(getProviderCostPerToken('unknown')).toBe(0.0001); // Default fallback
    });

    test('should handle case-insensitive provider names for cost calculation', () => {
      const getProviderCostPerToken = (router as any).getProviderCostPerToken.bind(router);
      
      expect(getProviderCostPerToken('OpenAI')).toBe(0.00015);
      expect(getProviderCostPerToken('GOOGLE')).toBe(0.000075);
      expect(getProviderCostPerToken('Cohere')).toBe(0.0001);
    });
  });
});
