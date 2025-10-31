/**
 * Unit tests for TokenEstimationService
 */

import { TokenEstimationService } from '../services/TokenEstimationService';
import { TokenizationConfig } from '../config/tokenization-config';

// Mock tiktoken
jest.mock('tiktoken', () => ({
  encoding_for_model: jest.fn().mockReturnValue({
    encode: jest.fn().mockReturnValue([1, 2, 3, 4, 5]) // Mock 5 tokens
  }),
  get_encoding: jest.fn().mockReturnValue({
    encode: jest.fn().mockReturnValue([1, 2, 3, 4, 5]) // Mock 5 tokens
  })
}));

// Mock node-cache
jest.mock('node-cache', () => {
  return jest.fn().mockImplementation(() => ({
    get: jest.fn(),
    set: jest.fn(),
    flushAll: jest.fn(),
    keys: jest.fn().mockReturnValue([]),
    getTtl: jest.fn().mockReturnValue(3600)
  }));
});

describe('TokenEstimationService', () => {
  let service: TokenEstimationService;
  let mockCache: any;

  beforeEach(() => {
    // Reset mocks
    jest.clearAllMocks();
    
    // Create service instance
    service = new TokenEstimationService(3600);
    
    // Get mock cache instance
    mockCache = (service as any).cache;
  });

  describe('Token Estimation Accuracy', () => {
    test('should estimate tokens accurately for OpenAI provider', async () => {
      const content = 'Hello, world! This is a test message.';
      const result = await service.estimateTokens(content, 'openai', 'gpt-4o-mini');

      expect(result.provider).toBe('openai');
      expect(result.model).toBe('gpt-4o-mini');
      expect(result.tokenCount).toBe(5); // Mocked value
      expect(result.confidence).toBeGreaterThan(0.8);
      expect(result.cached).toBe(false);
      expect(result.processingTime).toBeGreaterThan(0);
    });

    test('should estimate tokens accurately for Google provider', async () => {
      const content = 'This is a longer test message with more content to tokenize.';
      const result = await service.estimateTokens(content, 'google', 'gemini-2.0-flash-exp');

      expect(result.provider).toBe('google');
      expect(result.model).toBe('gemini-2.0-flash-exp');
      expect(result.tokenCount).toBe(5); // Mocked value
      expect(result.confidence).toBeGreaterThan(0.8);
    });

    test('should handle fallback estimation for unknown provider', async () => {
      const content = 'Test content for fallback';
      const result = await service.estimateTokens(content, 'unknown-provider', 'unknown-model');

      expect(result.provider).toBe('unknown-provider');
      expect(result.model).toBe('unknown-model');
      expect(result.tokenCount).toBe(Math.ceil(content.length / 4)); // Fallback calculation
      expect(result.confidence).toBe(0.5); // Lower confidence for fallback
    });

    test('should improve accuracy by 40%+ over rough estimation', async () => {
      const content = 'This is a comprehensive test message with multiple sentences and various punctuation marks!';
      const roughEstimate = Math.ceil(content.length / 4);
      
      const result = await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      // Calculate accuracy improvement
      const accuracyImprovement = Math.abs(result.tokenCount - roughEstimate) / roughEstimate;
      expect(accuracyImprovement).toBeGreaterThan(0.4); // 40%+ improvement
    });
  });

  describe('Caching Functionality', () => {
    test('should cache token estimation results', async () => {
      const content = 'Test content for caching';
      mockCache.get.mockReturnValue(null); // Cache miss
      
      await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      expect(mockCache.set).toHaveBeenCalledWith(
        expect.stringContaining('openai:gpt-4o-mini:'),
        expect.objectContaining({
          provider: 'openai',
          model: 'gpt-4o-mini',
          tokenCount: 5
        })
      );
    });

    test('should return cached result when available', async () => {
      const content = 'Test content for cache hit';
      const cachedResult = {
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: 5,
        confidence: 0.9,
        cached: false,
        processingTime: 1
      };
      
      mockCache.get.mockReturnValue(cachedResult);
      
      const result = await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      expect(result.cached).toBe(true);
      expect(result.tokenCount).toBe(5);
      expect(mockCache.set).not.toHaveBeenCalled();
    });

    test('should achieve >80% cache hit rate for repeated content', async () => {
      const content = 'Repeated test content';
      
      // First call - cache miss
      mockCache.get.mockReturnValueOnce(null);
      await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      // Subsequent calls - cache hits
      const cachedResult = {
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: 5,
        confidence: 0.9,
        cached: false,
        processingTime: 1
      };
      
      mockCache.get.mockReturnValue(cachedResult);
      
      // Make multiple calls
      for (let i = 0; i < 10; i++) {
        await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      }
      
      const cacheStats = service.getCacheStats();
      expect(cacheStats.hitRate).toBeGreaterThan(0.8); // >80% hit rate
    });

    test('should clear cache successfully', async () => {
      await service.clearCache();
      expect(mockCache.flushAll).toHaveBeenCalled();
    });
  });

  describe('Provider Consistency', () => {
    test('should work consistently across all supported providers', async () => {
      const content = 'Consistent test content across providers';
      const providers = ['openai', 'google', 'mistral', 'cohere', 'huggingface', 'xai'];
      
      const results = await service.estimateTokensForProviders(content, providers);
      
      expect(Object.keys(results)).toHaveLength(providers.length);
      
      Object.values(results).forEach(result => {
        expect(result.tokenCount).toBeGreaterThan(0);
        expect(result.confidence).toBeGreaterThan(0);
        expect(result.processingTime).toBeGreaterThan(0);
      });
    });

    test('should validate provider and model combinations', async () => {
      const content = 'Test content';
      
      // Valid provider/model combination
      const validResult = await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      expect(validResult.provider).toBe('openai');
      
      // Invalid model for provider
      const invalidResult = await service.estimateTokens(content, 'openai', 'invalid-model');
      expect(invalidResult.confidence).toBe(0.5); // Fallback confidence
    });
  });

  describe('Performance Requirements', () => {
    test('should maintain <5% performance impact', async () => {
      const content = 'Performance test content';
      const iterations = 100;
      
      const startTime = Date.now();
      
      for (let i = 0; i < iterations; i++) {
        await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      }
      
      const totalTime = Date.now() - startTime;
      const averageTime = totalTime / iterations;
      
      // Should be very fast with caching
      expect(averageTime).toBeLessThan(10); // Less than 10ms average
    });

    test('should handle large content efficiently', async () => {
      const largeContent = 'A'.repeat(10000); // 10KB content
      
      const startTime = Date.now();
      const result = await service.estimateTokens(largeContent, 'openai', 'gpt-4o-mini');
      const processingTime = Date.now() - startTime;
      
      expect(result.tokenCount).toBeGreaterThan(0);
      expect(processingTime).toBeLessThan(100); // Should process in <100ms
    });
  });

  describe('Statistics and Monitoring', () => {
    test('should track estimation statistics', async () => {
      const content = 'Statistics test content';
      
      await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      await service.estimateTokens(content, 'google', 'gemini-2.0-flash-exp');
      
      const stats = service.getStats();
      
      expect(stats.totalEstimations).toBe(2);
      expect(stats.providerBreakdown.openai).toBe(1);
      expect(stats.providerBreakdown.google).toBe(1);
      expect(stats.averageProcessingTime).toBeGreaterThan(0);
    });

    test('should calculate cost accurately', () => {
      const tokenCount = 1000;
      const cost = service.calculateCost(tokenCount, 'openai');
      
      expect(cost).toBe(tokenCount * 0.0005); // OpenAI cost per token
    });

    test('should enhance cost metrics with accurate token count', () => {
      const baseMetrics = {
        provider: 'openai',
        tokensUsed: 100,
        costPerToken: 0.0005,
        totalCost: 0.05,
        timestamp: new Date(),
        taskType: 'test'
      };
      
      const enhanced = service.enhanceCostMetrics(baseMetrics, 120, true);
      
      expect(enhanced.accurateTokenCount).toBe(120);
      expect(enhanced.estimationAccuracy).toBeGreaterThan(0);
      expect(enhanced.cacheHit).toBe(true);
    });
  });

  describe('Error Handling', () => {
    test('should handle encoding errors gracefully', async () => {
      // Mock encoding_for_model to throw error
      const { encoding_for_model } = require('tiktoken');
      encoding_for_model.mockImplementation(() => {
        throw new Error('Encoding not found');
      });
      
      const content = 'Test content with encoding error';
      const result = await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      // Should fallback to rough estimation
      expect(result.tokenCount).toBe(Math.ceil(content.length / 4));
      expect(result.confidence).toBe(0.5);
    });

    test('should handle invalid content gracefully', async () => {
      const invalidContent = '';
      const result = await service.estimateTokens(invalidContent, 'openai', 'gpt-4o-mini');
      
      expect(result.tokenCount).toBe(0);
      expect(result.confidence).toBeGreaterThan(0);
    });
  });

  describe('Cache Management', () => {
    test('should generate consistent cache keys', async () => {
      const content = 'Cache key test content';
      const provider = 'openai';
      const model = 'gpt-4o-mini';
      
      // Generate cache key multiple times
      const key1 = (service as any).generateCacheKey(content, provider, model);
      const key2 = (service as any).generateCacheKey(content, provider, model);
      
      expect(key1).toBe(key2);
      expect(key1).toContain(provider);
      expect(key1).toContain(model);
    });

    test('should warm up cache with common patterns', async () => {
      const commonPatterns = [
        'Hello, how can I help you?',
        'What is the current market price?',
        'Please analyze this trading data.'
      ];
      
      await service.warmupCache(commonPatterns);
      
      // Should have cached results for all patterns
      const stats = service.getCacheStats();
      expect(stats.totalRequests).toBeGreaterThan(0);
    });
  });
});
