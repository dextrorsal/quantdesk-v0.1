/**
 * Unit tests for Token Estimation Caching
 */

import { TokenEstimationService } from '../services/TokenEstimationService';

// Mock tiktoken
jest.mock('tiktoken', () => ({
  encoding_for_model: jest.fn().mockReturnValue({
    encode: jest.fn().mockReturnValue([1, 2, 3, 4, 5])
  }),
  get_encoding: jest.fn().mockReturnValue({
    encode: jest.fn().mockReturnValue([1, 2, 3, 4, 5])
  })
}));

// Mock node-cache
jest.mock('node-cache', () => {
  return jest.fn().mockImplementation(() => ({
    get: jest.fn(),
    set: jest.fn(),
    flushAll: jest.fn(),
    keys: jest.fn().mockReturnValue(['key1', 'key2']),
    getTtl: jest.fn().mockReturnValue(3600)
  }));
});

describe('Token Estimation Caching', () => {
  let service: TokenEstimationService;
  let mockCache: any;

  beforeEach(() => {
    jest.clearAllMocks();
    service = new TokenEstimationService(3600);
    mockCache = (service as any).cache;
  });

  describe('Cache Key Generation', () => {
    test('should generate consistent cache keys for same content', () => {
      const content = 'Test content for cache key generation';
      const provider = 'openai';
      const model = 'gpt-4o-mini';
      
      const key1 = (service as any).generateCacheKey(content, provider, model);
      const key2 = (service as any).generateCacheKey(content, provider, model);
      
      expect(key1).toBe(key2);
      expect(key1).toMatch(/^openai:gpt-4o-mini:[a-z0-9]+$/);
    });

    test('should generate different cache keys for different content', () => {
      const content1 = 'First test content';
      const content2 = 'Second test content';
      const provider = 'openai';
      const model = 'gpt-4o-mini';
      
      const key1 = (service as any).generateCacheKey(content1, provider, model);
      const key2 = (service as any).generateCacheKey(content2, provider, model);
      
      expect(key1).not.toBe(key2);
    });

    test('should generate different cache keys for different providers', () => {
      const content = 'Same content, different providers';
      const model = 'gpt-4o-mini';
      
      const key1 = (service as any).generateCacheKey(content, 'openai', model);
      const key2 = (service as any).generateCacheKey(content, 'google', model);
      
      expect(key1).not.toBe(key2);
    });

    test('should generate different cache keys for different models', () => {
      const content = 'Same content, different models';
      const provider = 'openai';
      
      const key1 = (service as any).generateCacheKey(content, provider, 'gpt-4o-mini');
      const key2 = (service as any).generateCacheKey(content, provider, 'gpt-4o');
      
      expect(key1).not.toBe(key2);
    });
  });

  describe('Cache Hit/Miss Behavior', () => {
    test('should return cached result on cache hit', async () => {
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
      expect(result.processingTime).toBeGreaterThan(0);
      expect(mockCache.set).not.toHaveBeenCalled();
    });

    test('should cache result on cache miss', async () => {
      const content = 'Test content for cache miss';
      mockCache.get.mockReturnValue(null);
      
      const result = await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      expect(result.cached).toBe(false);
      expect(mockCache.set).toHaveBeenCalledWith(
        expect.stringContaining('openai:gpt-4o-mini:'),
        expect.objectContaining({
          provider: 'openai',
          model: 'gpt-4o-mini',
          tokenCount: 5
        })
      );
    });

    test('should handle cache miss gracefully', async () => {
      const content = 'Test content for cache miss handling';
      mockCache.get.mockReturnValue(null);
      
      const result = await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      expect(result).toBeDefined();
      expect(result.provider).toBe('openai');
      expect(result.model).toBe('gpt-4o-mini');
      expect(result.tokenCount).toBeGreaterThan(0);
    });
  });

  describe('Cache Statistics', () => {
    test('should track cache hits and misses', async () => {
      const content = 'Test content for cache statistics';
      
      // First call - cache miss
      mockCache.get.mockReturnValueOnce(null);
      await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      // Second call - cache hit
      const cachedResult = {
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: 5,
        confidence: 0.9,
        cached: false,
        processingTime: 1
      };
      mockCache.get.mockReturnValue(cachedResult);
      await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      const cacheStats = service.getCacheStats();
      expect(cacheStats.hits).toBe(1);
      expect(cacheStats.misses).toBe(1);
      expect(cacheStats.totalRequests).toBe(2);
      expect(cacheStats.hitRate).toBe(0.5);
    });

    test('should calculate hit rate correctly', async () => {
      const content = 'Test content for hit rate calculation';
      
      // 3 cache misses
      mockCache.get.mockReturnValue(null);
      await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      await service.estimateTokens(content, 'google', 'gemini-2.0-flash-exp');
      await service.estimateTokens(content, 'mistral', 'mistral-large');
      
      // 7 cache hits
      const cachedResult = {
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: 5,
        confidence: 0.9,
        cached: false,
        processingTime: 1
      };
      mockCache.get.mockReturnValue(cachedResult);
      
      for (let i = 0; i < 7; i++) {
        await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      }
      
      const cacheStats = service.getCacheStats();
      expect(cacheStats.hits).toBe(7);
      expect(cacheStats.misses).toBe(3);
      expect(cacheStats.totalRequests).toBe(10);
      expect(cacheStats.hitRate).toBe(0.7); // 70% hit rate
    });
  });

  describe('Cache Management', () => {
    test('should clear cache successfully', async () => {
      await service.clearCache();
      
      expect(mockCache.flushAll).toHaveBeenCalled();
      
      const cacheStats = service.getCacheStats();
      expect(cacheStats.hits).toBe(0);
      expect(cacheStats.misses).toBe(0);
      expect(cacheStats.totalRequests).toBe(0);
      expect(cacheStats.hitRate).toBe(0);
    });

    test('should get cache configuration', () => {
      const config = service.getCacheConfig();
      
      expect(config.ttl).toBe(3600);
      expect(config.keys).toBe(2); // Mocked value
    });

    test('should warm up cache with common patterns', async () => {
      const commonPatterns = [
        'Hello, how can I help you?',
        'What is the current market price?',
        'Please analyze this trading data.'
      ];
      
      mockCache.get.mockReturnValue(null); // All cache misses for warmup
      
      await service.warmupCache(commonPatterns);
      
      // Should have attempted to cache results for all patterns and providers
      expect(mockCache.set).toHaveBeenCalled();
    });
  });

  describe('Cache Performance', () => {
    test('should improve performance with cache hits', async () => {
      const content = 'Performance test content';
      
      // First call - cache miss (slower)
      mockCache.get.mockReturnValueOnce(null);
      const startTime1 = Date.now();
      await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      const time1 = Date.now() - startTime1;
      
      // Second call - cache hit (faster)
      const cachedResult = {
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: 5,
        confidence: 0.9,
        cached: false,
        processingTime: 1
      };
      mockCache.get.mockReturnValue(cachedResult);
      const startTime2 = Date.now();
      await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      const time2 = Date.now() - startTime2;
      
      // Cache hit should be faster (though both are mocked to be fast)
      expect(time2).toBeLessThanOrEqual(time1);
    });

    test('should maintain cache effectiveness over time', async () => {
      const content = 'Long-term cache test content';
      
      // Simulate multiple cache hits over time
      const cachedResult = {
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: 5,
        confidence: 0.9,
        cached: false,
        processingTime: 1
      };
      mockCache.get.mockReturnValue(cachedResult);
      
      // Make many requests
      for (let i = 0; i < 100; i++) {
        await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      }
      
      const cacheStats = service.getCacheStats();
      expect(cacheStats.hitRate).toBe(1.0); // 100% hit rate
      expect(cacheStats.hits).toBe(100);
      expect(cacheStats.misses).toBe(0);
    });
  });

  describe('Cache Invalidation', () => {
    test('should handle cache expiration gracefully', async () => {
      const content = 'Cache expiration test content';
      
      // Simulate expired cache entry
      mockCache.get.mockReturnValue(null);
      
      const result = await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      expect(result.cached).toBe(false);
      expect(mockCache.set).toHaveBeenCalled();
    });

    test('should handle cache errors gracefully', async () => {
      const content = 'Cache error test content';
      
      // Simulate cache error
      mockCache.get.mockImplementation(() => {
        throw new Error('Cache error');
      });
      
      // Should still work despite cache error
      const result = await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      
      expect(result).toBeDefined();
      expect(result.provider).toBe('openai');
      expect(result.tokenCount).toBeGreaterThan(0);
    });
  });

  describe('Cache Memory Management', () => {
    test('should not leak memory with repeated cache operations', async () => {
      const content = 'Memory leak test content';
      
      // Simulate many cache operations
      for (let i = 0; i < 1000; i++) {
        mockCache.get.mockReturnValue(null);
        await service.estimateTokens(content, 'openai', 'gpt-4o-mini');
      }
      
      const cacheStats = service.getCacheStats();
      expect(cacheStats.totalRequests).toBe(1000);
      
      // Should still be able to get stats without memory issues
      const stats = service.getStats();
      expect(stats.totalEstimations).toBe(1000);
    });
  });
});
