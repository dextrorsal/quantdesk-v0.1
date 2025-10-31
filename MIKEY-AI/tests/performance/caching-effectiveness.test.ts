/**
 * Caching Effectiveness Tests
 * Validates >80% cache hit rate and performance improvements
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

// Mock node-cache with realistic behavior
jest.mock('node-cache', () => {
  const cache = new Map();
  return jest.fn().mockImplementation(() => ({
    get: jest.fn().mockImplementation((key: string) => {
      return cache.get(key) || null;
    }),
    set: jest.fn().mockImplementation((key: string, value: any) => {
      cache.set(key, value);
    }),
    flushAll: jest.fn().mockImplementation(() => {
      cache.clear();
    }),
    keys: jest.fn().mockImplementation(() => {
      return Array.from(cache.keys());
    }),
    getTtl: jest.fn().mockReturnValue(3600)
  }));
});

describe('Token Estimation Caching Effectiveness', () => {
  let tokenService: TokenEstimationService;

  beforeEach(() => {
    tokenService = new TokenEstimationService(3600); // 1 hour TTL
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Cache Hit Rate Requirements', () => {
    test('should achieve >80% cache hit rate for repeated content', async () => {
      const testContent = 'Repeated test content for cache effectiveness';
      const iterations = 100;
      
      // First iteration - all cache misses
      for (let i = 0; i < iterations; i++) {
        await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      }
      
      // Second iteration - should be cache hits
      for (let i = 0; i < iterations; i++) {
        await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      }
      
      const cacheStats = tokenService.getCacheStats();
      expect(cacheStats.hitRate).toBeGreaterThan(0.8); // >80% hit rate
      expect(cacheStats.hits).toBe(iterations);
      expect(cacheStats.misses).toBe(iterations);
    });

    test('should maintain high cache hit rate with mixed content', async () => {
      const repeatedContent = 'This content will be repeated many times';
      const uniqueContent = 'This content is unique each time';
      const iterations = 50;
      
      // Mix of repeated and unique content
      for (let i = 0; i < iterations; i++) {
        // 80% repeated content
        if (i % 5 !== 0) {
          await tokenService.estimateTokens(repeatedContent, 'openai', 'gpt-4o-mini');
        } else {
          await tokenService.estimateTokens(`${uniqueContent} ${i}`, 'openai', 'gpt-4o-mini');
        }
      }
      
      const cacheStats = tokenService.getCacheStats();
      expect(cacheStats.hitRate).toBeGreaterThan(0.6); // Should still have good hit rate
    });

    test('should handle cache invalidation correctly', async () => {
      const testContent = 'Cache invalidation test content';
      
      // First call - cache miss
      await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      
      // Second call - cache hit
      await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      
      // Clear cache
      await tokenService.clearCache();
      
      // Third call - should be cache miss again
      await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      
      const cacheStats = tokenService.getCacheStats();
      expect(cacheStats.hits).toBe(1);
      expect(cacheStats.misses).toBe(2);
    });
  });

  describe('Cache Performance Benefits', () => {
    test('should improve performance with cache hits', async () => {
      const testContent = 'Cache performance test content';
      
      // First call - cache miss (slower)
      const missStart = Date.now();
      await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      const missTime = Date.now() - missStart;
      
      // Second call - cache hit (faster)
      const hitStart = Date.now();
      await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      const hitTime = Date.now() - hitStart;
      
      // Cache hit should be significantly faster
      expect(hitTime).toBeLessThan(missTime);
      
      const cacheStats = tokenService.getCacheStats();
      expect(cacheStats.hits).toBe(1);
      expect(cacheStats.misses).toBe(1);
    });

    test('should maintain performance benefits with multiple providers', async () => {
      const testContent = 'Multi-provider cache test';
      const providers = ['openai', 'google', 'mistral', 'cohere'];
      
      // First iteration - cache misses
      const missStart = Date.now();
      for (const provider of providers) {
        await tokenService.estimateTokens(testContent, provider, 'default-model');
      }
      const missTime = Date.now() - missStart;
      
      // Second iteration - cache hits
      const hitStart = Date.now();
      for (const provider of providers) {
        await tokenService.estimateTokens(testContent, provider, 'default-model');
      }
      const hitTime = Date.now() - hitStart;
      
      expect(hitTime).toBeLessThan(missTime);
      
      const cacheStats = tokenService.getCacheStats();
      expect(cacheStats.hits).toBe(providers.length);
      expect(cacheStats.misses).toBe(providers.length);
    });
  });

  describe('Cache Memory Management', () => {
    test('should handle cache size efficiently', async () => {
      const patterns = Array.from({ length: 100 }, (_, i) => `Pattern ${i}`);
      
      // Warm up cache with many patterns
      await tokenService.warmupCache(patterns);
      
      const cacheConfig = tokenService.getCacheConfig();
      expect(cacheConfig.keys).toBeGreaterThan(0);
      expect(cacheConfig.keys).toBeLessThan(10000); // Reasonable cache size
    });

    test('should not leak memory with repeated cache operations', async () => {
      const testContent = 'Memory leak test content';
      const iterations = 1000;
      
      // Perform many cache operations
      for (let i = 0; i < iterations; i++) {
        await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      }
      
      const cacheStats = tokenService.getCacheStats();
      expect(cacheStats.totalRequests).toBe(iterations);
      
      // Should still be able to get stats without memory issues
      const stats = tokenService.getStats();
      expect(stats.totalEstimations).toBe(iterations);
    });
  });

  describe('Cache Key Generation', () => {
    test('should generate consistent cache keys', async () => {
      const testContent = 'Consistent cache key test';
      const provider = 'openai';
      const model = 'gpt-4o-mini';
      
      // Generate cache key multiple times
      const key1 = (tokenService as any).generateCacheKey(testContent, provider, model);
      const key2 = (tokenService as any).generateCacheKey(testContent, provider, model);
      
      expect(key1).toBe(key2);
    });

    test('should generate different cache keys for different content', async () => {
      const content1 = 'First test content';
      const content2 = 'Second test content';
      const provider = 'openai';
      const model = 'gpt-4o-mini';
      
      const key1 = (tokenService as any).generateCacheKey(content1, provider, model);
      const key2 = (tokenService as any).generateCacheKey(content2, provider, model);
      
      expect(key1).not.toBe(key2);
    });

    test('should generate different cache keys for different providers', async () => {
      const testContent = 'Provider-specific cache test';
      const model = 'gpt-4o-mini';
      
      const key1 = (tokenService as any).generateCacheKey(testContent, 'openai', model);
      const key2 = (tokenService as any).generateCacheKey(testContent, 'google', model);
      
      expect(key1).not.toBe(key2);
    });
  });

  describe('Cache Warmup Effectiveness', () => {
    test('should warm up cache with common patterns', async () => {
      const commonPatterns = [
        'Hello, how can I help you?',
        'What is the current market price?',
        'Please analyze this trading data.',
        'Can you explain this strategy?',
        'What are the risks involved?'
      ];
      
      await tokenService.warmupCache(commonPatterns);
      
      const cacheStats = tokenService.getCacheStats();
      expect(cacheStats.totalRequests).toBeGreaterThan(0);
      
      // Test that warmed up patterns are cached
      const result = await tokenService.estimateTokens(commonPatterns[0], 'openai', 'gpt-4o-mini');
      expect(result.cached).toBe(true);
    });

    test('should improve performance after cache warmup', async () => {
      const patterns = ['Pattern 1', 'Pattern 2', 'Pattern 3'];
      
      // Warm up cache
      await tokenService.warmupCache(patterns);
      
      // Test performance with warmed up patterns
      const startTime = Date.now();
      for (const pattern of patterns) {
        await tokenService.estimateTokens(pattern, 'openai', 'gpt-4o-mini');
      }
      const warmupTime = Date.now() - startTime;
      
      // Should be fast due to cache hits
      expect(warmupTime).toBeLessThan(100);
    });
  });

  describe('Cache Statistics Accuracy', () => {
    test('should track cache statistics accurately', async () => {
      const testContent = 'Statistics accuracy test';
      
      // First call - cache miss
      await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      
      let cacheStats = tokenService.getCacheStats();
      expect(cacheStats.hits).toBe(0);
      expect(cacheStats.misses).toBe(1);
      expect(cacheStats.totalRequests).toBe(1);
      expect(cacheStats.hitRate).toBe(0);
      
      // Second call - cache hit
      await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      
      cacheStats = tokenService.getCacheStats();
      expect(cacheStats.hits).toBe(1);
      expect(cacheStats.misses).toBe(1);
      expect(cacheStats.totalRequests).toBe(2);
      expect(cacheStats.hitRate).toBe(0.5);
    });

    test('should maintain accurate statistics with mixed operations', async () => {
      const repeatedContent = 'Repeated content';
      const uniqueContent = 'Unique content';
      
      // Mix of operations
      await tokenService.estimateTokens(repeatedContent, 'openai', 'gpt-4o-mini'); // Miss
      await tokenService.estimateTokens(uniqueContent, 'openai', 'gpt-4o-mini'); // Miss
      await tokenService.estimateTokens(repeatedContent, 'openai', 'gpt-4o-mini'); // Hit
      await tokenService.estimateTokens(repeatedContent, 'google', 'gemini-2.0-flash-exp'); // Miss (different provider)
      await tokenService.estimateTokens(repeatedContent, 'google', 'gemini-2.0-flash-exp'); // Hit
      
      const cacheStats = tokenService.getCacheStats();
      expect(cacheStats.hits).toBe(2);
      expect(cacheStats.misses).toBe(3);
      expect(cacheStats.totalRequests).toBe(5);
      expect(cacheStats.hitRate).toBe(0.4);
    });
  });
});
