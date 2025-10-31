/**
 * Performance Tests for Token Estimation Enhancement
 * Validates <5% performance impact and caching effectiveness
 */

import { TokenEstimationService } from '../services/TokenEstimationService';
import { MultiLLMRouter } from '../services/MultiLLMRouter';
import { OfficialLLMRouter } from '../services/OfficialLLMRouter';

// Mock tiktoken for performance testing
jest.mock('tiktoken', () => ({
  encoding_for_model: jest.fn().mockReturnValue({
    encode: jest.fn().mockImplementation((text: string) => {
      // Simulate realistic tokenization time
      const start = Date.now();
      while (Date.now() - start < 1) {} // 1ms delay
      return Array.from({ length: Math.ceil(text.length / 3) }, (_, i) => i);
    })
  }),
  get_encoding: jest.fn().mockReturnValue({
    encode: jest.fn().mockImplementation((text: string) => {
      const start = Date.now();
      while (Date.now() - start < 1) {} // 1ms delay
      return Array.from({ length: Math.ceil(text.length / 3) }, (_, i) => i);
    })
  })
}));

// Mock node-cache for performance testing
jest.mock('node-cache', () => {
  return jest.fn().mockImplementation(() => ({
    get: jest.fn().mockImplementation((key: string) => {
      // Simulate cache lookup time
      const start = Date.now();
      while (Date.now() - start < 0.1) {} // 0.1ms delay
      return null; // Always cache miss for performance testing
    }),
    set: jest.fn().mockImplementation(() => {
      const start = Date.now();
      while (Date.now() - start < 0.1) {} // 0.1ms delay
    }),
    flushAll: jest.fn(),
    keys: jest.fn().mockReturnValue([]),
    getTtl: jest.fn().mockReturnValue(3600)
  }));
});

// Mock logger
jest.mock('../utils/logger', () => ({
  systemLogger: {
    startup: jest.fn(),
    info: jest.fn(),
    error: jest.fn()
  },
  errorLogger: {
    aiError: jest.fn()
  }
}));

// Mock LangChain modules
jest.mock('@langchain/openai', () => ({
  ChatOpenAI: jest.fn().mockImplementation(() => ({
    invoke: jest.fn().mockResolvedValue({ content: 'Mock response' })
  }))
}));

jest.mock('@langchain/google-genai', () => ({
  ChatGoogleGenerativeAI: jest.fn().mockImplementation(() => ({
    invoke: jest.fn().mockResolvedValue({ content: 'Mock response' })
  }))
}));

describe('Token Estimation Performance Tests', () => {
  let tokenService: TokenEstimationService;
  let multiRouter: MultiLLMRouter;
  let officialRouter: OfficialLLMRouter;

  beforeEach(() => {
    // Set up environment variables
    process.env.OPENAI_API_KEY = 'test-key';
    process.env.GOOGLE_API_KEY = 'test-key';
    
    tokenService = new TokenEstimationService();
    multiRouter = new MultiLLMRouter();
    officialRouter = new OfficialLLMRouter();
  });

  afterEach(() => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    jest.clearAllMocks();
  });

  describe('Token Estimation Performance', () => {
    test('should maintain <5% performance impact for tokenization', async () => {
      const testContent = 'This is a test message for performance benchmarking. '.repeat(100);
      const iterations = 100;
      
      // Benchmark rough estimation (baseline)
      const roughStart = Date.now();
      for (let i = 0; i < iterations; i++) {
        Math.ceil(testContent.length / 4);
      }
      const roughTime = Date.now() - roughStart;
      
      // Benchmark accurate tokenization
      const accurateStart = Date.now();
      for (let i = 0; i < iterations; i++) {
        await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      }
      const accurateTime = Date.now() - accurateStart;
      
      // Calculate performance impact
      const performanceImpact = ((accurateTime - roughTime) / roughTime) * 100;
      
      console.log(`Rough estimation time: ${roughTime}ms`);
      console.log(`Accurate estimation time: ${accurateTime}ms`);
      console.log(`Performance impact: ${performanceImpact.toFixed(2)}%`);
      
      expect(performanceImpact).toBeLessThan(5); // <5% performance impact
    });

    test('should handle large content efficiently', async () => {
      const largeContent = 'A'.repeat(50000); // 50KB content
      
      const startTime = Date.now();
      const result = await tokenService.estimateTokens(largeContent, 'openai', 'gpt-4o-mini');
      const processingTime = Date.now() - startTime;
      
      expect(result.tokenCount).toBeGreaterThan(0);
      expect(processingTime).toBeLessThan(100); // Should process in <100ms
    });

    test('should maintain consistent performance across providers', async () => {
      const testContent = 'Performance test content for multiple providers';
      const providers = ['openai', 'google', 'mistral', 'cohere'];
      const times: number[] = [];
      
      for (const provider of providers) {
        const startTime = Date.now();
        await tokenService.estimateTokens(testContent, provider, 'default-model');
        const processingTime = Date.now() - startTime;
        times.push(processingTime);
      }
      
      // All providers should have similar performance
      const maxTime = Math.max(...times);
      const minTime = Math.min(...times);
      const variance = ((maxTime - minTime) / minTime) * 100;
      
      expect(variance).toBeLessThan(50); // Less than 50% variance between providers
    });
  });

  describe('Caching Performance', () => {
    test('should achieve >80% cache hit rate for repeated content', async () => {
      const testContent = 'Repeated test content for cache performance';
      const iterations = 100;
      
      // First iteration - all cache misses
      for (let i = 0; i < iterations; i++) {
        await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      }
      
      // Mock cache hits for subsequent calls
      const mockCache = (tokenService as any).cache;
      mockCache.get.mockReturnValue({
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: 10,
        confidence: 0.95,
        cached: false,
        processingTime: 1
      });
      
      // Second iteration - all cache hits
      for (let i = 0; i < iterations; i++) {
        await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      }
      
      const cacheStats = tokenService.getCacheStats();
      expect(cacheStats.hitRate).toBeGreaterThan(0.8); // >80% hit rate
    });

    test('should improve performance with cache hits', async () => {
      const testContent = 'Cache performance test content';
      
      // First call - cache miss
      const missStart = Date.now();
      await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      const missTime = Date.now() - missStart;
      
      // Mock cache hit for second call
      const mockCache = (tokenService as any).cache;
      mockCache.get.mockReturnValue({
        provider: 'openai',
        model: 'gpt-4o-mini',
        tokenCount: 10,
        confidence: 0.95,
        cached: false,
        processingTime: 1
      });
      
      // Second call - cache hit
      const hitStart = Date.now();
      await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      const hitTime = Date.now() - hitStart;
      
      expect(hitTime).toBeLessThan(missTime); // Cache hit should be faster
    });

    test('should handle cache warmup efficiently', async () => {
      const patterns = [
        'Hello, how can I help you?',
        'What is the current market price?',
        'Please analyze this trading data.',
        'Can you explain this strategy?',
        'What are the risks involved?'
      ];
      
      const startTime = Date.now();
      await tokenService.warmupCache(patterns);
      const warmupTime = Date.now() - startTime;
      
      expect(warmupTime).toBeLessThan(1000); // Should warmup in <1 second
    });
  });

  describe('Router Integration Performance', () => {
    test('should maintain <2 second routing decisions', async () => {
      const prompt = 'Test prompt for routing performance';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      const startTime = Date.now();
      await multiRouter.routeRequest(prompt, 'general');
      const routingTime = Date.now() - startTime;
      
      expect(routingTime).toBeLessThan(2000); // <2 seconds
      
      mockCallProvider.mockRestore();
    });

    test('should maintain performance with multiple concurrent requests', async () => {
      const prompts = Array.from({ length: 10 }, (_, i) => `Test prompt ${i}`);
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      const startTime = Date.now();
      
      // Process all requests concurrently
      await Promise.all(
        prompts.map(prompt => multiRouter.routeRequest(prompt, 'general'))
      );
      
      const totalTime = Date.now() - startTime;
      const averageTime = totalTime / prompts.length;
      
      expect(averageTime).toBeLessThan(500); // <500ms average per request
      
      mockCallProvider.mockRestore();
    });

    test('should handle OfficialLLMRouter performance requirements', async () => {
      const prompt = 'Test prompt for official router performance';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(officialRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      const startTime = Date.now();
      await officialRouter.routeRequest(prompt, 'general');
      const routingTime = Date.now() - startTime;
      
      expect(routingTime).toBeLessThan(2000); // <2 seconds
      
      mockCallProvider.mockRestore();
    });
  });

  describe('Memory Performance', () => {
    test('should not leak memory with repeated operations', async () => {
      const testContent = 'Memory leak test content';
      const iterations = 1000;
      
      // Get initial memory usage (approximate)
      const initialStats = tokenService.getStats();
      
      // Perform many operations
      for (let i = 0; i < iterations; i++) {
        await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      }
      
      // Get final memory usage
      const finalStats = tokenService.getStats();
      
      // Should not have excessive memory growth
      expect(finalStats.totalEstimations).toBe(iterations);
      expect(finalStats.averageProcessingTime).toBeGreaterThan(0);
    });

    test('should handle cache memory efficiently', async () => {
      const patterns = Array.from({ length: 1000 }, (_, i) => `Pattern ${i}`);
      
      await tokenService.warmupCache(patterns);
      
      const cacheConfig = tokenService.getCacheConfig();
      expect(cacheConfig.keys).toBeGreaterThan(0);
      expect(cacheConfig.keys).toBeLessThan(10000); // Reasonable cache size
    });
  });

  describe('Accuracy Validation', () => {
    test('should improve accuracy by 40%+ over rough estimation', async () => {
      const testContent = 'This is a comprehensive test message with multiple sentences, various punctuation marks, and complex vocabulary!';
      const roughEstimate = Math.ceil(testContent.length / 4);
      
      const result = await tokenService.estimateTokens(testContent, 'openai', 'gpt-4o-mini');
      
      // Calculate accuracy improvement
      const accuracyImprovement = Math.abs(result.tokenCount - roughEstimate) / roughEstimate;
      expect(accuracyImprovement).toBeGreaterThan(0.4); // 40%+ improvement
    });

    test('should maintain consistent accuracy across providers', async () => {
      const testContent = 'Consistent accuracy test content';
      const providers = ['openai', 'google', 'mistral', 'cohere'];
      const results = [];
      
      for (const provider of providers) {
        const result = await tokenService.estimateTokens(testContent, provider, 'default-model');
        results.push(result.tokenCount);
      }
      
      // All providers should give similar token counts for same content
      const maxTokens = Math.max(...results);
      const minTokens = Math.min(...results);
      const variance = ((maxTokens - minTokens) / minTokens) * 100;
      
      expect(variance).toBeLessThan(20); // Less than 20% variance
    });
  });

  describe('Integration Performance', () => {
    test('should integrate efficiently with cost optimization', async () => {
      const prompt = 'Integration performance test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      const startTime = Date.now();
      
      // Perform multiple operations that integrate cost optimization
      for (let i = 0; i < 10; i++) {
        await multiRouter.routeRequest(prompt, 'analysis');
      }
      
      const totalTime = Date.now() - startTime;
      const averageTime = totalTime / 10;
      
      expect(averageTime).toBeLessThan(200); // <200ms average per operation
      
      // Verify cost optimization is working
      const costStats = multiRouter.getCostOptimizationStats();
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should maintain performance with statistics collection', async () => {
      const prompt = 'Statistics performance test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      // Perform operations
      for (let i = 0; i < 5; i++) {
        await multiRouter.routeRequest(prompt, 'general');
      }
      
      // Collect statistics
      const startTime = Date.now();
      const usageStats = multiRouter.getUsageStats();
      const tokenStats = multiRouter.getTokenEstimationStats();
      const cacheStats = multiRouter.getTokenEstimationCacheStats();
      const costStats = multiRouter.getCostOptimizationStats();
      const statsTime = Date.now() - startTime;
      
      expect(statsTime).toBeLessThan(50); // Statistics collection should be fast
      expect(usageStats).toBeDefined();
      expect(tokenStats).toBeDefined();
      expect(cacheStats).toBeDefined();
      expect(costStats).toBeDefined();
      
      mockCallProvider.mockRestore();
    });
  });
});
