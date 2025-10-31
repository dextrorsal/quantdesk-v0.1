/**
 * Integration Tests for Token Estimation with Cost Tracking
 * Validates integration with existing cost optimization systems
 */

import { TokenEstimationService } from '../services/TokenEstimationService';
import { MultiLLMRouter } from '../services/MultiLLMRouter';
import { OfficialLLMRouter } from '../services/OfficialLLMRouter';
import { CostOptimizationEngine } from '../services/CostOptimizationEngine';

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
    get: jest.fn().mockReturnValue(null),
    set: jest.fn(),
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

describe('Token Estimation Integration with Cost Tracking', () => {
  let tokenService: TokenEstimationService;
  let multiRouter: MultiLLMRouter;
  let officialRouter: OfficialLLMRouter;
  let costEngine: CostOptimizationEngine;

  beforeEach(() => {
    // Set up environment variables
    process.env.OPENAI_API_KEY = 'test-key';
    process.env.GOOGLE_API_KEY = 'test-key';
    
    tokenService = new TokenEstimationService();
    multiRouter = new MultiLLMRouter();
    officialRouter = new OfficialLLMRouter();
    costEngine = new CostOptimizationEngine();
  });

  afterEach(() => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    jest.clearAllMocks();
  });

  describe('Cost Optimization Engine Integration', () => {
    test('should integrate with CostOptimizationEngine for accurate cost tracking', async () => {
      const testContent = 'Integration test content for cost tracking';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      // Perform request
      await multiRouter.routeRequest(testContent, 'analysis');
      
      // Verify cost optimization integration
      const costStats = multiRouter.getCostOptimizationStats();
      expect(costStats.totalCost).toBeGreaterThan(0);
      expect(costStats.averageCostPerToken).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should track accurate token counts in cost metrics', async () => {
      const testContent = 'Accurate token count tracking test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await multiRouter.routeRequest(testContent, 'analysis');
      
      // Verify usage history has accurate token counts
      const usageStats = multiRouter.getUsageStats();
      expect(usageStats.totalTokensUsed).toBeGreaterThan(0);
      expect(usageStats.recentUsage[0].tokensUsed).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should maintain cost optimization with multiple providers', async () => {
      const testContent = 'Multi-provider cost optimization test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      // Use different providers
      await multiRouter.routeRequest(testContent, 'analysis');
      await multiRouter.routeRequest(testContent, 'general');
      
      const costStats = multiRouter.getCostOptimizationStats();
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      const usageStats = multiRouter.getUsageStats();
      expect(usageStats.totalTokensUsed).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });
  });

  describe('MultiLLMRouter Integration', () => {
    test('should enhance MultiLLMRouter with accurate token estimation', async () => {
      const prompt = 'MultiLLMRouter integration test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      const result = await multiRouter.routeRequest(prompt, 'analysis');
      
      expect(result).toBeDefined();
      
      // Verify token estimation integration
      const tokenStats = multiRouter.getTokenEstimationStats();
      expect(tokenStats.totalEstimations).toBeGreaterThan(0);
      
      // Verify cost optimization integration
      const costStats = multiRouter.getCostOptimizationStats();
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should maintain backward compatibility with existing MultiLLMRouter API', async () => {
      const prompt = 'Backward compatibility test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      // Test existing API methods
      const result = await multiRouter.routeRequest(prompt, 'general');
      expect(result).toBeDefined();
      
      const usageStats = multiRouter.getUsageStats();
      expect(usageStats).toBeDefined();
      
      const providerStatus = multiRouter.getProviderStatus();
      expect(Array.isArray(providerStatus)).toBe(true);
      
      mockCallProvider.mockRestore();
    });

    test('should provide enhanced analytics through new methods', async () => {
      const prompt = 'Enhanced analytics test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await multiRouter.routeRequest(prompt, 'analysis');
      
      // Test new analytics methods
      const tokenStats = multiRouter.getTokenEstimationStats();
      expect(tokenStats).toBeDefined();
      
      const cacheStats = multiRouter.getTokenEstimationCacheStats();
      expect(cacheStats).toBeDefined();
      
      const costStats = multiRouter.getCostOptimizationStats();
      expect(costStats).toBeDefined();
      
      mockCallProvider.mockRestore();
    });
  });

  describe('OfficialLLMRouter Integration', () => {
    test('should enhance OfficialLLMRouter with accurate token estimation', async () => {
      const prompt = 'OfficialLLMRouter integration test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(officialRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      const result = await officialRouter.routeRequest(prompt, 'analysis');
      
      expect(result).toBeDefined();
      expect(result.response).toBeDefined();
      expect(result.provider).toBeDefined();
      
      // Verify token estimation integration
      const tokenStats = officialRouter.getTokenEstimationStats();
      expect(tokenStats.totalEstimations).toBeGreaterThan(0);
      
      // Verify cost optimization integration
      const costStats = officialRouter.getCostOptimizationStats();
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should maintain backward compatibility with existing OfficialLLMRouter API', async () => {
      const prompt = 'OfficialLLMRouter backward compatibility test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(officialRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      // Test existing API methods
      const result = await officialRouter.routeRequest(prompt, 'general');
      expect(result).toBeDefined();
      
      const costStats = officialRouter.getCostOptimizationStats();
      expect(costStats).toBeDefined();
      
      const costConfig = officialRouter.getCostOptimizationConfig();
      expect(costConfig).toBeDefined();
      
      mockCallProvider.mockRestore();
    });

    test('should provide enhanced analytics through new methods', async () => {
      const prompt = 'OfficialLLMRouter enhanced analytics test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(officialRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await officialRouter.routeRequest(prompt, 'analysis');
      
      // Test new analytics methods
      const tokenStats = officialRouter.getTokenEstimationStats();
      expect(tokenStats).toBeDefined();
      
      const cacheStats = officialRouter.getTokenEstimationCacheStats();
      expect(cacheStats).toBeDefined();
      
      const costStats = officialRouter.getCostOptimizationStats();
      expect(costStats).toBeDefined();
      
      mockCallProvider.mockRestore();
    });
  });

  describe('End-to-End Integration', () => {
    test('should work end-to-end with accurate token estimation and cost tracking', async () => {
      const prompts = [
        'What is the current market price?',
        'Please analyze this trading data.',
        'Can you explain this strategy?'
      ];
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      // Process multiple requests
      for (const prompt of prompts) {
        await multiRouter.routeRequest(prompt, 'analysis');
      }
      
      // Verify end-to-end integration
      const usageStats = multiRouter.getUsageStats();
      expect(usageStats.totalTokensUsed).toBeGreaterThan(0);
      expect(usageStats.recentUsage).toHaveLength(prompts.length);
      
      const tokenStats = multiRouter.getTokenEstimationStats();
      expect(tokenStats.totalEstimations).toBe(prompts.length);
      
      const costStats = multiRouter.getCostOptimizationStats();
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      const cacheStats = multiRouter.getTokenEstimationCacheStats();
      expect(cacheStats.totalRequests).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should maintain performance with integrated systems', async () => {
      const prompt = 'Performance integration test';
      const iterations = 10;
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      const startTime = Date.now();
      
      // Perform multiple integrated operations
      for (let i = 0; i < iterations; i++) {
        await multiRouter.routeRequest(prompt, 'general');
      }
      
      const totalTime = Date.now() - startTime;
      const averageTime = totalTime / iterations;
      
      expect(averageTime).toBeLessThan(200); // Should be fast with integration
      
      // Verify all systems are working
      const usageStats = multiRouter.getUsageStats();
      const tokenStats = multiRouter.getTokenEstimationStats();
      const costStats = multiRouter.getCostOptimizationStats();
      
      expect(usageStats.totalTokensUsed).toBeGreaterThan(0);
      expect(tokenStats.totalEstimations).toBe(iterations);
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should handle errors gracefully in integrated systems', async () => {
      const prompt = 'Error handling integration test';
      
      // Mock provider calls to throw error
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockRejectedValue(new Error('Provider error'));
      
      // Should handle error gracefully
      const result = await multiRouter.routeRequest(prompt, 'general');
      expect(result).toBeDefined();
      
      // Systems should still be functional
      const usageStats = multiRouter.getUsageStats();
      const tokenStats = multiRouter.getTokenEstimationStats();
      const costStats = multiRouter.getCostOptimizationStats();
      
      expect(usageStats).toBeDefined();
      expect(tokenStats).toBeDefined();
      expect(costStats).toBeDefined();
      
      mockCallProvider.mockRestore();
    });
  });

  describe('Cache Integration', () => {
    test('should integrate caching with cost tracking', async () => {
      const prompt = 'Cache integration test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      // First call - cache miss
      await multiRouter.routeRequest(prompt, 'analysis');
      
      // Second call - should use cache
      await multiRouter.routeRequest(prompt, 'analysis');
      
      const cacheStats = multiRouter.getTokenEstimationCacheStats();
      expect(cacheStats.totalRequests).toBeGreaterThan(0);
      
      // Cost tracking should still work with caching
      const costStats = multiRouter.getCostOptimizationStats();
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });

    test('should allow cache management in integrated systems', async () => {
      const prompt = 'Cache management integration test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await multiRouter.routeRequest(prompt, 'analysis');
      
      // Clear cache
      await multiRouter.clearTokenEstimationCache();
      
      // Verify cache was cleared
      const cacheStats = multiRouter.getTokenEstimationCacheStats();
      expect(cacheStats.hits).toBe(0);
      expect(cacheStats.misses).toBe(0);
      
      mockCallProvider.mockRestore();
    });
  });

  describe('Statistics Integration', () => {
    test('should provide comprehensive statistics across all systems', async () => {
      const prompt = 'Statistics integration test';
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      await multiRouter.routeRequest(prompt, 'analysis');
      
      // Collect statistics from all systems
      const usageStats = multiRouter.getUsageStats();
      const tokenStats = multiRouter.getTokenEstimationStats();
      const cacheStats = multiRouter.getTokenEstimationCacheStats();
      const costStats = multiRouter.getCostOptimizationStats();
      
      // All statistics should be available and consistent
      expect(usageStats.totalTokensUsed).toBeGreaterThan(0);
      expect(tokenStats.totalEstimations).toBeGreaterThan(0);
      expect(cacheStats.totalRequests).toBeGreaterThan(0);
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      // Statistics should be consistent across systems
      expect(usageStats.totalTokensUsed).toBe(tokenStats.totalEstimations);
      
      mockCallProvider.mockRestore();
    });

    test('should maintain statistics accuracy with multiple operations', async () => {
      const prompts = ['Prompt 1', 'Prompt 2', 'Prompt 3'];
      
      // Mock provider calls
      const mockCallProvider = jest.spyOn(multiRouter as any, 'callProvider');
      mockCallProvider.mockResolvedValue('Mock response');
      
      // Perform multiple operations
      for (const prompt of prompts) {
        await multiRouter.routeRequest(prompt, 'analysis');
      }
      
      // Verify statistics accuracy
      const usageStats = multiRouter.getUsageStats();
      const tokenStats = multiRouter.getTokenEstimationStats();
      const cacheStats = multiRouter.getTokenEstimationCacheStats();
      const costStats = multiRouter.getCostOptimizationStats();
      
      expect(usageStats.recentUsage).toHaveLength(prompts.length);
      expect(tokenStats.totalEstimations).toBe(prompts.length);
      expect(cacheStats.totalRequests).toBe(prompts.length);
      expect(costStats.totalCost).toBeGreaterThan(0);
      
      mockCallProvider.mockRestore();
    });
  });
});
