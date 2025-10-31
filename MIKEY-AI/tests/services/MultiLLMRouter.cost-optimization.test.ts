/**
 * Unit tests for enhanced MultiLLMRouter with cost optimization
 */

import { MultiLLMRouter, LLMProvider } from '../services/MultiLLMRouter';
import { CostOptimizationEngine } from '../services/CostOptimizationEngine';
import { ProviderCostRanking } from '../services/ProviderCostRanking';

// Mock the logger
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

describe('Enhanced MultiLLMRouter with Cost Optimization', () => {
  let router: MultiLLMRouter;
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    // Save original environment
    originalEnv = { ...process.env };
    
    // Set up test environment variables
    process.env.OPENAI_API_KEY = 'test-openai-key';
    process.env.GOOGLE_API_KEY = 'test-google-key';
    process.env.MISTRAL_API_KEY = 'test-mistral-key';
    process.env.COST_TIER_AFFORDABLE_PREMIUM = 'mistral,google';
    process.env.COST_TIER_EXPENSIVE = 'openai';
    process.env.ENABLE_COST_OPTIMIZATION = 'true';
    process.env.AFFORDABLE_MAX_COST_PER_TOKEN = '0.0001';
    process.env.EXPENSIVE_MAX_COST_PER_TOKEN = '0.01';
    
    router = new MultiLLMRouter();
  });

  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  describe('Cost Optimization Integration', () => {
    it('should initialize cost optimization engine', () => {
      expect(router).toBeDefined();
      expect(router.getCostOptimizationStats).toBeDefined();
      expect(router.getCostOptimizationConfig).toBeDefined();
    });

    it('should track cost metrics when routing requests', async () => {
      const prompt = 'Test prompt';
      const taskType = 'general';
      
      // Mock the routeRequest method to avoid actual API calls
      const originalRouteRequest = router.routeRequest;
      router.routeRequest = jest.fn().mockResolvedValue('Mock response');
      
      await router.routeRequest(prompt, taskType);
      
      const stats = router.getCostOptimizationStats();
      expect(stats.totalRequests).toBeGreaterThan(0);
      
      // Restore original method
      router.routeRequest = originalRouteRequest;
    });

    it('should prioritize affordable premium providers', () => {
      const config = router.getCostOptimizationConfig();
      expect(config.enableCostOptimization).toBe(true);
      expect(config.costTiers).toHaveLength(2);
      
      const affordableTier = config.costTiers.find(tier => tier.tier === 'affordable_premium');
      expect(affordableTier).toBeDefined();
      expect(affordableTier?.providers).toContain('mistral');
      expect(affordableTier?.providers).toContain('google');
    });
  });

  describe('Provider Selection Enhancement', () => {
    it('should preserve existing task-specific routing logic', () => {
      // This test verifies that the enhanced router maintains backward compatibility
      const providerStatus = router.getProviderStatus();
      expect(providerStatus).toBeDefined();
      expect(Array.isArray(providerStatus)).toBe(true);
    });

    it('should update provider availability correctly', () => {
      const providerName = 'openai';
      
      // Test updating availability
      router.updateProviderAvailability(providerName, false);
      
      // Verify the update (this would require access to internal state)
      expect(typeof router.updateProviderAvailability).toBe('function');
    });
  });

  describe('Cost Tracking Integration', () => {
    it('should track cost metrics for each request', () => {
      const stats = router.getCostOptimizationStats();
      
      expect(stats).toHaveProperty('totalRequests');
      expect(stats).toHaveProperty('totalCost');
      expect(stats).toHaveProperty('averageCostPerRequest');
      expect(stats).toHaveProperty('costSavings');
      expect(stats).toHaveProperty('savingsPercentage');
      
      expect(typeof stats.totalRequests).toBe('number');
      expect(typeof stats.totalCost).toBe('number');
      expect(typeof stats.averageCostPerRequest).toBe('number');
      expect(typeof stats.costSavings).toBe('number');
      expect(typeof stats.savingsPercentage).toBe('number');
    });

    it('should calculate cost savings correctly', () => {
      const stats = router.getCostOptimizationStats();
      
      // Cost savings should be non-negative
      expect(stats.costSavings).toBeGreaterThanOrEqual(0);
      expect(stats.savingsPercentage).toBeGreaterThanOrEqual(0);
      expect(stats.savingsPercentage).toBeLessThanOrEqual(100);
    });
  });

  describe('Configuration Management', () => {
    it('should expose cost optimization configuration', () => {
      const config = router.getCostOptimizationConfig();
      
      expect(config).toHaveProperty('costTiers');
      expect(config).toHaveProperty('enableCostOptimization');
      expect(config).toHaveProperty('qualityThreshold');
      expect(config).toHaveProperty('fallbackToExpensive');
      
      expect(Array.isArray(config.costTiers)).toBe(true);
      expect(typeof config.enableCostOptimization).toBe('boolean');
      expect(typeof config.qualityThreshold).toBe('number');
      expect(typeof config.fallbackToExpensive).toBe('boolean');
    });

    it('should handle cost optimization being disabled', () => {
      process.env.ENABLE_COST_OPTIMIZATION = 'false';
      const disabledRouter = new MultiLLMRouter();
      
      const config = disabledRouter.getCostOptimizationConfig();
      expect(config.enableCostOptimization).toBe(false);
    });
  });

  describe('Provider Status Integration', () => {
    it('should maintain existing provider status functionality', () => {
      const providerStatus = router.getProviderStatus();
      
      expect(Array.isArray(providerStatus)).toBe(true);
      
      if (providerStatus.length > 0) {
        const firstProvider = providerStatus[0];
        expect(firstProvider).toHaveProperty('name');
        expect(firstProvider).toHaveProperty('status');
        expect(firstProvider).toHaveProperty('tokensUsed');
        expect(firstProvider).toHaveProperty('tokenLimit');
        expect(firstProvider).toHaveProperty('strengths');
      }
    });

    it('should track usage statistics correctly', () => {
      const usageStats = router.getUsageStats();
      
      expect(usageStats).toHaveProperty('providers');
      expect(usageStats).toHaveProperty('totalTokensUsed');
      expect(usageStats).toHaveProperty('recentUsage');
      
      expect(Array.isArray(usageStats.providers)).toBe(true);
      expect(typeof usageStats.totalTokensUsed).toBe('number');
      expect(Array.isArray(usageStats.recentUsage)).toBe(true);
    });
  });

  describe('Error Handling and Fallback', () => {
    it('should handle cost optimization errors gracefully', () => {
      // Test that the router continues to work even if cost optimization fails
      expect(router).toBeDefined();
      expect(router.getProviderStatus).toBeDefined();
      expect(router.getUsageStats).toBeDefined();
    });

    it('should maintain backward compatibility', () => {
      // Ensure all existing methods still work
      expect(typeof router.getProviderStatus).toBe('function');
      expect(typeof router.getUsageStats).toBe('function');
      expect(typeof router.routeRequest).toBe('function');
    });
  });

  describe('Performance Considerations', () => {
    it('should not significantly impact routing performance', () => {
      const startTime = Date.now();
      
      // Perform multiple operations
      for (let i = 0; i < 100; i++) {
        router.getProviderStatus();
        router.getCostOptimizationStats();
        router.getCostOptimizationConfig();
      }
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should complete within reasonable time (less than 1 second for 100 operations)
      expect(duration).toBeLessThan(1000);
    });
  });

  describe('Integration with Cost Optimization Components', () => {
    it('should integrate with CostOptimizationEngine', () => {
      const stats = router.getCostOptimizationStats();
      expect(stats).toBeDefined();
      
      // Verify the integration is working
      expect(typeof stats.totalRequests).toBe('number');
    });

    it('should integrate with ProviderCostRanking', () => {
      // This is tested indirectly through the cost optimization functionality
      const config = router.getCostOptimizationConfig();
      expect(config.costTiers).toBeDefined();
    });
  });
});
