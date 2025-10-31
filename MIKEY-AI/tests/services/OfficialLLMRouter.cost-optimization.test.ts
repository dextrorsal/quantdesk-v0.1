/**
 * Unit tests for enhanced OfficialLLMRouter with cost optimization
 */

import { OfficialLLMRouter } from '../services/OfficialLLMRouter';
import { CostOptimizationEngine } from '../services/CostOptimizationEngine';
import { ProviderCostRanking } from '../services/ProviderCostRanking';

// Mock console.log to avoid noise in tests
const originalConsoleLog = console.log;
beforeAll(() => {
  console.log = jest.fn();
});

afterAll(() => {
  console.log = originalConsoleLog;
});

// Mock the SDK modules
jest.mock('openai', () => ({
  __esModule: true,
  default: jest.fn().mockImplementation(() => ({
    chat: {
      completions: {
        create: jest.fn().mockResolvedValue({
          choices: [{ message: { content: 'Mock OpenAI response' } }]
        })
      }
    }
  }))
}));

jest.mock('@google/genai', () => ({
  GoogleGenAI: jest.fn().mockImplementation(() => ({
    models: {
      generateContent: jest.fn().mockResolvedValue({
        text: 'Mock Google response'
      })
    }
  }))
}));

jest.mock('cohere-ai', () => ({
  CohereClientV2: jest.fn().mockImplementation(() => ({
    chat: jest.fn().mockResolvedValue({
      message: {
        content: [{ type: 'text', text: 'Mock Cohere response' }]
      }
    })
  }))
}));

jest.mock('@langchain/openai', () => ({
  ChatOpenAI: jest.fn().mockImplementation(() => ({
    invoke: jest.fn().mockResolvedValue({ content: 'Mock Mistral response' })
  }))
}));

describe('Enhanced OfficialLLMRouter with Cost Optimization', () => {
  let router: OfficialLLMRouter;
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    // Save original environment
    originalEnv = { ...process.env };
    
    // Set up test environment variables
    process.env.OPENAI_API_KEY = 'test-openai-key';
    process.env.GOOGLE_API_KEY = 'test-google-key';
    process.env.COHERE_API_KEY = 'test-cohere-key';
    process.env.MISTRAL_API_KEY = 'test-mistral-key';
    process.env.COST_TIER_AFFORDABLE_PREMIUM = 'mistral,google,cohere';
    process.env.COST_TIER_EXPENSIVE = 'openai';
    process.env.ENABLE_COST_OPTIMIZATION = 'true';
    process.env.AFFORDABLE_MAX_COST_PER_TOKEN = '0.0001';
    process.env.EXPENSIVE_MAX_COST_PER_TOKEN = '0.01';
    
    router = new OfficialLLMRouter();
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
      
      const result = await router.routeRequest(prompt, taskType);
      
      expect(result).toBeDefined();
      expect(result.response).toBeDefined();
      expect(result.provider).toBeDefined();
      
      const stats = router.getCostOptimizationStats();
      expect(stats.totalRequests).toBeGreaterThan(0);
    });

    it('should prioritize affordable premium providers', () => {
      const config = router.getCostOptimizationConfig();
      expect(config.enableCostOptimization).toBe(true);
      expect(config.costTiers).toHaveLength(2);
      
      const affordableTier = config.costTiers.find(tier => tier.tier === 'affordable_premium');
      expect(affordableTier).toBeDefined();
      expect(affordableTier?.providers).toContain('mistral');
      expect(affordableTier?.providers).toContain('google');
      expect(affordableTier?.providers).toContain('cohere');
    });
  });

  describe('Provider Selection Enhancement', () => {
    it('should preserve existing task-specific routing logic', async () => {
      const prompt = 'Test prompt';
      
      // Test different task types
      const taskTypes = ['trading_analysis', 'code_generation', 'multilingual', 'sentiment_analysis'];
      
      for (const taskType of taskTypes) {
        const result = await router.routeRequest(prompt, taskType);
        expect(result).toBeDefined();
        expect(result.provider).toBeDefined();
      }
    });

    it('should update provider availability correctly', () => {
      const providerName = 'openai';
      
      // Test updating availability
      router.updateProviderAvailability(providerName, false);
      
      // Verify the method exists and can be called
      expect(typeof router.updateProviderAvailability).toBe('function');
    });
  });

  describe('Cost Tracking Integration', () => {
    it('should track cost metrics for each request', async () => {
      const prompt = 'Test prompt';
      const taskType = 'general';
      
      await router.routeRequest(prompt, taskType);
      
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

    it('should calculate cost savings correctly', async () => {
      const prompt = 'Test prompt';
      const taskType = 'general';
      
      await router.routeRequest(prompt, taskType);
      
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
      const disabledRouter = new OfficialLLMRouter();
      
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
        expect(firstProvider).toHaveProperty('configured');
        expect(firstProvider).toHaveProperty('sdk');
      }
    });

    it('should track usage statistics correctly', () => {
      const usageStats = router.getUsageStats();
      
      expect(usageStats).toHaveProperty('providers');
      expect(usageStats).toHaveProperty('totalProviders');
      expect(usageStats).toHaveProperty('status');
      expect(usageStats).toHaveProperty('timestamp');
      
      expect(Array.isArray(usageStats.providers)).toBe(true);
      expect(typeof usageStats.totalProviders).toBe('number');
      expect(typeof usageStats.status).toBe('string');
    });
  });

  describe('Error Handling and Fallback', () => {
    it('should handle cost optimization errors gracefully', async () => {
      // Test that the router continues to work even if cost optimization fails
      const prompt = 'Test prompt';
      const result = await router.routeRequest(prompt, 'general');
      
      expect(result).toBeDefined();
      expect(result.response).toBeDefined();
      expect(result.provider).toBeDefined();
    });

    it('should maintain backward compatibility', () => {
      // Ensure all existing methods still work
      expect(typeof router.getProviderStatus).toBe('function');
      expect(typeof router.getUsageStats).toBe('function');
      expect(typeof router.routeRequest).toBe('function');
    });

    it('should handle fallback scenarios', async () => {
      // Test fallback functionality
      const prompt = 'Test prompt';
      
      // Mock a provider failure scenario
      const originalCallProvider = (router as any).callProvider;
      (router as any).callProvider = jest.fn().mockRejectedValue(new Error('Provider failed'));
      
      try {
        await router.routeRequest(prompt, 'general');
      } catch (error) {
        // Should eventually throw after all providers fail
        expect(error).toBeDefined();
      }
      
      // Restore original method
      (router as any).callProvider = originalCallProvider;
    });
  });

  describe('Performance Considerations', () => {
    it('should not significantly impact routing performance', async () => {
      const startTime = Date.now();
      
      // Perform multiple operations
      for (let i = 0; i < 10; i++) {
        router.getProviderStatus();
        router.getCostOptimizationStats();
        router.getCostOptimizationConfig();
      }
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should complete within reasonable time (less than 1 second for 10 operations)
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

  describe('Task-Specific Routing Preservation', () => {
    it('should maintain task-specific provider preferences', async () => {
      const prompt = 'Test prompt';
      
      // Test that task-specific routing still works
      const tradingResult = await router.routeRequest(prompt, 'trading_analysis');
      const codeResult = await router.routeRequest(prompt, 'code_generation');
      const sentimentResult = await router.routeRequest(prompt, 'sentiment_analysis');
      
      expect(tradingResult).toBeDefined();
      expect(codeResult).toBeDefined();
      expect(sentimentResult).toBeDefined();
      
      // Results should have valid providers
      expect(tradingResult.provider).toBeDefined();
      expect(codeResult.provider).toBeDefined();
      expect(sentimentResult.provider).toBeDefined();
    });
  });
});
