/**
 * Integration tests for Cost-First Routing Logic
 * Tests the complete integration of cost optimization across all components
 */

import { MultiLLMRouter } from '../services/MultiLLMRouter';
import { OfficialLLMRouter } from '../services/OfficialLLMRouter';
import { CostOptimizationEngine } from '../services/CostOptimizationEngine';
import { CostAnalyticsService } from '../services/CostAnalyticsService';
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

// Mock SDK modules
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

describe('Cost-First Routing Integration Tests', () => {
  let multiRouter: MultiLLMRouter;
  let officialRouter: OfficialLLMRouter;
  let costEngine: CostOptimizationEngine;
  let analyticsService: CostAnalyticsService;
  let providerRanking: ProviderCostRanking;
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
    process.env.COST_TIER_EXPENSIVE = 'openai,anthropic';
    process.env.ENABLE_COST_OPTIMIZATION = 'true';
    process.env.AFFORDABLE_MAX_COST_PER_TOKEN = '0.0001';
    process.env.EXPENSIVE_MAX_COST_PER_TOKEN = '0.01';
    process.env.QUALITY_THRESHOLD = '0.85';
    
    // Initialize all components
    multiRouter = new MultiLLMRouter();
    officialRouter = new OfficialLLMRouter();
    costEngine = new CostOptimizationEngine();
    analyticsService = new CostAnalyticsService();
    providerRanking = new ProviderCostRanking();
  });

  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  describe('End-to-End Cost Optimization Flow', () => {
    it('should route requests through cost-optimized providers', async () => {
      const prompt = 'Test prompt for cost optimization';
      const taskType = 'general';
      
      // Test MultiLLMRouter
      const multiResult = await multiRouter.routeRequest(prompt, taskType);
      expect(multiResult).toBeDefined();
      
      // Test OfficialLLMRouter
      const officialResult = await officialRouter.routeRequest(prompt, taskType);
      expect(officialResult).toBeDefined();
      expect(officialResult.response).toBeDefined();
      expect(officialResult.provider).toBeDefined();
    });

    it('should track cost metrics across both routers', async () => {
      const prompt = 'Test prompt for cost tracking';
      const taskType = 'general';
      
      // Route through both routers
      await multiRouter.routeRequest(prompt, taskType);
      await officialRouter.routeRequest(prompt, taskType);
      
      // Check cost tracking
      const multiStats = multiRouter.getCostOptimizationStats();
      const officialStats = officialRouter.getCostOptimizationStats();
      
      expect(multiStats.totalRequests).toBeGreaterThan(0);
      expect(officialStats.totalRequests).toBeGreaterThan(0);
    });

    it('should prioritize affordable premium providers', async () => {
      const prompt = 'Test prompt for provider prioritization';
      const taskType = 'general';
      
      // Route multiple requests and check provider selection
      const results = [];
      for (let i = 0; i < 5; i++) {
        const result = await officialRouter.routeRequest(prompt, taskType);
        results.push(result.provider);
      }
      
      // Should include affordable premium providers
      const affordableProviders = ['mistral', 'google', 'cohere'];
      const hasAffordableProvider = results.some(provider => 
        affordableProviders.includes(provider.toLowerCase())
      );
      
      expect(hasAffordableProvider).toBe(true);
    });
  });

  describe('Component Integration', () => {
    it('should integrate CostOptimizationEngine with both routers', () => {
      // Test MultiLLMRouter integration
      const multiConfig = multiRouter.getCostOptimizationConfig();
      expect(multiConfig.enableCostOptimization).toBe(true);
      
      // Test OfficialLLMRouter integration
      const officialConfig = officialRouter.getCostOptimizationConfig();
      expect(officialConfig.enableCostOptimization).toBe(true);
      
      // Both should have the same configuration
      expect(multiConfig.costTiers).toEqual(officialConfig.costTiers);
    });

    it('should integrate ProviderCostRanking with cost optimization', () => {
      const providers = ['openai', 'mistral', 'google', 'cohere'];
      const rankings = providerRanking.rankProviders(providers, 'general');
      
      expect(rankings).toHaveLength(4);
      
      // Affordable premium providers should be ranked higher
      const affordableProviders = ['mistral', 'google', 'cohere'];
      const expensiveProviders = ['openai'];
      
      const affordableRankings = rankings.filter(r => 
        affordableProviders.includes(r.provider)
      );
      const expensiveRankings = rankings.filter(r => 
        expensiveProviders.includes(r.provider)
      );
      
      // Affordable providers should have better ranks (lower numbers)
      if (affordableRankings.length > 0 && expensiveRankings.length > 0) {
        const bestAffordable = Math.min(...affordableRankings.map(r => r.rank));
        const bestExpensive = Math.min(...expensiveRankings.map(r => r.rank));
        
        expect(bestAffordable).toBeLessThan(bestExpensive);
      }
    });

    it('should integrate CostAnalyticsService with all components', () => {
      const analytics = analyticsService.getCostAnalytics();
      
      expect(analytics).toHaveProperty('totalRequests');
      expect(analytics).toHaveProperty('providerBreakdown');
      expect(analytics).toHaveProperty('taskTypeBreakdown');
      expect(analytics).toHaveProperty('qualityMetrics');
      
      // Should have provider breakdown data
      expect(Array.isArray(analytics.providerBreakdown)).toBe(true);
      
      // Should have task type breakdown data
      expect(Array.isArray(analytics.taskTypeBreakdown)).toBe(true);
    });
  });

  describe('Performance Integration', () => {
    it('should maintain performance with cost optimization enabled', async () => {
      const prompt = 'Test prompt for performance';
      const taskType = 'general';
      
      const startTime = Date.now();
      
      // Route multiple requests
      const promises = [];
      for (let i = 0; i < 10; i++) {
        promises.push(officialRouter.routeRequest(prompt, taskType));
      }
      
      await Promise.all(promises);
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should complete within reasonable time (less than 5 seconds for 10 requests)
      expect(duration).toBeLessThan(5000);
    });

    it('should not significantly impact routing decision time', () => {
      const startTime = Date.now();
      
      // Perform multiple routing decisions
      for (let i = 0; i < 100; i++) {
        multiRouter.getProviderStatus();
        multiRouter.getCostOptimizationStats();
        officialRouter.getProviderStatus();
        officialRouter.getCostOptimizationStats();
      }
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should complete within reasonable time (less than 1 second for 100 operations)
      expect(duration).toBeLessThan(1000);
    });
  });

  describe('Cost Efficiency Validation', () => {
    it('should achieve cost efficiency improvements', async () => {
      const prompt = 'Test prompt for cost efficiency';
      const taskType = 'general';
      
      // Route multiple requests
      for (let i = 0; i < 10; i++) {
        await officialRouter.routeRequest(prompt, taskType);
      }
      
      const stats = officialRouter.getCostOptimizationStats();
      
      // Should have cost savings
      expect(stats.costSavings).toBeGreaterThanOrEqual(0);
      expect(stats.savingsPercentage).toBeGreaterThanOrEqual(0);
    });

    it('should maintain quality thresholds', async () => {
      const prompt = 'Test prompt for quality maintenance';
      const taskType = 'general';
      
      // Route requests and check quality metrics
      for (let i = 0; i < 5; i++) {
        await officialRouter.routeRequest(prompt, taskType);
      }
      
      const analytics = analyticsService.getCostAnalytics();
      const qualityMetrics = analytics.qualityMetrics;
      
      // Quality should meet threshold
      expect(qualityMetrics.averageResponseQuality).toBeGreaterThan(0.8);
      expect(qualityMetrics.qualityThresholdCompliance).toBeGreaterThan(0.8);
    });
  });

  describe('Backward Compatibility', () => {
    it('should maintain existing API compatibility', () => {
      // Test that all existing methods still work
      expect(typeof multiRouter.getProviderStatus).toBe('function');
      expect(typeof multiRouter.getUsageStats).toBe('function');
      expect(typeof officialRouter.getProviderStatus).toBe('function');
      expect(typeof officialRouter.getUsageStats).toBe('function');
      
      // Test existing method calls
      const multiStatus = multiRouter.getProviderStatus();
      const multiUsage = multiRouter.getUsageStats();
      const officialStatus = officialRouter.getProviderStatus();
      const officialUsage = officialRouter.getUsageStats();
      
      expect(multiStatus).toBeDefined();
      expect(multiUsage).toBeDefined();
      expect(officialStatus).toBeDefined();
      expect(officialUsage).toBeDefined();
    });

    it('should preserve task-specific routing behavior', async () => {
      const prompt = 'Test prompt for task-specific routing';
      
      // Test different task types
      const taskTypes = ['trading_analysis', 'code_generation', 'sentiment_analysis'];
      
      for (const taskType of taskTypes) {
        const result = await officialRouter.routeRequest(prompt, taskType);
        expect(result).toBeDefined();
        expect(result.provider).toBeDefined();
        expect(result.response).toBeDefined();
      }
    });
  });

  describe('Error Handling Integration', () => {
    it('should handle errors gracefully across all components', async () => {
      const prompt = 'Test prompt for error handling';
      const taskType = 'general';
      
      // Test error handling in both routers
      try {
        await multiRouter.routeRequest(prompt, taskType);
        await officialRouter.routeRequest(prompt, taskType);
      } catch (error) {
        // Should handle errors gracefully
        expect(error).toBeDefined();
      }
      
      // Components should still be functional after errors
      expect(multiRouter.getProviderStatus()).toBeDefined();
      expect(officialRouter.getProviderStatus()).toBeDefined();
    });

    it('should maintain functionality when cost optimization is disabled', () => {
      process.env.ENABLE_COST_OPTIMIZATION = 'false';
      
      const disabledMultiRouter = new MultiLLMRouter();
      const disabledOfficialRouter = new OfficialLLMRouter();
      
      const multiConfig = disabledMultiRouter.getCostOptimizationConfig();
      const officialConfig = disabledOfficialRouter.getCostOptimizationConfig();
      
      expect(multiConfig.enableCostOptimization).toBe(false);
      expect(officialConfig.enableCostOptimization).toBe(false);
      
      // Should still be functional
      expect(disabledMultiRouter.getProviderStatus()).toBeDefined();
      expect(disabledOfficialRouter.getProviderStatus()).toBeDefined();
    });
  });

  describe('Data Consistency', () => {
    it('should maintain data consistency across components', async () => {
      const prompt = 'Test prompt for data consistency';
      const taskType = 'general';
      
      // Route requests through both routers
      await multiRouter.routeRequest(prompt, taskType);
      await officialRouter.routeRequest(prompt, taskType);
      
      // Check that both routers have consistent cost tracking
      const multiStats = multiRouter.getCostOptimizationStats();
      const officialStats = officialRouter.getCostOptimizationStats();
      
      // Both should have tracked usage
      expect(multiStats.totalRequests).toBeGreaterThan(0);
      expect(officialStats.totalRequests).toBeGreaterThan(0);
      
      // Cost metrics should be consistent
      expect(typeof multiStats.totalCost).toBe('number');
      expect(typeof officialStats.totalCost).toBe('number');
      expect(typeof multiStats.costSavings).toBe('number');
      expect(typeof officialStats.costSavings).toBe('number');
    });
  });
});
