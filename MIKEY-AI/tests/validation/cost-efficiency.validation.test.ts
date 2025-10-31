/**
 * Cost Efficiency Validation Tests
 * Validates that cost-first routing achieves the required 60%+ cost savings
 */

import { MultiLLMRouter } from '../services/MultiLLMRouter';
import { OfficialLLMRouter } from '../services/OfficialLLMRouter';
import { CostOptimizationEngine } from '../services/CostOptimizationEngine';
import { CostAnalyticsService } from '../services/CostAnalyticsService';

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

describe('Cost Efficiency Validation Tests', () => {
  let multiRouter: MultiLLMRouter;
  let officialRouter: OfficialLLMRouter;
  let costEngine: CostOptimizationEngine;
  let analyticsService: CostAnalyticsService;
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    // Save original environment
    originalEnv = { ...process.env };
    
    // Set up test environment variables for cost optimization
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
    
    // Initialize components
    multiRouter = new MultiLLMRouter();
    officialRouter = new OfficialLLMRouter();
    costEngine = new CostOptimizationEngine();
    analyticsService = new CostAnalyticsService();
  });

  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  describe('Cost Savings Validation', () => {
    it('should achieve 60%+ cost savings compared to expensive-only routing', async () => {
      const prompt = 'Test prompt for cost savings validation';
      const taskType = 'general';
      
      // Route multiple requests to accumulate cost data
      const requests = 50;
      for (let i = 0; i < requests; i++) {
        await officialRouter.routeRequest(prompt, taskType);
      }
      
      const stats = officialRouter.getCostOptimizationStats();
      
      // Should have achieved significant cost savings
      expect(stats.savingsPercentage).toBeGreaterThanOrEqual(60);
      expect(stats.costSavings).toBeGreaterThan(0);
    });

    it('should demonstrate cost efficiency improvement over time', async () => {
      const prompt = 'Test prompt for cost efficiency over time';
      const taskType = 'general';
      
      // Track cost efficiency over multiple batches
      const batches = 5;
      const requestsPerBatch = 20;
      const efficiencyMeasurements: number[] = [];
      
      for (let batch = 0; batch < batches; batch++) {
        // Route requests in this batch
        for (let i = 0; i < requestsPerBatch; i++) {
          await officialRouter.routeRequest(prompt, taskType);
        }
        
        // Measure efficiency after each batch
        const stats = officialRouter.getCostOptimizationStats();
        efficiencyMeasurements.push(stats.savingsPercentage);
      }
      
      // Should maintain or improve efficiency over time
      const averageEfficiency = efficiencyMeasurements.reduce((sum, eff) => sum + eff, 0) / efficiencyMeasurements.length;
      expect(averageEfficiency).toBeGreaterThanOrEqual(60);
      
      // Efficiency should be consistent (not declining significantly)
      const minEfficiency = Math.min(...efficiencyMeasurements);
      const maxEfficiency = Math.max(...efficiencyMeasurements);
      const efficiencyVariance = maxEfficiency - minEfficiency;
      
      expect(efficiencyVariance).toBeLessThan(20); // Should not vary by more than 20%
    });

    it('should prioritize affordable premium providers for cost efficiency', async () => {
      const prompt = 'Test prompt for provider prioritization';
      const taskType = 'general';
      
      // Route many requests and track provider usage
      const requests = 100;
      const providerUsage: Record<string, number> = {};
      
      for (let i = 0; i < requests; i++) {
        const result = await officialRouter.routeRequest(prompt, taskType);
        const provider = result.provider.toLowerCase();
        providerUsage[provider] = (providerUsage[provider] || 0) + 1;
      }
      
      // Affordable premium providers should be used more frequently
      const affordableProviders = ['mistral', 'google', 'cohere'];
      const expensiveProviders = ['openai', 'anthropic'];
      
      const affordableUsage = affordableProviders.reduce((sum, provider) => sum + (providerUsage[provider] || 0), 0);
      const expensiveUsage = expensiveProviders.reduce((sum, provider) => sum + (providerUsage[provider] || 0), 0);
      
      // Affordable providers should be used more than expensive ones
      expect(affordableUsage).toBeGreaterThan(expensiveUsage);
      
      // Affordable providers should account for at least 60% of usage
      const totalUsage = affordableUsage + expensiveUsage;
      const affordablePercentage = (affordableUsage / totalUsage) * 100;
      expect(affordablePercentage).toBeGreaterThanOrEqual(60);
    });
  });

  describe('Quality Threshold Compliance', () => {
    it('should maintain 85%+ user satisfaction with response quality', async () => {
      const prompt = 'Test prompt for quality validation';
      const taskType = 'general';
      
      // Route requests and track quality metrics
      const requests = 50;
      for (let i = 0; i < requests; i++) {
        await officialRouter.routeRequest(prompt, taskType);
      }
      
      const analytics = analyticsService.getCostAnalytics();
      const qualityMetrics = analytics.qualityMetrics;
      
      // Should maintain quality threshold
      expect(qualityMetrics.averageResponseQuality).toBeGreaterThanOrEqual(0.85);
      expect(qualityMetrics.qualityThresholdCompliance).toBeGreaterThanOrEqual(0.85);
    });

    it('should balance cost efficiency with quality requirements', async () => {
      const prompt = 'Test prompt for cost-quality balance';
      const taskType = 'general';
      
      // Route requests and measure both cost and quality
      const requests = 30;
      for (let i = 0; i < requests; i++) {
        await officialRouter.routeRequest(prompt, taskType);
      }
      
      const stats = officialRouter.getCostOptimizationStats();
      const analytics = analyticsService.getCostAnalytics();
      
      // Should achieve both cost savings and quality
      expect(stats.savingsPercentage).toBeGreaterThanOrEqual(60);
      expect(analytics.qualityMetrics.averageResponseQuality).toBeGreaterThanOrEqual(0.85);
      
      // Quality should not be sacrificed for cost savings
      expect(analytics.qualityMetrics.qualityThresholdCompliance).toBeGreaterThanOrEqual(0.85);
    });
  });

  describe('Cost Optimization Configuration Validation', () => {
    it('should use correct cost tier configurations', () => {
      const config = officialRouter.getCostOptimizationConfig();
      
      // Should have correct cost tiers
      expect(config.costTiers).toHaveLength(2);
      
      const affordableTier = config.costTiers.find(tier => tier.tier === 'affordable_premium');
      const expensiveTier = config.costTiers.find(tier => tier.tier === 'expensive');
      
      expect(affordableTier).toBeDefined();
      expect(expensiveTier).toBeDefined();
      
      // Affordable tier should have lower cost per token
      expect(affordableTier!.maxCostPerToken).toBeLessThan(expensiveTier!.maxCostPerToken);
      
      // Should include correct providers
      expect(affordableTier!.providers).toContain('mistral');
      expect(affordableTier!.providers).toContain('google');
      expect(affordableTier!.providers).toContain('cohere');
      expect(expensiveTier!.providers).toContain('openai');
    });

    it('should respect cost optimization settings', () => {
      const config = officialRouter.getCostOptimizationConfig();
      
      // Should have cost optimization enabled
      expect(config.enableCostOptimization).toBe(true);
      
      // Should have appropriate quality threshold
      expect(config.qualityThreshold).toBeGreaterThanOrEqual(0.85);
      
      // Should allow fallback to expensive providers
      expect(config.fallbackToExpensive).toBe(true);
    });
  });

  describe('Provider Cost Efficiency Analysis', () => {
    it('should rank providers by cost efficiency correctly', () => {
      const providers = ['openai', 'mistral', 'google', 'cohere'];
      const rankings = costEngine.rankProvidersByCost(providers, 'general');
      
      // Should rank affordable premium providers first
      const affordableProviders = ['mistral', 'google', 'cohere'];
      const expensiveProviders = ['openai'];
      
      // First few providers should be affordable premium
      const firstThreeProviders = rankings.slice(0, 3);
      const affordableCount = firstThreeProviders.filter(provider => 
        affordableProviders.includes(provider)
      ).length;
      
      expect(affordableCount).toBeGreaterThanOrEqual(2); // At least 2 of first 3 should be affordable
    });

    it('should calculate cost savings accurately', () => {
      // Create test cost metrics
      const metrics = [
        {
          provider: 'mistral',
          tokensUsed: 1000,
          costPerToken: 0.0001,
          totalCost: 0.1,
          timestamp: new Date(),
          taskType: 'general'
        },
        {
          provider: 'google',
          tokensUsed: 1000,
          costPerToken: 0.000075,
          totalCost: 0.075,
          timestamp: new Date(),
          taskType: 'general'
        }
      ];
      
      // Track metrics
      metrics.forEach(metric => {
        costEngine.trackCostMetrics(metric);
      });
      
      const savings = costEngine.calculateCostSavings(metrics);
      
      // Should calculate positive savings
      expect(savings).toBeGreaterThan(0);
    });
  });

  describe('Analytics and Reporting Validation', () => {
    it('should provide accurate cost analytics', async () => {
      const prompt = 'Test prompt for analytics validation';
      const taskType = 'general';
      
      // Route requests to generate analytics data
      const requests = 25;
      for (let i = 0; i < requests; i++) {
        await officialRouter.routeRequest(prompt, taskType);
      }
      
      const analytics = analyticsService.getCostAnalytics();
      
      // Should have accurate analytics
      expect(analytics.totalRequests).toBeGreaterThan(0);
      expect(analytics.totalCost).toBeGreaterThan(0);
      expect(analytics.costSavings).toBeGreaterThanOrEqual(0);
      expect(analytics.savingsPercentage).toBeGreaterThanOrEqual(0);
      
      // Should have provider breakdown
      expect(Array.isArray(analytics.providerBreakdown)).toBe(true);
      expect(analytics.providerBreakdown.length).toBeGreaterThan(0);
      
      // Should have task type breakdown
      expect(Array.isArray(analytics.taskTypeBreakdown)).toBe(true);
      expect(analytics.taskTypeBreakdown.length).toBeGreaterThan(0);
    });

    it('should generate actionable cost efficiency reports', async () => {
      const prompt = 'Test prompt for efficiency reports';
      const taskType = 'general';
      
      // Route requests to generate data
      const requests = 20;
      for (let i = 0; i < requests; i++) {
        await officialRouter.routeRequest(prompt, taskType);
      }
      
      const report = analyticsService.getCostEfficiencyReport();
      
      // Should have efficiency metrics
      expect(report.currentEfficiency).toBeGreaterThanOrEqual(0);
      expect(report.targetEfficiency).toBeGreaterThanOrEqual(60);
      expect(typeof report.efficiencyGap).toBe('number');
      
      // Should have actionable recommendations
      expect(Array.isArray(report.recommendations)).toBe(true);
      report.recommendations.forEach(recommendation => {
        expect(typeof recommendation).toBe('string');
        expect(recommendation.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Edge Cases and Error Scenarios', () => {
    it('should handle cost optimization gracefully when providers are unavailable', async () => {
      // Disable some providers
      officialRouter.updateProviderAvailability('mistral', false);
      officialRouter.updateProviderAvailability('google', false);
      
      const prompt = 'Test prompt for unavailable providers';
      const taskType = 'general';
      
      // Should still route successfully
      const result = await officialRouter.routeRequest(prompt, taskType);
      expect(result).toBeDefined();
      expect(result.provider).toBeDefined();
      
      // Should still track cost metrics
      const stats = officialRouter.getCostOptimizationStats();
      expect(stats.totalRequests).toBeGreaterThan(0);
    });

    it('should maintain cost efficiency with different task types', async () => {
      const prompt = 'Test prompt for different task types';
      const taskTypes = ['general', 'trading_analysis', 'code_generation', 'sentiment_analysis'];
      
      // Route requests for different task types
      for (const taskType of taskTypes) {
        await officialRouter.routeRequest(prompt, taskType);
      }
      
      const stats = officialRouter.getCostOptimizationStats();
      
      // Should maintain cost efficiency across task types
      expect(stats.savingsPercentage).toBeGreaterThanOrEqual(0);
      expect(stats.costSavings).toBeGreaterThanOrEqual(0);
    });
  });
});
