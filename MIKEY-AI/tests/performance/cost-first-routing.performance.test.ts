/**
 * Performance tests for Cost-First Routing Logic
 * Validates performance requirements and benchmarks
 */

import { MultiLLMRouter } from '../services/MultiLLMRouter';
import { OfficialLLMRouter } from '../services/OfficialLLMRouter';
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

describe('Cost-First Routing Performance Tests', () => {
  let multiRouter: MultiLLMRouter;
  let officialRouter: OfficialLLMRouter;
  let costEngine: CostOptimizationEngine;
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
    
    // Initialize components
    multiRouter = new MultiLLMRouter();
    officialRouter = new OfficialLLMRouter();
    costEngine = new CostOptimizationEngine();
    providerRanking = new ProviderCostRanking();
  });

  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  describe('Routing Decision Performance', () => {
    it('should complete routing decisions within 2 seconds', async () => {
      const prompt = 'Test prompt for routing performance';
      const taskType = 'general';
      
      const startTime = Date.now();
      
      // Test routing decision time
      const result = await officialRouter.routeRequest(prompt, taskType);
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should complete within 2 seconds (requirement from story)
      expect(duration).toBeLessThan(2000);
      expect(result).toBeDefined();
    });

    it('should handle multiple concurrent routing decisions efficiently', async () => {
      const prompt = 'Test prompt for concurrent routing';
      const taskType = 'general';
      
      const startTime = Date.now();
      
      // Test concurrent routing decisions
      const promises = [];
      for (let i = 0; i < 20; i++) {
        promises.push(officialRouter.routeRequest(prompt, taskType));
      }
      
      const results = await Promise.all(promises);
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should complete within reasonable time for 20 concurrent requests
      expect(duration).toBeLessThan(10000); // 10 seconds for 20 requests
      expect(results).toHaveLength(20);
      
      // All results should be valid
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.response).toBeDefined();
        expect(result.provider).toBeDefined();
      });
    });

    it('should maintain performance with cost optimization enabled', () => {
      const startTime = Date.now();
      
      // Test cost optimization performance
      const providers = ['openai', 'mistral', 'google', 'cohere'];
      const taskTypes = ['general', 'trading_analysis', 'code_generation'];
      
      // Perform multiple cost optimization operations
      for (let i = 0; i < 100; i++) {
        const taskType = taskTypes[i % taskTypes.length];
        const rankedProviders = costEngine.rankProvidersByCost(providers, taskType);
        expect(rankedProviders).toBeDefined();
      }
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should complete within reasonable time (less than 1 second for 100 operations)
      expect(duration).toBeLessThan(1000);
    });
  });

  describe('Memory Usage Performance', () => {
    it('should not cause memory leaks with repeated operations', () => {
      const initialMemory = process.memoryUsage().heapUsed;
      
      // Perform many operations
      for (let i = 0; i < 1000; i++) {
        const providers = ['openai', 'mistral', 'google', 'cohere'];
        const rankings = providerRanking.rankProviders(providers, 'general');
        expect(rankings).toBeDefined();
      }
      
      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;
      
      // Memory increase should be reasonable (less than 10MB)
      expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024);
    });

    it('should limit stored metrics to prevent memory issues', () => {
      // Track many cost metrics
      for (let i = 0; i < 2000; i++) {
        const metrics = {
          provider: 'mistral',
          tokensUsed: 100,
          costPerToken: 0.0001,
          totalCost: 0.01,
          timestamp: new Date(),
          taskType: 'general'
        };
        
        costEngine.trackCostMetrics(metrics);
      }
      
      const stats = costEngine.getCostStatistics();
      
      // Should limit stored metrics (as implemented in CostOptimizationEngine)
      expect(stats.totalRequests).toBeLessThanOrEqual(1000);
    });
  });

  describe('Scalability Performance', () => {
    it('should scale linearly with number of providers', () => {
      const smallProviderSet = ['openai', 'mistral'];
      const largeProviderSet = ['openai', 'mistral', 'google', 'cohere', 'anthropic', 'huggingface'];
      
      const startTime = Date.now();
      
      // Test with small provider set
      for (let i = 0; i < 100; i++) {
        costEngine.rankProvidersByCost(smallProviderSet, 'general');
      }
      
      const smallSetTime = Date.now() - startTime;
      
      const startTime2 = Date.now();
      
      // Test with large provider set
      for (let i = 0; i < 100; i++) {
        costEngine.rankProvidersByCost(largeProviderSet, 'general');
      }
      
      const largeSetTime = Date.now() - startTime2;
      
      // Performance should scale reasonably (not exponentially)
      const performanceRatio = largeSetTime / smallSetTime;
      expect(performanceRatio).toBeLessThan(3); // Should not be more than 3x slower
    });

    it('should handle high-frequency requests efficiently', async () => {
      const prompt = 'Test prompt for high-frequency requests';
      const taskType = 'general';
      
      const startTime = Date.now();
      
      // Simulate high-frequency requests
      const promises = [];
      for (let i = 0; i < 50; i++) {
        promises.push(officialRouter.routeRequest(prompt, taskType));
      }
      
      const results = await Promise.all(promises);
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should handle high-frequency requests efficiently
      expect(duration).toBeLessThan(15000); // 15 seconds for 50 requests
      expect(results).toHaveLength(50);
      
      // All results should be valid
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.response).toBeDefined();
        expect(result.provider).toBeDefined();
      });
    });
  });

  describe('Cost Optimization Performance Impact', () => {
    it('should have minimal performance impact when cost optimization is disabled', () => {
      process.env.ENABLE_COST_OPTIMIZATION = 'false';
      
      const disabledRouter = new OfficialLLMRouter();
      const providers = ['openai', 'mistral', 'google', 'cohere'];
      
      const startTime = Date.now();
      
      // Test routing performance with cost optimization disabled
      for (let i = 0; i < 100; i++) {
        const rankedProviders = disabledRouter.getCostOptimizationConfig();
        expect(rankedProviders).toBeDefined();
      }
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should be fast even with cost optimization disabled
      expect(duration).toBeLessThan(500);
    });

    it('should maintain performance with different task types', () => {
      const providers = ['openai', 'mistral', 'google', 'cohere'];
      const taskTypes = ['general', 'trading_analysis', 'code_generation', 'sentiment_analysis', 'multilingual'];
      
      const startTime = Date.now();
      
      // Test performance across different task types
      for (let i = 0; i < 100; i++) {
        const taskType = taskTypes[i % taskTypes.length];
        const rankings = providerRanking.rankProviders(providers, taskType);
        expect(rankings).toBeDefined();
      }
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should maintain consistent performance across task types
      expect(duration).toBeLessThan(1000);
    });
  });

  describe('System Resource Usage', () => {
    it('should not consume excessive CPU resources', () => {
      const startTime = Date.now();
      const startCpuUsage = process.cpuUsage();
      
      // Perform intensive operations
      for (let i = 0; i < 1000; i++) {
        const providers = ['openai', 'mistral', 'google', 'cohere'];
        const rankings = providerRanking.rankProviders(providers, 'general');
        expect(rankings).toBeDefined();
      }
      
      const endTime = Date.now();
      const endCpuUsage = process.cpuUsage(startCpuUsage);
      const duration = endTime - startTime;
      
      // CPU usage should be reasonable
      const cpuTimeMs = (endCpuUsage.user + endCpuUsage.system) / 1000;
      const cpuPercentage = (cpuTimeMs / duration) * 100;
      
      expect(cpuPercentage).toBeLessThan(50); // Should not use more than 50% CPU
    });

    it('should maintain performance under load', async () => {
      const prompt = 'Test prompt for load testing';
      const taskType = 'general';
      
      const startTime = Date.now();
      
      // Simulate load with many concurrent requests
      const promises = [];
      for (let i = 0; i < 30; i++) {
        promises.push(officialRouter.routeRequest(prompt, taskType));
      }
      
      const results = await Promise.all(promises);
      
      const endTime = Date.now();
      const duration = endTime - startTime;
      
      // Should maintain performance under load
      expect(duration).toBeLessThan(20000); // 20 seconds for 30 requests
      expect(results).toHaveLength(30);
      
      // All results should be valid
      results.forEach(result => {
        expect(result).toBeDefined();
        expect(result.response).toBeDefined();
        expect(result.provider).toBeDefined();
      });
    });
  });
});
