/**
 * Performance and satisfaction tests for Quality Threshold System
 */

import { QualityThresholdManager } from '../services/QualityThresholdManager';
import { MultiLLMRouter } from '../services/MultiLLMRouter';
import { OfficialLLMRouter } from '../services/OfficialLLMRouter';

// Mock dependencies
jest.mock('../services/CostOptimizationEngine', () => {
  return {
    CostOptimizationEngine: jest.fn().mockImplementation(() => ({
      rankProvidersByCost: jest.fn().mockReturnValue(['google', 'mistral', 'openai']),
      trackCostMetrics: jest.fn(),
      getCostStatistics: jest.fn().mockReturnValue({ totalCost: 0.1, averageCost: 0.05 }),
      getConfiguration: jest.fn().mockReturnValue({ enabled: true }),
      updateProviderAvailability: jest.fn()
    }))
  };
});

jest.mock('../services/ProviderCostRanking', () => {
  return {
    ProviderCostRanking: jest.fn().mockImplementation(() => ({
      rankProviders: jest.fn().mockReturnValue(['google', 'mistral', 'openai'])
    }))
  };
});

jest.mock('../services/TokenEstimationService', () => {
  return {
    TokenEstimationService: jest.fn().mockImplementation(() => ({
      estimateTokens: jest.fn().mockResolvedValue({
        tokenCount: 100,
        confidence: 0.95,
        cached: false,
        processingTime: 5
      }),
      getStats: jest.fn().mockReturnValue({ estimations: 10, cacheHits: 5 }),
      getCacheStats: jest.fn().mockReturnValue({ hits: 5, misses: 5 }),
      clearCache: jest.fn().mockResolvedValue(undefined),
      warmupCache: jest.fn().mockResolvedValue(undefined)
    }))
  };
});

describe('Quality Threshold System - Performance and Satisfaction Tests', () => {
  let qualityManager: QualityThresholdManager;
  let multiRouter: MultiLLMRouter;
  let officialRouter: OfficialLLMRouter;

  beforeEach(() => {
    // Set up environment variables
    process.env.GLOBAL_MIN_QUALITY = '0.7';
    process.env.ESCALATION_THRESHOLD = '0.6';
    process.env.QUALITY_EVALUATION_ENABLED = 'true';
    process.env.MAX_ESCALATIONS_PER_SESSION = '3';
    process.env.QUALITY_EVALUATION_TIMEOUT = '100';
    
    qualityManager = new QualityThresholdManager();
    
    // Set up routers with mocked providers
    process.env.OPENAI_API_KEY = 'test-openai-key';
    process.env.GOOGLE_API_KEY = 'test-google-key';
    process.env.MISTRAL_API_KEY = 'test-mistral-key';
    
    multiRouter = new MultiLLMRouter();
    officialRouter = new OfficialLLMRouter();
  });

  afterEach(() => {
    delete process.env.GLOBAL_MIN_QUALITY;
    delete process.env.ESCALATION_THRESHOLD;
    delete process.env.QUALITY_EVALUATION_ENABLED;
    delete process.env.MAX_ESCALATIONS_PER_SESSION;
    delete process.env.QUALITY_EVALUATION_TIMEOUT;
    delete process.env.OPENAI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    delete process.env.MISTRAL_API_KEY;
    jest.clearAllMocks();
  });

  describe('Performance Testing', () => {
    test('should evaluate quality within performance requirements (<100ms)', async () => {
      const response = 'This is a comprehensive analysis of the trading data with detailed insights and recommendations.';
      const taskType = 'analysis';
      const provider = 'openai';
      
      const startTime = Date.now();
      const result = await qualityManager.evaluateQuality(response, taskType, provider);
      const endTime = Date.now();
      const processingTime = endTime - startTime;
      
      expect(processingTime).toBeLessThan(100); // <100ms requirement
      expect(result.qualityScore).toBeGreaterThan(0);
      expect(result.confidence).toBeGreaterThan(0);
    });

    test('should handle multiple concurrent quality evaluations efficiently', async () => {
      const responses = [
        'High quality response with detailed analysis.',
        'Medium quality response with basic information.',
        'Low quality response with minimal detail.',
        'Excellent quality response with comprehensive insights.',
        'Poor quality response with errors and inconsistencies.'
      ];
      
      const taskType = 'analysis';
      const provider = 'openai';
      
      const startTime = Date.now();
      
      // Run evaluations concurrently
      const promises = responses.map(response => 
        qualityManager.evaluateQuality(response, taskType, provider)
      );
      
      const results = await Promise.all(promises);
      const endTime = Date.now();
      const totalTime = endTime - startTime;
      
      // Should complete all evaluations efficiently
      expect(results).toHaveLength(5);
      expect(totalTime).toBeLessThan(500); // Should be much faster than sequential
      
      // Verify quality scores are reasonable
      results.forEach(result => {
        expect(result.qualityScore).toBeGreaterThanOrEqual(0);
        expect(result.qualityScore).toBeLessThanOrEqual(1);
        expect(result.confidence).toBeGreaterThanOrEqual(0);
      });
    });

    test('should maintain performance with large responses', async () => {
      // Create a large response (1000+ characters)
      const largeResponse = 'This is a very detailed and comprehensive analysis '.repeat(50) + 
        'that covers multiple aspects of the topic in great depth with extensive ' +
        'explanations, examples, and recommendations for future actions.';
      
      const taskType = 'analysis';
      const provider = 'openai';
      
      const startTime = Date.now();
      const result = await qualityManager.evaluateQuality(largeResponse, taskType, provider);
      const endTime = Date.now();
      const processingTime = endTime - startTime;
      
      expect(processingTime).toBeLessThan(100); // Still <100ms for large responses
      expect(result.qualityScore).toBeGreaterThan(0);
      expect(result.evaluationCriteria).toBeDefined();
    });

    test('should handle escalation decisions efficiently', async () => {
      const sessionId = 'performance-test-session';
      const currentProvider = 'openai';
      const qualityScore = 0.4; // Below threshold
      
      const startTime = Date.now();
      const decision = await qualityManager.makeEscalationDecision(
        currentProvider, 
        qualityScore, 
        sessionId
      );
      const endTime = Date.now();
      const processingTime = endTime - startTime;
      
      expect(processingTime).toBeLessThan(50); // Escalation decisions should be very fast
      expect(decision.shouldEscalate).toBe(true);
      expect(decision.suggestedProvider).toBeDefined();
      expect(decision.confidence).toBeGreaterThan(0);
    });

    test('should maintain performance with frequent escalations', async () => {
      const sessionId = 'frequent-escalation-test';
      const providers = ['openai', 'google', 'mistral', 'cohere'];
      
      const startTime = Date.now();
      
      // Make multiple escalation decisions
      const promises = providers.map((provider, index) => 
        qualityManager.makeEscalationDecision(provider, 0.3 + index * 0.1, sessionId)
      );
      
      const decisions = await Promise.all(promises);
      const endTime = Date.now();
      const totalTime = endTime - startTime;
      
      expect(decisions).toHaveLength(4);
      expect(totalTime).toBeLessThan(200); // Should handle multiple escalations efficiently
      
      // Verify escalation counts are tracked correctly
      const escalationCount = qualityManager.getEscalationCount(sessionId);
      expect(escalationCount).toBeGreaterThan(0);
    });
  });

  describe('User Satisfaction Testing', () => {
    test('should maintain 85%+ user satisfaction rate', async () => {
      // Simulate various response qualities
      const testResponses = [
        { response: 'Excellent comprehensive analysis with detailed insights.', expectedQuality: 0.9 },
        { response: 'Good analysis with clear explanations and recommendations.', expectedQuality: 0.8 },
        { response: 'Decent analysis with basic information provided.', expectedQuality: 0.7 },
        { response: 'Adequate response with some useful information.', expectedQuality: 0.6 },
        { response: 'Poor response with minimal detail and errors.', expectedQuality: 0.4 },
        { response: 'Very poor response with significant issues.', expectedQuality: 0.2 },
        { response: 'Outstanding analysis with exceptional depth and clarity.', expectedQuality: 0.95 },
        { response: 'High quality response with thorough coverage.', expectedQuality: 0.85 },
        { response: 'Good response with solid analysis.', expectedQuality: 0.75 },
        { response: 'Acceptable response with basic coverage.', expectedQuality: 0.65 }
      ];
      
      const taskType = 'analysis';
      const provider = 'openai';
      
      // Evaluate all responses
      const results = await Promise.all(
        testResponses.map(test => 
          qualityManager.evaluateQuality(test.response, taskType, provider)
        )
      );
      
      // Calculate satisfaction metrics
      const highQualityResponses = results.filter(r => r.qualityScore >= 0.7).length;
      const totalResponses = results.length;
      const satisfactionRate = highQualityResponses / totalResponses;
      
      expect(satisfactionRate).toBeGreaterThanOrEqual(0.85); // 85%+ requirement
      
      // Verify quality scores align with expectations
      results.forEach((result, index) => {
        const expected = testResponses[index].expectedQuality;
        const actual = result.qualityScore;
        
        // Allow some variance but should be in reasonable range
        expect(Math.abs(actual - expected)).toBeLessThan(0.3);
      });
    });

    test('should escalate appropriately to maintain quality', async () => {
      const sessionId = 'satisfaction-test-session';
      
      // Test responses that should trigger escalation
      const lowQualityResponses = [
        'Bad response.',
        'Poor analysis.',
        'Inadequate information.',
        'Low quality content.',
        'Substandard response.'
      ];
      
      let escalationCount = 0;
      
      for (const response of lowQualityResponses) {
        const qualityResult = await qualityManager.evaluateQuality(response, 'analysis', 'huggingface');
        
        if (qualityResult.shouldEscalate) {
          const escalationDecision = await qualityManager.makeEscalationDecision(
            'huggingface',
            qualityResult.qualityScore,
            sessionId
          );
          
          if (escalationDecision.shouldEscalate) {
            escalationCount++;
          }
        }
      }
      
      // Should escalate for low quality responses
      expect(escalationCount).toBeGreaterThan(0);
      
      // Verify escalation count is tracked
      const trackedEscalations = qualityManager.getEscalationCount(sessionId);
      expect(trackedEscalations).toBe(escalationCount);
    });

    test('should provide consistent quality evaluation across providers', async () => {
      const response = 'This is a comprehensive analysis with detailed insights and clear recommendations.';
      const taskType = 'analysis';
      const providers = ['openai', 'google', 'mistral', 'cohere', 'huggingface', 'xai'];
      
      // Evaluate same response across different providers
      const results = await Promise.all(
        providers.map(provider => 
          qualityManager.evaluateQuality(response, taskType, provider)
        )
      );
      
      // Quality scores should be consistent (within reasonable variance)
      const qualityScores = results.map(r => r.qualityScore);
      const averageQuality = qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length;
      const variance = qualityScores.reduce((sum, score) => sum + Math.pow(score - averageQuality, 2), 0) / qualityScores.length;
      const standardDeviation = Math.sqrt(variance);
      
      // Should have reasonable consistency (low standard deviation)
      expect(standardDeviation).toBeLessThan(0.2);
      
      // All scores should be in reasonable range
      qualityScores.forEach(score => {
        expect(score).toBeGreaterThanOrEqual(0);
        expect(score).toBeLessThanOrEqual(1);
      });
    });

    test('should maintain quality thresholds effectively', async () => {
      const providers = ['openai', 'google', 'mistral', 'cohere', 'huggingface', 'xai'];
      
      // Test responses of varying quality
      const testCases = [
        { response: 'Excellent response.', expectedEscalation: false },
        { response: 'Good response.', expectedEscalation: false },
        { response: 'Average response.', expectedEscalation: false },
        { response: 'Poor response.', expectedEscalation: true },
        { response: 'Very poor response.', expectedEscalation: true }
      ];
      
      for (const provider of providers) {
        for (const testCase of testCases) {
          const qualityResult = await qualityManager.evaluateQuality(
            testCase.response, 
            'general', 
            provider
          );
          
          // Verify escalation decision aligns with expected behavior
          if (testCase.expectedEscalation) {
            expect(qualityResult.shouldEscalate).toBe(true);
          } else {
            expect(qualityResult.shouldEscalate).toBe(false);
          }
        }
      }
    });
  });

  describe('Integration Performance Testing', () => {
    test('should maintain routing performance with quality monitoring', async () => {
      const prompt = 'Test prompt for routing performance';
      const taskType = 'analysis';
      
      // Mock provider responses
      const mockResponse = 'High quality response for performance testing.';
      
      // Mock MultiLLMRouter provider
      const mockProvider = {
        name: 'OpenAI',
        model: { invoke: jest.fn().mockResolvedValue({ content: mockResponse }) },
        isAvailable: true,
        tokensUsed: 0,
        tokenLimit: 1000
      };
      
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      
      const startTime = Date.now();
      const result = await multiRouter.routeRequest(prompt, taskType);
      const endTime = Date.now();
      const routingTime = endTime - startTime;
      
      // Routing with quality monitoring should still be fast
      expect(routingTime).toBeLessThan(2000); // <2 seconds requirement
      expect(result).toBe(mockResponse);
    });

    test('should handle quality escalation without significant performance impact', async () => {
      const prompt = 'Test prompt for escalation performance';
      const taskType = 'analysis';
      const sessionId = 'escalation-performance-test';
      
      const mockResponse = 'Low quality response.';
      const escalatedResponse = 'High quality escalated response.';
      
      // Mock provider responses
      const mockProvider = {
        name: 'HuggingFace',
        model: { invoke: jest.fn().mockResolvedValue({ content: mockResponse }) },
        isAvailable: true,
        tokensUsed: 0,
        tokenLimit: 1000
      };
      
      const escalatedProvider = {
        name: 'Google',
        model: { invoke: jest.fn().mockResolvedValue({ content: escalatedResponse }) },
        isAvailable: true,
        tokensUsed: 0,
        tokenLimit: 1000
      };
      
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue(escalatedResponse);
      
      const startTime = Date.now();
      const result = await multiRouter.routeRequest(prompt, taskType, sessionId);
      const endTime = Date.now();
      const routingTime = endTime - startTime;
      
      // Even with escalation, should maintain performance
      expect(routingTime).toBeLessThan(3000); // Allow slightly more time for escalation
      expect(result).toBe(escalatedResponse);
    });

    test('should maintain performance with OfficialLLMRouter quality integration', async () => {
      const prompt = 'Test prompt for official router performance';
      const taskType = 'general';
      
      const mockResponse = 'Official router response.';
      
      jest.spyOn(officialRouter as any, 'callProvider').mockResolvedValue(mockResponse);
      
      const startTime = Date.now();
      const result = await officialRouter.routeRequest(prompt, taskType);
      const endTime = Date.now();
      const routingTime = endTime - startTime;
      
      expect(routingTime).toBeLessThan(2000); // <2 seconds requirement
      expect(result.response).toBe(mockResponse);
      expect(result.provider).toBeDefined();
    });
  });

  describe('Stress Testing', () => {
    test('should handle high volume of quality evaluations', async () => {
      const responses = Array.from({ length: 100 }, (_, i) => 
        `Response ${i}: This is a test response with varying quality levels.`
      );
      
      const taskType = 'general';
      const provider = 'openai';
      
      const startTime = Date.now();
      
      // Process all responses
      const results = await Promise.all(
        responses.map(response => 
          qualityManager.evaluateQuality(response, taskType, provider)
        )
      );
      
      const endTime = Date.now();
      const totalTime = endTime - startTime;
      const averageTime = totalTime / responses.length;
      
      expect(results).toHaveLength(100);
      expect(averageTime).toBeLessThan(50); // Average <50ms per evaluation
      expect(totalTime).toBeLessThan(5000); // Total <5 seconds for 100 evaluations
      
      // Verify all results are valid
      results.forEach(result => {
        expect(result.qualityScore).toBeGreaterThanOrEqual(0);
        expect(result.qualityScore).toBeLessThanOrEqual(1);
        expect(result.confidence).toBeGreaterThanOrEqual(0);
      });
    });

    test('should handle concurrent sessions with escalation limits', async () => {
      const sessions = Array.from({ length: 10 }, (_, i) => `session-${i}`);
      const provider = 'openai';
      const qualityScore = 0.3; // Below threshold
      
      const startTime = Date.now();
      
      // Make escalation decisions for all sessions
      const promises = sessions.map(sessionId => 
        qualityManager.makeEscalationDecision(provider, qualityScore, sessionId)
      );
      
      const decisions = await Promise.all(promises);
      const endTime = Date.now();
      const totalTime = endTime - startTime;
      
      expect(decisions).toHaveLength(10);
      expect(totalTime).toBeLessThan(1000); // Should handle concurrent sessions efficiently
      
      // Verify escalation counts are tracked per session
      sessions.forEach(sessionId => {
        const count = qualityManager.getEscalationCount(sessionId);
        expect(count).toBe(1); // Each session should have 1 escalation
      });
    });

    test('should maintain memory efficiency with large quality history', async () => {
      const taskType = 'analysis';
      const provider = 'openai';
      
      // Generate large number of evaluations
      for (let i = 0; i < 1000; i++) {
        await qualityManager.evaluateQuality(
          `Response ${i}: This is a test response for memory efficiency testing.`,
          taskType,
          provider
        );
      }
      
      // Verify statistics are still accurate
      const stats = qualityManager.getQualityStats();
      expect(stats.totalEvaluations).toBe(1000);
      expect(stats.averageQualityScore).toBeGreaterThan(0);
      expect(stats.escalationCount).toBeGreaterThanOrEqual(0);
      
      // Verify history can be retrieved efficiently
      const history = qualityManager.getQualityHistory(100);
      expect(history).toHaveLength(100);
      
      // Clear history and verify memory is freed
      qualityManager.clearQualityHistory();
      const clearedStats = qualityManager.getQualityStats();
      expect(clearedStats.totalEvaluations).toBe(0);
    });
  });

  describe('Quality Threshold Effectiveness Validation', () => {
    test('should achieve 85%+ user satisfaction requirement', async () => {
      // Simulate realistic usage patterns
      const usagePatterns = [
        { response: 'Comprehensive analysis with detailed insights.', taskType: 'analysis' },
        { response: 'Clear explanation with examples.', taskType: 'general' },
        { response: 'Good response with useful information.', taskType: 'code' },
        { response: 'Thorough analysis of the market trends.', taskType: 'trading' },
        { response: 'Excellent explanation with step-by-step guide.', taskType: 'general' },
        { response: 'Detailed code review with suggestions.', taskType: 'code' },
        { response: 'Comprehensive trading strategy analysis.', taskType: 'trading' },
        { response: 'Well-structured analysis with conclusions.', taskType: 'analysis' },
        { response: 'Clear and concise explanation.', taskType: 'general' },
        { response: 'High-quality code implementation.', taskType: 'code' }
      ];
      
      const providers = ['openai', 'google', 'mistral', 'cohere'];
      let totalEvaluations = 0;
      let satisfactoryEvaluations = 0;
      
      // Evaluate across different providers and task types
      for (const pattern of usagePatterns) {
        for (const provider of providers) {
          const result = await qualityManager.evaluateQuality(
            pattern.response,
            pattern.taskType,
            provider
          );
          
          totalEvaluations++;
          if (result.qualityScore >= 0.7) {
            satisfactoryEvaluations++;
          }
        }
      }
      
      const satisfactionRate = satisfactoryEvaluations / totalEvaluations;
      expect(satisfactionRate).toBeGreaterThanOrEqual(0.85); // 85%+ requirement
    });

    test('should maintain quality consistency across different task types', async () => {
      const response = 'This is a comprehensive analysis with detailed insights and clear recommendations.';
      const taskTypes = ['analysis', 'general', 'code', 'trading'];
      const provider = 'openai';
      
      const results = await Promise.all(
        taskTypes.map(taskType => 
          qualityManager.evaluateQuality(response, taskType, provider)
        )
      );
      
      // Quality scores should be consistent across task types
      const qualityScores = results.map(r => r.qualityScore);
      const averageQuality = qualityScores.reduce((sum, score) => sum + score, 0) / qualityScores.length;
      
      // All scores should be close to average (consistent evaluation)
      qualityScores.forEach(score => {
        expect(Math.abs(score - averageQuality)).toBeLessThan(0.2);
      });
      
      // Average quality should be high for a good response
      expect(averageQuality).toBeGreaterThan(0.7);
    });
  });
});
