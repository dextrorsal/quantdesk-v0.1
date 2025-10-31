/**
 * Integration tests for Quality Threshold System with existing fallback mechanisms
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

describe('Quality Threshold System - Integration Tests', () => {
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

  describe('Integration with Existing Fallback Mechanisms', () => {
    test('should work with MultiLLMRouter fallback system', async () => {
      const prompt = 'Test prompt for fallback integration';
      const taskType = 'analysis';
      const sessionId = 'fallback-integration-test';
      
      const mockResponse = 'Low quality response.';
      const fallbackResponse = 'High quality fallback response.';
      
      // Mock first provider to fail, fallback to succeed
      const mockProvider = {
        name: 'HuggingFace',
        model: { invoke: jest.fn().mockResolvedValue({ content: mockResponse }) },
        isAvailable: true,
        tokensUsed: 0,
        tokenLimit: 1000
      };
      
      const fallbackProvider = {
        name: 'Google',
        model: { invoke: jest.fn().mockResolvedValue({ content: fallbackResponse }) },
        isAvailable: true,
        tokensUsed: 0,
        tokenLimit: 1000
      };
      
      jest.spyOn(multiRouter as any, 'selectBestProvider')
        .mockReturnValueOnce(mockProvider)
        .mockReturnValueOnce(fallbackProvider);
      
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue(fallbackResponse);
      
      const result = await multiRouter.routeRequest(prompt, taskType, sessionId);
      
      expect(result).toBe(fallbackResponse);
    });

    test('should work with OfficialLLMRouter fallback system', async () => {
      const prompt = 'Test prompt for official router fallback';
      const taskType = 'general';
      const sessionId = 'official-fallback-test';
      
      const mockResponse = 'Low quality response.';
      const fallbackResponse = 'High quality fallback response.';
      
      // Mock first provider to fail, fallback to succeed
      jest.spyOn(officialRouter as any, 'callProvider')
        .mockRejectedValueOnce(new Error('First provider failed'))
        .mockResolvedValueOnce(fallbackResponse);
      
      const result = await officialRouter.routeRequest(prompt, taskType, sessionId);
      
      expect(result.response).toBe(fallbackResponse);
      expect(result.provider).toBeDefined();
    });

    test('should integrate quality escalation with existing fallback chains', async () => {
      const prompt = 'Test prompt for escalation-fallback integration';
      const taskType = 'analysis';
      const sessionId = 'escalation-fallback-test';
      
      const lowQualityResponse = 'Low quality response.';
      const escalatedResponse = 'High quality escalated response.';
      const finalFallbackResponse = 'Final fallback response.';
      
      // Mock quality evaluation to trigger escalation
      const mockQualityManager = (multiRouter as any).qualityThresholdManager;
      mockQualityManager.evaluateQuality.mockResolvedValue({
        provider: 'huggingface',
        qualityScore: 0.3, // Below threshold
        shouldEscalate: true,
        confidence: 0.8,
        timestamp: new Date()
      });
      
      mockQualityManager.makeEscalationDecision.mockResolvedValue({
        shouldEscalate: true,
        reason: 'Quality score below threshold',
        currentProvider: 'huggingface',
        suggestedProvider: 'google',
        qualityScore: 0.3,
        threshold: 0.6,
        confidence: 0.9
      });
      
      // Mock provider responses
      const mockProvider = {
        name: 'HuggingFace',
        model: { invoke: jest.fn().mockResolvedValue({ content: lowQualityResponse }) },
        isAvailable: true,
        tokensUsed: 0,
        tokenLimit: 1000
      };
      
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      
      // Mock escalation to fail, then fallback to succeed
      jest.spyOn(multiRouter as any, 'routeToProvider')
        .mockRejectedValueOnce(new Error('Escalation failed'))
        .mockResolvedValueOnce(finalFallbackResponse);
      
      const result = await multiRouter.routeRequest(prompt, taskType, sessionId);
      
      // Should eventually get a response through fallback
      expect(result).toBe(finalFallbackResponse);
    });
  });

  describe('End-to-End Quality Flow Integration', () => {
    test('should complete full quality evaluation and escalation flow', async () => {
      const prompt = 'Test prompt for full quality flow';
      const taskType = 'analysis';
      const sessionId = 'full-quality-flow-test';
      
      const lowQualityResponse = 'Low quality response.';
      const escalatedResponse = 'High quality escalated response.';
      
      // Mock quality evaluation to trigger escalation
      const mockQualityManager = (multiRouter as any).qualityThresholdManager;
      mockQualityManager.evaluateQuality.mockResolvedValue({
        provider: 'huggingface',
        qualityScore: 0.4, // Below threshold
        evaluationCriteria: { coherence: 0.4, relevance: 0.4 },
        shouldEscalate: true,
        confidence: 0.8,
        timestamp: new Date()
      });
      
      mockQualityManager.makeEscalationDecision.mockResolvedValue({
        shouldEscalate: true,
        reason: 'Quality score below threshold',
        currentProvider: 'huggingface',
        suggestedProvider: 'google',
        qualityScore: 0.4,
        threshold: 0.6,
        confidence: 0.9
      });
      
      // Mock provider responses
      const mockProvider = {
        name: 'HuggingFace',
        model: { invoke: jest.fn().mockResolvedValue({ content: lowQualityResponse }) },
        isAvailable: true,
        tokensUsed: 0,
        tokenLimit: 1000
      };
      
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'routeToProvider').mockResolvedValue(escalatedResponse);
      
      const result = await multiRouter.routeRequest(prompt, taskType, sessionId);
      
      // Verify full flow was executed
      expect(mockQualityManager.evaluateQuality).toHaveBeenCalledWith(
        lowQualityResponse,
        taskType,
        'huggingface'
      );
      
      expect(mockQualityManager.makeEscalationDecision).toHaveBeenCalledWith(
        'huggingface',
        0.4,
        sessionId
      );
      
      expect(result).toBe(escalatedResponse);
    });

    test('should handle quality evaluation disabled scenario', async () => {
      const prompt = 'Test prompt for disabled quality evaluation';
      const taskType = 'general';
      
      const mockResponse = 'Response without quality evaluation.';
      
      // Disable quality evaluation
      const mockQualityManager = (multiRouter as any).qualityThresholdManager;
      mockQualityManager.isQualityEvaluationEnabled.mockReturnValue(false);
      
      const mockProvider = {
        name: 'OpenAI',
        model: { invoke: jest.fn().mockResolvedValue({ content: mockResponse }) },
        isAvailable: true,
        tokensUsed: 0,
        tokenLimit: 1000
      };
      
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      
      const result = await multiRouter.routeRequest(prompt, taskType);
      
      // Should not call quality evaluation
      expect(mockQualityManager.evaluateQuality).not.toHaveBeenCalled();
      expect(result).toBe(mockResponse);
    });

    test('should maintain quality statistics across multiple requests', async () => {
      const prompts = [
        'First test prompt.',
        'Second test prompt.',
        'Third test prompt.',
        'Fourth test prompt.',
        'Fifth test prompt.'
      ];
      
      const taskType = 'analysis';
      const sessionId = 'stats-test-session';
      
      const mockResponse = 'Quality response.';
      
      // Mock provider
      const mockProvider = {
        name: 'OpenAI',
        model: { invoke: jest.fn().mockResolvedValue({ content: mockResponse }) },
        isAvailable: true,
        tokensUsed: 0,
        tokenLimit: 1000
      };
      
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      
      // Process multiple requests
      for (const prompt of prompts) {
        await multiRouter.routeRequest(prompt, taskType, sessionId);
      }
      
      // Verify statistics are maintained
      const stats = multiRouter.getQualityStats();
      expect(stats.totalEvaluations).toBeGreaterThan(0);
      expect(stats.averageQualityScore).toBeGreaterThan(0);
      expect(stats.escalationCount).toBeGreaterThanOrEqual(0);
      
      // Verify history is maintained
      const history = multiRouter.getQualityHistory();
      expect(history.length).toBeGreaterThan(0);
    });
  });

  describe('Quality Threshold Configuration Integration', () => {
    test('should apply provider-specific quality thresholds', async () => {
      const response = 'Test response for provider-specific thresholds.';
      const taskType = 'analysis';
      
      const providers = ['openai', 'google', 'mistral', 'cohere', 'huggingface', 'xai'];
      
      for (const provider of providers) {
        const result = await qualityManager.evaluateQuality(response, taskType, provider);
        
        // Each provider should have appropriate thresholds applied
        expect(result.provider).toBe(provider);
        expect(result.qualityScore).toBeGreaterThanOrEqual(0);
        expect(result.qualityScore).toBeLessThanOrEqual(1);
        expect(result.confidence).toBeGreaterThanOrEqual(0);
      }
    });

    test('should update quality thresholds dynamically', async () => {
      const provider = 'openai';
      const newConfig = {
        minQualityScore: 0.9,
        escalationThreshold: 0.8,
        enabled: true
      };
      
      await qualityManager.updateQualityThresholds(provider, newConfig);
      
      // Verify configuration was updated
      const threshold = (qualityManager as any).config.getThreshold(provider);
      expect(threshold.minQualityScore).toBe(0.9);
      expect(threshold.escalationThreshold).toBe(0.8);
    });

    test('should handle configuration validation', async () => {
      const validation = (qualityManager as any).config.validateConfiguration();
      
      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });
  });

  describe('Quality Metrics and Analytics Integration', () => {
    test('should provide comprehensive quality analytics', async () => {
      // Generate some test data
      const responses = [
        'High quality comprehensive analysis.',
        'Medium quality response with basic information.',
        'Low quality response with minimal detail.',
        'Excellent quality response with detailed insights.',
        'Poor quality response with errors.'
      ];
      
      const taskType = 'analysis';
      const provider = 'openai';
      
      // Evaluate all responses
      for (const response of responses) {
        await qualityManager.evaluateQuality(response, taskType, provider);
      }
      
      // Get comprehensive analytics
      const stats = qualityManager.getQualityStats();
      const profiles = qualityManager.getProviderProfiles();
      const history = qualityManager.getQualityHistory();
      
      expect(stats.totalEvaluations).toBe(5);
      expect(stats.averageQualityScore).toBeGreaterThan(0);
      expect(stats.escalationCount).toBeGreaterThanOrEqual(0);
      expect(stats.escalationRate).toBeGreaterThanOrEqual(0);
      expect(stats.userSatisfactionRate).toBeGreaterThan(0);
      
      expect(Array.isArray(profiles)).toBe(true);
      expect(profiles.length).toBeGreaterThan(0);
      
      expect(Array.isArray(history)).toBe(true);
      expect(history.length).toBe(5);
    });

    test('should track quality metrics per provider', async () => {
      const response = 'Test response for provider tracking.';
      const taskType = 'general';
      const providers = ['openai', 'google', 'mistral'];
      
      // Evaluate same response across different providers
      for (const provider of providers) {
        await qualityManager.evaluateQuality(response, taskType, provider);
      }
      
      // Get provider profiles
      const profiles = qualityManager.getProviderProfiles();
      
      // Should have profiles for all providers
      expect(profiles.length).toBeGreaterThanOrEqual(providers.length);
      
      // Each profile should have quality metrics
      profiles.forEach(profile => {
        expect(profile.provider).toBeDefined();
        expect(profile.averageQuality).toBeGreaterThanOrEqual(0);
        expect(profile.qualityConsistency).toBeGreaterThanOrEqual(0);
        expect(profile.escalationRate).toBeGreaterThanOrEqual(0);
        expect(profile.userSatisfaction).toBeGreaterThanOrEqual(0);
      });
    });

    test('should provide quality history with filtering', async () => {
      // Generate test data
      for (let i = 0; i < 10; i++) {
        await qualityManager.evaluateQuality(
          `Test response ${i}.`,
          'general',
          'openai'
        );
      }
      
      // Test different history limits
      const fullHistory = qualityManager.getQualityHistory();
      const limitedHistory = qualityManager.getQualityHistory(5);
      
      expect(fullHistory).toHaveLength(10);
      expect(limitedHistory).toHaveLength(5);
      
      // Limited history should be the most recent entries
      expect(limitedHistory[0]).toBe(fullHistory[5]);
      expect(limitedHistory[4]).toBe(fullHistory[9]);
    });
  });

  describe('Error Handling and Resilience', () => {
    test('should handle quality evaluation errors gracefully', async () => {
      const response = 'Test response for error handling.';
      const taskType = 'analysis';
      const provider = 'openai';
      
      // Mock evaluation criteria to throw error
      const mockConfig = (qualityManager as any).config;
      const originalGetEvaluationCriteria = mockConfig.getEvaluationCriteria;
      mockConfig.getEvaluationCriteria = jest.fn().mockReturnValue([
        {
          name: 'test',
          weight: 1.0,
          evaluator: jest.fn().mockRejectedValue(new Error('Evaluation failed')),
          description: 'Test criterion'
        }
      ]);
      
      const result = await qualityManager.evaluateQuality(response, taskType, provider);
      
      // Should return fallback result
      expect(result.qualityScore).toBe(0.5); // Fallback score
      expect(result.confidence).toBe(0.0); // Low confidence
      
      // Restore original method
      mockConfig.getEvaluationCriteria = originalGetEvaluationCriteria;
    });

    test('should handle escalation decision errors gracefully', async () => {
      const sessionId = 'error-handling-test';
      const provider = 'openai';
      const qualityScore = 0.3;
      
      // Mock escalation decision to throw error
      const mockShouldEscalate = jest.spyOn(qualityManager, 'shouldEscalate');
      mockShouldEscalate.mockRejectedValue(new Error('Escalation check failed'));
      
      const decision = await qualityManager.makeEscalationDecision(provider, qualityScore, sessionId);
      
      // Should return safe fallback decision
      expect(decision.shouldEscalate).toBe(false);
      expect(decision.reason).toContain('error');
      
      mockShouldEscalate.mockRestore();
    });

    test('should handle provider configuration errors gracefully', async () => {
      const provider = 'unknown-provider';
      const response = 'Test response.';
      const taskType = 'general';
      
      const result = await qualityManager.evaluateQuality(response, taskType, provider);
      
      // Should return fallback result for unknown provider
      expect(result.provider).toBe(provider);
      expect(result.qualityScore).toBe(0.5); // Fallback score
      expect(result.confidence).toBe(0.0); // Low confidence
    });
  });
});
