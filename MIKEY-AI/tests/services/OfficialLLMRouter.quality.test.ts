/**
 * Integration tests for OfficialLLMRouter with Quality Threshold System
 */

import { OfficialLLMRouter } from '../services/OfficialLLMRouter';
import { QualityThresholdManager } from '../services/QualityThresholdManager';

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

jest.mock('../services/QualityThresholdManager', () => {
  return {
    QualityThresholdManager: jest.fn().mockImplementation(() => ({
      evaluateQuality: jest.fn().mockResolvedValue({
        provider: 'openai',
        qualityScore: 0.8,
        evaluationCriteria: { coherence: 0.8, relevance: 0.8 },
        shouldEscalate: false,
        confidence: 0.9,
        timestamp: new Date()
      }),
      makeEscalationDecision: jest.fn().mockResolvedValue({
        shouldEscalate: false,
        reason: 'Quality score meets threshold',
        currentProvider: 'openai',
        suggestedProvider: 'openai',
        qualityScore: 0.8,
        threshold: 0.6,
        confidence: 1.0
      }),
      isQualityEvaluationEnabled: jest.fn().mockReturnValue(true),
      getQualityStats: jest.fn().mockReturnValue({
        totalEvaluations: 10,
        averageQualityScore: 0.8,
        escalationCount: 2,
        escalationRate: 0.2,
        userSatisfactionRate: 0.75
      }),
      getProviderProfiles: jest.fn().mockReturnValue([
        {
          provider: 'openai',
          averageQuality: 0.8,
          qualityConsistency: 0.9,
          escalationRate: 0.1,
          userSatisfaction: 0.8,
          strengths: ['reasoning', 'analysis'],
          weaknesses: ['cost']
        }
      ]),
      getQualityHistory: jest.fn().mockReturnValue([]),
      updateQualityThresholds: jest.fn().mockResolvedValue(undefined),
      setQualityEvaluationEnabled: jest.fn(),
      resetEscalationCount: jest.fn(),
      getEscalationCount: jest.fn().mockReturnValue(0),
      clearQualityHistory: jest.fn(),
      getQualityMetrics: jest.fn().mockResolvedValue({
        responseLength: 100,
        coherenceScore: 0.8,
        relevanceScore: 0.8,
        completenessScore: 0.7,
        errorRate: 0.0,
        userSatisfactionScore: 0.8
      })
    }))
  };
});

describe('OfficialLLMRouter - Quality Integration', () => {
  let router: OfficialLLMRouter;
  let mockQualityManager: any;

  beforeEach(() => {
    // Set up environment variables
    process.env.OPENAI_API_KEY = 'test-openai-key';
    process.env.GOOGLE_API_KEY = 'test-google-key';
    process.env.MISTRAL_API_KEY = 'test-mistral-key';
    
    router = new OfficialLLMRouter();
    mockQualityManager = (router as any).qualityThresholdManager;
  });

  afterEach(() => {
    delete process.env.OPENAI_API_KEY;
    delete process.env.GOOGLE_API_KEY;
    delete process.env.MISTRAL_API_KEY;
    jest.clearAllMocks();
  });

  describe('Quality Monitoring Integration', () => {
    test('should evaluate quality after routing request', async () => {
      const prompt = 'Test prompt for quality evaluation';
      const taskType = 'analysis';
      
      // Mock successful response
      const mockResponse = 'This is a high-quality response with detailed analysis.';
      
      // Mock provider call
      jest.spyOn(router as any, 'callProvider').mockResolvedValue(mockResponse);
      
      const result = await router.routeRequest(prompt, taskType);
      
      // Verify quality evaluation was called
      expect(mockQualityManager.evaluateQuality).toHaveBeenCalledWith(
        mockResponse,
        taskType,
        expect.any(String)
      );
      
      expect(result.response).toBe(mockResponse);
      expect(result.provider).toBeDefined();
    });

    test('should log quality metrics after evaluation', async () => {
      const prompt = 'Test prompt for quality logging';
      const taskType = 'general';
      
      const mockResponse = 'Quality response for logging.';
      
      jest.spyOn(router as any, 'callProvider').mockResolvedValue(mockResponse);
      
      await router.routeRequest(prompt, taskType);
      
      // Verify quality evaluation was called
      expect(mockQualityManager.evaluateQuality).toHaveBeenCalled();
    });

    test('should handle quality evaluation errors gracefully', async () => {
      const prompt = 'Test prompt for error handling';
      const taskType = 'analysis';
      
      // Mock quality evaluation to throw error
      mockQualityManager.evaluateQuality.mockRejectedValue(new Error('Quality evaluation failed'));
      
      const mockResponse = 'Response despite quality error.';
      
      jest.spyOn(router as any, 'callProvider').mockResolvedValue(mockResponse);
      
      // Should not throw error
      const result = await router.routeRequest(prompt, taskType);
      expect(result.response).toBe(mockResponse);
    });
  });

  describe('Quality-Based Escalation', () => {
    test('should escalate when quality is below threshold', async () => {
      const prompt = 'Test prompt for escalation';
      const taskType = 'analysis';
      const sessionId = 'test-session';
      
      // Mock low quality response
      const mockResponse = 'Low quality response.';
      const escalatedResponse = 'High quality escalated response.';
      
      // Mock quality evaluation to indicate escalation needed
      mockQualityManager.evaluateQuality.mockResolvedValue({
        provider: 'huggingface',
        qualityScore: 0.4, // Below threshold
        evaluationCriteria: { coherence: 0.4, relevance: 0.4 },
        shouldEscalate: true,
        confidence: 0.8,
        timestamp: new Date()
      });
      
      // Mock escalation decision
      mockQualityManager.makeEscalationDecision.mockResolvedValue({
        shouldEscalate: true,
        reason: 'Quality score below threshold',
        currentProvider: 'huggingface',
        suggestedProvider: 'google',
        qualityScore: 0.4,
        threshold: 0.6,
        confidence: 0.9
      });
      
      // Mock provider calls
      jest.spyOn(router as any, 'callProvider')
        .mockResolvedValueOnce(mockResponse) // First call
        .mockResolvedValueOnce(escalatedResponse); // Escalated call
      
      const result = await router.routeRequest(prompt, taskType, sessionId);
      
      // Verify escalation was attempted
      expect(mockQualityManager.makeEscalationDecision).toHaveBeenCalledWith(
        expect.any(String),
        0.4,
        sessionId
      );
      
      // Should return escalated response
      expect(result.response).toBe(escalatedResponse);
      expect(result.provider).toBe('google');
    });

    test('should not escalate when quality meets threshold', async () => {
      const prompt = 'Test prompt for no escalation';
      const taskType = 'general';
      
      const mockResponse = 'High quality response.';
      
      // Mock quality evaluation to indicate no escalation needed
      mockQualityManager.evaluateQuality.mockResolvedValue({
        provider: 'openai',
        qualityScore: 0.8, // Above threshold
        evaluationCriteria: { coherence: 0.8, relevance: 0.8 },
        shouldEscalate: false,
        confidence: 0.9,
        timestamp: new Date()
      });
      
      jest.spyOn(router as any, 'callProvider').mockResolvedValue(mockResponse);
      
      const result = await router.routeRequest(prompt, taskType);
      
      // Should not call escalation decision
      expect(mockQualityManager.makeEscalationDecision).not.toHaveBeenCalled();
      
      // Should return original response
      expect(result.response).toBe(mockResponse);
    });

    test('should handle escalation failures gracefully', async () => {
      const prompt = 'Test prompt for escalation failure';
      const taskType = 'analysis';
      
      const mockResponse = 'Original response.';
      
      // Mock quality evaluation to indicate escalation needed
      mockQualityManager.evaluateQuality.mockResolvedValue({
        provider: 'mistral',
        qualityScore: 0.3,
        shouldEscalate: true,
        confidence: 0.7,
        timestamp: new Date()
      });
      
      // Mock escalation decision
      mockQualityManager.makeEscalationDecision.mockResolvedValue({
        shouldEscalate: true,
        reason: 'Quality score below threshold',
        currentProvider: 'mistral',
        suggestedProvider: 'openai',
        qualityScore: 0.3,
        threshold: 0.6,
        confidence: 0.8
      });
      
      // Mock provider calls - first succeeds, escalation fails
      jest.spyOn(router as any, 'callProvider')
        .mockResolvedValueOnce(mockResponse) // First call succeeds
        .mockRejectedValueOnce(new Error('Escalation failed')); // Escalation fails
      
      const result = await router.routeRequest(prompt, taskType);
      
      // Should return original response when escalation fails
      expect(result.response).toBe(mockResponse);
    });
  });

  describe('Quality Metrics Integration', () => {
    test('should provide quality statistics', () => {
      const stats = router.getQualityStats();
      
      expect(stats).toBeDefined();
      expect(stats.totalEvaluations).toBe(10);
      expect(stats.averageQualityScore).toBe(0.8);
      expect(stats.escalationCount).toBe(2);
      expect(stats.escalationRate).toBe(0.2);
      expect(stats.userSatisfactionRate).toBe(0.75);
    });

    test('should provide provider quality profiles', () => {
      const profiles = router.getProviderQualityProfiles();
      
      expect(Array.isArray(profiles)).toBe(true);
      expect(profiles.length).toBeGreaterThan(0);
      
      const profile = profiles[0];
      expect(profile.provider).toBe('openai');
      expect(profile.averageQuality).toBe(0.8);
      expect(profile.qualityConsistency).toBe(0.9);
      expect(profile.escalationRate).toBe(0.1);
      expect(profile.userSatisfaction).toBe(0.8);
      expect(Array.isArray(profile.strengths)).toBe(true);
      expect(Array.isArray(profile.weaknesses)).toBe(true);
    });

    test('should provide quality history', () => {
      const history = router.getQualityHistory(5);
      
      expect(Array.isArray(history)).toBe(true);
      expect(mockQualityManager.getQualityHistory).toHaveBeenCalledWith(5);
    });

    test('should update quality thresholds', async () => {
      const config = {
        minQualityScore: 0.9,
        escalationThreshold: 0.7,
        enabled: true
      };
      
      await router.updateQualityThresholds('openai', config);
      
      expect(mockQualityManager.updateQualityThresholds).toHaveBeenCalledWith('openai', config);
    });

    test('should enable/disable quality evaluation', () => {
      router.setQualityEvaluationEnabled(false);
      expect(mockQualityManager.setQualityEvaluationEnabled).toHaveBeenCalledWith(false);
      
      router.setQualityEvaluationEnabled(true);
      expect(mockQualityManager.setQualityEvaluationEnabled).toHaveBeenCalledWith(true);
    });

    test('should check if quality evaluation is enabled', () => {
      const isEnabled = router.isQualityEvaluationEnabled();
      expect(isEnabled).toBe(true);
      expect(mockQualityManager.isQualityEvaluationEnabled).toHaveBeenCalled();
    });

    test('should manage escalation counts per session', () => {
      const sessionId = 'test-session';
      
      router.resetEscalationCount(sessionId);
      expect(mockQualityManager.resetEscalationCount).toHaveBeenCalledWith(sessionId);
      
      const count = router.getEscalationCount(sessionId);
      expect(count).toBe(0);
      expect(mockQualityManager.getEscalationCount).toHaveBeenCalledWith(sessionId);
    });

    test('should clear quality history', () => {
      router.clearQualityHistory();
      expect(mockQualityManager.clearQualityHistory).toHaveBeenCalled();
    });

    test('should get quality metrics for responses', async () => {
      const response = 'Test response for quality metrics';
      const taskType = 'analysis';
      
      const metrics = await router.getQualityMetrics(response, taskType);
      
      expect(metrics).toBeDefined();
      expect(metrics.responseLength).toBe(100);
      expect(metrics.coherenceScore).toBe(0.8);
      expect(metrics.relevanceScore).toBe(0.8);
      expect(metrics.completenessScore).toBe(0.7);
      expect(metrics.errorRate).toBe(0.0);
      expect(metrics.userSatisfactionScore).toBe(0.8);
      
      expect(mockQualityManager.getQualityMetrics).toHaveBeenCalledWith(response, taskType);
    });
  });

  describe('Quality Configuration Integration', () => {
    test('should respect quality evaluation enabled setting', async () => {
      const prompt = 'Test prompt for configuration';
      const taskType = 'general';
      
      // Disable quality evaluation
      mockQualityManager.isQualityEvaluationEnabled.mockReturnValue(false);
      
      const mockResponse = 'Response without quality evaluation.';
      
      jest.spyOn(router as any, 'callProvider').mockResolvedValue(mockResponse);
      
      await router.routeRequest(prompt, taskType);
      
      // Should not call quality evaluation
      expect(mockQualityManager.evaluateQuality).not.toHaveBeenCalled();
    });

    test('should handle quality evaluation timeout', async () => {
      const prompt = 'Test prompt for timeout';
      const taskType = 'analysis';
      
      // Mock quality evaluation to timeout
      mockQualityManager.evaluateQuality.mockImplementation(() => 
        new Promise((resolve) => setTimeout(resolve, 200))
      );
      
      const mockResponse = 'Response with timeout.';
      
      jest.spyOn(router as any, 'callProvider').mockResolvedValue(mockResponse);
      
      // Should not throw error even with timeout
      const result = await router.routeRequest(prompt, taskType);
      expect(result.response).toBe(mockResponse);
    });
  });

  describe('Quality and Cost Integration', () => {
    test('should integrate quality thresholds with cost optimization', async () => {
      const prompt = 'Test prompt for cost-quality integration';
      const taskType = 'analysis';
      
      const mockResponse = 'Response for cost-quality integration.';
      
      jest.spyOn(router as any, 'callProvider').mockResolvedValue(mockResponse);
      
      await router.routeRequest(prompt, taskType);
      
      // Both quality evaluation and cost tracking should be called
      expect(mockQualityManager.evaluateQuality).toHaveBeenCalled();
      // Cost tracking is handled by trackUsageWithAccurateTokens
    });

    test('should balance quality and cost in provider selection', async () => {
      const prompt = 'Test prompt for balanced selection';
      const taskType = 'general';
      
      const mockResponse = 'Balanced response.';
      
      // Mock provider selection to consider both cost and quality
      jest.spyOn(router as any, 'callProvider').mockResolvedValue(mockResponse);
      
      const result = await router.routeRequest(prompt, taskType);
      
      // Should select provider that balances cost and quality
      expect(result.response).toBe(mockResponse);
    });
  });

  describe('Fallback Integration with Quality', () => {
    test('should handle fallback with quality monitoring', async () => {
      const prompt = 'Test prompt for fallback with quality';
      const taskType = 'analysis';
      const sessionId = 'test-session';
      
      const fallbackResponse = 'Fallback response.';
      
      // Mock first provider to fail, fallback to succeed
      jest.spyOn(router as any, 'callProvider')
        .mockRejectedValueOnce(new Error('First provider failed'))
        .mockResolvedValueOnce(fallbackResponse);
      
      const result = await router.routeRequest(prompt, taskType, sessionId);
      
      expect(result.response).toBe(fallbackResponse);
      expect(result.provider).toBeDefined();
    });

    test('should handle multiple fallback attempts with quality', async () => {
      const prompt = 'Test prompt for multiple fallbacks';
      const taskType = 'general';
      
      const finalResponse = 'Final fallback response.';
      
      // Mock multiple failures, then success
      jest.spyOn(router as any, 'callProvider')
        .mockRejectedValueOnce(new Error('Provider 1 failed'))
        .mockRejectedValueOnce(new Error('Provider 2 failed'))
        .mockResolvedValueOnce(finalResponse);
      
      const result = await router.routeRequest(prompt, taskType);
      
      expect(result.response).toBe(finalResponse);
    });
  });
});
