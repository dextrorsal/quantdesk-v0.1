/**
 * Unit tests for QualityThresholdManager
 */

import { QualityThresholdManager } from '../services/QualityThresholdManager';
import { QualityConfig } from '../config/quality-config';

// Mock CostOptimizationEngine
jest.mock('../services/CostOptimizationEngine', () => {
  return {
    CostOptimizationEngine: jest.fn().mockImplementation(() => ({
      getProviderCostTier: jest.fn().mockReturnValue({
        tier: 'affordable_premium',
        providers: ['google', 'mistral'],
        maxCostPerToken: 0.0002
      })
    }))
  };
});

describe('QualityThresholdManager', () => {
  let manager: QualityThresholdManager;
  let mockConfig: any;

  beforeEach(() => {
    // Set up environment variables for testing
    process.env.GLOBAL_MIN_QUALITY = '0.7';
    process.env.ESCALATION_THRESHOLD = '0.6';
    process.env.QUALITY_EVALUATION_ENABLED = 'true';
    process.env.OPENAI_MIN_QUALITY = '0.8';
    process.env.OPENAI_ESCALATION_THRESHOLD = '0.6';
    
    manager = new QualityThresholdManager();
    mockConfig = (manager as any).config;
  });

  afterEach(() => {
    delete process.env.GLOBAL_MIN_QUALITY;
    delete process.env.ESCALATION_THRESHOLD;
    delete process.env.QUALITY_EVALUATION_ENABLED;
    delete process.env.OPENAI_MIN_QUALITY;
    delete process.env.OPENAI_ESCALATION_THRESHOLD;
    jest.clearAllMocks();
  });

  describe('Quality Evaluation', () => {
    test('should evaluate quality of a response', async () => {
      const response = 'This is a comprehensive analysis of the trading data. The market shows strong bullish trends with increasing volume and positive momentum indicators.';
      const taskType = 'analysis';
      const provider = 'openai';

      const result = await manager.evaluateQuality(response, taskType, provider);

      expect(result.provider).toBe(provider);
      expect(result.qualityScore).toBeGreaterThan(0);
      expect(result.qualityScore).toBeLessThanOrEqual(1);
      expect(result.evaluationCriteria).toBeDefined();
      expect(result.shouldEscalate).toBeDefined();
      expect(result.confidence).toBeGreaterThanOrEqual(0);
      expect(result.timestamp).toBeInstanceOf(Date);
    });

    test('should handle different task types appropriately', async () => {
      const responses = {
        analysis: 'Based on the market data analysis, I can see clear trends indicating bullish momentum.',
        general: 'Hello! How can I help you today?',
        code: 'function calculatePrice(amount, rate) { return amount * rate; }'
      };

      for (const [taskType, response] of Object.entries(responses)) {
        const result = await manager.evaluateQuality(response, taskType, 'openai');
        
        expect(result.qualityScore).toBeGreaterThan(0);
        expect(result.evaluationCriteria).toBeDefined();
      }
    });

    test('should handle empty or invalid responses gracefully', async () => {
      const invalidResponses = ['', '   ', 'a', 'This is too short'];

      for (const response of invalidResponses) {
        const result = await manager.evaluateQuality(response, 'general', 'openai');
        
        expect(result.qualityScore).toBeGreaterThanOrEqual(0);
        expect(result.qualityScore).toBeLessThanOrEqual(1);
        expect(result.confidence).toBeGreaterThanOrEqual(0);
      }
    });

    test('should track quality history', async () => {
      const response = 'Test response for quality tracking';
      
      // Clear history first
      manager.clearQualityHistory();
      
      await manager.evaluateQuality(response, 'general', 'openai');
      await manager.evaluateQuality(response, 'analysis', 'google');
      
      const history = manager.getQualityHistory();
      expect(history).toHaveLength(2);
      expect(history[0].provider).toBe('openai');
      expect(history[1].provider).toBe('google');
    });
  });

  describe('Escalation Logic', () => {
    test('should determine escalation correctly', async () => {
      // High quality - should not escalate
      const highQualityResult = await manager.shouldEscalate(0.8, 0.6);
      expect(highQualityResult).toBe(false);

      // Low quality - should escalate
      const lowQualityResult = await manager.shouldEscalate(0.4, 0.6);
      expect(lowQualityResult).toBe(true);

      // Borderline quality - should escalate
      const borderlineResult = await manager.shouldEscalate(0.6, 0.6);
      expect(borderlineResult).toBe(false); // Equal to threshold, no escalation
    });

    test('should get optimal fallback provider', async () => {
      const currentProvider = 'openai';
      const fallbackProvider = await manager.getOptimalFallbackProvider(currentProvider);
      
      expect(fallbackProvider).toBeDefined();
      expect(fallbackProvider).not.toBe(currentProvider);
      expect(['google', 'mistral', 'cohere', 'huggingface', 'xai']).toContain(fallbackProvider);
    });

    test('should handle case when no fallback providers available', async () => {
      // Mock to return empty array
      const getAvailableProviders = jest.spyOn(manager as any, 'getAvailableProviders');
      getAvailableProviders.mockReturnValue(['openai']);

      const fallbackProvider = await manager.getOptimalFallbackProvider('openai');
      expect(fallbackProvider).toBe('openai'); // Should return current provider

      getAvailableProviders.mockRestore();
    });

    test('should make escalation decision correctly', async () => {
      const sessionId = 'test-session';
      
      // Test escalation decision
      const decision = await manager.makeEscalationDecision('openai', 0.4, sessionId);
      
      expect(decision.shouldEscalate).toBe(true);
      expect(decision.currentProvider).toBe('openai');
      expect(decision.suggestedProvider).toBeDefined();
      expect(decision.qualityScore).toBe(0.4);
      expect(decision.confidence).toBeGreaterThan(0);
      expect(decision.reason).toContain('Quality score');
    });

    test('should respect maximum escalations per session', async () => {
      const sessionId = 'test-session-max';
      
      // Set max escalations to 2
      manager.updateQualityThresholds('openai', { maxEscalationsPerSession: 2 });
      
      // First escalation
      const decision1 = await manager.makeEscalationDecision('openai', 0.3, sessionId);
      expect(decision1.shouldEscalate).toBe(true);
      
      // Second escalation
      const decision2 = await manager.makeEscalationDecision('openai', 0.3, sessionId);
      expect(decision2.shouldEscalate).toBe(true);
      
      // Third escalation should be blocked
      const decision3 = await manager.makeEscalationDecision('openai', 0.3, sessionId);
      expect(decision3.shouldEscalate).toBe(false);
      expect(decision3.reason).toContain('Maximum escalations');
    });
  });

  describe('Configuration Management', () => {
    test('should update quality thresholds', async () => {
      const newConfig = {
        minQualityScore: 0.9,
        escalationThreshold: 0.7,
        enabled: true
      };

      await manager.updateQualityThresholds('openai', newConfig);
      
      // Verify configuration was updated
      const threshold = mockConfig.getThreshold('openai');
      expect(threshold.minQualityScore).toBe(0.9);
      expect(threshold.escalationThreshold).toBe(0.7);
    });

    test('should check if quality evaluation is enabled', () => {
      expect(manager.isQualityEvaluationEnabled()).toBe(true);
    });

    test('should enable/disable quality evaluation', () => {
      manager.setQualityEvaluationEnabled(false);
      expect(manager.isQualityEvaluationEnabled()).toBe(false);
      
      manager.setQualityEvaluationEnabled(true);
      expect(manager.isQualityEvaluationEnabled()).toBe(true);
    });

    test('should reset escalation count for session', () => {
      const sessionId = 'test-session-reset';
      
      // Make some escalations
      manager.makeEscalationDecision('openai', 0.3, sessionId);
      manager.makeEscalationDecision('openai', 0.3, sessionId);
      
      expect(manager.getEscalationCount(sessionId)).toBe(2);
      
      // Reset escalation count
      manager.resetEscalationCount(sessionId);
      expect(manager.getEscalationCount(sessionId)).toBe(0);
    });
  });

  describe('Statistics and Analytics', () => {
    test('should provide quality statistics', async () => {
      // Generate some test data
      await manager.evaluateQuality('Test response 1', 'general', 'openai');
      await manager.evaluateQuality('Test response 2', 'analysis', 'google');
      
      const stats = manager.getQualityStats();
      
      expect(stats.totalEvaluations).toBeGreaterThan(0);
      expect(stats.averageQualityScore).toBeGreaterThan(0);
      expect(stats.averageQualityScore).toBeLessThanOrEqual(1);
      expect(stats.escalationCount).toBeGreaterThanOrEqual(0);
      expect(stats.escalationRate).toBeGreaterThanOrEqual(0);
      expect(stats.escalationRate).toBeLessThanOrEqual(1);
      expect(stats.userSatisfactionRate).toBeGreaterThanOrEqual(0);
      expect(stats.providerQualityBreakdown).toBeDefined();
    });

    test('should provide provider quality profiles', () => {
      const profiles = manager.getProviderProfiles();
      
      expect(Array.isArray(profiles)).toBe(true);
      expect(profiles.length).toBeGreaterThan(0);
      
      profiles.forEach(profile => {
        expect(profile.provider).toBeDefined();
        expect(profile.averageQuality).toBeGreaterThanOrEqual(0);
        expect(profile.averageQuality).toBeLessThanOrEqual(1);
        expect(profile.qualityConsistency).toBeGreaterThanOrEqual(0);
        expect(profile.escalationRate).toBeGreaterThanOrEqual(0);
        expect(profile.userSatisfaction).toBeGreaterThanOrEqual(0);
        expect(Array.isArray(profile.strengths)).toBe(true);
        expect(Array.isArray(profile.weaknesses)).toBe(true);
      });
    });

    test('should get quality history with limit', async () => {
      // Clear history first
      manager.clearQualityHistory();
      
      // Generate test data
      for (let i = 0; i < 5; i++) {
        await manager.evaluateQuality(`Test response ${i}`, 'general', 'openai');
      }
      
      const fullHistory = manager.getQualityHistory();
      const limitedHistory = manager.getQualityHistory(3);
      
      expect(fullHistory).toHaveLength(5);
      expect(limitedHistory).toHaveLength(3);
      expect(limitedHistory[0]).toBe(fullHistory[2]); // Should get last 3
    });

    test('should clear quality history', async () => {
      // Generate some test data
      await manager.evaluateQuality('Test response', 'general', 'openai');
      
      expect(manager.getQualityHistory()).toHaveLength(1);
      
      // Clear history
      manager.clearQualityHistory();
      
      expect(manager.getQualityHistory()).toHaveLength(0);
    });
  });

  describe('Quality Metrics', () => {
    test('should get quality metrics for a response', async () => {
      const response = 'This is a comprehensive analysis with detailed explanations and clear conclusions.';
      const taskType = 'analysis';
      
      const metrics = await manager.getQualityMetrics(response, taskType);
      
      expect(metrics.responseLength).toBe(response.length);
      expect(metrics.coherenceScore).toBeGreaterThanOrEqual(0);
      expect(metrics.coherenceScore).toBeLessThanOrEqual(1);
      expect(metrics.relevanceScore).toBeGreaterThanOrEqual(0);
      expect(metrics.relevanceScore).toBeLessThanOrEqual(1);
      expect(metrics.completenessScore).toBeGreaterThanOrEqual(0);
      expect(metrics.completenessScore).toBeLessThanOrEqual(1);
      expect(metrics.errorRate).toBeGreaterThanOrEqual(0);
      expect(metrics.userSatisfactionScore).toBeGreaterThanOrEqual(0);
      expect(metrics.userSatisfactionScore).toBeLessThanOrEqual(1);
    });

    test('should handle different response lengths appropriately', async () => {
      const responses = [
        'Short response.',
        'This is a medium length response with some detail.',
        'This is a very long response with extensive detail and comprehensive analysis that covers multiple aspects of the topic in great depth.'
      ];

      for (const response of responses) {
        const metrics = await manager.getQualityMetrics(response, 'general');
        
        expect(metrics.responseLength).toBe(response.length);
        expect(metrics.completenessScore).toBeGreaterThanOrEqual(0);
        expect(metrics.completenessScore).toBeLessThanOrEqual(1);
      }
    });
  });

  describe('Error Handling', () => {
    test('should handle missing provider configuration gracefully', async () => {
      const response = 'Test response';
      const taskType = 'general';
      const unknownProvider = 'unknown-provider';

      const result = await manager.evaluateQuality(response, taskType, unknownProvider);
      
      // Should return fallback result
      expect(result.provider).toBe(unknownProvider);
      expect(result.qualityScore).toBe(0.5); // Fallback score
      expect(result.confidence).toBe(0.0); // Low confidence
    });

    test('should handle evaluation errors gracefully', async () => {
      // Mock evaluation criteria to throw error
      const mockEvaluator = jest.fn().mockRejectedValue(new Error('Evaluation failed'));
      const mockCriteria = [{
        name: 'test',
        weight: 1.0,
        evaluator: mockEvaluator,
        description: 'Test criterion'
      }];

      jest.spyOn(mockConfig, 'getEvaluationCriteria').mockReturnValue(mockCriteria);

      const result = await manager.evaluateQuality('Test response', 'general', 'openai');
      
      // Should return fallback result
      expect(result.qualityScore).toBe(0.5);
      expect(result.confidence).toBe(0.0);
    });
  });

  describe('Provider Profile Updates', () => {
    test('should update provider profiles based on quality evaluations', async () => {
      const provider = 'openai';
      
      // Initial profile
      const initialProfiles = manager.getProviderProfiles();
      const initialProfile = initialProfiles.find(p => p.provider === provider);
      expect(initialProfile).toBeDefined();
      
      // Evaluate quality multiple times
      await manager.evaluateQuality('High quality response', 'analysis', provider);
      await manager.evaluateQuality('Another high quality response', 'general', provider);
      
      // Check if profile was updated
      const updatedProfiles = manager.getProviderProfiles();
      const updatedProfile = updatedProfiles.find(p => p.provider === provider);
      
      expect(updatedProfile).toBeDefined();
      expect(updatedProfile!.averageQuality).toBeGreaterThanOrEqual(0);
    });
  });
});
