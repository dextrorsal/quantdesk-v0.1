/**
 * Unit tests for QualityThresholdManager Escalation Logic
 */

import { QualityThresholdManager } from '../services/QualityThresholdManager';

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

describe('QualityThresholdManager - Escalation Logic', () => {
  let manager: QualityThresholdManager;

  beforeEach(() => {
    // Set up environment variables for testing
    process.env.GLOBAL_MIN_QUALITY = '0.7';
    process.env.ESCALATION_THRESHOLD = '0.6';
    process.env.QUALITY_EVALUATION_ENABLED = 'true';
    process.env.MAX_ESCALATIONS_PER_SESSION = '3';
    process.env.OPENAI_MIN_QUALITY = '0.8';
    process.env.OPENAI_ESCALATION_THRESHOLD = '0.6';
    process.env.GOOGLE_MIN_QUALITY = '0.75';
    process.env.GOOGLE_ESCALATION_THRESHOLD = '0.6';
    process.env.MISTRAL_MIN_QUALITY = '0.7';
    process.env.MISTRAL_ESCALATION_THRESHOLD = '0.5';
    
    manager = new QualityThresholdManager();
  });

  afterEach(() => {
    delete process.env.GLOBAL_MIN_QUALITY;
    delete process.env.ESCALATION_THRESHOLD;
    delete process.env.QUALITY_EVALUATION_ENABLED;
    delete process.env.MAX_ESCALATIONS_PER_SESSION;
    delete process.env.OPENAI_MIN_QUALITY;
    delete process.env.OPENAI_ESCALATION_THRESHOLD;
    delete process.env.GOOGLE_MIN_QUALITY;
    delete process.env.GOOGLE_ESCALATION_THRESHOLD;
    delete process.env.MISTRAL_MIN_QUALITY;
    delete process.env.MISTRAL_ESCALATION_THRESHOLD;
    jest.clearAllMocks();
  });

  describe('Escalation Decision Engine', () => {
    test('should escalate when quality score is below threshold', async () => {
      const decision = await manager.makeEscalationDecision('openai', 0.4, 'test-session');
      
      expect(decision.shouldEscalate).toBe(true);
      expect(decision.qualityScore).toBe(0.4);
      expect(decision.threshold).toBe(0.6);
      expect(decision.reason).toContain('Quality score');
      expect(decision.suggestedProvider).toBeDefined();
      expect(decision.suggestedProvider).not.toBe('openai');
    });

    test('should not escalate when quality score meets threshold', async () => {
      const decision = await manager.makeEscalationDecision('openai', 0.7, 'test-session');
      
      expect(decision.shouldEscalate).toBe(false);
      expect(decision.qualityScore).toBe(0.7);
      expect(decision.threshold).toBe(0.6);
      expect(decision.reason).toContain('Quality score meets threshold');
      expect(decision.suggestedProvider).toBe('openai');
    });

    test('should not escalate when quality score equals threshold', async () => {
      const decision = await manager.makeEscalationDecision('openai', 0.6, 'test-session');
      
      expect(decision.shouldEscalate).toBe(false);
      expect(decision.qualityScore).toBe(0.6);
      expect(decision.threshold).toBe(0.6);
    });

    test('should handle different provider thresholds', async () => {
      // Test OpenAI threshold (0.6)
      const openaiDecision = await manager.makeEscalationDecision('openai', 0.55, 'test-session');
      expect(openaiDecision.shouldEscalate).toBe(true);
      expect(openaiDecision.threshold).toBe(0.6);

      // Test Mistral threshold (0.5)
      const mistralDecision = await manager.makeEscalationDecision('mistral', 0.55, 'test-session');
      expect(mistralDecision.shouldEscalate).toBe(false);
      expect(mistralDecision.threshold).toBe(0.5);
    });

    test('should provide confidence score for escalation decisions', async () => {
      const decision = await manager.makeEscalationDecision('openai', 0.3, 'test-session');
      
      expect(decision.confidence).toBeGreaterThan(0);
      expect(decision.confidence).toBeLessThanOrEqual(1);
    });
  });

  describe('Multi-Provider Fallback Logic', () => {
    test('should try multiple affordable providers before escalating', async () => {
      const currentProvider = 'openai';
      const fallbackProvider = await manager.getOptimalFallbackProvider(currentProvider);
      
      expect(fallbackProvider).toBeDefined();
      expect(fallbackProvider).not.toBe(currentProvider);
      
      // Should prefer affordable premium providers
      const affordableProviders = ['google', 'mistral', 'cohere'];
      expect(affordableProviders).toContain(fallbackProvider);
    });

    test('should exclude current provider from fallback options', async () => {
      const providers = ['openai', 'google', 'mistral', 'cohere', 'huggingface', 'xai'];
      
      for (const provider of providers) {
        const fallbackProvider = await manager.getOptimalFallbackProvider(provider);
        expect(fallbackProvider).not.toBe(provider);
        expect(providers).toContain(fallbackProvider);
      }
    });

    test('should handle case when only one provider is available', async () => {
      // Mock to return only one provider
      const getAvailableProviders = jest.spyOn(manager as any, 'getAvailableProviders');
      getAvailableProviders.mockReturnValue(['openai']);

      const fallbackProvider = await manager.getOptimalFallbackProvider('openai');
      expect(fallbackProvider).toBe('openai'); // Should return current provider

      getAvailableProviders.mockRestore();
    });

    test('should prioritize providers based on quality and cost', async () => {
      // Generate some quality history to influence provider selection
      await manager.evaluateQuality('High quality response', 'analysis', 'google');
      await manager.evaluateQuality('High quality response', 'analysis', 'google');
      await manager.evaluateQuality('Medium quality response', 'analysis', 'mistral');
      
      const fallbackProvider = await manager.getOptimalFallbackProvider('openai');
      
      // Should prefer Google due to higher quality history
      expect(fallbackProvider).toBe('google');
    });
  });

  describe('Escalation Limits and Session Management', () => {
    test('should respect maximum escalations per session', async () => {
      const sessionId = 'test-session-limits';
      
      // Make escalations up to the limit
      for (let i = 0; i < 3; i++) {
        const decision = await manager.makeEscalationDecision('openai', 0.3, sessionId);
        expect(decision.shouldEscalate).toBe(true);
      }
      
      // Next escalation should be blocked
      const blockedDecision = await manager.makeEscalationDecision('openai', 0.3, sessionId);
      expect(blockedDecision.shouldEscalate).toBe(false);
      expect(blockedDecision.reason).toContain('Maximum escalations');
    });

    test('should track escalation count per session', async () => {
      const sessionId = 'test-session-count';
      
      expect(manager.getEscalationCount(sessionId)).toBe(0);
      
      await manager.makeEscalationDecision('openai', 0.3, sessionId);
      expect(manager.getEscalationCount(sessionId)).toBe(1);
      
      await manager.makeEscalationDecision('openai', 0.3, sessionId);
      expect(manager.getEscalationCount(sessionId)).toBe(2);
    });

    test('should reset escalation count for session', async () => {
      const sessionId = 'test-session-reset';
      
      // Make some escalations
      await manager.makeEscalationDecision('openai', 0.3, sessionId);
      await manager.makeEscalationDecision('openai', 0.3, sessionId);
      
      expect(manager.getEscalationCount(sessionId)).toBe(2);
      
      // Reset escalation count
      manager.resetEscalationCount(sessionId);
      expect(manager.getEscalationCount(sessionId)).toBe(0);
      
      // Should be able to escalate again
      const decision = await manager.makeEscalationDecision('openai', 0.3, sessionId);
      expect(decision.shouldEscalate).toBe(true);
    });

    test('should handle different sessions independently', async () => {
      const session1 = 'session-1';
      const session2 = 'session-2';
      
      // Make escalations in session 1
      await manager.makeEscalationDecision('openai', 0.3, session1);
      await manager.makeEscalationDecision('openai', 0.3, session1);
      
      // Make escalations in session 2
      await manager.makeEscalationDecision('openai', 0.3, session2);
      
      expect(manager.getEscalationCount(session1)).toBe(2);
      expect(manager.getEscalationCount(session2)).toBe(1);
      
      // Reset session 1
      manager.resetEscalationCount(session1);
      expect(manager.getEscalationCount(session1)).toBe(0);
      expect(manager.getEscalationCount(session2)).toBe(1); // Should remain unchanged
    });
  });

  describe('Escalation Triggers and Thresholds', () => {
    test('should use provider-specific escalation thresholds', async () => {
      // Test different providers with different thresholds
      const providers = [
        { name: 'openai', threshold: 0.6 },
        { name: 'google', threshold: 0.6 },
        { name: 'mistral', threshold: 0.5 },
        { name: 'cohere', threshold: 0.5 },
        { name: 'huggingface', threshold: 0.45 },
        { name: 'xai', threshold: 0.55 }
      ];

      for (const provider of providers) {
        const decision = await manager.makeEscalationDecision(provider.name, 0.4, 'test-session');
        expect(decision.threshold).toBe(provider.threshold);
      }
    });

    test('should handle missing provider configuration', async () => {
      const decision = await manager.makeEscalationDecision('unknown-provider', 0.4, 'test-session');
      
      // Should use global escalation threshold
      expect(decision.threshold).toBe(0.6);
      expect(decision.shouldEscalate).toBe(true);
    });

    test('should update escalation thresholds dynamically', async () => {
      // Update OpenAI threshold
      await manager.updateQualityThresholds('openai', { escalationThreshold: 0.8 });
      
      const decision = await manager.makeEscalationDecision('openai', 0.7, 'test-session');
      expect(decision.threshold).toBe(0.8);
      expect(decision.shouldEscalate).toBe(true); // 0.7 < 0.8
    });
  });

  describe('Escalation Decision Quality', () => {
    test('should provide detailed escalation reasons', async () => {
      const decision = await manager.makeEscalationDecision('openai', 0.3, 'test-session');
      
      expect(decision.reason).toContain('Quality score');
      expect(decision.reason).toContain('0.300');
      expect(decision.reason).toContain('0.600');
    });

    test('should suggest appropriate fallback providers', async () => {
      const decision = await manager.makeEscalationDecision('openai', 0.3, 'test-session');
      
      expect(decision.suggestedProvider).toBeDefined();
      expect(decision.suggestedProvider).not.toBe('openai');
      expect(['google', 'mistral', 'cohere', 'huggingface', 'xai']).toContain(decision.suggestedProvider);
    });

    test('should maintain escalation decision consistency', async () => {
      // Same inputs should produce same decisions
      const decision1 = await manager.makeEscalationDecision('openai', 0.4, 'test-session');
      const decision2 = await manager.makeEscalationDecision('openai', 0.4, 'test-session');
      
      expect(decision1.shouldEscalate).toBe(decision2.shouldEscalate);
      expect(decision1.threshold).toBe(decision2.threshold);
      expect(decision1.suggestedProvider).toBe(decision2.suggestedProvider);
    });
  });

  describe('Integration with Quality Evaluation', () => {
    test('should integrate escalation with quality evaluation', async () => {
      const response = 'This is a low quality response with minimal detail.';
      const taskType = 'analysis';
      const provider = 'openai';
      
      // Evaluate quality
      const qualityResult = await manager.evaluateQuality(response, taskType, provider);
      
      // Make escalation decision based on quality result
      const escalationDecision = await manager.makeEscalationDecision(
        provider, 
        qualityResult.qualityScore, 
        'test-session'
      );
      
      expect(escalationDecision.qualityScore).toBe(qualityResult.qualityScore);
      expect(escalationDecision.shouldEscalate).toBe(qualityResult.shouldEscalate);
    });

    test('should track escalation statistics', async () => {
      // Generate some escalation data
      await manager.makeEscalationDecision('openai', 0.3, 'session-1');
      await manager.makeEscalationDecision('google', 0.4, 'session-2');
      await manager.makeEscalationDecision('mistral', 0.2, 'session-3');
      
      const stats = manager.getQualityStats();
      
      expect(stats.escalationCount).toBeGreaterThan(0);
      expect(stats.escalationRate).toBeGreaterThan(0);
      expect(stats.escalationRate).toBeLessThanOrEqual(1);
    });
  });

  describe('Error Handling in Escalation', () => {
    test('should handle errors in escalation decision gracefully', async () => {
      // Mock to throw error
      const mockShouldEscalate = jest.spyOn(manager, 'shouldEscalate');
      mockShouldEscalate.mockRejectedValue(new Error('Escalation check failed'));

      const decision = await manager.makeEscalationDecision('openai', 0.3, 'test-session');
      
      // Should return safe fallback decision
      expect(decision.shouldEscalate).toBe(false);
      expect(decision.reason).toContain('error');
      
      mockShouldEscalate.mockRestore();
    });

    test('should handle errors in fallback provider selection', async () => {
      // Mock to throw error
      const mockGetOptimalFallbackProvider = jest.spyOn(manager, 'getOptimalFallbackProvider');
      mockGetOptimalFallbackProvider.mockRejectedValue(new Error('Fallback selection failed'));

      const decision = await manager.makeEscalationDecision('openai', 0.3, 'test-session');
      
      // Should return current provider as fallback
      expect(decision.suggestedProvider).toBe('openai');
      
      mockGetOptimalFallbackProvider.mockRestore();
    });
  });
});
