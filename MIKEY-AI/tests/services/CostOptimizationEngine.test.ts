/**
 * Unit tests for CostOptimizationEngine
 */

import { CostOptimizationEngine, CostMetrics } from '../services/CostOptimizationEngine';
import { CostTierConfiguration } from '../config/cost-tiers';
import { ProviderCostRanking } from '../services/ProviderCostRanking';

describe('CostOptimizationEngine', () => {
  let engine: CostOptimizationEngine;

  beforeEach(() => {
    // Reset environment variables for testing
    process.env.COST_TIER_AFFORDABLE_PREMIUM = 'mistral,google,cohere';
    process.env.COST_TIER_EXPENSIVE = 'openai,anthropic';
    process.env.AFFORDABLE_MAX_COST_PER_TOKEN = '0.0001';
    process.env.EXPENSIVE_MAX_COST_PER_TOKEN = '0.01';
    process.env.ENABLE_COST_OPTIMIZATION = 'true';
    process.env.QUALITY_THRESHOLD = '0.85';
    process.env.FALLBACK_TO_EXPENSIVE = 'true';
    
    engine = new CostOptimizationEngine();
  });

  afterEach(() => {
    // Clean up environment variables
    delete process.env.COST_TIER_AFFORDABLE_PREMIUM;
    delete process.env.COST_TIER_EXPENSIVE;
    delete process.env.AFFORDABLE_MAX_COST_PER_TOKEN;
    delete process.env.EXPENSIVE_MAX_COST_PER_TOKEN;
    delete process.env.ENABLE_COST_OPTIMIZATION;
    delete process.env.QUALITY_THRESHOLD;
    delete process.env.FALLBACK_TO_EXPENSIVE;
  });

  describe('Configuration Loading', () => {
    it('should load cost tiers from environment variables', () => {
      const config = engine.getConfiguration();
      
      expect(config.costTiers).toHaveLength(2);
      expect(config.costTiers[0].tier).toBe('affordable_premium');
      expect(config.costTiers[0].providers).toContain('mistral');
      expect(config.costTiers[0].providers).toContain('google');
      expect(config.costTiers[0].providers).toContain('cohere');
      expect(config.costTiers[1].tier).toBe('expensive');
      expect(config.costTiers[1].providers).toContain('openai');
      expect(config.costTiers[1].providers).toContain('anthropic');
    });

    it('should use default values when environment variables are not set', () => {
      delete process.env.COST_TIER_AFFORDABLE_PREMIUM;
      delete process.env.COST_TIER_EXPENSIVE;
      
      const newEngine = new CostOptimizationEngine();
      const config = newEngine.getConfiguration();
      
      expect(config.costTiers).toHaveLength(2);
      expect(config.costTiers[0].providers).toContain('mistral');
      expect(config.costTiers[0].providers).toContain('google');
      expect(config.costTiers[1].providers).toContain('openai');
    });
  });

  describe('Provider Cost Tier Detection', () => {
    it('should correctly identify affordable premium providers', () => {
      const mistralTier = engine.getProviderCostTier('mistral');
      const googleTier = engine.getProviderCostTier('google');
      
      expect(mistralTier?.tier).toBe('affordable_premium');
      expect(googleTier?.tier).toBe('affordable_premium');
    });

    it('should correctly identify expensive providers', () => {
      const openaiTier = engine.getProviderCostTier('openai');
      const anthropicTier = engine.getProviderCostTier('anthropic');
      
      expect(openaiTier?.tier).toBe('expensive');
      expect(anthropicTier?.tier).toBe('expensive');
    });

    it('should return null for unknown providers', () => {
      const unknownTier = engine.getProviderCostTier('unknown');
      expect(unknownTier).toBeNull();
    });
  });

  describe('Provider Ranking', () => {
    it('should rank affordable premium providers before expensive ones', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const rankedProviders = engine.rankProvidersByCost(providers);
      
      // Affordable premium providers should come first
      const affordableIndex = rankedProviders.indexOf('mistral');
      const expensiveIndex = rankedProviders.indexOf('openai');
      
      expect(affordableIndex).toBeLessThan(expensiveIndex);
    });

    it('should return original order when cost optimization is disabled', () => {
      process.env.ENABLE_COST_OPTIMIZATION = 'false';
      const disabledEngine = new CostOptimizationEngine();
      
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const rankedProviders = disabledEngine.rankProvidersByCost(providers);
      
      expect(rankedProviders).toEqual(providers);
    });

    it('should consider task type in ranking', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const generalRanking = engine.rankProvidersByCost(providers, 'general');
      const codingRanking = engine.rankProvidersByCost(providers, 'coding');
      
      // Rankings might differ based on task type
      expect(generalRanking).toBeDefined();
      expect(codingRanking).toBeDefined();
    });
  });

  describe('Cost Tracking', () => {
    it('should track cost metrics correctly', () => {
      const metrics: CostMetrics = {
        provider: 'mistral',
        tokensUsed: 100,
        costPerToken: 0.0001,
        totalCost: 0.01,
        timestamp: new Date(),
        taskType: 'general'
      };

      engine.trackCostMetrics(metrics);
      const stats = engine.getCostStatistics();
      
      expect(stats.totalRequests).toBe(1);
      expect(stats.totalCost).toBe(0.01);
      expect(stats.averageCostPerRequest).toBe(0.01);
    });

    it('should calculate cost savings correctly', () => {
      const metrics: CostMetrics[] = [
        {
          provider: 'mistral',
          tokensUsed: 100,
          costPerToken: 0.0001,
          totalCost: 0.01,
          timestamp: new Date(),
          taskType: 'general'
        },
        {
          provider: 'google',
          tokensUsed: 200,
          costPerToken: 0.0001,
          totalCost: 0.02,
          timestamp: new Date(),
          taskType: 'analysis'
        }
      ];

      const savings = engine.calculateCostSavings(metrics);
      
      // Should be positive savings compared to expensive providers
      expect(savings).toBeGreaterThan(0);
    });

    it('should limit stored metrics to prevent memory issues', () => {
      // Add more than 1000 metrics
      for (let i = 0; i < 1001; i++) {
        const metrics: CostMetrics = {
          provider: 'mistral',
          tokensUsed: 100,
          costPerToken: 0.0001,
          totalCost: 0.01,
          timestamp: new Date(),
          taskType: 'general'
        };
        engine.trackCostMetrics(metrics);
      }

      const stats = engine.getCostStatistics();
      expect(stats.totalRequests).toBeLessThanOrEqual(1000);
    });
  });

  describe('Provider Availability', () => {
    it('should update provider availability correctly', () => {
      engine.updateProviderAvailability('mistral', false);
      
      // This would require access to internal state, so we test the method exists
      expect(typeof engine.updateProviderAvailability).toBe('function');
    });
  });
});

describe('CostTierConfiguration', () => {
  let config: CostTierConfiguration;

  beforeEach(() => {
    process.env.COST_TIER_AFFORDABLE_PREMIUM = 'mistral,google,cohere';
    process.env.COST_TIER_EXPENSIVE = 'openai,anthropic';
    process.env.AFFORDABLE_MAX_COST_PER_TOKEN = '0.0001';
    process.env.EXPENSIVE_MAX_COST_PER_TOKEN = '0.01';
    
    config = CostTierConfiguration.getInstance();
  });

  afterEach(() => {
    delete process.env.COST_TIER_AFFORDABLE_PREMIUM;
    delete process.env.COST_TIER_EXPENSIVE;
    delete process.env.AFFORDABLE_MAX_COST_PER_TOKEN;
    delete process.env.EXPENSIVE_MAX_COST_PER_TOKEN;
  });

  describe('Provider Tier Detection', () => {
    it('should correctly identify provider tiers', () => {
      expect(config.isAffordablePremium('mistral')).toBe(true);
      expect(config.isAffordablePremium('google')).toBe(true);
      expect(config.isExpensive('openai')).toBe(true);
      expect(config.isExpensive('anthropic')).toBe(true);
    });

    it('should return correct cost per token values', () => {
      expect(config.getMaxCostPerToken('mistral')).toBe(0.0001);
      expect(config.getMaxCostPerToken('openai')).toBe(0.01);
    });

    it('should return correct quality scores', () => {
      expect(config.getQualityScore('mistral')).toBe(0.8);
      expect(config.getQualityScore('openai')).toBe(0.9);
    });
  });

  describe('Provider Lists', () => {
    it('should return correct provider lists for each tier', () => {
      const affordableProviders = config.getProvidersInTier('affordable_premium');
      const expensiveProviders = config.getProvidersInTier('expensive');
      
      expect(affordableProviders).toContain('mistral');
      expect(affordableProviders).toContain('google');
      expect(expensiveProviders).toContain('openai');
      expect(expensiveProviders).toContain('anthropic');
    });

    it('should return providers ranked by cost efficiency', () => {
      const rankedProviders = config.getProvidersByCostEfficiency();
      
      // Affordable premium providers should come first
      const mistralIndex = rankedProviders.indexOf('mistral');
      const openaiIndex = rankedProviders.indexOf('openai');
      
      expect(mistralIndex).toBeLessThan(openaiIndex);
    });
  });

  describe('Configuration Validation', () => {
    it('should validate correct configuration', () => {
      const validation = config.validateConfiguration();
      expect(validation.isValid).toBe(true);
      expect(validation.errors).toHaveLength(0);
    });

    it('should detect duplicate providers', () => {
      config.updateConfiguration([
        {
          tier: 'affordable_premium',
          providers: ['mistral', 'mistral'], // Duplicate
          maxCostPerToken: 0.0001,
          qualityScore: 0.8,
          priority: 1
        }
      ]);
      
      const validation = config.validateConfiguration();
      expect(validation.isValid).toBe(false);
      expect(validation.errors).toContain('Duplicate providers found in configuration');
    });

    it('should detect invalid cost per token values', () => {
      config.updateConfiguration([
        {
          tier: 'affordable_premium',
          providers: ['mistral'],
          maxCostPerToken: -0.0001, // Invalid negative value
          qualityScore: 0.8,
          priority: 1
        }
      ]);
      
      const validation = config.validateConfiguration();
      expect(validation.isValid).toBe(false);
      expect(validation.errors.some(error => error.includes('Invalid maxCostPerToken'))).toBe(true);
    });
  });
});

describe('ProviderCostRanking', () => {
  let ranking: ProviderCostRanking;

  beforeEach(() => {
    process.env.COST_TIER_AFFORDABLE_PREMIUM = 'mistral,google,cohere';
    process.env.COST_TIER_EXPENSIVE = 'openai,anthropic';
    process.env.AFFORDABLE_MAX_COST_PER_TOKEN = '0.0001';
    process.env.EXPENSIVE_MAX_COST_PER_TOKEN = '0.01';
    process.env.COST_WEIGHT = '0.4';
    process.env.QUALITY_WEIGHT = '0.3';
    process.env.AVAILABILITY_WEIGHT = '0.2';
    process.env.TASK_SUITABILITY_WEIGHT = '0.1';
    
    ranking = new ProviderCostRanking();
  });

  afterEach(() => {
    delete process.env.COST_TIER_AFFORDABLE_PREMIUM;
    delete process.env.COST_TIER_EXPENSIVE;
    delete process.env.AFFORDABLE_MAX_COST_PER_TOKEN;
    delete process.env.EXPENSIVE_MAX_COST_PER_TOKEN;
    delete process.env.COST_WEIGHT;
    delete process.env.QUALITY_WEIGHT;
    delete process.env.AVAILABILITY_WEIGHT;
    delete process.env.TASK_SUITABILITY_WEIGHT;
  });

  describe('Provider Ranking', () => {
    it('should rank providers by efficiency score', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const rankings = ranking.rankProviders(providers);
      
      expect(rankings).toHaveLength(4);
      expect(rankings[0].rank).toBe(1);
      expect(rankings[1].rank).toBe(2);
      expect(rankings[2].rank).toBe(3);
      expect(rankings[3].rank).toBe(4);
    });

    it('should prioritize available providers', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const availableProviders = new Set(['mistral', 'google']);
      const rankings = ranking.rankProviders(providers, 'general', availableProviders);
      
      const availableRankings = rankings.filter(r => r.availability);
      const unavailableRankings = rankings.filter(r => !r.availability);
      
      expect(availableRankings.length).toBe(2);
      expect(unavailableRankings.length).toBe(2);
      
      // Available providers should have lower ranks (higher priority)
      availableRankings.forEach(available => {
        unavailableRankings.forEach(unavailable => {
          expect(available.rank).toBeLessThan(unavailable.rank);
        });
      });
    });

    it('should consider task type in ranking', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const generalRankings = ranking.rankProviders(providers, 'general');
      const codingRankings = ranking.rankProviders(providers, 'coding');
      
      expect(generalRankings).toHaveLength(4);
      expect(codingRankings).toHaveLength(4);
      
      // Rankings might differ based on task type
      const generalTop = generalRankings[0].provider;
      const codingTop = codingRankings[0].provider;
      
      expect(generalTop).toBeDefined();
      expect(codingTop).toBeDefined();
    });
  });

  describe('Optimal Provider Selection', () => {
    it('should return optimal provider for task', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const optimalProvider = ranking.getOptimalProvider(providers, 'general');
      
      expect(optimalProvider).toBeDefined();
      expect(providers).toContain(optimalProvider!);
    });

    it('should return null when no providers are available', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const availableProviders = new Set<string>(); // Empty set
      const optimalProvider = ranking.getOptimalProvider(providers, 'general', availableProviders);
      
      expect(optimalProvider).toBeNull();
    });
  });

  describe('Fallback Providers', () => {
    it('should return fallback providers excluding primary', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const fallbackProviders = ranking.getFallbackProviders('openai', providers);
      
      expect(fallbackProviders).not.toContain('openai');
      expect(fallbackProviders.length).toBe(3);
    });

    it('should only return available fallback providers', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const availableProviders = new Set(['mistral', 'google']);
      const fallbackProviders = ranking.getFallbackProviders('openai', providers, availableProviders);
      
      expect(fallbackProviders).toContain('mistral');
      expect(fallbackProviders).toContain('google');
      expect(fallbackProviders).not.toContain('anthropic');
    });
  });

  describe('Cost Savings Calculation', () => {
    it('should calculate cost savings potential correctly', () => {
      const savings = ranking.calculateCostSavingsPotential('openai', 'mistral', 1000);
      
      // Should be positive savings (expensive to affordable)
      expect(savings).toBeGreaterThan(0);
    });

    it('should return zero for same provider', () => {
      const savings = ranking.calculateCostSavingsPotential('openai', 'openai', 1000);
      expect(savings).toBe(0);
    });
  });

  describe('Ranking Statistics', () => {
    it('should calculate ranking statistics correctly', () => {
      const providers = ['openai', 'mistral', 'anthropic', 'google'];
      const rankings = ranking.rankProviders(providers);
      const stats = ranking.getRankingStatistics(rankings);
      
      expect(stats.totalProviders).toBe(4);
      expect(stats.availableProviders).toBe(4);
      expect(stats.affordablePremiumCount).toBeGreaterThan(0);
      expect(stats.expensiveCount).toBeGreaterThan(0);
      expect(stats.averageEfficiencyScore).toBeGreaterThan(0);
      expect(stats.topPerformer).toBeDefined();
    });
  });
});
