/**
 * Provider Cost Ranking Algorithm
 * Intelligent ranking system for LLM providers based on cost efficiency and quality
 */

import { CostTierConfiguration, CostTierConfig } from '../config/cost-tiers';

export interface ProviderRankingResult {
  provider: string;
  rank: number;
  costTier: 'affordable_premium' | 'expensive';
  costPerToken: number;
  qualityScore: number;
  efficiencyScore: number;
  availability: boolean;
  taskSuitability: number;
}

export interface RankingCriteria {
  costWeight: number;
  qualityWeight: number;
  availabilityWeight: number;
  taskSuitabilityWeight: number;
}

export class ProviderCostRanking {
  private costConfig: CostTierConfiguration;
  private criteria: RankingCriteria;

  constructor() {
    this.costConfig = CostTierConfiguration.getInstance();
    this.criteria = this.loadRankingCriteria();
  }

  /**
   * Load ranking criteria from environment variables
   */
  private loadRankingCriteria(): RankingCriteria {
    return {
      costWeight: parseFloat(process.env.COST_WEIGHT || '0.4'),
      qualityWeight: parseFloat(process.env.QUALITY_WEIGHT || '0.3'),
      availabilityWeight: parseFloat(process.env.AVAILABILITY_WEIGHT || '0.2'),
      taskSuitabilityWeight: parseFloat(process.env.TASK_SUITABILITY_WEIGHT || '0.1')
    };
  }

  /**
   * Rank providers by cost efficiency and quality
   */
  public rankProviders(
    providers: string[], 
    taskType: string = 'general',
    availableProviders?: Set<string>
  ): ProviderRankingResult[] {
    const rankings: ProviderRankingResult[] = [];

    for (const provider of providers) {
      const tier = this.costConfig.getProviderTier(provider);
      if (!tier) continue;

      const isAvailable = availableProviders ? availableProviders.has(provider) : true;
      const taskSuitability = this.calculateTaskSuitability(provider, taskType);
      const efficiencyScore = this.calculateEfficiencyScore(provider, tier, taskSuitability);

      rankings.push({
        provider,
        rank: 0, // Will be set after sorting
        costTier: tier.tier,
        costPerToken: tier.maxCostPerToken,
        qualityScore: tier.qualityScore,
        efficiencyScore,
        availability: isAvailable,
        taskSuitability
      });
    }

    // Sort by efficiency score (higher is better) and availability
    rankings.sort((a, b) => {
      // Available providers first
      if (a.availability !== b.availability) {
        return a.availability ? -1 : 1;
      }
      
      // Then by efficiency score
      return b.efficiencyScore - a.efficiencyScore;
    });

    // Assign ranks
    rankings.forEach((ranking, index) => {
      ranking.rank = index + 1;
    });

    return rankings;
  }

  /**
   * Calculate efficiency score combining cost, quality, and task suitability
   */
  private calculateEfficiencyScore(
    provider: string, 
    tier: CostTierConfig, 
    taskSuitability: number
  ): number {
    // Normalize cost (lower is better, so invert)
    const normalizedCost = 1 - Math.min(tier.maxCostPerToken / 0.01, 1);
    
    // Quality score is already normalized (0-1)
    const qualityScore = tier.qualityScore;
    
    // Task suitability is already normalized (0-1)
    const suitabilityScore = taskSuitability;

    // Calculate weighted efficiency score
    const efficiencyScore = 
      (normalizedCost * this.criteria.costWeight) +
      (qualityScore * this.criteria.qualityWeight) +
      (suitabilityScore * this.criteria.taskSuitabilityWeight);

    return Math.max(0, Math.min(1, efficiencyScore));
  }

  /**
   * Calculate task suitability for different task types
   */
  private calculateTaskSuitability(provider: string, taskType: string): number {
    const taskSuitabilityMap: Record<string, Record<string, number>> = {
      'openai': {
        'general': 0.9,
        'coding': 0.95,
        'analysis': 0.9,
        'creative': 0.85,
        'reasoning': 0.95
      },
      'anthropic': {
        'general': 0.9,
        'coding': 0.9,
        'analysis': 0.95,
        'creative': 0.9,
        'reasoning': 0.95
      },
      'google': {
        'general': 0.85,
        'coding': 0.8,
        'analysis': 0.9,
        'creative': 0.85,
        'reasoning': 0.85
      },
      'mistral': {
        'general': 0.8,
        'coding': 0.85,
        'analysis': 0.8,
        'creative': 0.8,
        'reasoning': 0.8
      },
      'cohere': {
        'general': 0.8,
        'coding': 0.75,
        'analysis': 0.85,
        'creative': 0.8,
        'reasoning': 0.8
      },
      'huggingface': {
        'general': 0.75,
        'coding': 0.8,
        'analysis': 0.75,
        'creative': 0.8,
        'reasoning': 0.75
      }
    };

    const providerSuitability = taskSuitabilityMap[provider.toLowerCase()];
    if (!providerSuitability) {
      return 0.8; // Default suitability score
    }

    return providerSuitability[taskType.toLowerCase()] || providerSuitability['general'];
  }

  /**
   * Get optimal provider for a specific task
   */
  public getOptimalProvider(
    providers: string[], 
    taskType: string = 'general',
    availableProviders?: Set<string>
  ): string | null {
    const rankings = this.rankProviders(providers, taskType, availableProviders);
    
    // Return the highest ranked available provider
    const availableRankings = rankings.filter(r => r.availability);
    return availableRankings.length > 0 ? availableRankings[0].provider : null;
  }

  /**
   * Get fallback providers in order of preference
   */
  public getFallbackProviders(
    primaryProvider: string,
    allProviders: string[],
    availableProviders?: Set<string>
  ): string[] {
    const rankings = this.rankProviders(allProviders, 'general', availableProviders);
    
    // Filter out the primary provider and return the rest
    return rankings
      .filter(r => r.provider !== primaryProvider && r.availability)
      .map(r => r.provider);
  }

  /**
   * Calculate cost savings potential for a provider switch
   */
  public calculateCostSavingsPotential(
    currentProvider: string,
    targetProvider: string,
    estimatedTokens: number
  ): number {
    const currentCost = this.costConfig.getMaxCostPerToken(currentProvider);
    const targetCost = this.costConfig.getMaxCostPerToken(targetProvider);
    
    const currentTotalCost = currentCost * estimatedTokens;
    const targetTotalCost = targetCost * estimatedTokens;
    
    return Math.max(0, currentTotalCost - targetTotalCost);
  }

  /**
   * Get ranking statistics
   */
  public getRankingStatistics(rankings: ProviderRankingResult[]): {
    totalProviders: number;
    availableProviders: number;
    affordablePremiumCount: number;
    expensiveCount: number;
    averageEfficiencyScore: number;
    topPerformer: string;
  } {
    const availableRankings = rankings.filter(r => r.availability);
    const affordablePremium = rankings.filter(r => r.costTier === 'affordable_premium');
    const expensive = rankings.filter(r => r.costTier === 'expensive');
    
    const averageEfficiencyScore = rankings.length > 0 
      ? rankings.reduce((sum, r) => sum + r.efficiencyScore, 0) / rankings.length 
      : 0;
    
    const topPerformer = rankings.length > 0 ? rankings[0].provider : '';

    return {
      totalProviders: rankings.length,
      availableProviders: availableRankings.length,
      affordablePremiumCount: affordablePremium.length,
      expensiveCount: expensive.length,
      averageEfficiencyScore,
      topPerformer
    };
  }

  /**
   * Update ranking criteria
   */
  public updateRankingCriteria(newCriteria: Partial<RankingCriteria>): void {
    this.criteria = { ...this.criteria, ...newCriteria };
  }

  /**
   * Get current ranking criteria
   */
  public getRankingCriteria(): RankingCriteria {
    return { ...this.criteria };
  }
}
