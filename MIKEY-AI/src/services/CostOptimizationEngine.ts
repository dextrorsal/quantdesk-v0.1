/**
 * Cost Optimization Engine for LLM Router
 * Provides intelligent cost-first routing with configurable provider tiers
 */

export interface ProviderCostTier {
  tier: 'affordable_premium' | 'expensive';
  providers: string[];
  maxCostPerToken: number;
}

export interface CostMetrics {
  provider: string;
  tokensUsed: number;
  costPerToken: number;
  totalCost: number;
  timestamp: Date;
  taskType: string;
}

export interface CostOptimizationConfig {
  costTiers: ProviderCostTier[];
  enableCostOptimization: boolean;
  qualityThreshold: number;
  fallbackToExpensive: boolean;
}

export interface ProviderRanking {
  provider: string;
  costTier: string;
  costPerToken: number;
  qualityScore: number;
  availability: boolean;
  rank: number;
}

export class CostOptimizationEngine {
  private config: CostOptimizationConfig;
  private costMetrics: CostMetrics[] = [];
  private providerRankings: Map<string, ProviderRanking> = new Map();

  constructor() {
    this.config = this.loadConfiguration();
    this.initializeProviderRankings();
  }

  /**
   * Load cost optimization configuration from environment variables
   */
  private loadConfiguration(): CostOptimizationConfig {
    const affordableProviders = process.env.COST_TIER_AFFORDABLE_PREMIUM?.split(',') || [
      'mistral', 'google', 'cohere', 'huggingface'
    ];
    
    const expensiveProviders = process.env.COST_TIER_EXPENSIVE?.split(',') || [
      'openai', 'anthropic'
    ];

    return {
      costTiers: [
        {
          tier: 'affordable_premium',
          providers: affordableProviders,
          maxCostPerToken: parseFloat(process.env.AFFORDABLE_MAX_COST_PER_TOKEN || '0.0001')
        },
        {
          tier: 'expensive',
          providers: expensiveProviders,
          maxCostPerToken: parseFloat(process.env.EXPENSIVE_MAX_COST_PER_TOKEN || '0.01')
        }
      ],
      enableCostOptimization: process.env.ENABLE_COST_OPTIMIZATION === 'true',
      qualityThreshold: parseFloat(process.env.QUALITY_THRESHOLD || '0.85'),
      fallbackToExpensive: process.env.FALLBACK_TO_EXPENSIVE === 'true'
    };
  }

  /**
   * Initialize provider rankings based on cost tiers
   */
  private initializeProviderRankings(): void {
    let rank = 1;
    
    // Rank affordable premium providers first
    const affordableTier = this.config.costTiers.find(tier => tier.tier === 'affordable_premium');
    if (affordableTier) {
      affordableTier.providers.forEach(provider => {
        this.providerRankings.set(provider, {
          provider,
          costTier: 'affordable_premium',
          costPerToken: affordableTier.maxCostPerToken,
          qualityScore: 0.8, // Default quality score
          availability: true,
          rank: rank++
        });
      });
    }

    // Rank expensive providers second
    const expensiveTier = this.config.costTiers.find(tier => tier.tier === 'expensive');
    if (expensiveTier) {
      expensiveTier.providers.forEach(provider => {
        this.providerRankings.set(provider, {
          provider,
          costTier: 'expensive',
          costPerToken: expensiveTier.maxCostPerToken,
          qualityScore: 0.9, // Higher quality score for expensive providers
          availability: true,
          rank: rank++
        });
      });
    }
  }

  /**
   * Get provider cost tier information
   */
  public getProviderCostTier(provider: string): ProviderCostTier | null {
    const tier = this.config.costTiers.find(t => t.providers.includes(provider));
    return tier || null;
  }

  /**
   * Rank providers by cost efficiency while considering task requirements
   */
  public rankProvidersByCost(providers: string[], taskType: string = 'general'): string[] {
    if (!this.config.enableCostOptimization) {
      return providers; // Return original order if cost optimization disabled
    }

    const rankedProviders = providers
      .map(provider => this.providerRankings.get(provider))
      .filter(ranking => ranking && ranking.availability)
      .sort((a, b) => {
        // Primary sort: cost tier (affordable_premium first)
        if (a!.costTier !== b!.costTier) {
          return a!.costTier === 'affordable_premium' ? -1 : 1;
        }
        // Secondary sort: cost per token (lower is better)
        return a!.costPerToken - b!.costPerToken;
      })
      .map(ranking => ranking!.provider);

    return rankedProviders;
  }

  /**
   * Calculate cost savings compared to using only expensive providers
   */
  public calculateCostSavings(metrics: CostMetrics[]): number {
    if (metrics.length === 0) return 0;

    const totalActualCost = metrics.reduce((sum, metric) => sum + metric.totalCost, 0);
    
    // Calculate what it would have cost using only expensive providers
    const expensiveCostPerToken = this.config.costTiers
      .find(tier => tier.tier === 'expensive')?.maxCostPerToken || 0.01;
    
    const totalExpensiveCost = metrics.reduce((sum, metric) => 
      sum + (metric.tokensUsed * expensiveCostPerToken), 0);

    const savings = totalExpensiveCost - totalActualCost;
    return Math.max(0, savings);
  }

  /**
   * Track cost metrics for analytics
   */
  public trackCostMetrics(metrics: CostMetrics): void {
    this.costMetrics.push(metrics);
    
    // Keep only last 1000 metrics to prevent memory issues
    if (this.costMetrics.length > 1000) {
      this.costMetrics = this.costMetrics.slice(-1000);
    }
  }

  /**
   * Get cost optimization statistics
   */
  public getCostStatistics(): {
    totalRequests: number;
    totalCost: number;
    averageCostPerRequest: number;
    costSavings: number;
    savingsPercentage: number;
  } {
    const totalRequests = this.costMetrics.length;
    const totalCost = this.costMetrics.reduce((sum, metric) => sum + metric.totalCost, 0);
    const averageCostPerRequest = totalRequests > 0 ? totalCost / totalRequests : 0;
    const costSavings = this.calculateCostSavings(this.costMetrics);
    const savingsPercentage = totalCost > 0 ? (costSavings / (totalCost + costSavings)) * 100 : 0;

    return {
      totalRequests,
      totalCost,
      averageCostPerRequest,
      costSavings,
      savingsPercentage
    };
  }

  /**
   * Update provider availability status
   */
  public updateProviderAvailability(provider: string, isAvailable: boolean): void {
    const ranking = this.providerRankings.get(provider);
    if (ranking) {
      ranking.availability = isAvailable;
    }
  }

  /**
   * Get configuration for external access
   */
  public getConfiguration(): CostOptimizationConfig {
    return { ...this.config };
  }
}
