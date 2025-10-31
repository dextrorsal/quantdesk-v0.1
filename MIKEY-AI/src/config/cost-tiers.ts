/**
 * Cost Tier Configuration
 * Environment-based configuration for LLM provider cost tiers
 */

export interface CostTierConfig {
  tier: 'affordable_premium' | 'expensive';
  providers: string[];
  maxCostPerToken: number;
  qualityScore: number;
  priority: number;
}

export interface CostOptimizationSettings {
  enableCostOptimization: boolean;
  qualityThreshold: number;
  fallbackToExpensive: boolean;
  maxRetriesPerTier: number;
  costSavingsTarget: number;
}

export class CostTierConfiguration {
  private static instance: CostTierConfiguration;
  private config: CostTierConfig[];
  private settings: CostOptimizationSettings;

  private constructor() {
    this.config = this.loadCostTierConfig();
    this.settings = this.loadOptimizationSettings();
  }

  public static getInstance(): CostTierConfiguration {
    if (!CostTierConfiguration.instance) {
      CostTierConfiguration.instance = new CostTierConfiguration();
    }
    return CostTierConfiguration.instance;
  }

  /**
   * Load cost tier configuration from environment variables
   */
  private loadCostTierConfig(): CostTierConfig[] {
    // Affordable Premium Tier (Priority 1)
    const affordableProviders = this.parseProviderList(
      process.env.COST_TIER_AFFORDABLE_PREMIUM || 'mistral,google,cohere,huggingface'
    );
    
    // Expensive Tier (Priority 2)
    const expensiveProviders = this.parseProviderList(
      process.env.COST_TIER_EXPENSIVE || 'openai,anthropic'
    );

    return [
      {
        tier: 'affordable_premium',
        providers: affordableProviders,
        maxCostPerToken: parseFloat(process.env.AFFORDABLE_MAX_COST_PER_TOKEN || '0.0001'),
        qualityScore: parseFloat(process.env.AFFORDABLE_QUALITY_SCORE || '0.8'),
        priority: 1
      },
      {
        tier: 'expensive',
        providers: expensiveProviders,
        maxCostPerToken: parseFloat(process.env.EXPENSIVE_MAX_COST_PER_TOKEN || '0.01'),
        qualityScore: parseFloat(process.env.EXPENSIVE_QUALITY_SCORE || '0.9'),
        priority: 2
      }
    ];
  }

  /**
   * Load optimization settings from environment variables
   */
  private loadOptimizationSettings(): CostOptimizationSettings {
    return {
      enableCostOptimization: process.env.ENABLE_COST_OPTIMIZATION !== 'false',
      qualityThreshold: parseFloat(process.env.QUALITY_THRESHOLD || '0.85'),
      fallbackToExpensive: process.env.FALLBACK_TO_EXPENSIVE !== 'false',
      maxRetriesPerTier: parseInt(process.env.MAX_RETRIES_PER_TIER || '2'),
      costSavingsTarget: parseFloat(process.env.COST_SAVINGS_TARGET || '60')
    };
  }

  /**
   * Parse comma-separated provider list from environment variable
   */
  private parseProviderList(providerString: string): string[] {
    return providerString
      .split(',')
      .map(provider => provider.trim().toLowerCase())
      .filter(provider => provider.length > 0);
  }

  /**
   * Get cost tier configuration for a specific provider
   */
  public getProviderTier(provider: string): CostTierConfig | null {
    return this.config.find(tier => tier.providers.includes(provider.toLowerCase())) || null;
  }

  /**
   * Get all providers in a specific tier
   */
  public getProvidersInTier(tier: 'affordable_premium' | 'expensive'): string[] {
    const tierConfig = this.config.find(t => t.tier === tier);
    return tierConfig ? tierConfig.providers : [];
  }

  /**
   * Get providers ranked by cost efficiency
   */
  public getProvidersByCostEfficiency(): string[] {
    const rankedProviders: string[] = [];
    
    // Add affordable premium providers first
    const affordableProviders = this.getProvidersInTier('affordable_premium');
    rankedProviders.push(...affordableProviders);
    
    // Add expensive providers second
    const expensiveProviders = this.getProvidersInTier('expensive');
    rankedProviders.push(...expensiveProviders);
    
    return rankedProviders;
  }

  /**
   * Check if provider is in affordable premium tier
   */
  public isAffordablePremium(provider: string): boolean {
    const tier = this.getProviderTier(provider);
    return tier?.tier === 'affordable_premium';
  }

  /**
   * Check if provider is in expensive tier
   */
  public isExpensive(provider: string): boolean {
    const tier = this.getProviderTier(provider);
    return tier?.tier === 'expensive';
  }

  /**
   * Get maximum cost per token for a provider
   */
  public getMaxCostPerToken(provider: string): number {
    const tier = this.getProviderTier(provider);
    return tier?.maxCostPerToken || 0.01; // Default to expensive tier cost
  }

  /**
   * Get quality score for a provider
   */
  public getQualityScore(provider: string): number {
    const tier = this.getProviderTier(provider);
    return tier?.qualityScore || 0.8; // Default quality score
  }

  /**
   * Get optimization settings
   */
  public getSettings(): CostOptimizationSettings {
    return { ...this.settings };
  }

  /**
   * Update configuration at runtime (for testing)
   */
  public updateConfiguration(newConfig: Partial<CostTierConfig[]>, newSettings?: Partial<CostOptimizationSettings>): void {
    if (newConfig) {
      this.config = newConfig as CostTierConfig[];
    }
    if (newSettings) {
      this.settings = { ...this.settings, ...newSettings };
    }
  }

  /**
   * Validate configuration
   */
  public validateConfiguration(): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Check for duplicate providers
    const allProviders = this.config.flatMap(tier => tier.providers);
    const uniqueProviders = new Set(allProviders);
    if (allProviders.length !== uniqueProviders.size) {
      errors.push('Duplicate providers found in configuration');
    }

    // Check for valid cost per token values
    this.config.forEach(tier => {
      if (tier.maxCostPerToken <= 0) {
        errors.push(`Invalid maxCostPerToken for tier ${tier.tier}: ${tier.maxCostPerToken}`);
      }
      if (tier.qualityScore < 0 || tier.qualityScore > 1) {
        errors.push(`Invalid qualityScore for tier ${tier.tier}: ${tier.qualityScore}`);
      }
    });

    // Check settings validity
    if (this.settings.qualityThreshold < 0 || this.settings.qualityThreshold > 1) {
      errors.push(`Invalid qualityThreshold: ${this.settings.qualityThreshold}`);
    }
    if (this.settings.maxRetriesPerTier < 0) {
      errors.push(`Invalid maxRetriesPerTier: ${this.settings.maxRetriesPerTier}`);
    }
    if (this.settings.costSavingsTarget < 0 || this.settings.costSavingsTarget > 100) {
      errors.push(`Invalid costSavingsTarget: ${this.settings.costSavingsTarget}`);
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }
}
