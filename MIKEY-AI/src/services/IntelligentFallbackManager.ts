/**
 * Intelligent Fallback Logic
 * Advanced fallback mechanisms with cost-quality balanced selection
 */

import { 
  FallbackDecision, 
  FallbackStrategy, 
  ProviderStatus, 
  CircuitBreakerState 
} from '../types/provider-health';
import { ProviderHealthMonitor } from './ProviderHealthMonitor';
import { CostOptimizationEngine } from './CostOptimizationEngine';
import { QualityThresholdManager } from './QualityThresholdManager';
import { FallbackConfiguration } from '../config/fallback-config';
import { systemLogger, errorLogger } from '../utils/logger';

export class IntelligentFallbackManager {
  private healthMonitor: ProviderHealthMonitor;
  private costEngine: CostOptimizationEngine;
  private qualityManager: QualityThresholdManager;
  private config: FallbackConfiguration;
  private fallbackHistory: Map<string, number> = new Map(); // Track fallback frequency per provider

  constructor() {
    this.healthMonitor = new ProviderHealthMonitor();
    this.costEngine = new CostOptimizationEngine();
    this.qualityManager = new QualityThresholdManager();
    this.config = FallbackConfiguration.getInstance();
    
    systemLogger.startup('IntelligentFallbackManager', 'Initialized with intelligent fallback logic');
  }

  /**
   * Make intelligent fallback decision
   */
  public async makeFallbackDecision(
    originalProvider: string,
    error: Error,
    retryCount: number,
    taskType: string = 'general'
  ): Promise<FallbackDecision> {
    try {
      const strategy = this.config.getFallbackStrategy();
      const maxRetries = this.config.getMaxRetries();
      
      // Check if we should fallback
      if (retryCount >= maxRetries) {
        return {
          shouldFallback: false,
          reason: 'Maximum retries exceeded',
          suggestedProvider: originalProvider,
          retryCount,
          maxRetries,
          estimatedDelay: 0
        };
      }

      // Get available providers
      const availableProviders = await this.getAvailableProviders(originalProvider);
      
      if (availableProviders.length === 0) {
        return {
          shouldFallback: false,
          reason: 'No available providers for fallback',
          suggestedProvider: originalProvider,
          retryCount,
          maxRetries,
          estimatedDelay: 0
        };
      }

      // Select best fallback provider based on strategy
      const suggestedProvider = await this.selectBestFallbackProvider(
        availableProviders,
        strategy,
        taskType,
        error
      );

      if (!suggestedProvider) {
        return {
          shouldFallback: false,
          reason: 'No suitable fallback provider found',
          suggestedProvider: originalProvider,
          retryCount,
          maxRetries,
          estimatedDelay: 0
        };
      }

      // Calculate estimated delay
      const estimatedDelay = this.calculateRetryDelay(retryCount);

      // Update fallback history
      this.updateFallbackHistory(suggestedProvider);

      return {
        shouldFallback: true,
        reason: this.getFallbackReason(error, originalProvider, suggestedProvider),
        suggestedProvider,
        retryCount,
        maxRetries,
        estimatedDelay
      };

    } catch (error) {
      errorLogger.aiError(error as Error, 'Fallback decision making');
      
      const configRetries = this.config.getMaxRetries();
      const maxRetriesValue = configRetries || 3;
      return {
        shouldFallback: false,
        reason: 'Fallback decision error',
        suggestedProvider: originalProvider,
        retryCount,
        maxRetries: maxRetriesValue,
        estimatedDelay: 0
      };
    }
  }

  /**
   * Get available providers for fallback
   */
  public async getAvailableProviders(excludeProvider: string): Promise<string[]> {
    try {
      const healthyProviders = await this.healthMonitor.getHealthyProviders();
      const fallbackOrder = this.config.getFallbackOrder();
      
      // Filter out the original provider and sort by fallback order
      const availableProviders = healthyProviders
        .filter(provider => provider !== excludeProvider)
        .sort((a, b) => {
          const aIndex = fallbackOrder.indexOf(a);
          const bIndex = fallbackOrder.indexOf(b);
          
          // Providers not in fallback order go to the end
          if (aIndex === -1 && bIndex === -1) return 0;
          if (aIndex === -1) return 1;
          if (bIndex === -1) return -1;
          
          return aIndex - bIndex;
        });

      return availableProviders;

    } catch (error) {
      errorLogger.aiError(error as Error, 'Getting available providers');
      return [];
    }
  }

  /**
   * Select best fallback provider based on strategy
   */
  private async selectBestFallbackProvider(
    availableProviders: string[],
    strategy: FallbackStrategy,
    taskType: string,
    error: Error
  ): Promise<string | null> {
    try {
      switch (strategy.strategy) {
        case 'cost-first':
          return await this.selectCostFirstProvider(availableProviders, strategy);
          
        case 'quality-first':
          return await this.selectQualityFirstProvider(availableProviders, strategy, taskType);
          
        case 'availability-first':
          return await this.selectAvailabilityFirstProvider(availableProviders);
          
        case 'balanced':
        default:
          return await this.selectBalancedProvider(availableProviders, strategy, taskType);
      }
    } catch (error) {
      errorLogger.aiError(error as Error, 'Selecting fallback provider');
      return null;
    }
  }

  /**
   * Select provider based on cost optimization
   */
  private async selectCostFirstProvider(
    availableProviders: string[],
    strategy: FallbackStrategy
  ): Promise<string | null> {
    try {
      const costStats = this.costEngine.getCostStatistics();
      const providerCosts = new Map<string, number>();
      
      // Get cost information for each provider
      for (const provider of availableProviders) {
        const providerCost = await this.getProviderCost(provider);
        providerCosts.set(provider, providerCost);
      }
      
      // Sort by cost (ascending)
      const sortedProviders = availableProviders.sort((a, b) => {
        const costA = providerCosts.get(a) || 0;
        const costB = providerCosts.get(b) || 0;
        return costA - costB;
      });
      
      // Filter by cost threshold if specified
      if (strategy.costThreshold > 0) {
        const filteredProviders = sortedProviders.filter(provider => {
          const cost = providerCosts.get(provider) || 0;
          return cost <= strategy.costThreshold;
        });
        
        if (filteredProviders.length > 0) {
          return filteredProviders[0];
        }
      }
      
      return sortedProviders[0] || null;
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Cost-first provider selection');
      return null;
    }
  }

  /**
   * Select provider based on quality optimization
   */
  private async selectQualityFirstProvider(
    availableProviders: string[],
    strategy: FallbackStrategy,
    taskType: string
  ): Promise<string | null> {
    try {
      const providerQualities = new Map<string, number>();
      
      // Get quality information for each provider
      for (const provider of availableProviders) {
        const quality = await this.getProviderQuality(provider, taskType);
        providerQualities.set(provider, quality);
      }
      
      // Sort by quality (descending)
      const sortedProviders = availableProviders.sort((a, b) => {
        const qualityA = providerQualities.get(a) || 0;
        const qualityB = providerQualities.get(b) || 0;
        return qualityB - qualityA;
      });
      
      // Filter by quality threshold if specified
      if (strategy.qualityThreshold > 0) {
        const filteredProviders = sortedProviders.filter(provider => {
          const quality = providerQualities.get(provider) || 0;
          return quality >= strategy.qualityThreshold;
        });
        
        if (filteredProviders.length > 0) {
          return filteredProviders[0];
        }
      }
      
      return sortedProviders[0] || null;
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Quality-first provider selection');
      return null;
    }
  }

  /**
   * Select provider based on availability
   */
  private async selectAvailabilityFirstProvider(
    availableProviders: string[]
  ): Promise<string | null> {
    try {
      const providerAvailabilities = new Map<string, number>();
      
      // Get availability information for each provider
      for (const provider of availableProviders) {
        const metrics = this.healthMonitor.getProviderHealthMetrics(provider);
        if (metrics) {
          providerAvailabilities.set(provider, metrics.availability);
        }
      }
      
      // Sort by availability (descending)
      const sortedProviders = availableProviders.sort((a, b) => {
        const availabilityA = providerAvailabilities.get(a) || 0;
        const availabilityB = providerAvailabilities.get(b) || 0;
        return availabilityB - availabilityA;
      });
      
      return sortedProviders[0] || null;
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Availability-first provider selection');
      return null;
    }
  }

  /**
   * Select provider using balanced approach
   */
  private async selectBalancedProvider(
    availableProviders: string[],
    strategy: FallbackStrategy,
    taskType: string
  ): Promise<string | null> {
    try {
      const providerScores = new Map<string, number>();
      
      // Calculate balanced score for each provider
      for (const provider of availableProviders) {
        const cost = await this.getProviderCost(provider);
        const quality = await this.getProviderQuality(provider, taskType);
        const availability = this.getProviderAvailability(provider);
        
        // Normalize scores (0-1 range)
        const normalizedCost = Math.max(0, 1 - (cost / 0.1)); // Assume max cost is 0.1
        const normalizedQuality = quality;
        const normalizedAvailability = availability;
        
        // Calculate balanced score (equal weights)
        const balancedScore = (normalizedCost + normalizedQuality + normalizedAvailability) / 3;
        
        providerScores.set(provider, balancedScore);
      }
      
      // Sort by balanced score (descending)
      const sortedProviders = availableProviders.sort((a, b) => {
        const scoreA = providerScores.get(a) || 0;
        const scoreB = providerScores.get(b) || 0;
        return scoreB - scoreA;
      });
      
      return sortedProviders[0] || null;
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Balanced provider selection');
      return null;
    }
  }

  /**
   * Calculate retry delay with exponential backoff
   */
  private calculateRetryDelay(retryCount: number): number {
    const baseDelay = this.config.getRetryDelay();
    const multiplier = this.config.getRetryBackoffMultiplier();
    const maxDelay = this.config.getMaxRetryDelay();
    
    const delay = baseDelay * Math.pow(multiplier, retryCount);
    return Math.min(delay, maxDelay);
  }

  /**
   * Get provider cost
   */
  private async getProviderCost(provider: string): Promise<number> {
    try {
      // Get cost from cost optimization engine
      const costStats = this.costEngine.getCostStatistics();
      
      // Provider-specific cost mapping (simplified)
      const providerCosts: Record<string, number> = {
        'openai': 0.0005,
        'google': 0.0003,
        'mistral': 0.0002,
        'cohere': 0.0004,
        'anthropic': 0.0006
      };
      
      return providerCosts[provider.toLowerCase()] || 0.0005;
      
    } catch (error) {
      errorLogger.aiError(error as Error, `Getting cost for ${provider}`);
      return 0.0005; // Default cost
    }
  }

  /**
   * Get provider quality score
   */
  private async getProviderQuality(provider: string, taskType: string): Promise<number> {
    try {
      // Get quality from quality threshold manager
      const qualityStats = this.qualityManager.getQualityStats();
      
      // Provider-specific quality mapping (simplified)
      const providerQualities: Record<string, number> = {
        'openai': 0.9,
        'google': 0.85,
        'mistral': 0.8,
        'cohere': 0.82,
        'anthropic': 0.88
      };
      
      return providerQualities[provider.toLowerCase()] || 0.8;
      
    } catch (error) {
      errorLogger.aiError(error as Error, `Getting quality for ${provider}`);
      return 0.8; // Default quality
    }
  }

  /**
   * Get provider availability
   */
  private getProviderAvailability(provider: string): number {
    try {
      const metrics = this.healthMonitor.getProviderHealthMetrics(provider);
      return metrics ? metrics.availability : 0.5; // Default availability
    } catch (error) {
      errorLogger.aiError(error as Error, `Getting availability for ${provider}`);
      return 0.5; // Default availability
    }
  }

  /**
   * Update fallback history
   */
  private updateFallbackHistory(provider: string): void {
    const currentCount = this.fallbackHistory.get(provider) || 0;
    this.fallbackHistory.set(provider, currentCount + 1);
  }

  /**
   * Get fallback reason
   */
  private getFallbackReason(error: Error, originalProvider: string, fallbackProvider: string): string {
    const errorMessage = error.message.toLowerCase();
    
    if (errorMessage.includes('timeout')) {
      return `Timeout on ${originalProvider}, falling back to ${fallbackProvider}`;
    } else if (errorMessage.includes('rate limit')) {
      return `Rate limit on ${originalProvider}, falling back to ${fallbackProvider}`;
    } else if (errorMessage.includes('network')) {
      return `Network error on ${originalProvider}, falling back to ${fallbackProvider}`;
    } else if (errorMessage.includes('circuit breaker')) {
      return `Circuit breaker open on ${originalProvider}, falling back to ${fallbackProvider}`;
    } else {
      return `Error on ${originalProvider}, falling back to ${fallbackProvider}`;
    }
  }

  /**
   * Get fallback statistics
   */
  public getFallbackStatistics(): {
    totalFallbacks: number;
    providerFallbackCounts: Record<string, number>;
    averageFallbacksPerProvider: number;
  } {
    const totalFallbacks = Array.from(this.fallbackHistory.values()).reduce((sum, count) => sum + count, 0);
    const providerCount = this.fallbackHistory.size;
    
    return {
      totalFallbacks,
      providerFallbackCounts: Object.fromEntries(this.fallbackHistory),
      averageFallbacksPerProvider: providerCount > 0 ? totalFallbacks / providerCount : 0
    };
  }

  /**
   * Reset fallback history
   */
  public resetFallbackHistory(): void {
    this.fallbackHistory.clear();
    systemLogger.startup('IntelligentFallbackManager', 'Reset fallback history');
  }

  /**
   * Get health monitor instance
   */
  public getHealthMonitor(): ProviderHealthMonitor {
    return this.healthMonitor;
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.healthMonitor.destroy();
  }
}
