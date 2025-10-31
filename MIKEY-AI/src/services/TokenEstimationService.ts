/**
 * Token Estimation Service
 * Accurate token estimation using tiktoken with provider-specific encodings
 */

import { encoding_for_model, get_encoding } from 'tiktoken';
import NodeCache from 'node-cache';
import { 
  TokenEstimationResult, 
  TokenEstimationConfig, 
  TokenEstimationStats, 
  CacheStats,
  EnhancedCostMetrics 
} from '../types/token-estimation';
import { TokenizationConfig } from '../config/tokenization-config';

export class TokenEstimationService {
  private cache: NodeCache;
  private config: TokenizationConfig;
  private stats: TokenEstimationStats;
  private cacheStats: CacheStats;

  constructor(cacheTTL: number = 3600) { // 1 hour default TTL
    this.cache = new NodeCache({ 
      stdTTL: cacheTTL,
      checkperiod: cacheTTL * 0.2,
      useClones: false
    });
    this.config = TokenizationConfig.getInstance();
    this.stats = {
      totalEstimations: 0,
      averageAccuracy: 0,
      averageProcessingTime: 0,
      cacheStats: { hits: 0, misses: 0, hitRate: 0, totalRequests: 0 },
      providerBreakdown: {}
    };
    this.cacheStats = { hits: 0, misses: 0, hitRate: 0, totalRequests: 0 };
  }

  /**
   * Estimate tokens for content using provider-specific encoding
   */
  public async estimateTokens(
    content: string, 
    provider: string, 
    model: string
  ): Promise<TokenEstimationResult> {
    const startTime = Date.now();
    
    // Check cache first
    const cacheKey = this.generateCacheKey(content, provider, model);
    const cachedResult = this.cache.get<TokenEstimationResult>(cacheKey);
    
    if (cachedResult) {
      this.cacheStats.hits++;
      this.stats.cacheStats.hits++;
      this.updateStats(cachedResult, true);
      return {
        ...cachedResult,
        cached: true,
        processingTime: Date.now() - startTime
      };
    }

    this.cacheStats.misses++;
    this.stats.cacheStats.misses++;

    try {
      // Get provider configuration
      const providerConfig = this.config.getProvider(provider);
      if (!providerConfig) {
        throw new Error(`Unknown provider: ${provider}`);
      }

      // Validate model
      if (!this.config.isValidModel(provider, model)) {
        throw new Error(`Invalid model ${model} for provider ${provider}`);
      }

      // Get encoding for the model
      let encoding;
      try {
        encoding = encoding_for_model(model as any);
      } catch (error) {
        // Fallback to provider-specific encoding
        encoding = get_encoding(providerConfig.encoding as any);
      }

      // Calculate token count
      const tokenCount = encoding.encode(content).length;
      const confidence = this.calculateConfidence(content, tokenCount, provider);

      const result: TokenEstimationResult = {
        provider,
        model,
        tokenCount,
        confidence,
        cached: false,
        processingTime: Date.now() - startTime
      };

      // Cache the result
      this.cache.set(cacheKey, result);

      // Update statistics
      this.updateStats(result, false);

      return result;

    } catch (error) {
      console.error(`Token estimation failed for ${provider}/${model}:`, error);
      
      // Fallback to rough estimation (4 chars per token)
      const fallbackTokenCount = Math.ceil(content.length / 4);
      
      const fallbackResult: TokenEstimationResult = {
        provider,
        model,
        tokenCount: fallbackTokenCount,
        confidence: 0.5, // Lower confidence for fallback
        cached: false,
        processingTime: Date.now() - startTime
      };

      this.updateStats(fallbackResult, false);
      return fallbackResult;
    }
  }

  /**
   * Get cached estimation if available
   */
  public async getCachedEstimation(
    content: string, 
    provider: string, 
    model: string
  ): Promise<TokenEstimationResult | null> {
    const cacheKey = this.generateCacheKey(content, provider, model);
    return this.cache.get<TokenEstimationResult>(cacheKey) || null;
  }

  /**
   * Clear all cached estimations
   */
  public async clearCache(): Promise<void> {
    this.cache.flushAll();
    this.cacheStats = { hits: 0, misses: 0, hitRate: 0, totalRequests: 0 };
    this.stats.cacheStats = { hits: 0, misses: 0, hitRate: 0, totalRequests: 0 };
  }

  /**
   * Get estimation accuracy metrics
   */
  public async getEstimationAccuracy(): Promise<number> {
    return this.stats.averageAccuracy;
  }

  /**
   * Get comprehensive statistics
   */
  public getStats(): TokenEstimationStats {
    this.updateCacheHitRate();
    return { ...this.stats };
  }

  /**
   * Get cache statistics
   */
  public getCacheStats(): CacheStats {
    this.updateCacheHitRate();
    return { ...this.cacheStats };
  }

  /**
   * Estimate tokens for multiple providers
   */
  public async estimateTokensForProviders(
    content: string, 
    providers: string[], 
    model?: string
  ): Promise<Record<string, TokenEstimationResult>> {
    const results: Record<string, TokenEstimationResult> = {};
    
    const promises = providers.map(async (provider) => {
      const defaultModel = model || this.config.getDefaultModel(provider);
      if (defaultModel) {
        results[provider] = await this.estimateTokens(content, provider, defaultModel);
      }
    });

    await Promise.all(promises);
    return results;
  }

  /**
   * Calculate cost for token estimation
   */
  public calculateCost(tokenCount: number, provider: string): number {
    const costPerToken = this.config.getCostPerToken(provider);
    return tokenCount * costPerToken;
  }

  /**
   * Enhance cost metrics with accurate token count
   */
  public enhanceCostMetrics(
    baseMetrics: any, 
    accurateTokenCount: number, 
    cacheHit: boolean
  ): EnhancedCostMetrics {
    const estimationAccuracy = this.calculateEstimationAccuracy(
      baseMetrics.tokensUsed, 
      accurateTokenCount
    );

    return {
      ...baseMetrics,
      accurateTokenCount,
      estimationAccuracy,
      cacheHit
    };
  }

  /**
   * Generate cache key for content
   */
  private generateCacheKey(content: string, provider: string, model: string): string {
    // Create a hash of the content for the cache key
    const contentHash = this.simpleHash(content);
    return `${provider}:${model}:${contentHash}`;
  }

  /**
   * Simple hash function for content
   */
  private simpleHash(str: string): string {
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  /**
   * Calculate confidence score for estimation
   */
  private calculateConfidence(content: string, tokenCount: number, provider: string): number {
    // Base confidence on content characteristics
    let confidence = 0.9; // High confidence for tiktoken

    // Adjust based on content length
    if (content.length < 10) {
      confidence -= 0.1; // Lower confidence for very short content
    }

    // Adjust based on provider reliability
    const providerConfig = this.config.getProvider(provider);
    if (providerConfig && providerConfig.costPerToken < 0.0002) {
      confidence += 0.05; // Slightly higher confidence for premium providers
    }

    return Math.min(1.0, Math.max(0.0, confidence));
  }

  /**
   * Calculate estimation accuracy compared to rough estimation
   */
  private calculateEstimationAccuracy(roughCount: number, accurateCount: number): number {
    if (roughCount === 0) return 1.0;
    const accuracy = Math.abs(accurateCount - roughCount) / roughCount;
    return Math.max(0, 1 - accuracy);
  }

  /**
   * Update statistics
   */
  private updateStats(result: TokenEstimationResult, fromCache: boolean): void {
    this.stats.totalEstimations++;
    
    // Update provider breakdown
    if (!this.stats.providerBreakdown[result.provider]) {
      this.stats.providerBreakdown[result.provider] = 0;
    }
    this.stats.providerBreakdown[result.provider]++;

    // Update average processing time
    this.stats.averageProcessingTime = 
      (this.stats.averageProcessingTime * (this.stats.totalEstimations - 1) + result.processingTime) / 
      this.stats.totalEstimations;

    // Update cache stats
    this.stats.cacheStats.totalRequests++;
    this.updateCacheHitRate();
  }

  /**
   * Update cache hit rate
   */
  private updateCacheHitRate(): void {
    const total = this.cacheStats.hits + this.cacheStats.misses;
    this.cacheStats.hitRate = total > 0 ? this.cacheStats.hits / total : 0;
    this.stats.cacheStats.hitRate = this.cacheStats.hitRate;
  }

  /**
   * Get cache configuration
   */
  public getCacheConfig(): { ttl: number; keys: number } {
    const keys = this.cache.keys().length;
    return {
      ttl: 3600, // Default TTL
      keys
    };
  }

  /**
   * Warm up cache with common content patterns
   */
  public async warmupCache(commonPatterns: string[]): Promise<void> {
    const providers = this.config.getProviderNames();
    
    for (const pattern of commonPatterns) {
      for (const provider of providers) {
        const model = this.config.getDefaultModel(provider);
        if (model) {
          await this.estimateTokens(pattern, provider, model);
        }
      }
    }
  }
}
