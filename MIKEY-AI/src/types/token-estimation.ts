/**
 * Token Estimation Types
 * Type definitions for accurate token estimation service
 */

export interface TokenEstimationConfig {
  provider: string;
  model: string;
  encoding: string;
  cacheEnabled: boolean;
  cacheTTL: number;
}

export interface TokenEstimationResult {
  provider: string;
  model: string;
  tokenCount: number;
  confidence: number;
  cached: boolean;
  processingTime: number;
}

export interface EnhancedCostMetrics {
  provider: string;
  tokensUsed: number;
  costPerToken: number;
  totalCost: number;
  timestamp: Date;
  taskType: string;
  accurateTokenCount: number;
  estimationAccuracy: number;
  cacheHit: boolean;
}

export interface TokenizationProvider {
  name: string;
  encoding: string;
  models: string[];
  costPerToken: number;
}

export interface CacheStats {
  hits: number;
  misses: number;
  hitRate: number;
  totalRequests: number;
}

export interface TokenEstimationStats {
  totalEstimations: number;
  averageAccuracy: number;
  averageProcessingTime: number;
  cacheStats: CacheStats;
  providerBreakdown: Record<string, number>;
}
