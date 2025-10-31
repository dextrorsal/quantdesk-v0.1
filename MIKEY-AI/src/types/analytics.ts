/**
 * Analytics Types
 * Type definitions for comprehensive usage analytics and monitoring
 */

export interface RequestMetrics {
  requestId: string;
  provider: string;
  tokensUsed: number;
  cost: number;
  qualityScore: number;
  responseTime: number;
  timestamp: Date;
  taskType: string;
  sessionId?: string;
  escalationCount?: number;
  fallbackUsed?: boolean;
}

export interface TimeRange {
  start: Date;
  end: Date;
}

export interface CostReport {
  timeRange: TimeRange;
  totalCost: number;
  costSavings: number;
  providerBreakdown: ProviderCostBreakdown[];
  qualityMetrics: any; // QualityMetrics;
  utilizationMetrics: UtilizationMetrics;
}

export interface ProviderCostBreakdown {
  provider: string;
  totalCost: number;
  tokenCount: number;
  requestCount: number;
  averageCostPerToken: number;
  averageCostPerRequest: number;
  costPercentage: number;
}

export interface ProviderUtilization {
  provider: string;
  requestCount: number;
  totalTokens: number;
  utilizationPercentage: number;
  averageResponseTime: number;
  successRate: number;
  errorRate: number;
  lastUsed: Date;
}

export interface SatisfactionMetrics {
  averageQualityScore: number;
  userSatisfactionRate: number;
  escalationRate: number;
  fallbackRate: number;
  responseTimeMetrics: ResponseTimeMetrics;
  qualityDistribution: QualityDistribution;
}

export interface ResponseTimeMetrics {
  average: number;
  median: number;
  p95: number;
  p99: number;
  min: number;
  max: number;
}

export interface QualityDistribution {
  excellent: number; // >0.8
  good: number; // 0.6-0.8
  average: number; // 0.4-0.6
  poor: number; // <0.4
}

export interface UtilizationMetrics {
  totalRequests: number;
  totalTokens: number;
  averageRequestsPerHour: number;
  peakHourlyRequests: number;
  providerDistribution: Record<string, number>;
}

export interface AnalyticsConfig {
  dataRetentionDays: number;
  privacyCompliance: boolean;
  anonymizeData: boolean;
  enableRealTimeTracking: boolean;
  batchSize: number;
  flushInterval: number;
}

export interface AnalyticsStats {
  totalRequests: number;
  totalCost: number;
  totalSavings: number;
  averageQualityScore: number;
  dataPointsCollected: number;
  lastUpdated: Date;
}

export interface CostSavingsReport {
  timeRange: TimeRange;
  baselineCost: number; // Cost if using single expensive provider
  actualCost: number;
  savingsAmount: number;
  savingsPercentage: number;
  roi: number; // Return on investment
  breakEvenPoint?: Date;
}

export interface AnalyticsDashboard {
  overview: AnalyticsStats;
  costReport: CostReport;
  providerUtilization: ProviderUtilization[];
  satisfactionMetrics: SatisfactionMetrics;
  costSavings: CostSavingsReport;
  trends: AnalyticsTrends;
}

export interface AnalyticsTrends {
  costTrend: TrendData[];
  qualityTrend: TrendData[];
  utilizationTrend: TrendData[];
  savingsTrend: TrendData[];
}

export interface TrendData {
  timestamp: Date;
  value: number;
  label?: string;
}

export interface AnalyticsFilter {
  timeRange?: TimeRange;
  providers?: string[];
  taskTypes?: string[];
  qualityThreshold?: number;
  costThreshold?: number;
}

export interface AnalyticsQuery {
  filter: AnalyticsFilter;
  groupBy?: 'hour' | 'day' | 'week' | 'month';
  limit?: number;
  offset?: number;
  sortBy?: 'timestamp' | 'cost' | 'quality' | 'responseTime';
  sortOrder?: 'asc' | 'desc';
}
