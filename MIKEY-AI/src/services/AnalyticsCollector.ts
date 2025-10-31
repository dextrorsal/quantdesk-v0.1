/**
 * Analytics Collector
 * Comprehensive usage analytics collection and reporting system
 */

import { 
  RequestMetrics, 
  CostReport, 
  ProviderUtilization, 
  SatisfactionMetrics,
  AnalyticsStats,
  CostSavingsReport,
  AnalyticsDashboard,
  TimeRange,
  AnalyticsFilter,
  AnalyticsQuery
} from '../types/analytics';
import { AnalyticsConfiguration } from '../config/analytics-config';
import { CostOptimizationEngine } from './CostOptimizationEngine';
import { QualityThresholdManager } from './QualityThresholdManager';
import { TokenEstimationService } from './TokenEstimationService';
import { systemLogger, errorLogger } from '../utils/logger';

export class AnalyticsCollector {
  private config: AnalyticsConfiguration;
  private costEngine: CostOptimizationEngine;
  private qualityManager: QualityThresholdManager;
  private tokenService: TokenEstimationService;
  
  // In-memory storage for real-time analytics
  private requestMetrics: RequestMetrics[] = [];
  private analyticsStats: AnalyticsStats;
  private providerStats: Map<string, ProviderUtilization> = new Map();
  
  // Batch processing
  private batchBuffer: RequestMetrics[] = [];
  private flushTimer?: NodeJS.Timeout;

  constructor() {
    this.config = AnalyticsConfiguration.getInstance();
    this.costEngine = new CostOptimizationEngine();
    this.qualityManager = new QualityThresholdManager();
    this.tokenService = new TokenEstimationService();
    
    this.analyticsStats = this.initializeStats();
    this.initializeBatchProcessing();
    
    systemLogger.startup('AnalyticsCollector', 'Initialized with real-time tracking');
  }

  /**
   * Track request metrics in real-time
   */
  public async trackRequestMetrics(metrics: RequestMetrics): Promise<void> {
    try {
      // Validate metrics
      this.validateMetrics(metrics);
      
      // Anonymize data if required
      const processedMetrics = this.config.isDataAnonymizationEnabled() 
        ? this.anonymizeMetrics(metrics) 
        : metrics;
      
      // Add to real-time storage
      this.requestMetrics.push(processedMetrics);
      
      // Update provider stats
      this.updateProviderStats(processedMetrics);
      
      // Update overall stats
      this.updateAnalyticsStats(processedMetrics);
      
      // Add to batch buffer for persistence
      if (this.config.isRealTimeTrackingEnabled()) {
        this.batchBuffer.push(processedMetrics);
        
        // Flush if batch is full
        if (this.batchBuffer.length >= this.config.getBatchSize()) {
          await this.flushBatch();
        }
      }
      
      systemLogger.startup('AnalyticsCollector', 
        `Tracked metrics for ${metrics.provider}: ${metrics.cost.toFixed(4)} cost, ${metrics.qualityScore.toFixed(3)} quality`
      );
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Analytics collection');
    }
  }

  /**
   * Generate comprehensive cost report
   */
  public async generateCostReport(timeRange: TimeRange): Promise<CostReport> {
    try {
      const filteredMetrics = this.filterMetricsByTimeRange(timeRange);
      
      const totalCost = filteredMetrics.reduce((sum, m) => sum + m.cost, 0);
      const totalTokens = filteredMetrics.reduce((sum, m) => sum + m.tokensUsed, 0);
      
      // Calculate cost savings compared to single expensive provider
      const baselineCost = await this.calculateBaselineCost(filteredMetrics);
      const costSavings = baselineCost - totalCost;
      
      // Provider breakdown
      const providerBreakdown = this.calculateProviderBreakdown(filteredMetrics, totalCost);
      
      // Quality metrics
      const qualityMetrics = this.calculateQualityMetrics(filteredMetrics);
      
      // Utilization metrics
      const utilizationMetrics = this.calculateUtilizationMetrics(filteredMetrics);
      
      return {
        timeRange,
        totalCost,
        costSavings,
        providerBreakdown,
        qualityMetrics,
        utilizationMetrics
      };
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Cost report generation');
      throw error;
    }
  }

  /**
   * Get provider utilization metrics
   */
  public async getProviderUtilization(): Promise<ProviderUtilization[]> {
    try {
      const utilization: ProviderUtilization[] = [];
      
      this.providerStats.forEach((stats, provider) => {
        utilization.push({
          ...stats,
          provider
        });
      });
      
      // Sort by utilization percentage (descending)
      return utilization.sort((a, b) => b.utilizationPercentage - a.utilizationPercentage);
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Provider utilization calculation');
      throw error;
    }
  }

  /**
   * Get user satisfaction metrics
   */
  public async getUserSatisfactionMetrics(): Promise<SatisfactionMetrics> {
    try {
      const recentMetrics = this.requestMetrics.slice(-1000); // Last 1000 requests
      
      if (recentMetrics.length === 0) {
        return this.getDefaultSatisfactionMetrics();
      }
      
      const averageQualityScore = recentMetrics.reduce((sum, m) => sum + m.qualityScore, 0) / recentMetrics.length;
      const userSatisfactionRate = Math.max(0, averageQualityScore - 0.1);
      
      // Calculate escalation and fallback rates
      const escalationCount = recentMetrics.filter(m => m.escalationCount && m.escalationCount > 0).length;
      const fallbackCount = recentMetrics.filter(m => m.fallbackUsed).length;
      
      const escalationRate = escalationCount / recentMetrics.length;
      const fallbackRate = fallbackCount / recentMetrics.length;
      
      // Response time metrics
      const responseTimes = recentMetrics.map(m => m.responseTime).sort((a, b) => a - b);
      const responseTimeMetrics = this.calculateResponseTimeMetrics(responseTimes);
      
      // Quality distribution
      const qualityDistribution = this.calculateQualityDistribution(recentMetrics);
      
      return {
        averageQualityScore,
        userSatisfactionRate,
        escalationRate,
        fallbackRate,
        responseTimeMetrics,
        qualityDistribution
      };
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Satisfaction metrics calculation');
      throw error;
    }
  }

  /**
   * Get comprehensive analytics dashboard
   */
  public async getAnalyticsDashboard(timeRange?: TimeRange): Promise<AnalyticsDashboard> {
    try {
      const reportTimeRange = timeRange || this.getDefaultTimeRange();
      
      const [
        costReport,
        providerUtilization,
        satisfactionMetrics,
        costSavings
      ] = await Promise.all([
        this.generateCostReport(reportTimeRange),
        this.getProviderUtilization(),
        this.getUserSatisfactionMetrics(),
        this.generateCostSavingsReport(reportTimeRange)
      ]);
      
      return {
        overview: this.analyticsStats,
        costReport,
        providerUtilization,
        satisfactionMetrics,
        costSavings,
        trends: await this.generateTrends(reportTimeRange)
      };
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Analytics dashboard generation');
      throw error;
    }
  }

  /**
   * Generate cost savings report
   */
  public async generateCostSavingsReport(timeRange: TimeRange): Promise<CostSavingsReport> {
    try {
      const filteredMetrics = this.filterMetricsByTimeRange(timeRange);
      const actualCost = filteredMetrics.reduce((sum, m) => sum + m.cost, 0);
      const baselineCost = await this.calculateBaselineCost(filteredMetrics);
      
      const savingsAmount = baselineCost - actualCost;
      const savingsPercentage = baselineCost > 0 ? (savingsAmount / baselineCost) * 100 : 0;
      
      // Calculate ROI (simplified)
      const roi = actualCost > 0 ? (savingsAmount / actualCost) * 100 : 0;
      
      return {
        timeRange,
        baselineCost,
        actualCost,
        savingsAmount,
        savingsPercentage,
        roi
      };
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Cost savings report generation');
      throw error;
    }
  }

  /**
   * Get analytics statistics
   */
  public getAnalyticsStats(): AnalyticsStats {
    return { ...this.analyticsStats };
  }

  /**
   * Clear old data based on retention policy
   */
  public async clearOldData(): Promise<void> {
    try {
      if (!this.config.isDataRetentionEnabled()) {
        return;
      }
      
      const retentionDate = new Date();
      retentionDate.setDate(retentionDate.getDate() - this.config.getDataRetentionDays());
      
      // Remove old metrics
      this.requestMetrics = this.requestMetrics.filter(m => m.timestamp >= retentionDate);
      
      // Update stats
      this.analyticsStats.lastUpdated = new Date();
      
      systemLogger.startup('AnalyticsCollector', 
        `Cleared data older than ${this.config.getDataRetentionDays()} days`
      );
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Data cleanup');
    }
  }

  /**
   * Export analytics data
   */
  public async exportAnalyticsData(query: AnalyticsQuery): Promise<RequestMetrics[]> {
    try {
      let filteredMetrics = [...this.requestMetrics];
      
      // Apply filters
      if (query.filter.timeRange) {
        filteredMetrics = this.filterMetricsByTimeRange(query.filter.timeRange, filteredMetrics);
      }
      
      if (query.filter.providers && query.filter.providers.length > 0) {
        filteredMetrics = filteredMetrics.filter(m => query.filter.providers!.includes(m.provider));
      }
      
      if (query.filter.taskTypes && query.filter.taskTypes.length > 0) {
        filteredMetrics = filteredMetrics.filter(m => query.filter.taskTypes!.includes(m.taskType));
      }
      
      if (query.filter.qualityThreshold) {
        filteredMetrics = filteredMetrics.filter(m => m.qualityScore >= query.filter.qualityThreshold!);
      }
      
      if (query.filter.costThreshold) {
        filteredMetrics = filteredMetrics.filter(m => m.cost >= query.filter.costThreshold!);
      }
      
      // Apply sorting
      if (query.sortBy) {
        filteredMetrics.sort((a, b) => {
          const aVal = this.getSortValue(a, query.sortBy!);
          const bVal = this.getSortValue(b, query.sortBy!);
          return query.sortOrder === 'desc' ? bVal - aVal : aVal - bVal;
        });
      }
      
      // Apply pagination
      const offset = query.offset || 0;
      const limit = query.limit || filteredMetrics.length;
      
      return filteredMetrics.slice(offset, offset + limit);
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Analytics data export');
      throw error;
    }
  }

  // Private helper methods
  private initializeStats(): AnalyticsStats {
    return {
      totalRequests: 0,
      totalCost: 0,
      totalSavings: 0,
      averageQualityScore: 0,
      dataPointsCollected: 0,
      lastUpdated: new Date()
    };
  }

  private initializeBatchProcessing(): void {
    if (this.config.isRealTimeTrackingEnabled()) {
      this.flushTimer = setInterval(async () => {
        await this.flushBatch();
      }, this.config.getFlushInterval());
    }
  }

  private validateMetrics(metrics: RequestMetrics): void {
    if (!metrics.requestId || !metrics.provider) {
      throw new Error('Invalid metrics: missing required fields');
    }
    
    if (metrics.cost < 0 || metrics.tokensUsed < 0 || metrics.responseTime < 0) {
      throw new Error('Invalid metrics: negative values not allowed');
    }
    
    if (metrics.qualityScore < 0 || metrics.qualityScore > 1) {
      throw new Error('Invalid metrics: quality score must be between 0 and 1');
    }
  }

  private anonymizeMetrics(metrics: RequestMetrics): RequestMetrics {
    return {
      ...metrics,
      requestId: this.hashString(metrics.requestId),
      sessionId: metrics.sessionId ? this.hashString(metrics.sessionId) : undefined
    };
  }

  private hashString(str: string): string {
    // Simple hash function for anonymization
    let hash = 0;
    for (let i = 0; i < str.length; i++) {
      const char = str.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(36);
  }

  private updateProviderStats(metrics: RequestMetrics): void {
    const existing = this.providerStats.get(metrics.provider) || {
      provider: metrics.provider,
      requestCount: 0,
      totalTokens: 0,
      utilizationPercentage: 0,
      averageResponseTime: 0,
      successRate: 1,
      errorRate: 0,
      lastUsed: new Date()
    };
    
    existing.requestCount++;
    existing.totalTokens += metrics.tokensUsed;
    existing.averageResponseTime = (existing.averageResponseTime + metrics.responseTime) / 2;
    existing.lastUsed = metrics.timestamp;
    
    this.providerStats.set(metrics.provider, existing);
  }

  private updateAnalyticsStats(metrics: RequestMetrics): void {
    this.analyticsStats.totalRequests++;
    this.analyticsStats.totalCost += metrics.cost;
    this.analyticsStats.dataPointsCollected++;
    
    // Update average quality score
    const totalQuality = this.analyticsStats.averageQualityScore * (this.analyticsStats.totalRequests - 1);
    this.analyticsStats.averageQualityScore = (totalQuality + metrics.qualityScore) / this.analyticsStats.totalRequests;
    
    this.analyticsStats.lastUpdated = new Date();
  }

  private filterMetricsByTimeRange(timeRange: TimeRange, metrics: RequestMetrics[] = this.requestMetrics): RequestMetrics[] {
    return metrics.filter(m => m.timestamp >= timeRange.start && m.timestamp <= timeRange.end);
  }

  private async calculateBaselineCost(metrics: RequestMetrics[]): Promise<number> {
    // Calculate what it would cost if all requests went to the most expensive provider
    const expensiveProviderCost = 0.0005; // OpenAI GPT-4 cost per token
    return metrics.reduce((sum, m) => sum + (m.tokensUsed * expensiveProviderCost), 0);
  }

  private calculateProviderBreakdown(metrics: RequestMetrics[], totalCost: number): any[] {
    const providerMap = new Map<string, any>();
    
    metrics.forEach(m => {
      const existing = providerMap.get(m.provider) || {
        provider: m.provider,
        totalCost: 0,
        tokenCount: 0,
        requestCount: 0
      };
      
      existing.totalCost += m.cost;
      existing.tokenCount += m.tokensUsed;
      existing.requestCount++;
      
      providerMap.set(m.provider, existing);
    });
    
    return Array.from(providerMap.values()).map(p => ({
      ...p,
      averageCostPerToken: p.tokenCount > 0 ? p.totalCost / p.tokenCount : 0,
      averageCostPerRequest: p.requestCount > 0 ? p.totalCost / p.requestCount : 0,
      costPercentage: totalCost > 0 ? (p.totalCost / totalCost) * 100 : 0
    }));
  }

  private calculateQualityMetrics(metrics: RequestMetrics[]): any {
    if (metrics.length === 0) {
      return { averageQuality: 0, qualityDistribution: { excellent: 0, good: 0, average: 0, poor: 0 } };
    }
    
    const averageQuality = metrics.reduce((sum, m) => sum + m.qualityScore, 0) / metrics.length;
    const qualityDistribution = this.calculateQualityDistribution(metrics);
    
    return { averageQuality, qualityDistribution };
  }

  private calculateUtilizationMetrics(metrics: RequestMetrics[]): any {
    const totalRequests = metrics.length;
    const totalTokens = metrics.reduce((sum, m) => sum + m.tokensUsed, 0);
    
    // Calculate hourly distribution
    const hourlyRequests = new Map<number, number>();
    metrics.forEach(m => {
      const hour = m.timestamp.getHours();
      hourlyRequests.set(hour, (hourlyRequests.get(hour) || 0) + 1);
    });
    
    const averageRequestsPerHour = totalRequests / 24;
    const peakHourlyRequests = Math.max(...hourlyRequests.values());
    
    // Provider distribution
    const providerDistribution: Record<string, number> = {};
    metrics.forEach(m => {
      providerDistribution[m.provider] = (providerDistribution[m.provider] || 0) + 1;
    });
    
    return {
      totalRequests,
      totalTokens,
      averageRequestsPerHour,
      peakHourlyRequests,
      providerDistribution
    };
  }

  private calculateResponseTimeMetrics(responseTimes: number[]): any {
    if (responseTimes.length === 0) {
      return { average: 0, median: 0, p95: 0, p99: 0, min: 0, max: 0 };
    }
    
    const sorted = responseTimes.sort((a, b) => a - b);
    const len = sorted.length;
    
    return {
      average: responseTimes.reduce((sum, t) => sum + t, 0) / len,
      median: sorted[Math.floor(len / 2)],
      p95: sorted[Math.floor(len * 0.95)],
      p99: sorted[Math.floor(len * 0.99)],
      min: sorted[0],
      max: sorted[len - 1]
    };
  }

  private calculateQualityDistribution(metrics: RequestMetrics[]): any {
    const distribution = { excellent: 0, good: 0, average: 0, poor: 0 };
    
    metrics.forEach(m => {
      if (m.qualityScore > 0.8) distribution.excellent++;
      else if (m.qualityScore > 0.6) distribution.good++;
      else if (m.qualityScore > 0.4) distribution.average++;
      else distribution.poor++;
    });
    
    return distribution;
  }

  private getDefaultSatisfactionMetrics(): SatisfactionMetrics {
    return {
      averageQualityScore: 0.5,
      userSatisfactionRate: 0.4,
      escalationRate: 0,
      fallbackRate: 0,
      responseTimeMetrics: { average: 0, median: 0, p95: 0, p99: 0, min: 0, max: 0 },
      qualityDistribution: { excellent: 0, good: 0, average: 0, poor: 0 }
    };
  }

  private getDefaultTimeRange(): TimeRange {
    const end = new Date();
    const start = new Date();
    start.setDate(start.getDate() - 7); // Last 7 days
    
    return { start, end };
  }

  private async generateTrends(timeRange: TimeRange): Promise<any> {
    // Simplified trend generation - would be more sophisticated in production
    return {
      costTrend: [],
      qualityTrend: [],
      utilizationTrend: [],
      savingsTrend: []
    };
  }

  private getSortValue(metrics: RequestMetrics, sortBy: string): number {
    switch (sortBy) {
      case 'timestamp': return metrics.timestamp.getTime();
      case 'cost': return metrics.cost;
      case 'quality': return metrics.qualityScore;
      case 'responseTime': return metrics.responseTime;
      default: return 0;
    }
  }

  private async flushBatch(): Promise<void> {
    if (this.batchBuffer.length === 0) {
      return;
    }
    
    try {
      // In a real implementation, this would persist to database
      // For now, we'll just log the batch
      systemLogger.startup('AnalyticsCollector', 
        `Flushed batch of ${this.batchBuffer.length} metrics`
      );
      
      this.batchBuffer = [];
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Batch flush');
    }
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    if (this.flushTimer) {
      clearInterval(this.flushTimer);
    }
    
    // Flush remaining batch
    this.flushBatch();
  }
}
