/**
 * Cost Analytics Service
 * Comprehensive cost tracking and analytics for LLM router optimization
 */

import { CostOptimizationEngine, CostMetrics } from './CostOptimizationEngine';
import { ProviderCostRanking } from './ProviderCostRanking';

export interface CostAnalytics {
  totalRequests: number;
  totalCost: number;
  averageCostPerRequest: number;
  costSavings: number;
  savingsPercentage: number;
  providerBreakdown: ProviderBreakdown[];
  taskTypeBreakdown: TaskTypeBreakdown[];
  timeSeriesData: TimeSeriesDataPoint[];
  qualityMetrics: QualityMetrics;
}

export interface ProviderBreakdown {
  provider: string;
  requests: number;
  totalCost: number;
  averageCostPerRequest: number;
  costPercentage: number;
  utilizationRate: number;
}

export interface TaskTypeBreakdown {
  taskType: string;
  requests: number;
  totalCost: number;
  averageCostPerRequest: number;
  preferredProvider: string;
  costEfficiency: number;
}

export interface TimeSeriesDataPoint {
  timestamp: Date;
  requests: number;
  totalCost: number;
  averageCostPerRequest: number;
  costSavings: number;
}

export interface QualityMetrics {
  averageResponseQuality: number;
  qualityThresholdCompliance: number;
  providerQualityScores: Record<string, number>;
  qualityTrend: 'improving' | 'stable' | 'declining';
}

export class CostAnalyticsService {
  private costOptimizationEngine: CostOptimizationEngine;
  private providerRanking: ProviderCostRanking;
  private timeSeriesData: TimeSeriesDataPoint[] = [];
  private qualityMetrics: QualityMetrics;

  constructor() {
    this.costOptimizationEngine = new CostOptimizationEngine();
    this.providerRanking = new ProviderCostRanking();
    this.qualityMetrics = this.initializeQualityMetrics();
    
    // Initialize time series data collection
    this.setupTimeSeriesCollection();
  }

  /**
   * Initialize quality metrics
   */
  private initializeQualityMetrics(): QualityMetrics {
    return {
      averageResponseQuality: 0.85,
      qualityThresholdCompliance: 0.9,
      providerQualityScores: {
        'openai': 0.9,
        'anthropic': 0.9,
        'google': 0.85,
        'mistral': 0.8,
        'cohere': 0.8,
        'huggingface': 0.75
      },
      qualityTrend: 'stable'
    };
  }

  /**
   * Setup time series data collection
   */
  private setupTimeSeriesCollection(): void {
    // Collect data every hour
    setInterval(() => {
      this.collectTimeSeriesData();
    }, 60 * 60 * 1000); // 1 hour
  }

  /**
   * Collect time series data point
   */
  private collectTimeSeriesData(): void {
    const stats = this.costOptimizationEngine.getCostStatistics();
    
    const dataPoint: TimeSeriesDataPoint = {
      timestamp: new Date(),
      requests: stats.totalRequests,
      totalCost: stats.totalCost,
      averageCostPerRequest: stats.averageCostPerRequest,
      costSavings: stats.costSavings
    };
    
    this.timeSeriesData.push(dataPoint);
    
    // Keep only last 168 hours (1 week) of data
    if (this.timeSeriesData.length > 168) {
      this.timeSeriesData = this.timeSeriesData.slice(-168);
    }
  }

  /**
   * Get comprehensive cost analytics
   */
  public getCostAnalytics(): CostAnalytics {
    const stats = this.costOptimizationEngine.getCostStatistics();
    
    return {
      totalRequests: stats.totalRequests,
      totalCost: stats.totalCost,
      averageCostPerRequest: stats.averageCostPerRequest,
      costSavings: stats.costSavings,
      savingsPercentage: stats.savingsPercentage,
      providerBreakdown: this.getProviderBreakdown(),
      taskTypeBreakdown: this.getTaskTypeBreakdown(),
      timeSeriesData: this.timeSeriesData,
      qualityMetrics: this.qualityMetrics
    };
  }

  /**
   * Get provider breakdown analytics
   */
  private getProviderBreakdown(): ProviderBreakdown[] {
    const config = this.costOptimizationEngine.getConfiguration();
    const breakdown: ProviderBreakdown[] = [];
    
    // This would typically come from actual usage data
    // For now, we'll simulate based on configuration
    config.costTiers.forEach(tier => {
      tier.providers.forEach(provider => {
        breakdown.push({
          provider,
          requests: Math.floor(Math.random() * 100), // Simulated data
          totalCost: Math.random() * 10, // Simulated data
          averageCostPerRequest: tier.maxCostPerToken,
          costPercentage: 0, // Would be calculated from actual data
          utilizationRate: Math.random() * 100 // Simulated data
        });
      });
    });
    
    return breakdown;
  }

  /**
   * Get task type breakdown analytics
   */
  private getTaskTypeBreakdown(): TaskTypeBreakdown[] {
    const taskTypes = ['general', 'trading_analysis', 'code_generation', 'sentiment_analysis', 'multilingual'];
    
    return taskTypes.map(taskType => ({
      taskType,
      requests: Math.floor(Math.random() * 50), // Simulated data
      totalCost: Math.random() * 5, // Simulated data
      averageCostPerRequest: Math.random() * 0.01, // Simulated data
      preferredProvider: this.getPreferredProviderForTask(taskType),
      costEfficiency: Math.random() * 100 // Simulated data
    }));
  }

  /**
   * Get preferred provider for a task type
   */
  private getPreferredProviderForTask(taskType: string): string {
    const rankings = this.providerRanking.rankProviders(['openai', 'google', 'mistral', 'cohere'], taskType);
    return rankings.length > 0 ? rankings[0].provider : 'openai';
  }

  /**
   * Track cost metrics with enhanced analytics
   */
  public trackCostMetrics(metrics: CostMetrics): void {
    this.costOptimizationEngine.trackCostMetrics(metrics);
    
    // Update quality metrics based on provider performance
    this.updateQualityMetrics(metrics);
  }

  /**
   * Update quality metrics based on usage
   */
  private updateQualityMetrics(metrics: CostMetrics): void {
    // This would typically involve actual quality assessment
    // For now, we'll simulate quality updates
    const providerQuality = this.qualityMetrics.providerQualityScores[metrics.provider] || 0.8;
    
    // Update average quality (weighted by usage)
    const totalRequests = this.costOptimizationEngine.getCostStatistics().totalRequests;
    if (totalRequests > 0) {
      this.qualityMetrics.averageResponseQuality = 
        (this.qualityMetrics.averageResponseQuality * (totalRequests - 1) + providerQuality) / totalRequests;
    }
    
    // Update quality threshold compliance
    const config = this.costOptimizationEngine.getConfiguration();
    this.qualityMetrics.qualityThresholdCompliance = 
      this.qualityMetrics.averageResponseQuality >= config.qualityThreshold ? 1.0 : 0.8;
  }

  /**
   * Get cost efficiency report
   */
  public getCostEfficiencyReport(): {
    currentEfficiency: number;
    targetEfficiency: number;
    efficiencyGap: number;
    recommendations: string[];
  } {
    const stats = this.costOptimizationEngine.getCostStatistics();
    const config = this.costOptimizationEngine.getConfiguration();
    
    const currentEfficiency = stats.savingsPercentage;
    const targetEfficiency = config.costTiers.find(tier => tier.tier === 'affordable_premium') ? 60 : 40;
    const efficiencyGap = targetEfficiency - currentEfficiency;
    
    const recommendations: string[] = [];
    
    if (efficiencyGap > 10) {
      recommendations.push('Consider increasing usage of affordable premium providers');
    }
    
    if (stats.averageCostPerRequest > 0.01) {
      recommendations.push('Review provider cost configurations');
    }
    
    if (this.qualityMetrics.qualityThresholdCompliance < 0.9) {
      recommendations.push('Monitor quality metrics and adjust thresholds if needed');
    }
    
    return {
      currentEfficiency,
      targetEfficiency,
      efficiencyGap,
      recommendations
    };
  }

  /**
   * Get performance trends
   */
  public getPerformanceTrends(): {
    costTrend: 'decreasing' | 'stable' | 'increasing';
    qualityTrend: 'improving' | 'stable' | 'declining';
    efficiencyTrend: 'improving' | 'stable' | 'declining';
  } {
    if (this.timeSeriesData.length < 2) {
      return {
        costTrend: 'stable',
        qualityTrend: 'stable',
        efficiencyTrend: 'stable'
      };
    }
    
    const recent = this.timeSeriesData.slice(-24); // Last 24 hours
    const older = this.timeSeriesData.slice(-48, -24); // Previous 24 hours
    
    const recentAvgCost = recent.reduce((sum, point) => sum + point.averageCostPerRequest, 0) / recent.length;
    const olderAvgCost = older.reduce((sum, point) => sum + point.averageCostPerRequest, 0) / older.length;
    
    const recentAvgSavings = recent.reduce((sum, point) => sum + point.costSavings, 0) / recent.length;
    const olderAvgSavings = older.reduce((sum, point) => sum + point.costSavings, 0) / older.length;
    
    return {
      costTrend: recentAvgCost < olderAvgCost ? 'decreasing' : 
                 recentAvgCost > olderAvgCost ? 'increasing' : 'stable',
      qualityTrend: this.qualityMetrics.qualityTrend,
      efficiencyTrend: recentAvgSavings > olderAvgSavings ? 'improving' :
                       recentAvgSavings < olderAvgSavings ? 'declining' : 'stable'
    };
  }

  /**
   * Export analytics data
   */
  public exportAnalytics(format: 'json' | 'csv'): string {
    const analytics = this.getCostAnalytics();
    
    if (format === 'json') {
      return JSON.stringify(analytics, null, 2);
    } else {
      // CSV format
      const csv = [
        'Metric,Value',
        `Total Requests,${analytics.totalRequests}`,
        `Total Cost,${analytics.totalCost}`,
        `Average Cost Per Request,${analytics.averageCostPerRequest}`,
        `Cost Savings,${analytics.costSavings}`,
        `Savings Percentage,${analytics.savingsPercentage}`,
        `Average Response Quality,${analytics.qualityMetrics.averageResponseQuality}`,
        `Quality Threshold Compliance,${analytics.qualityMetrics.qualityThresholdCompliance}`
      ].join('\n');
      
      return csv;
    }
  }

  /**
   * Reset analytics data
   */
  public resetAnalytics(): void {
    this.timeSeriesData = [];
    this.qualityMetrics = this.initializeQualityMetrics();
  }
}
