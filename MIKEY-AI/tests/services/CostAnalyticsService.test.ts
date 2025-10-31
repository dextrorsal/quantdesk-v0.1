/**
 * Unit tests for CostAnalyticsService
 */

import { CostAnalyticsService } from '../services/CostAnalyticsService';
import { CostMetrics } from '../services/CostOptimizationEngine';

describe('CostAnalyticsService', () => {
  let analyticsService: CostAnalyticsService;
  let originalEnv: NodeJS.ProcessEnv;

  beforeEach(() => {
    // Save original environment
    originalEnv = { ...process.env };
    
    // Set up test environment variables
    process.env.COST_TIER_AFFORDABLE_PREMIUM = 'mistral,google,cohere';
    process.env.COST_TIER_EXPENSIVE = 'openai,anthropic';
    process.env.ENABLE_COST_OPTIMIZATION = 'true';
    process.env.AFFORDABLE_MAX_COST_PER_TOKEN = '0.0001';
    process.env.EXPENSIVE_MAX_COST_PER_TOKEN = '0.01';
    process.env.QUALITY_THRESHOLD = '0.85';
    
    analyticsService = new CostAnalyticsService();
  });

  afterEach(() => {
    // Restore original environment
    process.env = originalEnv;
  });

  describe('Cost Analytics Generation', () => {
    it('should generate comprehensive cost analytics', () => {
      const analytics = analyticsService.getCostAnalytics();
      
      expect(analytics).toHaveProperty('totalRequests');
      expect(analytics).toHaveProperty('totalCost');
      expect(analytics).toHaveProperty('averageCostPerRequest');
      expect(analytics).toHaveProperty('costSavings');
      expect(analytics).toHaveProperty('savingsPercentage');
      expect(analytics).toHaveProperty('providerBreakdown');
      expect(analytics).toHaveProperty('taskTypeBreakdown');
      expect(analytics).toHaveProperty('timeSeriesData');
      expect(analytics).toHaveProperty('qualityMetrics');
      
      expect(typeof analytics.totalRequests).toBe('number');
      expect(typeof analytics.totalCost).toBe('number');
      expect(typeof analytics.averageCostPerRequest).toBe('number');
      expect(typeof analytics.costSavings).toBe('number');
      expect(typeof analytics.savingsPercentage).toBe('number');
      expect(Array.isArray(analytics.providerBreakdown)).toBe(true);
      expect(Array.isArray(analytics.taskTypeBreakdown)).toBe(true);
      expect(Array.isArray(analytics.timeSeriesData)).toBe(true);
      expect(typeof analytics.qualityMetrics).toBe('object');
    });

    it('should track cost metrics correctly', () => {
      const metrics: CostMetrics = {
        provider: 'mistral',
        tokensUsed: 100,
        costPerToken: 0.0001,
        totalCost: 0.01,
        timestamp: new Date(),
        taskType: 'general'
      };

      analyticsService.trackCostMetrics(metrics);
      
      const analytics = analyticsService.getCostAnalytics();
      expect(analytics.totalRequests).toBeGreaterThan(0);
    });
  });

  describe('Provider Breakdown Analytics', () => {
    it('should generate provider breakdown data', () => {
      const analytics = analyticsService.getCostAnalytics();
      
      expect(Array.isArray(analytics.providerBreakdown)).toBe(true);
      
      if (analytics.providerBreakdown.length > 0) {
        const breakdown = analytics.providerBreakdown[0];
        expect(breakdown).toHaveProperty('provider');
        expect(breakdown).toHaveProperty('requests');
        expect(breakdown).toHaveProperty('totalCost');
        expect(breakdown).toHaveProperty('averageCostPerRequest');
        expect(breakdown).toHaveProperty('costPercentage');
        expect(breakdown).toHaveProperty('utilizationRate');
        
        expect(typeof breakdown.provider).toBe('string');
        expect(typeof breakdown.requests).toBe('number');
        expect(typeof breakdown.totalCost).toBe('number');
        expect(typeof breakdown.averageCostPerRequest).toBe('number');
        expect(typeof breakdown.costPercentage).toBe('number');
        expect(typeof breakdown.utilizationRate).toBe('number');
      }
    });
  });

  describe('Task Type Breakdown Analytics', () => {
    it('should generate task type breakdown data', () => {
      const analytics = analyticsService.getCostAnalytics();
      
      expect(Array.isArray(analytics.taskTypeBreakdown)).toBe(true);
      
      if (analytics.taskTypeBreakdown.length > 0) {
        const breakdown = analytics.taskTypeBreakdown[0];
        expect(breakdown).toHaveProperty('taskType');
        expect(breakdown).toHaveProperty('requests');
        expect(breakdown).toHaveProperty('totalCost');
        expect(breakdown).toHaveProperty('averageCostPerRequest');
        expect(breakdown).toHaveProperty('preferredProvider');
        expect(breakdown).toHaveProperty('costEfficiency');
        
        expect(typeof breakdown.taskType).toBe('string');
        expect(typeof breakdown.requests).toBe('number');
        expect(typeof breakdown.totalCost).toBe('number');
        expect(typeof breakdown.averageCostPerRequest).toBe('number');
        expect(typeof breakdown.preferredProvider).toBe('string');
        expect(typeof breakdown.costEfficiency).toBe('number');
      }
    });
  });

  describe('Quality Metrics', () => {
    it('should track quality metrics', () => {
      const analytics = analyticsService.getCostAnalytics();
      const qualityMetrics = analytics.qualityMetrics;
      
      expect(qualityMetrics).toHaveProperty('averageResponseQuality');
      expect(qualityMetrics).toHaveProperty('qualityThresholdCompliance');
      expect(qualityMetrics).toHaveProperty('providerQualityScores');
      expect(qualityMetrics).toHaveProperty('qualityTrend');
      
      expect(typeof qualityMetrics.averageResponseQuality).toBe('number');
      expect(typeof qualityMetrics.qualityThresholdCompliance).toBe('number');
      expect(typeof qualityMetrics.providerQualityScores).toBe('object');
      expect(['improving', 'stable', 'declining']).toContain(qualityMetrics.qualityTrend);
    });

    it('should update quality metrics when tracking usage', () => {
      const metrics: CostMetrics = {
        provider: 'openai',
        tokensUsed: 100,
        costPerToken: 0.00015,
        totalCost: 0.015,
        timestamp: new Date(),
        taskType: 'general'
      };

      analyticsService.trackCostMetrics(metrics);
      
      const analytics = analyticsService.getCostAnalytics();
      expect(analytics.qualityMetrics.averageResponseQuality).toBeGreaterThan(0);
    });
  });

  describe('Cost Efficiency Report', () => {
    it('should generate cost efficiency report', () => {
      const report = analyticsService.getCostEfficiencyReport();
      
      expect(report).toHaveProperty('currentEfficiency');
      expect(report).toHaveProperty('targetEfficiency');
      expect(report).toHaveProperty('efficiencyGap');
      expect(report).toHaveProperty('recommendations');
      
      expect(typeof report.currentEfficiency).toBe('number');
      expect(typeof report.targetEfficiency).toBe('number');
      expect(typeof report.efficiencyGap).toBe('number');
      expect(Array.isArray(report.recommendations)).toBe(true);
    });

    it('should provide actionable recommendations', () => {
      const report = analyticsService.getCostEfficiencyReport();
      
      expect(Array.isArray(report.recommendations)).toBe(true);
      
      // Recommendations should be strings
      report.recommendations.forEach(recommendation => {
        expect(typeof recommendation).toBe('string');
        expect(recommendation.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Performance Trends', () => {
    it('should generate performance trends', () => {
      const trends = analyticsService.getPerformanceTrends();
      
      expect(trends).toHaveProperty('costTrend');
      expect(trends).toHaveProperty('qualityTrend');
      expect(trends).toHaveProperty('efficiencyTrend');
      
      expect(['decreasing', 'stable', 'increasing']).toContain(trends.costTrend);
      expect(['improving', 'stable', 'declining']).toContain(trends.qualityTrend);
      expect(['improving', 'stable', 'declining']).toContain(trends.efficiencyTrend);
    });
  });

  describe('Data Export', () => {
    it('should export analytics data in JSON format', () => {
      const jsonData = analyticsService.exportAnalytics('json');
      
      expect(typeof jsonData).toBe('string');
      
      // Should be valid JSON
      const parsed = JSON.parse(jsonData);
      expect(parsed).toHaveProperty('totalRequests');
      expect(parsed).toHaveProperty('totalCost');
      expect(parsed).toHaveProperty('qualityMetrics');
    });

    it('should export analytics data in CSV format', () => {
      const csvData = analyticsService.exportAnalytics('csv');
      
      expect(typeof csvData).toBe('string');
      expect(csvData).toContain('Metric,Value');
      expect(csvData).toContain('Total Requests');
      expect(csvData).toContain('Total Cost');
      expect(csvData).toContain('Average Cost Per Request');
    });
  });

  describe('Data Management', () => {
    it('should reset analytics data', () => {
      // Track some data first
      const metrics: CostMetrics = {
        provider: 'mistral',
        tokensUsed: 100,
        costPerToken: 0.0001,
        totalCost: 0.01,
        timestamp: new Date(),
        taskType: 'general'
      };
      
      analyticsService.trackCostMetrics(metrics);
      
      // Reset analytics
      analyticsService.resetAnalytics();
      
      const analytics = analyticsService.getCostAnalytics();
      expect(analytics.timeSeriesData).toHaveLength(0);
    });
  });

  describe('Time Series Data Collection', () => {
    it('should collect time series data', () => {
      const analytics = analyticsService.getCostAnalytics();
      
      expect(Array.isArray(analytics.timeSeriesData)).toBe(true);
      
      // Time series data should have the correct structure
      if (analytics.timeSeriesData.length > 0) {
        const dataPoint = analytics.timeSeriesData[0];
        expect(dataPoint).toHaveProperty('timestamp');
        expect(dataPoint).toHaveProperty('requests');
        expect(dataPoint).toHaveProperty('totalCost');
        expect(dataPoint).toHaveProperty('averageCostPerRequest');
        expect(dataPoint).toHaveProperty('costSavings');
        
        expect(dataPoint.timestamp instanceof Date).toBe(true);
        expect(typeof dataPoint.requests).toBe('number');
        expect(typeof dataPoint.totalCost).toBe('number');
        expect(typeof dataPoint.averageCostPerRequest).toBe('number');
        expect(typeof dataPoint.costSavings).toBe('number');
      }
    });
  });

  describe('Integration with Cost Optimization Components', () => {
    it('should integrate with CostOptimizationEngine', () => {
      const analytics = analyticsService.getCostAnalytics();
      
      // Should have access to cost optimization data
      expect(typeof analytics.totalRequests).toBe('number');
      expect(typeof analytics.costSavings).toBe('number');
    });

    it('should integrate with ProviderCostRanking', () => {
      const analytics = analyticsService.getCostAnalytics();
      
      // Should have provider breakdown data
      expect(Array.isArray(analytics.providerBreakdown)).toBe(true);
    });
  });
});
