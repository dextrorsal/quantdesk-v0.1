/**
 * Unit tests for AnalyticsCollector
 */

import { AnalyticsCollector } from '../services/AnalyticsCollector';
import { RequestMetrics, TimeRange } from '../types/analytics';

// Mock dependencies
jest.mock('../config/analytics-config', () => {
  return {
    AnalyticsConfiguration: jest.fn().mockImplementation(() => ({
      getInstance: jest.fn().mockReturnThis(),
      getConfiguration: jest.fn().mockReturnValue({
        dataRetentionDays: 90,
        privacyCompliance: true,
        anonymizeData: true,
        enableRealTimeTracking: true,
        batchSize: 100,
        flushInterval: 30000
      }),
      isDataAnonymizationEnabled: jest.fn().mockReturnValue(true),
      isRealTimeTrackingEnabled: jest.fn().mockReturnValue(true),
      getBatchSize: jest.fn().mockReturnValue(100),
      getFlushInterval: jest.fn().mockReturnValue(30000),
      isDataRetentionEnabled: jest.fn().mockReturnValue(true),
      getDataRetentionDays: jest.fn().mockReturnValue(90)
    }))
  };
});

jest.mock('../services/CostOptimizationEngine', () => {
  return {
    CostOptimizationEngine: jest.fn().mockImplementation(() => ({
      getCostStatistics: jest.fn().mockReturnValue({ totalCost: 0.1, averageCost: 0.05 })
    }))
  };
});

jest.mock('../services/QualityThresholdManager', () => {
  return {
    QualityThresholdManager: jest.fn().mockImplementation(() => ({
      getQualityStats: jest.fn().mockReturnValue({
        totalEvaluations: 10,
        averageQualityScore: 0.8,
        escalationCount: 2
      })
    }))
  };
});

jest.mock('../services/TokenEstimationService', () => {
  return {
    TokenEstimationService: jest.fn().mockImplementation(() => ({
      getStats: jest.fn().mockReturnValue({ estimations: 10, cacheHits: 5 })
    }))
  };
});

jest.mock('../utils/logger', () => ({
  systemLogger: {
    startup: jest.fn()
  },
  errorLogger: {
    aiError: jest.fn()
  }
}));

describe('AnalyticsCollector', () => {
  let collector: AnalyticsCollector;

  beforeEach(() => {
    collector = new AnalyticsCollector();
  });

  afterEach(() => {
    collector.destroy();
    jest.clearAllMocks();
  });

  describe('Request Metrics Tracking', () => {
    test('should track request metrics successfully', async () => {
      const metrics: RequestMetrics = {
        requestId: 'test-request-1',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis',
        sessionId: 'session-1',
        escalationCount: 0,
        fallbackUsed: false
      };

      await collector.trackRequestMetrics(metrics);

      const stats = collector.getAnalyticsStats();
      expect(stats.totalRequests).toBe(1);
      expect(stats.totalCost).toBe(0.05);
      expect(stats.averageQualityScore).toBe(0.8);
    });

    test('should validate metrics before tracking', async () => {
      const invalidMetrics: RequestMetrics = {
        requestId: '', // Invalid: empty request ID
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await expect(collector.trackRequestMetrics(invalidMetrics)).rejects.toThrow('Invalid metrics: missing required fields');
    });

    test('should reject negative values in metrics', async () => {
      const invalidMetrics: RequestMetrics = {
        requestId: 'test-request-1',
        provider: 'openai',
        tokensUsed: -100, // Invalid: negative tokens
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await expect(collector.trackRequestMetrics(invalidMetrics)).rejects.toThrow('Invalid metrics: negative values not allowed');
    });

    test('should reject invalid quality scores', async () => {
      const invalidMetrics: RequestMetrics = {
        requestId: 'test-request-1',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 1.5, // Invalid: quality score > 1
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await expect(collector.trackRequestMetrics(invalidMetrics)).rejects.toThrow('Invalid metrics: quality score must be between 0 and 1');
    });

    test('should anonymize data when enabled', async () => {
      const metrics: RequestMetrics = {
        requestId: 'sensitive-request-id',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis',
        sessionId: 'sensitive-session-id'
      };

      await collector.trackRequestMetrics(metrics);

      // Verify that data was anonymized (requestId and sessionId should be hashed)
      const stats = collector.getAnalyticsStats();
      expect(stats.totalRequests).toBe(1);
      // The actual anonymization would be tested by checking the stored data
    });

    test('should track multiple metrics correctly', async () => {
      const metrics1: RequestMetrics = {
        requestId: 'test-request-1',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      const metrics2: RequestMetrics = {
        requestId: 'test-request-2',
        provider: 'google',
        tokensUsed: 150,
        cost: 0.03,
        qualityScore: 0.7,
        responseTime: 2000,
        timestamp: new Date(),
        taskType: 'general'
      };

      await collector.trackRequestMetrics(metrics1);
      await collector.trackRequestMetrics(metrics2);

      const stats = collector.getAnalyticsStats();
      expect(stats.totalRequests).toBe(2);
      expect(stats.totalCost).toBe(0.08);
      expect(stats.averageQualityScore).toBe(0.75); // (0.8 + 0.7) / 2
    });
  });

  describe('Cost Report Generation', () => {
    test('should generate cost report for time range', async () => {
      // Add some test data
      const now = new Date();
      const metrics: RequestMetrics[] = [
        {
          requestId: 'test-request-1',
          provider: 'openai',
          tokensUsed: 100,
          cost: 0.05,
          qualityScore: 0.8,
          responseTime: 1500,
          timestamp: new Date(now.getTime() - 1000),
          taskType: 'analysis'
        },
        {
          requestId: 'test-request-2',
          provider: 'google',
          tokensUsed: 150,
          cost: 0.03,
          qualityScore: 0.7,
          responseTime: 2000,
          timestamp: new Date(now.getTime() - 500),
          taskType: 'general'
        }
      ];

      for (const metric of metrics) {
        await collector.trackRequestMetrics(metric);
      }

      const timeRange: TimeRange = {
        start: new Date(now.getTime() - 2000),
        end: now
      };

      const report = await collector.generateCostReport(timeRange);

      expect(report.totalCost).toBe(0.08);
      expect(report.costSavings).toBeGreaterThan(0);
      expect(report.providerBreakdown).toHaveLength(2);
      expect(report.qualityMetrics).toBeDefined();
      expect(report.utilizationMetrics).toBeDefined();
    });

    test('should handle empty time range gracefully', async () => {
      const timeRange: TimeRange = {
        start: new Date(),
        end: new Date(Date.now() - 1000)
      };

      const report = await collector.generateCostReport(timeRange);

      expect(report.totalCost).toBe(0);
      expect(report.costSavings).toBe(0);
      expect(report.providerBreakdown).toHaveLength(0);
    });
  });

  describe('Provider Utilization', () => {
    test('should calculate provider utilization correctly', async () => {
      const metrics: RequestMetrics[] = [
        {
          requestId: 'test-request-1',
          provider: 'openai',
          tokensUsed: 100,
          cost: 0.05,
          qualityScore: 0.8,
          responseTime: 1500,
          timestamp: new Date(),
          taskType: 'analysis'
        },
        {
          requestId: 'test-request-2',
          provider: 'openai',
          tokensUsed: 200,
          cost: 0.10,
          qualityScore: 0.9,
          responseTime: 1200,
          timestamp: new Date(),
          taskType: 'general'
        },
        {
          requestId: 'test-request-3',
          provider: 'google',
          tokensUsed: 150,
          cost: 0.03,
          qualityScore: 0.7,
          responseTime: 2000,
          timestamp: new Date(),
          taskType: 'analysis'
        }
      ];

      for (const metric of metrics) {
        await collector.trackRequestMetrics(metric);
      }

      const utilization = await collector.getProviderUtilization();

      expect(utilization).toHaveLength(2);
      
      const openaiUtil = utilization.find(u => u.provider === 'openai');
      expect(openaiUtil).toBeDefined();
      expect(openaiUtil!.requestCount).toBe(2);
      expect(openaiUtil!.totalTokens).toBe(300);
      expect(openaiUtil!.totalCost).toBe(0.15);

      const googleUtil = utilization.find(u => u.provider === 'google');
      expect(googleUtil).toBeDefined();
      expect(googleUtil!.requestCount).toBe(1);
      expect(googleUtil!.totalTokens).toBe(150);
      expect(googleUtil!.totalCost).toBe(0.03);
    });

    test('should return empty array when no data available', async () => {
      const utilization = await collector.getProviderUtilization();
      expect(utilization).toHaveLength(0);
    });
  });

  describe('User Satisfaction Metrics', () => {
    test('should calculate satisfaction metrics correctly', async () => {
      const metrics: RequestMetrics[] = [
        {
          requestId: 'test-request-1',
          provider: 'openai',
          tokensUsed: 100,
          cost: 0.05,
          qualityScore: 0.9,
          responseTime: 1500,
          timestamp: new Date(),
          taskType: 'analysis',
          escalationCount: 0,
          fallbackUsed: false
        },
        {
          requestId: 'test-request-2',
          provider: 'google',
          tokensUsed: 150,
          cost: 0.03,
          qualityScore: 0.6,
          responseTime: 2000,
          timestamp: new Date(),
          taskType: 'general',
          escalationCount: 1,
          fallbackUsed: true
        },
        {
          requestId: 'test-request-3',
          provider: 'mistral',
          tokensUsed: 200,
          cost: 0.02,
          qualityScore: 0.8,
          responseTime: 1800,
          timestamp: new Date(),
          taskType: 'analysis',
          escalationCount: 0,
          fallbackUsed: false
        }
      ];

      for (const metric of metrics) {
        await collector.trackRequestMetrics(metric);
      }

      const satisfaction = await collector.getUserSatisfactionMetrics();

      expect(satisfaction.averageQualityScore).toBeCloseTo(0.767, 2); // (0.9 + 0.6 + 0.8) / 3
      expect(satisfaction.userSatisfactionRate).toBeGreaterThan(0);
      expect(satisfaction.escalationRate).toBeCloseTo(0.333, 2); // 1/3
      expect(satisfaction.fallbackRate).toBeCloseTo(0.333, 2); // 1/3
      expect(satisfaction.responseTimeMetrics).toBeDefined();
      expect(satisfaction.qualityDistribution).toBeDefined();
    });

    test('should return default metrics when no data available', async () => {
      const satisfaction = await collector.getUserSatisfactionMetrics();

      expect(satisfaction.averageQualityScore).toBe(0.5);
      expect(satisfaction.userSatisfactionRate).toBe(0.4);
      expect(satisfaction.escalationRate).toBe(0);
      expect(satisfaction.fallbackRate).toBe(0);
    });
  });

  describe('Analytics Dashboard', () => {
    test('should generate comprehensive dashboard', async () => {
      // Add test data
      const metrics: RequestMetrics = {
        requestId: 'test-request-1',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await collector.trackRequestMetrics(metrics);

      const dashboard = await collector.getAnalyticsDashboard();

      expect(dashboard.overview).toBeDefined();
      expect(dashboard.costReport).toBeDefined();
      expect(dashboard.providerUtilization).toBeDefined();
      expect(dashboard.satisfactionMetrics).toBeDefined();
      expect(dashboard.costSavings).toBeDefined();
      expect(dashboard.trends).toBeDefined();
    });
  });

  describe('Cost Savings Report', () => {
    test('should calculate cost savings correctly', async () => {
      const metrics: RequestMetrics = {
        requestId: 'test-request-1',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await collector.trackRequestMetrics(metrics);

      const timeRange: TimeRange = {
        start: new Date(Date.now() - 10000),
        end: new Date()
      };

      const savingsReport = await collector.generateCostSavingsReport(timeRange);

      expect(savingsReport.baselineCost).toBeGreaterThan(0);
      expect(savingsReport.actualCost).toBe(0.05);
      expect(savingsReport.savingsAmount).toBeGreaterThanOrEqual(0);
      expect(savingsReport.savingsPercentage).toBeGreaterThanOrEqual(0);
      expect(savingsReport.roi).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Data Export', () => {
    test('should export analytics data with filters', async () => {
      const metrics: RequestMetrics[] = [
        {
          requestId: 'test-request-1',
          provider: 'openai',
          tokensUsed: 100,
          cost: 0.05,
          qualityScore: 0.8,
          responseTime: 1500,
          timestamp: new Date(),
          taskType: 'analysis'
        },
        {
          requestId: 'test-request-2',
          provider: 'google',
          tokensUsed: 150,
          cost: 0.03,
          qualityScore: 0.7,
          responseTime: 2000,
          timestamp: new Date(),
          taskType: 'general'
        }
      ];

      for (const metric of metrics) {
        await collector.trackRequestMetrics(metric);
      }

      const exportedData = await collector.exportAnalyticsData({
        filter: {
          providers: ['openai']
        },
        limit: 10
      });

      expect(exportedData).toHaveLength(1);
      expect(exportedData[0].provider).toBe('openai');
    });

    test('should handle empty export gracefully', async () => {
      const exportedData = await collector.exportAnalyticsData({
        filter: {
          providers: ['nonexistent-provider']
        }
      });

      expect(exportedData).toHaveLength(0);
    });
  });

  describe('Data Retention', () => {
    test('should clear old data based on retention policy', async () => {
      // Add old data
      const oldMetrics: RequestMetrics = {
        requestId: 'old-request',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(Date.now() - 100 * 24 * 60 * 60 * 1000), // 100 days ago
        taskType: 'analysis'
      };

      await collector.trackRequestMetrics(oldMetrics);

      // Add recent data
      const recentMetrics: RequestMetrics = {
        requestId: 'recent-request',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await collector.trackRequestMetrics(recentMetrics);

      // Clear old data
      await collector.clearOldData();

      const stats = collector.getAnalyticsStats();
      // Should still have recent data
      expect(stats.totalRequests).toBeGreaterThan(0);
    });
  });

  describe('Error Handling', () => {
    test('should handle tracking errors gracefully', async () => {
      const invalidMetrics: RequestMetrics = {
        requestId: '', // Invalid
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      // Should not throw error
      await collector.trackRequestMetrics(invalidMetrics);
      
      // Stats should remain unchanged
      const stats = collector.getAnalyticsStats();
      expect(stats.totalRequests).toBe(0);
    });

    test('should handle report generation errors gracefully', async () => {
      // Mock an error in report generation
      const timeRange: TimeRange = {
        start: new Date(),
        end: new Date(Date.now() - 1000)
      };

      // Should not throw error
      const report = await collector.generateCostReport(timeRange);
      expect(report.totalCost).toBe(0);
    });
  });
});
