/**
 * Integration tests for Analytics Collection
 */

import { MultiLLMRouter } from '../services/MultiLLMRouter';
import { OfficialLLMRouter } from '../services/OfficialLLMRouter';
import { AnalyticsCollector } from '../services/AnalyticsCollector';

// Mock dependencies
jest.mock('../config/analytics-config', () => {
  return {
    AnalyticsConfiguration: jest.fn().mockImplementation(() => ({
      getInstance: jest.fn().mockReturnThis(),
      getConfiguration: jest.fn().mockReturnValue({
        dataRetentionDays: 90,
        privacyCompliance: true,
        anonymizeData: false, // Disable for testing
        enableRealTimeTracking: true,
        batchSize: 10,
        flushInterval: 1000
      }),
      isDataAnonymizationEnabled: jest.fn().mockReturnValue(false),
      isRealTimeTrackingEnabled: jest.fn().mockReturnValue(true),
      getBatchSize: jest.fn().mockReturnValue(10),
      getFlushInterval: jest.fn().mockReturnValue(1000),
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
      isQualityEvaluationEnabled: jest.fn().mockReturnValue(true),
      evaluateQuality: jest.fn().mockResolvedValue({
        provider: 'openai',
        qualityScore: 0.8,
        shouldEscalate: false,
        confidence: 0.9,
        timestamp: new Date()
      }),
      makeEscalationDecision: jest.fn().mockResolvedValue({
        shouldEscalate: false,
        reason: 'Quality acceptable',
        currentProvider: 'openai',
        suggestedProvider: 'openai',
        qualityScore: 0.8,
        threshold: 0.7,
        confidence: 0.9
      })
    }))
  };
});

jest.mock('../services/TokenEstimationService', () => {
  return {
    TokenEstimationService: jest.fn().mockImplementation(() => ({
      estimateTokens: jest.fn().mockResolvedValue({
        totalTokens: 100,
        estimatedCost: 0.05,
        provider: 'openai',
        model: 'gpt-3.5-turbo'
      })
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

describe('Analytics Integration Tests', () => {
  let multiRouter: MultiLLMRouter;
  let officialRouter: OfficialLLMRouter;
  let analyticsCollector: AnalyticsCollector;

  beforeEach(() => {
    multiRouter = new MultiLLMRouter();
    officialRouter = new OfficialLLMRouter();
    analyticsCollector = new AnalyticsCollector();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('MultiLLMRouter Analytics Integration', () => {
    test('should track analytics metrics when routing requests', async () => {
      // Mock the provider to return a response
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockResolvedValue({ content: 'Test response' })
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      // Mock the selectBestProvider method
      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);

      // Mock the trackUsageWithAccurateTokens method
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});

      // Mock the getProviderModel method
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Track analytics calls
      const trackMetricsSpy = jest.spyOn(analyticsCollector, 'trackRequestMetrics');

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail due to missing API keys, but analytics should still be tracked
      }

      // Verify analytics tracking was called
      expect(trackMetricsSpy).toHaveBeenCalled();
    });

    test('should expose analytics methods', async () => {
      // Test that analytics methods are available
      expect(typeof multiRouter.getAnalyticsDashboard).toBe('function');
      expect(typeof multiRouter.getCostReport).toBe('function');
      expect(typeof multiRouter.getProviderUtilization).toBe('function');
      expect(typeof multiRouter.getUserSatisfactionMetrics).toBe('function');
      expect(typeof multiRouter.getAnalyticsStats).toBe('function');
    });

    test('should handle analytics errors gracefully', async () => {
      // Mock analytics to throw error
      jest.spyOn(analyticsCollector, 'trackRequestMetrics').mockRejectedValue(new Error('Analytics error'));

      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockResolvedValue({ content: 'Test response' })
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Should not throw error even if analytics fails
      await expect(multiRouter.routeRequest('Test prompt', 'analysis')).resolves.not.toThrow();
    });
  });

  describe('OfficialLLMRouter Analytics Integration', () => {
    test('should track analytics metrics when routing requests', async () => {
      // Mock the provider selection and calling
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockResolvedValue('Test response');
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Track analytics calls
      const trackMetricsSpy = jest.spyOn(analyticsCollector, 'trackRequestMetrics');

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail due to missing API keys, but analytics should still be tracked
      }

      // Verify analytics tracking was called
      expect(trackMetricsSpy).toHaveBeenCalled();
    });

    test('should expose analytics methods', async () => {
      // Test that analytics methods are available
      expect(typeof officialRouter.getAnalyticsDashboard).toBe('function');
      expect(typeof officialRouter.getCostReport).toBe('function');
      expect(typeof officialRouter.getProviderUtilization).toBe('function');
      expect(typeof officialRouter.getUserSatisfactionMetrics).toBe('function');
      expect(typeof officialRouter.getAnalyticsStats).toBe('function');
    });

    test('should handle analytics errors gracefully', async () => {
      // Mock analytics to throw error
      jest.spyOn(analyticsCollector, 'trackRequestMetrics').mockRejectedValue(new Error('Analytics error'));

      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockResolvedValue('Test response');
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Should not throw error even if analytics fails
      await expect(officialRouter.routeRequest('Test prompt', 'analysis')).resolves.not.toThrow();
    });
  });

  describe('AnalyticsCollector Integration', () => {
    test('should collect metrics from multiple sources', async () => {
      const metrics1 = {
        requestId: 'req-1',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      const metrics2 = {
        requestId: 'req-2',
        provider: 'google',
        tokensUsed: 150,
        cost: 0.03,
        qualityScore: 0.7,
        responseTime: 2000,
        timestamp: new Date(),
        taskType: 'general'
      };

      await analyticsCollector.trackRequestMetrics(metrics1);
      await analyticsCollector.trackRequestMetrics(metrics2);

      const stats = analyticsCollector.getAnalyticsStats();
      expect(stats.totalRequests).toBe(2);
      expect(stats.totalCost).toBe(0.08);
      expect(stats.averageQualityScore).toBe(0.75);
    });

    test('should generate comprehensive reports', async () => {
      // Add test data
      const metrics = {
        requestId: 'req-1',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      // Test cost report generation
      const timeRange = {
        start: new Date(Date.now() - 10000),
        end: new Date()
      };

      const costReport = await analyticsCollector.generateCostReport(timeRange);
      expect(costReport.totalCost).toBe(0.05);
      expect(costReport.providerBreakdown).toBeDefined();

      // Test provider utilization
      const utilization = await analyticsCollector.getProviderUtilization();
      expect(utilization).toBeDefined();

      // Test satisfaction metrics
      const satisfaction = await analyticsCollector.getUserSatisfactionMetrics();
      expect(satisfaction.averageQualityScore).toBeDefined();

      // Test dashboard
      const dashboard = await analyticsCollector.getAnalyticsDashboard();
      expect(dashboard.overview).toBeDefined();
      expect(dashboard.costReport).toBeDefined();
      expect(dashboard.providerUtilization).toBeDefined();
      expect(dashboard.satisfactionMetrics).toBeDefined();
    });

    test('should handle data retention policies', async () => {
      // Add old data
      const oldMetrics = {
        requestId: 'old-req',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(Date.now() - 100 * 24 * 60 * 60 * 1000), // 100 days ago
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(oldMetrics);

      // Add recent data
      const recentMetrics = {
        requestId: 'recent-req',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(recentMetrics);

      // Clear old data
      await analyticsCollector.clearOldData();

      const stats = analyticsCollector.getAnalyticsStats();
      // Should still have recent data
      expect(stats.totalRequests).toBeGreaterThan(0);
    });

    test('should export data with filters', async () => {
      // Add test data
      const metrics = [
        {
          requestId: 'req-1',
          provider: 'openai',
          tokensUsed: 100,
          cost: 0.05,
          qualityScore: 0.8,
          responseTime: 1500,
          timestamp: new Date(),
          taskType: 'analysis'
        },
        {
          requestId: 'req-2',
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
        await analyticsCollector.trackRequestMetrics(metric);
      }

      // Export with provider filter
      const exportedData = await analyticsCollector.exportAnalyticsData({
        filter: {
          providers: ['openai']
        },
        limit: 10
      });

      expect(exportedData).toHaveLength(1);
      expect(exportedData[0].provider).toBe('openai');
    });
  });

  describe('End-to-End Analytics Flow', () => {
    test('should track complete request lifecycle', async () => {
      // Mock all dependencies
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockResolvedValue({ content: 'Test response' })
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      // Track the complete flow
      const trackMetricsSpy = jest.spyOn(analyticsCollector, 'trackRequestMetrics');

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail due to missing API keys
      }

      // Verify analytics was tracked
      expect(trackMetricsSpy).toHaveBeenCalledWith(
        expect.objectContaining({
          provider: 'openai',
          taskType: 'analysis',
          sessionId: 'session-1'
        })
      );

      // Verify analytics methods work
      const stats = multiRouter.getAnalyticsStats();
      expect(stats).toBeDefined();
    });

    test('should maintain analytics consistency across routers', async () => {
      // Both routers should use the same analytics collector
      const multiStats = multiRouter.getAnalyticsStats();
      const officialStats = officialRouter.getAnalyticsStats();

      // Both should return valid stats objects
      expect(multiStats).toBeDefined();
      expect(officialStats).toBeDefined();
      expect(typeof multiStats.totalRequests).toBe('number');
      expect(typeof officialStats.totalRequests).toBe('number');
    });
  });

  describe('Performance Impact', () => {
    test('should not significantly impact routing performance', async () => {
      const mockProvider = {
        name: 'openai',
        model: {
          invoke: jest.fn().mockResolvedValue({ content: 'Test response' })
        },
        tokenLimit: 1000,
        tokensUsed: 0,
        costPerToken: 0.0005,
        strengths: ['general'],
        isAvailable: true
      };

      jest.spyOn(multiRouter as any, 'selectBestProvider').mockReturnValue(mockProvider);
      jest.spyOn(multiRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(multiRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      const startTime = Date.now();

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis');
      } catch (error) {
        // Expected to fail
      }

      const endTime = Date.now();
      const duration = endTime - startTime;

      // Analytics should not add more than 50ms overhead
      expect(duration).toBeLessThan(50);
    });
  });
});
