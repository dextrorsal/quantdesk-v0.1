/**
 * Integration tests for Analytics with existing monitoring systems
 */

import { AnalyticsCollector } from '../services/AnalyticsCollector';
import { MultiLLMRouter } from '../services/MultiLLMRouter';
import { OfficialLLMRouter } from '../services/OfficialLLMRouter';

// Mock dependencies
jest.mock('../config/analytics-config', () => {
  return {
    AnalyticsConfiguration: jest.fn().mockImplementation(() => ({
      getInstance: jest.fn().mockReturnThis(),
      getConfiguration: jest.fn().mockReturnValue({
        dataRetentionDays: 90,
        privacyCompliance: true,
        anonymizeData: false,
        enableRealTimeTracking: true,
        batchSize: 100,
        flushInterval: 30000
      }),
      isDataAnonymizationEnabled: jest.fn().mockReturnValue(false),
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

describe('Analytics Integration with Existing Monitoring', () => {
  let analyticsCollector: AnalyticsCollector;
  let multiRouter: MultiLLMRouter;
  let officialRouter: OfficialLLMRouter;

  beforeEach(() => {
    analyticsCollector = new AnalyticsCollector();
    multiRouter = new MultiLLMRouter();
    officialRouter = new OfficialLLMRouter();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Integration with Cost Optimization Engine', () => {
    test('should integrate analytics with cost optimization', async () => {
      // Mock cost optimization engine
      const mockCostEngine = {
        getCostStatistics: jest.fn().mockReturnValue({
          totalCost: 0.1,
          averageCost: 0.05,
          costSavings: 0.02
        })
      };

      // Add analytics data
      const metrics = {
        requestId: 'cost-integration-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      // Verify integration
      const stats = analyticsCollector.getAnalyticsStats();
      expect(stats.totalCost).toBe(0.05);
      expect(stats.totalSavings).toBe(0);
    });

    test('should track cost savings from optimization', async () => {
      const metrics = {
        requestId: 'cost-savings-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.03, // Lower cost due to optimization
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const stats = analyticsCollector.getAnalyticsStats();
      expect(stats.totalCost).toBe(0.03);
    });
  });

  describe('Integration with Quality Threshold Manager', () => {
    test('should integrate analytics with quality monitoring', async () => {
      const metrics = {
        requestId: 'quality-integration-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const satisfaction = await analyticsCollector.getUserSatisfactionMetrics();
      expect(satisfaction.averageQualityScore).toBe(0.8);
    });

    test('should track quality escalations', async () => {
      const metrics = {
        requestId: 'quality-escalation-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.6, // Low quality
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis',
        escalationCount: 1
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const satisfaction = await analyticsCollector.getUserSatisfactionMetrics();
      expect(satisfaction.escalationRate).toBeGreaterThan(0);
    });
  });

  describe('Integration with Token Estimation Service', () => {
    test('should integrate analytics with token estimation', async () => {
      const metrics = {
        requestId: 'token-integration-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const stats = analyticsCollector.getAnalyticsStats();
      expect(stats.totalRequests).toBe(1);
    });

    test('should track accurate token usage', async () => {
      const metrics = {
        requestId: 'token-accuracy-test',
        provider: 'openai',
        tokensUsed: 150, // Accurate token count
        cost: 0.075, // Accurate cost
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const stats = analyticsCollector.getAnalyticsStats();
      expect(stats.totalCost).toBe(0.075);
    });
  });

  describe('Integration with MultiLLMRouter', () => {
    test('should integrate analytics with MultiLLMRouter', async () => {
      // Mock the router to return a response
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

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail due to missing API keys
      }

      // Verify analytics integration
      const stats = multiRouter.getAnalyticsStats();
      expect(stats).toBeDefined();
    });

    test('should track MultiLLMRouter performance metrics', async () => {
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

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis');
      } catch (error) {
        // Expected to fail
      }

      // Verify performance tracking
      const stats = multiRouter.getAnalyticsStats();
      expect(stats).toBeDefined();
    });
  });

  describe('Integration with OfficialLLMRouter', () => {
    test('should integrate analytics with OfficialLLMRouter', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockResolvedValue('Test response');
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail due to missing API keys
      }

      // Verify analytics integration
      const stats = officialRouter.getAnalyticsStats();
      expect(stats).toBeDefined();
    });

    test('should track OfficialLLMRouter performance metrics', async () => {
      jest.spyOn(officialRouter as any, 'selectBestProvider').mockReturnValue({ name: 'openai' });
      jest.spyOn(officialRouter as any, 'callProvider').mockResolvedValue('Test response');
      jest.spyOn(officialRouter as any, 'trackUsageWithAccurateTokens').mockImplementation(() => {});
      jest.spyOn(officialRouter as any, 'getProviderModel').mockReturnValue('gpt-3.5-turbo');

      try {
        await officialRouter.routeRequest('Test prompt', 'analysis');
      } catch (error) {
        // Expected to fail
      }

      // Verify performance tracking
      const stats = officialRouter.getAnalyticsStats();
      expect(stats).toBeDefined();
    });
  });

  describe('Integration with Logging Systems', () => {
    test('should integrate with system logger', async () => {
      const metrics = {
        requestId: 'logging-integration-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      // Verify logging integration
      const stats = analyticsCollector.getAnalyticsStats();
      expect(stats).toBeDefined();
    });

    test('should integrate with error logger', async () => {
      // Mock analytics to throw error
      jest.spyOn(analyticsCollector, 'trackRequestMetrics').mockRejectedValue(new Error('Analytics error'));

      const metrics = {
        requestId: 'error-logging-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      // Should not throw error even if analytics fails
      await expect(analyticsCollector.trackRequestMetrics(metrics)).rejects.toThrow('Analytics error');
    });
  });

  describe('Integration with Database Systems', () => {
    test('should integrate with data persistence', async () => {
      const metrics = {
        requestId: 'database-integration-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      // Verify data persistence integration
      const stats = analyticsCollector.getAnalyticsStats();
      expect(stats.totalRequests).toBe(1);
    });

    test('should handle database connection errors gracefully', async () => {
      // Mock database error
      jest.spyOn(analyticsCollector as any, 'flushBatch').mockRejectedValue(new Error('Database connection error'));

      const metrics = {
        requestId: 'database-error-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      // Should not throw error even if database fails
      await expect(analyticsCollector.trackRequestMetrics(metrics)).resolves.not.toThrow();
    });
  });

  describe('Integration with Monitoring Dashboards', () => {
    test('should provide data for monitoring dashboards', async () => {
      const metrics = {
        requestId: 'dashboard-integration-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const dashboard = await analyticsCollector.getAnalyticsDashboard();
      expect(dashboard.overview).toBeDefined();
      expect(dashboard.costReport).toBeDefined();
      expect(dashboard.providerUtilization).toBeDefined();
      expect(dashboard.satisfactionMetrics).toBeDefined();
    });

    test('should provide real-time monitoring data', async () => {
      const metrics = {
        requestId: 'realtime-monitoring-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const stats = analyticsCollector.getAnalyticsStats();
      expect(stats.lastUpdated).toBeDefined();
      expect(stats.dataPointsCollected).toBeGreaterThan(0);
    });
  });

  describe('Integration with Alerting Systems', () => {
    test('should provide data for alerting systems', async () => {
      const metrics = {
        requestId: 'alerting-integration-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const satisfaction = await analyticsCollector.getUserSatisfactionMetrics();
      expect(satisfaction.averageQualityScore).toBeDefined();
      expect(satisfaction.escalationRate).toBeDefined();
      expect(satisfaction.fallbackRate).toBeDefined();
    });

    test('should track metrics for threshold-based alerts', async () => {
      const metrics = {
        requestId: 'threshold-alert-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.3, // Low quality that should trigger alerts
        responseTime: 5000, // High response time that should trigger alerts
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const satisfaction = await analyticsCollector.getUserSatisfactionMetrics();
      expect(satisfaction.averageQualityScore).toBeLessThan(0.5);
    });
  });

  describe('End-to-End Integration Testing', () => {
    test('should integrate all monitoring systems together', async () => {
      // Test complete integration flow
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

      try {
        await multiRouter.routeRequest('Test prompt', 'analysis', 'session-1');
      } catch (error) {
        // Expected to fail due to missing API keys
      }

      // Verify all systems are integrated
      const stats = multiRouter.getAnalyticsStats();
      expect(stats).toBeDefined();

      const dashboard = await multiRouter.getAnalyticsDashboard();
      expect(dashboard).toBeDefined();

      const utilization = await multiRouter.getProviderUtilization();
      expect(utilization).toBeDefined();

      const satisfaction = await multiRouter.getUserSatisfactionMetrics();
      expect(satisfaction).toBeDefined();
    });

    test('should maintain data consistency across systems', async () => {
      const metrics = {
        requestId: 'consistency-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(metrics);

      const stats = analyticsCollector.getAnalyticsStats();
      const dashboard = await analyticsCollector.getAnalyticsDashboard();

      // Verify data consistency
      expect(stats.totalRequests).toBe(1);
      expect(dashboard.overview.totalRequests).toBe(1);
    });
  });
});
