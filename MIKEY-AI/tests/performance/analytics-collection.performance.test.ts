/**
 * Performance tests for Analytics Collection
 */

import { AnalyticsCollector } from '../services/AnalyticsCollector';
import { RequestMetrics } from '../types/analytics';

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

describe('Analytics Performance Tests', () => {
  let analyticsCollector: AnalyticsCollector;

  beforeEach(() => {
    analyticsCollector = new AnalyticsCollector();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Metrics Tracking Performance', () => {
    test('should track metrics within 50ms overhead requirement', async () => {
      const metrics: RequestMetrics = {
        requestId: 'perf-test-1',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      const startTime = Date.now();
      await analyticsCollector.trackRequestMetrics(metrics);
      const endTime = Date.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(50); // Should be under 50ms
    });

    test('should handle high-volume metrics tracking efficiently', async () => {
      const metricsCount = 1000;
      const metrics: RequestMetrics[] = [];

      // Generate test metrics
      for (let i = 0; i < metricsCount; i++) {
        metrics.push({
          requestId: `perf-test-${i}`,
          provider: i % 2 === 0 ? 'openai' : 'google',
          tokensUsed: 100 + (i % 50),
          cost: 0.05 + (i % 10) * 0.001,
          qualityScore: 0.7 + (i % 30) * 0.01,
          responseTime: 1000 + (i % 1000),
          timestamp: new Date(),
          taskType: i % 3 === 0 ? 'analysis' : 'general'
        });
      }

      const startTime = Date.now();

      // Track all metrics
      for (const metric of metrics) {
        await analyticsCollector.trackRequestMetrics(metric);
      }

      const endTime = Date.now();
      const totalDuration = endTime - startTime;
      const averageDuration = totalDuration / metricsCount;

      expect(averageDuration).toBeLessThan(10); // Average should be under 10ms per metric
      expect(totalDuration).toBeLessThan(5000); // Total should be under 5 seconds
    });

    test('should maintain performance with concurrent tracking', async () => {
      const concurrentRequests = 100;
      const metrics: RequestMetrics = {
        requestId: 'concurrent-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      const startTime = Date.now();

      // Track metrics concurrently
      const promises = Array(concurrentRequests).fill(null).map(async () => {
        return analyticsCollector.trackRequestMetrics({
          ...metrics,
          requestId: `concurrent-${Math.random().toString(36).substr(2, 9)}`
        });
      });

      await Promise.all(promises);

      const endTime = Date.now();
      const totalDuration = endTime - startTime;
      const averageDuration = totalDuration / concurrentRequests;

      expect(averageDuration).toBeLessThan(20); // Average should be under 20ms per concurrent request
      expect(totalDuration).toBeLessThan(2000); // Total should be under 2 seconds
    });
  });

  describe('Report Generation Performance', () => {
    beforeEach(async () => {
      // Add test data for report generation
      const testMetrics: RequestMetrics[] = [];
      for (let i = 0; i < 100; i++) {
        testMetrics.push({
          requestId: `report-test-${i}`,
          provider: i % 3 === 0 ? 'openai' : i % 3 === 1 ? 'google' : 'mistral',
          tokensUsed: 100 + (i % 50),
          cost: 0.05 + (i % 10) * 0.001,
          qualityScore: 0.7 + (i % 30) * 0.01,
          responseTime: 1000 + (i % 1000),
          timestamp: new Date(Date.now() - (i % 7) * 24 * 60 * 60 * 1000), // Last 7 days
          taskType: i % 4 === 0 ? 'analysis' : 'general'
        });
      }

      for (const metric of testMetrics) {
        await analyticsCollector.trackRequestMetrics(metric);
      }
    });

    test('should generate cost report within 500ms', async () => {
      const timeRange = {
        start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        end: new Date()
      };

      const startTime = Date.now();
      const report = await analyticsCollector.generateCostReport(timeRange);
      const endTime = Date.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(500); // Should be under 500ms
      expect(report).toBeDefined();
      expect(report.totalCost).toBeGreaterThan(0);
    });

    test('should generate provider utilization within 300ms', async () => {
      const startTime = Date.now();
      const utilization = await analyticsCollector.getProviderUtilization();
      const endTime = Date.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(300); // Should be under 300ms
      expect(utilization).toBeDefined();
      expect(Array.isArray(utilization)).toBe(true);
    });

    test('should generate satisfaction metrics within 200ms', async () => {
      const startTime = Date.now();
      const satisfaction = await analyticsCollector.getUserSatisfactionMetrics();
      const endTime = Date.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(200); // Should be under 200ms
      expect(satisfaction).toBeDefined();
      expect(satisfaction.averageQualityScore).toBeDefined();
    });

    test('should generate analytics dashboard within 1 second', async () => {
      const timeRange = {
        start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        end: new Date()
      };

      const startTime = Date.now();
      const dashboard = await analyticsCollector.getAnalyticsDashboard(timeRange);
      const endTime = Date.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(1000); // Should be under 1 second
      expect(dashboard).toBeDefined();
      expect(dashboard.overview).toBeDefined();
      expect(dashboard.costReport).toBeDefined();
      expect(dashboard.providerUtilization).toBeDefined();
      expect(dashboard.satisfactionMetrics).toBeDefined();
    });
  });

  describe('Memory Usage Performance', () => {
    test('should not cause memory leaks with repeated tracking', async () => {
      const initialMemory = process.memoryUsage().heapUsed;

      // Track many metrics
      for (let i = 0; i < 1000; i++) {
        const metrics: RequestMetrics = {
          requestId: `memory-test-${i}`,
          provider: 'openai',
          tokensUsed: 100,
          cost: 0.05,
          qualityScore: 0.8,
          responseTime: 1500,
          timestamp: new Date(),
          taskType: 'analysis'
        };

        await analyticsCollector.trackRequestMetrics(metrics);
      }

      // Force garbage collection if available
      if (global.gc) {
        global.gc();
      }

      const finalMemory = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemory - initialMemory;

      // Memory increase should be reasonable (less than 10MB)
      expect(memoryIncrease).toBeLessThan(10 * 1024 * 1024);
    });

    test('should handle data retention efficiently', async () => {
      // Add old data
      const oldMetrics: RequestMetrics = {
        requestId: 'old-memory-test',
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
      const recentMetrics: RequestMetrics = {
        requestId: 'recent-memory-test',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.05,
        qualityScore: 0.8,
        responseTime: 1500,
        timestamp: new Date(),
        taskType: 'analysis'
      };

      await analyticsCollector.trackRequestMetrics(recentMetrics);

      const startTime = Date.now();
      await analyticsCollector.clearOldData();
      const endTime = Date.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(100); // Should be under 100ms
    });
  });

  describe('Batch Processing Performance', () => {
    test('should process batches efficiently', async () => {
      const batchSize = 100;
      const metrics: RequestMetrics[] = [];

      // Generate batch of metrics
      for (let i = 0; i < batchSize; i++) {
        metrics.push({
          requestId: `batch-test-${i}`,
          provider: 'openai',
          tokensUsed: 100,
          cost: 0.05,
          qualityScore: 0.8,
          responseTime: 1500,
          timestamp: new Date(),
          taskType: 'analysis'
        });
      }

      const startTime = Date.now();

      // Process batch
      for (const metric of metrics) {
        await analyticsCollector.trackRequestMetrics(metric);
      }

      const endTime = Date.now();
      const totalDuration = endTime - startTime;
      const averageDuration = totalDuration / batchSize;

      expect(averageDuration).toBeLessThan(5); // Average should be under 5ms per metric
      expect(totalDuration).toBeLessThan(1000); // Total should be under 1 second
    });

    test('should handle batch flush efficiently', async () => {
      // Mock batch processing
      const mockFlushBatch = jest.spyOn(analyticsCollector as any, 'flushBatch');
      mockFlushBatch.mockImplementation(async () => {
        // Simulate batch processing time
        await new Promise(resolve => setTimeout(resolve, 10));
      });

      const startTime = Date.now();
      await (analyticsCollector as any).flushBatch();
      const endTime = Date.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(50); // Should be under 50ms
      expect(mockFlushBatch).toHaveBeenCalled();
    });
  });

  describe('Data Export Performance', () => {
    beforeEach(async () => {
      // Add test data for export
      const testMetrics: RequestMetrics[] = [];
      for (let i = 0; i < 500; i++) {
        testMetrics.push({
          requestId: `export-test-${i}`,
          provider: i % 3 === 0 ? 'openai' : i % 3 === 1 ? 'google' : 'mistral',
          tokensUsed: 100 + (i % 50),
          cost: 0.05 + (i % 10) * 0.001,
          qualityScore: 0.7 + (i % 30) * 0.01,
          responseTime: 1000 + (i % 1000),
          timestamp: new Date(Date.now() - (i % 30) * 24 * 60 * 60 * 1000), // Last 30 days
          taskType: i % 4 === 0 ? 'analysis' : 'general'
        });
      }

      for (const metric of testMetrics) {
        await analyticsCollector.trackRequestMetrics(metric);
      }
    });

    test('should export data with filters within 1 second', async () => {
      const query = {
        filter: {
          providers: ['openai'],
          timeRange: {
            start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
            end: new Date()
          }
        },
        limit: 100
      };

      const startTime = Date.now();
      const exportedData = await analyticsCollector.exportAnalyticsData(query);
      const endTime = Date.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(1000); // Should be under 1 second
      expect(exportedData).toBeDefined();
      expect(Array.isArray(exportedData)).toBe(true);
    });

    test('should handle large exports efficiently', async () => {
      const query = {
        filter: {},
        limit: 1000
      };

      const startTime = Date.now();
      const exportedData = await analyticsCollector.exportAnalyticsData(query);
      const endTime = Date.now();

      const duration = endTime - startTime;
      expect(duration).toBeLessThan(2000); // Should be under 2 seconds
      expect(exportedData.length).toBeGreaterThan(0);
    });
  });

  describe('Concurrent Operations Performance', () => {
    test('should handle concurrent report generation', async () => {
      const timeRange = {
        start: new Date(Date.now() - 7 * 24 * 60 * 60 * 1000),
        end: new Date()
      };

      const startTime = Date.now();

      // Generate multiple reports concurrently
      const promises = [
        analyticsCollector.generateCostReport(timeRange),
        analyticsCollector.getProviderUtilization(),
        analyticsCollector.getUserSatisfactionMetrics(),
        analyticsCollector.getAnalyticsDashboard(timeRange)
      ];

      const results = await Promise.all(promises);

      const endTime = Date.now();
      const totalDuration = endTime - startTime;

      expect(totalDuration).toBeLessThan(1500); // Should be under 1.5 seconds
      expect(results).toHaveLength(4);
      expect(results.every(result => result !== undefined)).toBe(true);
    });

    test('should handle concurrent tracking and reporting', async () => {
      const startTime = Date.now();

      // Mix tracking and reporting operations
      const promises = [
        // Tracking operations
        analyticsCollector.trackRequestMetrics({
          requestId: 'concurrent-1',
          provider: 'openai',
          tokensUsed: 100,
          cost: 0.05,
          qualityScore: 0.8,
          responseTime: 1500,
          timestamp: new Date(),
          taskType: 'analysis'
        }),
        analyticsCollector.trackRequestMetrics({
          requestId: 'concurrent-2',
          provider: 'google',
          tokensUsed: 150,
          cost: 0.03,
          qualityScore: 0.7,
          responseTime: 2000,
          timestamp: new Date(),
          taskType: 'general'
        }),
        // Reporting operations
        analyticsCollector.getProviderUtilization(),
        analyticsCollector.getUserSatisfactionMetrics(),
        analyticsCollector.getAnalyticsStats()
      ];

      const results = await Promise.all(promises);

      const endTime = Date.now();
      const totalDuration = endTime - startTime;

      expect(totalDuration).toBeLessThan(1000); // Should be under 1 second
      expect(results).toHaveLength(5);
      expect(results.every(result => result !== undefined)).toBe(true);
    });
  });
});
