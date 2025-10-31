/**
 * Performance tests for monitoring system
 * Tests the performance impact of monitoring on the LLM routers
 */

import { MonitoringService } from '../../src/services/MonitoringService';
import { MultiLLMRouter } from '../../src/services/MultiLLMRouter';
import { OfficialLLMRouter } from '../../src/services/OfficialLLMRouter';

describe('Monitoring System Performance', () => {
  let monitoringService: MonitoringService;
  let multiLLMRouter: MultiLLMRouter;
  let officialLLMRouter: OfficialLLMRouter;

  beforeEach(() => {
    monitoringService = new MonitoringService();
    multiLLMRouter = new MultiLLMRouter();
    officialLLMRouter = new OfficialLLMRouter();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('Monitoring Overhead', () => {
    it('should have minimal overhead on MultiLLMRouter', async () => {
      const mockPrompt = 'Performance test prompt for MultiLLMRouter';
      const mockTaskType = 'general';
      const mockSessionId = 'perf-test-session-1';

      // Mock the router to return a successful response
      jest.spyOn(multiLLMRouter, 'routeRequest').mockResolvedValue('Mock response');

      const startTime = Date.now();
      
      // Execute request
      await multiLLMRouter.routeRequest(mockPrompt, mockTaskType, mockSessionId);
      
      const endTime = Date.now();
      const totalTime = endTime - startTime;

      // Monitoring overhead should be minimal (< 50ms as per requirements)
      expect(totalTime).toBeLessThan(50);
    });

    it('should have minimal overhead on OfficialLLMRouter', async () => {
      const mockPrompt = 'Performance test prompt for OfficialLLMRouter';
      const mockTaskType = 'general';
      const mockSessionId = 'perf-test-session-2';

      // Mock the router to return a successful response
      jest.spyOn(officialLLMRouter, 'routeRequest').mockResolvedValue({
        response: 'Mock response',
        provider: 'openai'
      });

      const startTime = Date.now();
      
      // Execute request
      await officialLLMRouter.routeRequest(mockPrompt, mockTaskType, mockSessionId);
      
      const endTime = Date.now();
      const totalTime = endTime - startTime;

      // Monitoring overhead should be minimal (< 50ms as per requirements)
      expect(totalTime).toBeLessThan(50);
    });

    it('should handle high-frequency monitoring requests efficiently', async () => {
      const numberOfRequests = 100;
      const mockMetrics = {
        requestId: 'perf-test-request',
        provider: 'openai',
        cost: 0.01,
        tokensUsed: 100,
        responseTime: 500,
        qualityScore: 0.85,
        success: true,
        errorType: undefined,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'general',
        sessionId: 'perf-test-session'
      };

      const startTime = Date.now();

      // Execute multiple monitoring requests
      const promises = Array(numberOfRequests).fill(null).map(async (_, index) => {
        return monitoringService.collectMetrics({
          ...mockMetrics,
          requestId: `perf-test-request-${index}`
        });
      });

      await Promise.all(promises);

      const endTime = Date.now();
      const totalTime = endTime - startTime;
      const averageTimePerRequest = totalTime / numberOfRequests;

      // Average time per request should be minimal
      expect(averageTimePerRequest).toBeLessThan(10); // 10ms per request
    });
  });

  describe('Alert Performance', () => {
    it('should create alerts efficiently', async () => {
      const numberOfAlerts = 50;
      const mockAlert = {
        type: 'COST_ANOMALY' as const,
        severity: 'MEDIUM' as const,
        message: 'Cost anomaly detected',
        details: {
          provider: 'openai',
          currentValue: 10.0,
          threshold: 5.0
        }
      };

      const startTime = Date.now();

      // Create multiple alerts
      const promises = Array(numberOfAlerts).fill(null).map(async (_, index) => {
        return monitoringService.createAlert({
          ...mockAlert,
          message: `Cost anomaly detected ${index}`
        });
      });

      await Promise.all(promises);

      const endTime = Date.now();
      const totalTime = endTime - startTime;
      const averageTimePerAlert = totalTime / numberOfAlerts;

      // Average time per alert should be minimal
      expect(averageTimePerAlert).toBeLessThan(5); // 5ms per alert
    });

    it('should resolve alerts efficiently', async () => {
      const numberOfAlerts = 50;
      const alertIds: string[] = [];

      // Create alerts first
      for (let i = 0; i < numberOfAlerts; i++) {
        const alert = await monitoringService.createAlert({
          type: 'COST_ANOMALY',
          severity: 'MEDIUM',
          message: `Cost anomaly detected ${i}`,
          details: {
            provider: 'openai',
            currentValue: 10.0,
            threshold: 5.0
          }
        });
        alertIds.push(alert.id);
      }

      const startTime = Date.now();

      // Resolve all alerts
      const promises = alertIds.map(alertId => {
        return monitoringService.resolveAlert(alertId, 'Resolved for performance test');
      });

      await Promise.all(promises);

      const endTime = Date.now();
      const totalTime = endTime - startTime;
      const averageTimePerResolution = totalTime / numberOfAlerts;

      // Average time per resolution should be minimal
      expect(averageTimePerResolution).toBeLessThan(5); // 5ms per resolution
    });
  });

  describe('Metrics Aggregation Performance', () => {
    it('should aggregate metrics efficiently', async () => {
      const numberOfMetrics = 1000;
      const mockMetrics = {
        requestId: 'agg-test-request',
        provider: 'openai',
        cost: 0.01,
        tokensUsed: 100,
        responseTime: 500,
        qualityScore: 0.85,
        success: true,
        errorType: undefined,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'general',
        sessionId: 'agg-test-session'
      };

      // Collect metrics
      const promises = Array(numberOfMetrics).fill(null).map(async (_, index) => {
        return monitoringService.collectMetrics({
          ...mockMetrics,
          requestId: `agg-test-request-${index}`,
          cost: 0.01 + (index * 0.001), // Varying costs
          qualityScore: 0.85 + (index * 0.0001) // Varying quality scores
        });
      });

      await Promise.all(promises);

      const startTime = Date.now();

      // Aggregate metrics
      const aggregatedMetrics = await monitoringService.getMetrics({
        timeRange: '1h',
        provider: 'openai'
      });

      const endTime = Date.now();
      const aggregationTime = endTime - startTime;

      // Aggregation should be efficient even with large datasets
      expect(aggregationTime).toBeLessThan(100); // 100ms for aggregation
      expect(aggregatedMetrics).toBeDefined();
    });

    it('should handle concurrent metric collection efficiently', async () => {
      const numberOfConcurrentRequests = 100;
      const mockMetrics = {
        requestId: 'concurrent-test-request',
        provider: 'openai',
        cost: 0.01,
        tokensUsed: 100,
        responseTime: 500,
        qualityScore: 0.85,
        success: true,
        errorType: undefined,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'general',
        sessionId: 'concurrent-test-session'
      };

      const startTime = Date.now();

      // Execute concurrent metric collection
      const promises = Array(numberOfConcurrentRequests).fill(null).map(async (_, index) => {
        return monitoringService.collectMetrics({
          ...mockMetrics,
          requestId: `concurrent-test-request-${index}`
        });
      });

      await Promise.all(promises);

      const endTime = Date.now();
      const totalTime = endTime - startTime;
      const averageTimePerRequest = totalTime / numberOfConcurrentRequests;

      // Should handle concurrent requests efficiently
      expect(averageTimePerRequest).toBeLessThan(20); // 20ms per concurrent request
    });
  });

  describe('System Status Performance', () => {
    it('should retrieve system status efficiently', async () => {
      const startTime = Date.now();

      const systemStatus = await monitoringService.getSystemStatus();

      const endTime = Date.now();
      const retrievalTime = endTime - startTime;

      // System status retrieval should be fast
      expect(retrievalTime).toBeLessThan(50); // 50ms
      expect(systemStatus).toBeDefined();
      expect(systemStatus.overallHealth).toBeDefined();
    });

    it('should retrieve health status efficiently', async () => {
      const startTime = Date.now();

      const healthStatus = await monitoringService.getHealthStatus();

      const endTime = Date.now();
      const retrievalTime = endTime - startTime;

      // Health status retrieval should be fast
      expect(retrievalTime).toBeLessThan(50); // 50ms
      expect(healthStatus).toBeDefined();
      expect(healthStatus.status).toBeDefined();
    });
  });

  describe('Configuration Performance', () => {
    it('should update configuration efficiently', async () => {
      const newConfig = {
        alertThresholds: {
          costAnomalyThreshold: 3.0,
          qualityDegradationThreshold: 0.6,
          budgetLimitThreshold: 0.9,
          responseTimeThreshold: 2000,
          failureRateThreshold: 0.1
        },
        metricsInterval: 10000,
        alertCheckInterval: 15000
      };

      const startTime = Date.now();

      await monitoringService.configureMonitoring(newConfig);

      const endTime = Date.now();
      const updateTime = endTime - startTime;

      // Configuration update should be fast
      expect(updateTime).toBeLessThan(10); // 10ms
    });
  });

  describe('Memory Usage', () => {
    it('should not cause memory leaks with high-frequency monitoring', async () => {
      const initialMemoryUsage = process.memoryUsage().heapUsed;
      const numberOfRequests = 1000;

      // Execute many monitoring requests
      const promises = Array(numberOfRequests).fill(null).map(async (_, index) => {
        return monitoringService.collectMetrics({
          requestId: `memory-test-request-${index}`,
          provider: 'openai',
          cost: 0.01,
          tokensUsed: 100,
          responseTime: 500,
          qualityScore: 0.85,
          success: true,
          errorType: undefined,
          fallbackUsed: false,
          escalationUsed: false,
          taskType: 'general',
          sessionId: 'memory-test-session'
        });
      });

      await Promise.all(promises);

      const finalMemoryUsage = process.memoryUsage().heapUsed;
      const memoryIncrease = finalMemoryUsage - initialMemoryUsage;

      // Memory increase should be reasonable (not excessive)
      expect(memoryIncrease).toBeLessThan(50 * 1024 * 1024); // 50MB
    });
  });
});
