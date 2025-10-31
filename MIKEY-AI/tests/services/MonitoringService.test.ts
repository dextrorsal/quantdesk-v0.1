/**
 * Unit tests for MonitoringService
 */

import { MonitoringService } from '../services/MonitoringService';
import { MonitoringMetrics, PerformanceMetrics, CostMetrics, QualityMetrics } from '../types/monitoring';

// Mock dependencies
jest.mock('../config/monitoring-config', () => {
  return {
    MonitoringConfigurationManager: jest.fn().mockImplementation(() => ({
      getInstance: jest.fn().mockReturnThis(),
      getMonitoringConfig: jest.fn().mockReturnValue({
        metricsInterval: 5000,
        alertThresholds: {
          costAnomalyThreshold: 0.1,
          qualityDegradationThreshold: 0.7,
          budgetLimitThreshold: 0.8,
          responseTimeThreshold: 5000,
          errorRateThreshold: 0.1,
          availabilityThreshold: 0.95,
          circuitBreakerThreshold: 5,
          fallbackRateThreshold: 0.2
        },
        retentionPeriod: 30,
        enabledMetrics: ['cost', 'quality', 'performance', 'availability'],
        enableRealTimeMonitoring: true,
        enablePerformanceTracking: true,
        enableCostTracking: true,
        enableQualityTracking: true
      }),
      getMonitoringConfiguration: jest.fn().mockReturnValue({
        enableMonitoring: true,
        enableAlerting: true,
        enablePerformanceTracking: true,
        enableCostTracking: true,
        enableQualityTracking: true,
        metricsCollectionInterval: 5000,
        alertCheckInterval: 10000,
        dataRetentionDays: 30,
        maxAlertsPerHour: 100,
        alertCooldownMinutes: 15
      }),
      getAlertThresholds: jest.fn().mockReturnValue({
        costAnomalyThreshold: 0.1,
        qualityDegradationThreshold: 0.7,
        budgetLimitThreshold: 0.8,
        responseTimeThreshold: 5000,
        errorRateThreshold: 0.1,
        availabilityThreshold: 0.95,
        circuitBreakerThreshold: 5,
        fallbackRateThreshold: 0.2
      }),
      isMonitoringEnabled: jest.fn().mockReturnValue(true),
      isAlertingEnabled: jest.fn().mockReturnValue(true),
      isPerformanceTrackingEnabled: jest.fn().mockReturnValue(true),
      isCostTrackingEnabled: jest.fn().mockReturnValue(true),
      isQualityTrackingEnabled: jest.fn().mockReturnValue(true),
      getMetricsCollectionInterval: jest.fn().mockReturnValue(5000),
      getAlertCheckInterval: jest.fn().mockReturnValue(10000),
      getDataRetentionDays: jest.fn().mockReturnValue(30),
      getMaxAlertsPerHour: jest.fn().mockReturnValue(100),
      getAlertCooldownMinutes: jest.fn().mockReturnValue(15)
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

jest.mock('../services/ProviderHealthMonitor', () => {
  return {
    ProviderHealthMonitor: jest.fn().mockImplementation(() => ({
      getProviderHealthSummary: jest.fn().mockReturnValue({
        totalProviders: 4,
        healthyProviders: 3,
        unhealthyProviders: 1,
        circuitBreakerOpen: 0,
        averageAvailability: 0.95,
        overallHealthScore: 0.9,
        lastUpdated: new Date()
      }),
      getAllProviderStatuses: jest.fn().mockReturnValue(new Map([
        ['openai', { provider: 'openai', isHealthy: true, circuitBreakerState: 'CLOSED', lastHealthCheck: new Date() }],
        ['google', { provider: 'google', isHealthy: true, circuitBreakerState: 'CLOSED', lastHealthCheck: new Date() }],
        ['mistral', { provider: 'mistral', isHealthy: true, circuitBreakerState: 'CLOSED', lastHealthCheck: new Date() }],
        ['cohere', { provider: 'cohere', isHealthy: false, circuitBreakerState: 'OPEN', lastHealthCheck: new Date() }]
      ])),
      getProviderHealthMetrics: jest.fn().mockReturnValue({
        provider: 'openai',
        availability: 0.95,
        averageResponseTime: 1500,
        errorRate: 0.05,
        successRate: 0.95,
        circuitBreakerState: 'CLOSED',
        lastUpdated: new Date()
      }),
      destroy: jest.fn()
    }))
  };
});

jest.mock('../services/AnalyticsCollector', () => {
  return {
    AnalyticsCollector: jest.fn().mockImplementation(() => ({
      trackRequestMetrics: jest.fn().mockResolvedValue(undefined),
      getAnalyticsStats: jest.fn().mockReturnValue({
        totalRequests: 0,
        totalCost: 0,
        averageQualityScore: 0,
        dataPointsCollected: 0,
        lastUpdated: new Date()
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

describe('MonitoringService', () => {
  let monitoringService: MonitoringService;

  beforeEach(() => {
    monitoringService = new MonitoringService();
  });

  afterEach(() => {
    monitoringService.destroy();
    jest.clearAllMocks();
  });

  describe('Metrics Collection', () => {
    test('should collect monitoring metrics', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis',
        sessionId: 'session-1'
      };

      await monitoringService.collectMetrics(metrics);

      const recentMetrics = monitoringService.getRecentMetrics(1);
      expect(recentMetrics).toHaveLength(1);
      expect(recentMetrics[0].requestId).toBe('req-123');
      expect(recentMetrics[0].provider).toBe('openai');
      expect(recentMetrics[0].cost).toBe(0.05);
      expect(recentMetrics[0].timestamp).toBeInstanceOf(Date);
    });

    test('should collect performance metrics', async () => {
      const metrics: Omit<PerformanceMetrics, 'timestamp'> = {
        routingDecisionTime: 50,
        providerSelectionTime: 25,
        tokenEstimationTime: 30,
        qualityEvaluationTime: 40,
        fallbackDecisionTime: 20,
        totalRequestTime: 1500,
        memoryUsage: 100,
        cpuUsage: 50
      };

      await monitoringService.collectPerformanceMetrics(metrics);

      const recentMetrics = monitoringService.getRecentPerformanceMetrics(1);
      expect(recentMetrics).toHaveLength(1);
      expect(recentMetrics[0].routingDecisionTime).toBe(50);
      expect(recentMetrics[0].totalRequestTime).toBe(1500);
      expect(recentMetrics[0].timestamp).toBeInstanceOf(Date);
    });

    test('should collect cost metrics', async () => {
      const metrics: Omit<CostMetrics, 'timestamp'> = {
        totalCost: 0.1,
        averageCost: 0.05,
        costPerToken: 0.0005,
        costSavings: 0.02,
        providerCostBreakdown: { 'openai': 0.05, 'google': 0.03 },
        dailyCost: 1.0,
        monthlyCost: 30.0,
        budgetUtilization: 0.5
      };

      await monitoringService.collectCostMetrics(metrics);

      const recentMetrics = monitoringService.getRecentCostMetrics(1);
      expect(recentMetrics).toHaveLength(1);
      expect(recentMetrics[0].totalCost).toBe(0.1);
      expect(recentMetrics[0].budgetUtilization).toBe(0.5);
      expect(recentMetrics[0].timestamp).toBeInstanceOf(Date);
    });

    test('should collect quality metrics', async () => {
      const metrics: Omit<QualityMetrics, 'timestamp'> = {
        averageQualityScore: 0.8,
        qualityDegradationRate: 0.05,
        escalationRate: 0.1,
        userSatisfactionScore: 0.85,
        providerQualityBreakdown: { 'openai': 0.9, 'google': 0.8 },
        qualityThresholdViolations: 2
      };

      await monitoringService.collectQualityMetrics(metrics);

      const recentMetrics = monitoringService.getRecentQualityMetrics(1);
      expect(recentMetrics).toHaveLength(1);
      expect(recentMetrics[0].averageQualityScore).toBe(0.8);
      expect(recentMetrics[0].escalationRate).toBe(0.1);
      expect(recentMetrics[0].timestamp).toBeInstanceOf(Date);
    });

    test('should handle metrics collection errors gracefully', async () => {
      // Mock error in metrics collection
      jest.spyOn(monitoringService as any, 'trimMetricsHistory').mockImplementation(() => {
        throw new Error('Trim error');
      });

      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      };

      // Should not throw error
      await expect(monitoringService.collectMetrics(metrics)).resolves.not.toThrow();
    });
  });

  describe('Alert System', () => {
    test('should create cost anomaly alert', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.15, // Above threshold of 0.1
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      };

      await monitoringService.collectMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const costAlerts = alerts.filter(alert => alert.alertType === 'cost_anomaly');
      
      expect(costAlerts.length).toBeGreaterThan(0);
      expect(costAlerts[0].severity).toBe('high');
      expect(costAlerts[0].message).toContain('Cost anomaly detected');
    });

    test('should create quality degradation alert', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.6, // Below threshold of 0.7
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      };

      await monitoringService.collectMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const qualityAlerts = alerts.filter(alert => alert.alertType === 'quality_degradation');
      
      expect(qualityAlerts.length).toBeGreaterThan(0);
      expect(qualityAlerts[0].severity).toBe('high');
      expect(qualityAlerts[0].message).toContain('Quality degradation detected');
    });

    test('should create response time alert', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 6000, // Above threshold of 5000
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      };

      await monitoringService.collectMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const responseTimeAlerts = alerts.filter(alert => alert.alertType === 'response_time_slow');
      
      expect(responseTimeAlerts.length).toBeGreaterThan(0);
      expect(responseTimeAlerts[0].severity).toBe('medium');
      expect(responseTimeAlerts[0].message).toContain('Slow response time');
    });

    test('should create budget limit alert', async () => {
      const metrics: Omit<CostMetrics, 'timestamp'> = {
        totalCost: 0.1,
        averageCost: 0.05,
        costPerToken: 0.0005,
        costSavings: 0.02,
        providerCostBreakdown: { 'openai': 0.05 },
        dailyCost: 1.0,
        monthlyCost: 30.0,
        budgetUtilization: 0.9 // Above threshold of 0.8
      };

      await monitoringService.collectCostMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const budgetAlerts = alerts.filter(alert => alert.alertType === 'budget_limit');
      
      expect(budgetAlerts.length).toBeGreaterThan(0);
      expect(budgetAlerts[0].severity).toBe('critical');
      expect(budgetAlerts[0].message).toContain('Budget limit reached');
    });

    test('should resolve alerts', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.15, // Above threshold
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      };

      await monitoringService.collectMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      expect(alerts.length).toBeGreaterThan(0);

      const alertId = alerts[0].alertId;
      const resolved = monitoringService.resolveAlert(alertId);
      
      expect(resolved).toBe(true);
      
      const updatedAlerts = monitoringService.getActiveAlerts();
      expect(updatedAlerts.find(alert => alert.alertId === alertId)?.resolved).toBe(true);
    });

    test('should respect alert cooldown', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.15, // Above threshold
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      };

      // First alert
      await monitoringService.collectMetrics(metrics);
      const firstAlerts = monitoringService.getActiveAlerts();
      const firstAlertCount = firstAlerts.length;

      // Second alert within cooldown
      await monitoringService.collectMetrics({ ...metrics, requestId: 'req-124' });
      const secondAlerts = monitoringService.getActiveAlerts();
      
      // Should not create duplicate alert due to cooldown
      expect(secondAlerts.length).toBe(firstAlertCount);
    });
  });

  describe('Monitoring Dashboard', () => {
    test('should generate monitoring dashboard', () => {
      const dashboard = monitoringService.getMonitoringDashboard();

      expect(dashboard).toBeDefined();
      expect(dashboard.status).toBeDefined();
      expect(dashboard.recentMetrics).toBeDefined();
      expect(dashboard.activeAlerts).toBeDefined();
      expect(dashboard.performanceTrends).toBeDefined();
      expect(dashboard.costTrends).toBeDefined();
      expect(dashboard.qualityTrends).toBeDefined();
      expect(dashboard.providerStats).toBeDefined();
    });

    test('should provide monitoring status', () => {
      const status = monitoringService.getMonitoringStatus();

      expect(status).toBeDefined();
      expect(typeof status.isHealthy).toBe('boolean');
      expect(typeof status.activeAlerts).toBe('number');
      expect(typeof status.criticalAlerts).toBe('number');
      expect(typeof status.metricsCollected).toBe('number');
      expect(typeof status.performanceScore).toBe('number');
      expect(typeof status.uptime).toBe('number');
      expect(typeof status.systemLoad).toBe('number');
      expect(status.lastUpdate).toBeInstanceOf(Date);
    });

    test('should provide provider statistics', () => {
      const providerStats = monitoringService.getProviderStats();

      expect(providerStats).toBeDefined();
      expect(typeof providerStats).toBe('object');
    });

    test('should provide monitoring statistics', () => {
      const stats = monitoringService.getMonitoringStats();

      expect(stats).toBeDefined();
      expect(typeof stats.totalMetricsCollected).toBe('number');
      expect(typeof stats.totalAlertsGenerated).toBe('number');
      expect(typeof stats.totalAlertsResolved).toBe('number');
      expect(typeof stats.averageResponseTime).toBe('number');
      expect(typeof stats.averageCost).toBe('number');
      expect(typeof stats.averageQualityScore).toBe('number');
      expect(typeof stats.systemUptime).toBe('number');
      expect(typeof stats.activeProviders).toBe('number');
      expect(typeof stats.healthyProviders).toBe('number');
      expect(stats.lastHealthCheck).toBeInstanceOf(Date);
    });
  });

  describe('Monitoring Control', () => {
    test('should start monitoring', () => {
      const service = new MonitoringService();
      
      service.startMonitoring();
      
      // Should not throw error
      expect(service).toBeDefined();
      
      service.destroy();
    });

    test('should stop monitoring', () => {
      const service = new MonitoringService();
      
      service.stopMonitoring();
      
      // Should not throw error
      expect(service).toBeDefined();
      
      service.destroy();
    });

    test('should clear all data', () => {
      // Add some data first
      monitoringService.collectMetrics({
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      });

      expect(monitoringService.getRecentMetrics(1).length).toBeGreaterThan(0);

      monitoringService.clearAllData();

      expect(monitoringService.getRecentMetrics(1).length).toBe(0);
      expect(monitoringService.getActiveAlerts().length).toBe(0);
    });
  });

  describe('Data Retention', () => {
    test('should trim metrics history based on retention period', async () => {
      // Mock old timestamp
      const oldDate = new Date(Date.now() - 40 * 24 * 60 * 60 * 1000); // 40 days ago
      
      // Add old metric
      const oldMetric: MonitoringMetrics = {
        requestId: 'req-old',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis',
        timestamp: oldDate
      };

      (monitoringService as any).metricsHistory.push(oldMetric);

      // Add new metric
      await monitoringService.collectMetrics({
        requestId: 'req-new',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      });

      const recentMetrics = monitoringService.getRecentMetrics(10);
      const oldMetrics = recentMetrics.filter(m => m.requestId === 'req-old');
      
      expect(oldMetrics.length).toBe(0); // Should be trimmed
    });
  });

  describe('Error Handling', () => {
    test('should handle alert creation errors gracefully', async () => {
      // Mock error in alert creation
      jest.spyOn(monitoringService as any, 'createAlert').mockRejectedValue(new Error('Alert creation error'));

      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.15, // Above threshold
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8,
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      };

      // Should not throw error
      await expect(monitoringService.collectMetrics(metrics)).resolves.not.toThrow();
    });

    test('should handle performance metrics collection errors', async () => {
      // Mock error in performance metrics collection
      jest.spyOn(monitoringService as any, 'trimPerformanceHistory').mockImplementation(() => {
        throw new Error('Performance trim error');
      });

      const metrics: Omit<PerformanceMetrics, 'timestamp'> = {
        routingDecisionTime: 50,
        providerSelectionTime: 25,
        tokenEstimationTime: 30,
        qualityEvaluationTime: 40,
        fallbackDecisionTime: 20,
        totalRequestTime: 1500,
        memoryUsage: 100,
        cpuUsage: 50
      };

      // Should not throw error
      await expect(monitoringService.collectPerformanceMetrics(metrics)).resolves.not.toThrow();
    });
  });
});
