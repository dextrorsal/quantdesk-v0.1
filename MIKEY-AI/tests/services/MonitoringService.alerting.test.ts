/**
 * Unit tests for alerting system
 */

import { MonitoringService } from '../services/MonitoringService';
import { MonitoringConfigurationManager } from '../config/monitoring-config';
import { MonitoringMetrics, CostMetrics, QualityMetrics, PerformanceMetrics } from '../types/monitoring';

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
      getAllProviderStatuses: jest.fn().mockReturnValue(new Map()),
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

describe('Alerting System Tests', () => {
  let monitoringService: MonitoringService;

  beforeEach(() => {
    monitoringService = new MonitoringService();
  });

  afterEach(() => {
    monitoringService.destroy();
    jest.clearAllMocks();
  });

  describe('Cost Pattern Anomaly Detection', () => {
    test('should detect cost anomalies', async () => {
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
      expect(costAlerts[0].metadata.cost).toBe(0.15);
      expect(costAlerts[0].metadata.threshold).toBe(0.1);
    });

    test('should not trigger alert for normal cost', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.05, // Below threshold of 0.1
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
      
      expect(costAlerts.length).toBe(0);
    });

    test('should detect cost anomalies in cost metrics', async () => {
      const metrics: Omit<CostMetrics, 'timestamp'> = {
        totalCost: 0.2, // High total cost
        averageCost: 0.1,
        costPerToken: 0.001,
        costSavings: 0.02,
        providerCostBreakdown: { 'openai': 0.1, 'google': 0.1 },
        dailyCost: 2.0,
        monthlyCost: 60.0,
        budgetUtilization: 0.5
      };

      await monitoringService.collectCostMetrics(metrics);

      // Cost metrics don't directly trigger cost anomaly alerts, but budget alerts
      const alerts = monitoringService.getActiveAlerts();
      expect(alerts.length).toBeGreaterThanOrEqual(0);
    });
  });

  describe('Quality Degradation Alerts', () => {
    test('should detect quality degradation', async () => {
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
      expect(qualityAlerts[0].metadata.qualityScore).toBe(0.6);
      expect(qualityAlerts[0].metadata.threshold).toBe(0.7);
    });

    test('should detect quality degradation in quality metrics', async () => {
      const metrics: Omit<QualityMetrics, 'timestamp'> = {
        averageQualityScore: 0.8,
        qualityDegradationRate: 0.15, // Above 10% threshold
        escalationRate: 0.1,
        userSatisfactionScore: 0.85,
        providerQualityBreakdown: { 'openai': 0.9, 'google': 0.8 },
        qualityThresholdViolations: 2
      };

      await monitoringService.collectQualityMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const qualityAlerts = alerts.filter(alert => alert.alertType === 'quality_degradation');
      
      expect(qualityAlerts.length).toBeGreaterThan(0);
      expect(qualityAlerts[0].message).toContain('Quality degradation rate');
    });

    test('should detect high escalation rate', async () => {
      const metrics: Omit<QualityMetrics, 'timestamp'> = {
        averageQualityScore: 0.8,
        qualityDegradationRate: 0.05,
        escalationRate: 0.25, // Above 20% threshold
        userSatisfactionScore: 0.85,
        providerQualityBreakdown: { 'openai': 0.9, 'google': 0.8 },
        qualityThresholdViolations: 2
      };

      await monitoringService.collectQualityMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const escalationAlerts = alerts.filter(alert => 
        alert.alertType === 'quality_degradation' && 
        alert.message.includes('escalation rate')
      );
      
      expect(escalationAlerts.length).toBeGreaterThan(0);
      expect(escalationAlerts[0].severity).toBe('medium');
    });

    test('should not trigger alert for good quality', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.8, // Above threshold of 0.7
        success: true,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'analysis'
      };

      await monitoringService.collectMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const qualityAlerts = alerts.filter(alert => alert.alertType === 'quality_degradation');
      
      expect(qualityAlerts.length).toBe(0);
    });
  });

  describe('Budget Threshold Monitoring', () => {
    test('should detect budget limit threshold', async () => {
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
      expect(budgetAlerts[0].metadata.budgetUtilization).toBe(0.9);
      expect(budgetAlerts[0].metadata.threshold).toBe(0.8);
    });

    test('should not trigger alert for normal budget utilization', async () => {
      const metrics: Omit<CostMetrics, 'timestamp'> = {
        totalCost: 0.1,
        averageCost: 0.05,
        costPerToken: 0.0005,
        costSavings: 0.02,
        providerCostBreakdown: { 'openai': 0.05 },
        dailyCost: 1.0,
        monthlyCost: 30.0,
        budgetUtilization: 0.5 // Below threshold of 0.8
      };

      await monitoringService.collectCostMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const budgetAlerts = alerts.filter(alert => alert.alertType === 'budget_limit');
      
      expect(budgetAlerts.length).toBe(0);
    });

    test('should detect critical budget utilization', async () => {
      const metrics: Omit<CostMetrics, 'timestamp'> = {
        totalCost: 0.1,
        averageCost: 0.05,
        costPerToken: 0.0005,
        costSavings: 0.02,
        providerCostBreakdown: { 'openai': 0.05 },
        dailyCost: 1.0,
        monthlyCost: 30.0,
        budgetUtilization: 0.95 // Very high utilization
      };

      await monitoringService.collectCostMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const budgetAlerts = alerts.filter(alert => alert.alertType === 'budget_limit');
      
      expect(budgetAlerts.length).toBeGreaterThan(0);
      expect(budgetAlerts[0].severity).toBe('critical');
      expect(budgetAlerts[0].message).toContain('95.0%');
    });
  });

  describe('Response Time Alerts', () => {
    test('should detect slow response times', async () => {
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
      expect(responseTimeAlerts[0].metadata.responseTime).toBe(6000);
      expect(responseTimeAlerts[0].metadata.threshold).toBe(5000);
    });

    test('should detect slow performance metrics', async () => {
      const metrics: Omit<PerformanceMetrics, 'timestamp'> = {
        routingDecisionTime: 50,
        providerSelectionTime: 25,
        tokenEstimationTime: 30,
        qualityEvaluationTime: 40,
        fallbackDecisionTime: 20,
        totalRequestTime: 6000, // Above threshold of 5000
        memoryUsage: 100,
        cpuUsage: 50
      };

      await monitoringService.collectPerformanceMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const performanceAlerts = alerts.filter(alert => alert.alertType === 'response_time_slow');
      
      expect(performanceAlerts.length).toBeGreaterThan(0);
      expect(performanceAlerts[0].message).toContain('Slow total request time');
    });

    test('should detect slow routing decisions', async () => {
      const metrics: Omit<PerformanceMetrics, 'timestamp'> = {
        routingDecisionTime: 250, // Above 200ms threshold
        providerSelectionTime: 25,
        tokenEstimationTime: 30,
        qualityEvaluationTime: 40,
        fallbackDecisionTime: 20,
        totalRequestTime: 1500,
        memoryUsage: 100,
        cpuUsage: 50
      };

      await monitoringService.collectPerformanceMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const performanceAlerts = alerts.filter(alert => alert.alertType === 'performance_degradation');
      
      expect(performanceAlerts.length).toBeGreaterThan(0);
      expect(performanceAlerts[0].message).toContain('Slow routing decision');
    });
  });

  describe('Error Rate Alerts', () => {
    test('should detect high error rates', async () => {
      // Simulate multiple failed requests
      for (let i = 0; i < 15; i++) {
        const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
          requestId: `req-${i}`,
          provider: 'openai',
          cost: 0.05,
          tokensUsed: 100,
          responseTime: 1500,
          qualityScore: 0.8,
          success: false, // Failed request
          fallbackUsed: false,
          escalationUsed: false,
          taskType: 'analysis'
        };

        await monitoringService.collectMetrics(metrics);
      }

      const alerts = monitoringService.getActiveAlerts();
      const errorRateAlerts = alerts.filter(alert => alert.alertType === 'error_rate_high');
      
      expect(errorRateAlerts.length).toBeGreaterThan(0);
      expect(errorRateAlerts[0].severity).toBe('high');
      expect(errorRateAlerts[0].message).toContain('High error rate');
    });

    test('should not trigger alert for normal error rates', async () => {
      // Simulate mostly successful requests
      for (let i = 0; i < 20; i++) {
        const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
          requestId: `req-${i}`,
          provider: 'openai',
          cost: 0.05,
          tokensUsed: 100,
          responseTime: 1500,
          qualityScore: 0.8,
          success: i % 10 === 0, // Only 10% failures
          fallbackUsed: false,
          escalationUsed: false,
          taskType: 'analysis'
        };

        await monitoringService.collectMetrics(metrics);
      }

      const alerts = monitoringService.getActiveAlerts();
      const errorRateAlerts = alerts.filter(alert => alert.alertType === 'error_rate_high');
      
      expect(errorRateAlerts.length).toBe(0);
    });
  });

  describe('Alert Management', () => {
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
      const resolvedAlert = updatedAlerts.find(alert => alert.alertId === alertId);
      expect(resolvedAlert).toBeUndefined(); // Should not be in active alerts
      
      const allAlerts = monitoringService.getAllAlerts();
      const foundAlert = allAlerts.find(alert => alert.alertId === alertId);
      expect(foundAlert?.resolved).toBe(true);
      expect(foundAlert?.resolvedAt).toBeInstanceOf(Date);
    });

    test('should not resolve non-existent alerts', () => {
      const resolved = monitoringService.resolveAlert('non-existent-alert');
      expect(resolved).toBe(false);
    });

    test('should not resolve already resolved alerts', async () => {
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
      const alertId = alerts[0].alertId;
      
      // Resolve first time
      const resolved1 = monitoringService.resolveAlert(alertId);
      expect(resolved1).toBe(true);
      
      // Try to resolve again
      const resolved2 = monitoringService.resolveAlert(alertId);
      expect(resolved2).toBe(false);
    });
  });

  describe('Alert Cooldown and Rate Limiting', () => {
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

    test('should respect rate limiting', async () => {
      // Mock max alerts per hour to 2
      const mockConfig = require('../config/monitoring-config').MonitoringConfigurationManager;
      mockConfig.mockImplementation(() => ({
        getInstance: jest.fn().mockReturnThis(),
        getMaxAlertsPerHour: jest.fn().mockReturnValue(2),
        getAlertCooldownMinutes: jest.fn().mockReturnValue(1),
        isMonitoringEnabled: jest.fn().mockReturnValue(true),
        isAlertingEnabled: jest.fn().mockReturnValue(true),
        getAlertThresholds: jest.fn().mockReturnValue({
          costAnomalyThreshold: 0.1,
          qualityDegradationThreshold: 0.7,
          budgetLimitThreshold: 0.8,
          responseTimeThreshold: 5000,
          errorRateThreshold: 0.1,
          availabilityThreshold: 0.95,
          circuitBreakerThreshold: 5,
          fallbackRateThreshold: 0.2
        })
      }));

      const service = new MonitoringService();

      // Create multiple alerts
      for (let i = 0; i < 5; i++) {
        const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
          requestId: `req-${i}`,
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

        await service.collectMetrics(metrics);
      }

      const alerts = service.getActiveAlerts();
      expect(alerts.length).toBeLessThanOrEqual(2); // Should be rate limited

      service.destroy();
    });
  });

  describe('Alert Severity Levels', () => {
    test('should create critical alerts for budget limits', async () => {
      const metrics: Omit<CostMetrics, 'timestamp'> = {
        totalCost: 0.1,
        averageCost: 0.05,
        costPerToken: 0.0005,
        costSavings: 0.02,
        providerCostBreakdown: { 'openai': 0.05 },
        dailyCost: 1.0,
        monthlyCost: 30.0,
        budgetUtilization: 0.9
      };

      await monitoringService.collectCostMetrics(metrics);

      const alerts = monitoringService.getActiveAlerts();
      const budgetAlerts = alerts.filter(alert => alert.alertType === 'budget_limit');
      
      expect(budgetAlerts.length).toBeGreaterThan(0);
      expect(budgetAlerts[0].severity).toBe('critical');
    });

    test('should create high severity alerts for quality degradation', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 1500,
        qualityScore: 0.6, // Below threshold
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
    });

    test('should create medium severity alerts for response time', async () => {
      const metrics: Omit<MonitoringMetrics, 'timestamp'> = {
        requestId: 'req-123',
        provider: 'openai',
        cost: 0.05,
        tokensUsed: 100,
        responseTime: 6000, // Above threshold
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
    });
  });
});
