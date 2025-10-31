/**
 * Integration tests for monitoring system
 * Tests the integration between MonitoringService and existing systems
 */

import { MonitoringService } from '../../src/services/MonitoringService';
import { MultiLLMRouter } from '../../src/services/MultiLLMRouter';
import { OfficialLLMRouter } from '../../src/services/OfficialLLMRouter';
import { ProviderHealthMonitor } from '../../src/services/ProviderHealthMonitor';
import { AnalyticsCollector } from '../../src/services/AnalyticsCollector';

describe('Monitoring System Integration', () => {
  let monitoringService: MonitoringService;
  let multiLLMRouter: MultiLLMRouter;
  let officialLLMRouter: OfficialLLMRouter;
  let providerHealthMonitor: ProviderHealthMonitor;
  let analyticsCollector: AnalyticsCollector;

  beforeEach(() => {
    monitoringService = new MonitoringService();
    multiLLMRouter = new MultiLLMRouter();
    officialLLMRouter = new OfficialLLMRouter();
    providerHealthMonitor = new ProviderHealthMonitor();
    analyticsCollector = new AnalyticsCollector();
  });

  afterEach(() => {
    jest.clearAllMocks();
  });

  describe('MonitoringService Integration', () => {
    it('should integrate with MultiLLMRouter for metrics collection', async () => {
      const mockPrompt = 'Test prompt for monitoring integration';
      const mockTaskType = 'general';
      const mockSessionId = 'test-session-123';

      // Mock the router to return a successful response
      jest.spyOn(multiLLMRouter, 'routeRequest').mockResolvedValue('Mock response');

      // Mock monitoring service methods
      const collectMetricsSpy = jest.spyOn(monitoringService, 'collectMetrics').mockResolvedValue();
      const collectPerformanceMetricsSpy = jest.spyOn(monitoringService, 'collectPerformanceMetrics').mockResolvedValue();
      const collectCostMetricsSpy = jest.spyOn(monitoringService, 'collectCostMetrics').mockResolvedValue();
      const collectQualityMetricsSpy = jest.spyOn(monitoringService, 'collectQualityMetrics').mockResolvedValue();

      // Execute request
      await multiLLMRouter.routeRequest(mockPrompt, mockTaskType, mockSessionId);

      // Verify monitoring methods were called
      expect(collectMetricsSpy).toHaveBeenCalled();
      expect(collectPerformanceMetricsSpy).toHaveBeenCalled();
      expect(collectCostMetricsSpy).toHaveBeenCalled();
      expect(collectQualityMetricsSpy).toHaveBeenCalled();
    });

    it('should integrate with OfficialLLMRouter for metrics collection', async () => {
      const mockPrompt = 'Test prompt for official router monitoring';
      const mockTaskType = 'general';
      const mockSessionId = 'test-session-456';

      // Mock the router to return a successful response
      jest.spyOn(officialLLMRouter, 'routeRequest').mockResolvedValue({
        response: 'Mock response',
        provider: 'openai'
      });

      // Mock monitoring service methods
      const collectMetricsSpy = jest.spyOn(monitoringService, 'collectMetrics').mockResolvedValue();
      const collectPerformanceMetricsSpy = jest.spyOn(monitoringService, 'collectPerformanceMetrics').mockResolvedValue();
      const collectCostMetricsSpy = jest.spyOn(monitoringService, 'collectCostMetrics').mockResolvedValue();
      const collectQualityMetricsSpy = jest.spyOn(monitoringService, 'collectQualityMetrics').mockResolvedValue();

      // Execute request
      await officialLLMRouter.routeRequest(mockPrompt, mockTaskType, mockSessionId);

      // Verify monitoring methods were called
      expect(collectMetricsSpy).toHaveBeenCalled();
      expect(collectPerformanceMetricsSpy).toHaveBeenCalled();
      expect(collectCostMetricsSpy).toHaveBeenCalled();
      expect(collectQualityMetricsSpy).toHaveBeenCalled();
    });

    it('should integrate with ProviderHealthMonitor for health tracking', async () => {
      const mockProvider = 'openai';
      const mockIsHealthy = true;
      const mockResponseTime = 500;
      const mockErrorMessage = 'Test error';

      // Mock monitoring service methods
      const collectMetricsSpy = jest.spyOn(monitoringService, 'collectMetrics').mockResolvedValue();

      // Update provider status
      await providerHealthMonitor.updateProviderStatus(
        mockProvider,
        mockIsHealthy,
        mockResponseTime,
        mockErrorMessage
      );

      // Verify monitoring integration
      expect(collectMetricsSpy).toHaveBeenCalled();
    });

    it('should integrate with AnalyticsCollector for data correlation', async () => {
      const mockRequestMetrics = {
        requestId: 'test-request-123',
        provider: 'openai',
        tokensUsed: 100,
        cost: 0.01,
        qualityScore: 0.85,
        responseTime: 500,
        timestamp: new Date(),
        taskType: 'general',
        sessionId: 'test-session-789',
        escalationCount: 0,
        fallbackUsed: false
      };

      // Mock analytics collector
      const trackRequestMetricsSpy = jest.spyOn(analyticsCollector, 'trackRequestMetrics').mockResolvedValue();

      // Mock monitoring service
      const collectMetricsSpy = jest.spyOn(monitoringService, 'collectMetrics').mockResolvedValue();

      // Track analytics
      await analyticsCollector.trackRequestMetrics(mockRequestMetrics);

      // Collect monitoring metrics
      await monitoringService.collectMetrics({
        requestId: mockRequestMetrics.requestId,
        provider: mockRequestMetrics.provider,
        cost: mockRequestMetrics.cost,
        tokensUsed: mockRequestMetrics.tokensUsed,
        responseTime: mockRequestMetrics.responseTime,
        qualityScore: mockRequestMetrics.qualityScore,
        success: true,
        errorType: undefined,
        fallbackUsed: mockRequestMetrics.fallbackUsed,
        escalationUsed: mockRequestMetrics.escalationCount > 0,
        taskType: mockRequestMetrics.taskType,
        sessionId: mockRequestMetrics.sessionId
      });

      // Verify both services were called
      expect(trackRequestMetricsSpy).toHaveBeenCalledWith(mockRequestMetrics);
      expect(collectMetricsSpy).toHaveBeenCalled();
    });
  });

  describe('Alert Integration', () => {
    it('should trigger alerts based on cost anomalies', async () => {
      const mockMetrics = {
        requestId: 'test-request-cost-anomaly',
        provider: 'openai',
        cost: 10.0, // High cost that should trigger alert
        tokensUsed: 1000,
        responseTime: 500,
        qualityScore: 0.85,
        success: true,
        errorType: undefined,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'general',
        sessionId: 'test-session-cost'
      };

      // Mock alert creation
      const createAlertSpy = jest.spyOn(monitoringService, 'createAlert').mockResolvedValue();

      // Collect metrics that should trigger cost anomaly alert
      await monitoringService.collectMetrics(mockMetrics);

      // Verify alert was created
      expect(createAlertSpy).toHaveBeenCalled();
    });

    it('should trigger alerts based on quality degradation', async () => {
      const mockMetrics = {
        requestId: 'test-request-quality-degradation',
        provider: 'openai',
        cost: 0.01,
        tokensUsed: 100,
        responseTime: 500,
        qualityScore: 0.3, // Low quality that should trigger alert
        success: true,
        errorType: undefined,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'general',
        sessionId: 'test-session-quality'
      };

      // Mock alert creation
      const createAlertSpy = jest.spyOn(monitoringService, 'createAlert').mockResolvedValue();

      // Collect metrics that should trigger quality degradation alert
      await monitoringService.collectMetrics(mockMetrics);

      // Verify alert was created
      expect(createAlertSpy).toHaveBeenCalled();
    });

    it('should trigger alerts based on response time threshold', async () => {
      const mockMetrics = {
        requestId: 'test-request-response-time',
        provider: 'openai',
        cost: 0.01,
        tokensUsed: 100,
        responseTime: 5000, // High response time that should trigger alert
        qualityScore: 0.85,
        success: true,
        errorType: undefined,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'general',
        sessionId: 'test-session-response-time'
      };

      // Mock alert creation
      const createAlertSpy = jest.spyOn(monitoringService, 'createAlert').mockResolvedValue();

      // Collect metrics that should trigger response time alert
      await monitoringService.collectMetrics(mockMetrics);

      // Verify alert was created
      expect(createAlertSpy).toHaveBeenCalled();
    });
  });

  describe('System Status Integration', () => {
    it('should provide comprehensive system status', async () => {
      const systemStatus = await monitoringService.getSystemStatus();

      expect(systemStatus).toHaveProperty('overallHealth');
      expect(systemStatus).toHaveProperty('componentStatus');
      expect(systemStatus).toHaveProperty('activeAlerts');
      expect(systemStatus).toHaveProperty('lastUpdated');

      expect(['OPERATIONAL', 'DEGRADED', 'OUTAGE']).toContain(systemStatus.overallHealth);
      expect(systemStatus.componentStatus).toHaveProperty('MultiLLMRouter');
      expect(systemStatus.componentStatus).toHaveProperty('OfficialLLMRouter');
      expect(systemStatus.componentStatus).toHaveProperty('ProviderHealthMonitor');
      expect(systemStatus.componentStatus).toHaveProperty('AnalyticsCollector');
    });

    it('should integrate health status with monitoring', async () => {
      const healthStatus = await monitoringService.getHealthStatus();

      expect(healthStatus).toHaveProperty('status');
      expect(healthStatus).toHaveProperty('timestamp');
      expect(healthStatus).toHaveProperty('components');

      expect(['healthy', 'degraded', 'unhealthy']).toContain(healthStatus.status);
    });
  });

  describe('Configuration Integration', () => {
    it('should allow dynamic configuration updates', async () => {
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

      await monitoringService.configureMonitoring(newConfig);

      // Verify configuration was updated
      const config = monitoringService.getConfiguration();
      expect(config.alertThresholds.costAnomalyThreshold).toBe(3.0);
      expect(config.alertThresholds.qualityDegradationThreshold).toBe(0.6);
      expect(config.metricsInterval).toBe(10000);
      expect(config.alertCheckInterval).toBe(15000);
    });
  });

  describe('Error Handling Integration', () => {
    it('should handle monitoring service errors gracefully', async () => {
      // Mock monitoring service to throw error
      jest.spyOn(monitoringService, 'collectMetrics').mockRejectedValue(new Error('Monitoring service error'));

      const mockPrompt = 'Test prompt for error handling';
      const mockTaskType = 'general';

      // Should not throw error even if monitoring fails
      await expect(multiLLMRouter.routeRequest(mockPrompt, mockTaskType)).resolves.toBeDefined();
    });

    it('should handle alert creation errors gracefully', async () => {
      // Mock alert creation to throw error
      jest.spyOn(monitoringService, 'createAlert').mockRejectedValue(new Error('Alert creation error'));

      const mockMetrics = {
        requestId: 'test-request-error',
        provider: 'openai',
        cost: 10.0, // High cost that should trigger alert
        tokensUsed: 1000,
        responseTime: 500,
        qualityScore: 0.85,
        success: true,
        errorType: undefined,
        fallbackUsed: false,
        escalationUsed: false,
        taskType: 'general',
        sessionId: 'test-session-error'
      };

      // Should not throw error even if alert creation fails
      await expect(monitoringService.collectMetrics(mockMetrics)).resolves.toBeUndefined();
    });
  });
});
