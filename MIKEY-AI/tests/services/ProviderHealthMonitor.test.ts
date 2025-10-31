/**
 * Unit tests for ProviderHealthMonitor
 */

import { ProviderHealthMonitor } from '../services/ProviderHealthMonitor';
import { CircuitBreakerState } from '../types/provider-health';

// Mock dependencies
jest.mock('../config/fallback-config', () => {
  return {
    FallbackConfiguration: jest.fn().mockImplementation(() => ({
      getInstance: jest.fn().mockReturnThis(),
      getFallbackConfig: jest.fn().mockReturnValue({
        maxRetries: 3,
        retryDelay: 1000,
        circuitBreakerThreshold: 5,
        circuitBreakerTimeout: 60000,
        healthCheckInterval: 30000,
        fallbackOrder: ['openai', 'google', 'mistral', 'cohere'],
        enableCircuitBreaker: true,
        enableHealthMonitoring: true,
        retryBackoffMultiplier: 2,
        maxRetryDelay: 10000
      }),
      getCircuitBreakerConfig: jest.fn().mockReturnValue({
        failureThreshold: 5,
        timeout: 60000,
        halfOpenMaxCalls: 3,
        enableAutoRecovery: true,
        recoveryCheckInterval: 30000
      }),
      getRetryConfig: jest.fn().mockReturnValue({
        maxRetries: 3,
        baseDelay: 1000,
        maxDelay: 10000,
        backoffMultiplier: 2,
        jitter: true,
        retryableErrors: ['timeout', 'network', 'rate_limit']
      }),
      getFallbackStrategy: jest.fn().mockReturnValue({
        strategy: 'balanced',
        maxFallbackDepth: 3,
        enableCascadingFallback: true,
        qualityThreshold: 0.7,
        costThreshold: 0.1
      }),
      getHealthMonitorConfig: jest.fn().mockReturnValue({
        checkInterval: 30000,
        timeout: 5000,
        enableContinuousMonitoring: true,
        alertThresholds: {
          errorRate: 0.1,
          responseTime: 5000,
          availability: 0.95
        }
      }),
      isHealthMonitoringEnabled: jest.fn().mockReturnValue(true),
      isCircuitBreakerEnabled: jest.fn().mockReturnValue(true),
      getHealthCheckInterval: jest.fn().mockReturnValue(30000),
      getCircuitBreakerThreshold: jest.fn().mockReturnValue(5),
      getCircuitBreakerTimeout: jest.fn().mockReturnValue(60000),
      getFallbackOrder: jest.fn().mockReturnValue(['openai', 'google', 'mistral', 'cohere'])
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

describe('ProviderHealthMonitor', () => {
  let healthMonitor: ProviderHealthMonitor;

  beforeEach(() => {
    healthMonitor = new ProviderHealthMonitor();
  });

  afterEach(() => {
    healthMonitor.destroy();
    jest.clearAllMocks();
  });

  describe('Provider Health Checking', () => {
    test('should check provider health successfully', async () => {
      const result = await healthMonitor.checkProviderHealth('openai');

      expect(result).toBeDefined();
      expect(result.provider).toBe('openai');
      expect(typeof result.isHealthy).toBe('boolean');
      expect(result.responseTime).toBeGreaterThanOrEqual(0);
      expect(result.timestamp).toBeInstanceOf(Date);
    });

    test('should handle health check failures gracefully', async () => {
      // Mock health check to fail
      jest.spyOn(healthMonitor as any, 'performHealthCheck').mockResolvedValue(false);

      const result = await healthMonitor.checkProviderHealth('openai');

      expect(result.isHealthy).toBe(false);
      expect(result.error).toBeDefined();
    });

    test('should update provider status after health check', async () => {
      const provider = 'openai';
      
      await healthMonitor.checkProviderHealth(provider);
      
      const status = healthMonitor.getProviderStatus(provider);
      expect(status).toBeDefined();
      expect(status!.provider).toBe(provider);
      expect(status!.totalRequests).toBe(1);
      expect(status!.lastHealthCheck).toBeInstanceOf(Date);
    });
  });

  describe('Provider Status Management', () => {
    test('should initialize provider statuses correctly', () => {
      const statuses = healthMonitor.getAllProviderStatuses();
      
      expect(statuses.size).toBeGreaterThan(0);
      expect(statuses.has('openai')).toBe(true);
      expect(statuses.has('google')).toBe(true);
      expect(statuses.has('mistral')).toBe(true);
      expect(statuses.has('cohere')).toBe(true);
    });

    test('should update provider status correctly', async () => {
      const provider = 'openai';
      
      await healthMonitor.updateProviderStatus(provider, true, 1500);
      
      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.isHealthy).toBe(true);
      expect(status!.responseTime).toBe(1500);
      expect(status!.totalRequests).toBe(1);
      expect(status!.successCount).toBe(1);
    });

    test('should track consecutive failures', async () => {
      const provider = 'openai';
      
      // Simulate multiple failures
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Network error');
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Timeout error');
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Rate limit error');
      
      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.consecutiveFailures).toBe(3);
      expect(status!.failureCount).toBe(3);
      expect(status!.lastFailure).toBeInstanceOf(Date);
    });

    test('should reset consecutive failures on success', async () => {
      const provider = 'openai';
      
      // Simulate failures
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Error');
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Error');
      
      // Then success
      await healthMonitor.updateProviderStatus(provider, true, 1500);
      
      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.consecutiveFailures).toBe(0);
      expect(status!.isHealthy).toBe(true);
    });
  });

  describe('Circuit Breaker Pattern', () => {
    test('should open circuit breaker after threshold failures', async () => {
      const provider = 'openai';
      const threshold = 5;
      
      // Simulate failures up to threshold
      for (let i = 0; i < threshold; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }
      
      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');
    });

    test('should move to half-open state after timeout', async () => {
      const provider = 'openai';
      
      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }
      
      // Mock time passage
      const status = healthMonitor.getProviderStatus(provider);
      if (status) {
        status.lastFailure = new Date(Date.now() - 70000); // 70 seconds ago
      }
      
      // Trigger state update
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Test error');
      
      const updatedStatus = healthMonitor.getProviderStatus(provider);
      expect(updatedStatus!.circuitBreakerState).toBe('HALF_OPEN');
    });

    test('should close circuit breaker on successful request in half-open state', async () => {
      const provider = 'openai';
      
      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }
      
      // Move to half-open
      const status = healthMonitor.getProviderStatus(provider);
      if (status) {
        status.circuitBreakerState = 'HALF_OPEN';
        status.lastFailure = new Date(Date.now() - 70000);
      }
      
      // Successful request
      await healthMonitor.updateProviderStatus(provider, true, 1500);
      
      const updatedStatus = healthMonitor.getProviderStatus(provider);
      expect(updatedStatus!.circuitBreakerState).toBe('CLOSED');
      expect(updatedStatus!.consecutiveFailures).toBe(0);
    });

    test('should reopen circuit breaker on failure in half-open state', async () => {
      const provider = 'openai';
      
      // Move to half-open state
      const status = healthMonitor.getProviderStatus(provider);
      if (status) {
        status.circuitBreakerState = 'HALF_OPEN';
        status.lastFailure = new Date(Date.now() - 70000);
      }
      
      // Failed request
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Still failing');
      
      const updatedStatus = healthMonitor.getProviderStatus(provider);
      expect(updatedStatus!.circuitBreakerState).toBe('OPEN');
    });

    test('should get circuit breaker state', async () => {
      const provider = 'openai';
      
      const state = await healthMonitor.getCircuitBreakerState(provider);
      expect(state).toBe('CLOSED');
    });
  });

  describe('Healthy Providers', () => {
    test('should return healthy providers', async () => {
      // Make all providers healthy
      const providers = ['openai', 'google', 'mistral', 'cohere'];
      
      for (const provider of providers) {
        await healthMonitor.updateProviderStatus(provider, true, 1500);
      }
      
      const healthyProviders = await healthMonitor.getHealthyProviders();
      expect(healthyProviders).toHaveLength(4);
      expect(healthyProviders).toContain('openai');
      expect(healthyProviders).toContain('google');
      expect(healthyProviders).toContain('mistral');
      expect(healthyProviders).toContain('cohere');
    });

    test('should exclude unhealthy providers', async () => {
      // Make some providers unhealthy
      await healthMonitor.updateProviderStatus('openai', false, 5000, 'Error');
      await healthMonitor.updateProviderStatus('google', true, 1500);
      await healthMonitor.updateProviderStatus('mistral', true, 1500);
      await healthMonitor.updateProviderStatus('cohere', true, 1500);
      
      const healthyProviders = await healthMonitor.getHealthyProviders();
      expect(healthyProviders).toHaveLength(3);
      expect(healthyProviders).not.toContain('openai');
      expect(healthyProviders).toContain('google');
    });

    test('should exclude providers with open circuit breaker', async () => {
      const provider = 'openai';
      
      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }
      
      const healthyProviders = await healthMonitor.getHealthyProviders();
      expect(healthyProviders).not.toContain(provider);
    });
  });

  describe('Health Metrics', () => {
    test('should calculate provider health metrics', async () => {
      const provider = 'openai';
      
      // Add some requests
      await healthMonitor.updateProviderStatus(provider, true, 1500);
      await healthMonitor.updateProviderStatus(provider, true, 1200);
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Error');
      
      const metrics = healthMonitor.getProviderHealthMetrics(provider);
      
      expect(metrics).toBeDefined();
      expect(metrics!.provider).toBe(provider);
      expect(metrics!.availability).toBeCloseTo(0.67, 1); // 2/3 success rate
      expect(metrics!.averageResponseTime).toBeGreaterThan(0);
      expect(metrics!.errorRate).toBeCloseTo(0.33, 1); // 1/3 error rate
      expect(metrics!.successRate).toBeCloseTo(0.67, 1);
      expect(metrics!.circuitBreakerState).toBe('CLOSED');
    });

    test('should calculate provider health summary', () => {
      const summary = healthMonitor.getProviderHealthSummary();
      
      expect(summary).toBeDefined();
      expect(summary.totalProviders).toBeGreaterThan(0);
      expect(summary.healthyProviders).toBeGreaterThanOrEqual(0);
      expect(summary.unhealthyProviders).toBeGreaterThanOrEqual(0);
      expect(summary.circuitBreakerOpen).toBeGreaterThanOrEqual(0);
      expect(summary.averageAvailability).toBeGreaterThanOrEqual(0);
      expect(summary.overallHealthScore).toBeGreaterThanOrEqual(0);
      expect(summary.lastUpdated).toBeInstanceOf(Date);
    });
  });

  describe('Health Alerts', () => {
    test('should create health alerts on failures', async () => {
      const provider = 'openai';
      
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Network error');
      
      const alerts = healthMonitor.getHealthAlerts();
      expect(alerts.length).toBeGreaterThan(0);
      
      const alert = alerts[alerts.length - 1];
      expect(alert.provider).toBe(provider);
      expect(alert.message).toContain('Network error');
      expect(alert.resolved).toBe(false);
    });

    test('should resolve health alerts', async () => {
      const provider = 'openai';
      
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Error');
      
      const alerts = healthMonitor.getUnresolvedHealthAlerts();
      expect(alerts.length).toBeGreaterThan(0);
      
      const alert = alerts[alerts.length - 1];
      const resolved = healthMonitor.resolveHealthAlert(alert.alertId);
      
      expect(resolved).toBe(true);
      
      const updatedAlert = healthMonitor.getHealthAlerts().find(a => a.alertId === alert.alertId);
      expect(updatedAlert!.resolved).toBe(true);
      expect(updatedAlert!.resolvedAt).toBeInstanceOf(Date);
    });

    test('should get unresolved health alerts', async () => {
      const provider = 'openai';
      
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Error');
      
      const unresolvedAlerts = healthMonitor.getUnresolvedHealthAlerts();
      expect(unresolvedAlerts.length).toBeGreaterThan(0);
      expect(unresolvedAlerts.every(alert => !alert.resolved)).toBe(true);
    });
  });

  describe('Fallback Events', () => {
    test('should record fallback events', () => {
      const event = {
        originalProvider: 'openai',
        fallbackProvider: 'google',
        reason: 'Circuit breaker open',
        timestamp: new Date(),
        success: true,
        responseTime: 2000,
        retryCount: 1
      };
      
      healthMonitor.recordFallbackEvent(event);
      
      const events = healthMonitor.getFallbackEvents();
      expect(events.length).toBe(1);
      
      const recordedEvent = events[0];
      expect(recordedEvent.originalProvider).toBe('openai');
      expect(recordedEvent.fallbackProvider).toBe('google');
      expect(recordedEvent.reason).toBe('Circuit breaker open');
      expect(recordedEvent.success).toBe(true);
      expect(recordedEvent.eventId).toBeDefined();
    });

    test('should limit fallback events to 1000', () => {
      // Record more than 1000 events
      for (let i = 0; i < 1001; i++) {
        healthMonitor.recordFallbackEvent({
          originalProvider: 'openai',
          fallbackProvider: 'google',
          reason: `Test ${i}`,
          timestamp: new Date(),
          success: true,
          responseTime: 1500,
          retryCount: 0
        });
      }
      
      const events = healthMonitor.getFallbackEvents();
      expect(events.length).toBe(1000);
    });
  });

  describe('Health Monitoring', () => {
    test('should start health monitoring', () => {
      const monitor = new ProviderHealthMonitor();
      
      // Health monitoring should start automatically if enabled
      expect(monitor).toBeDefined();
      
      monitor.destroy();
    });

    test('should stop health monitoring', () => {
      const monitor = new ProviderHealthMonitor();
      
      monitor.stopHealthMonitoring();
      
      // Should not throw error
      expect(monitor).toBeDefined();
      
      monitor.destroy();
    });
  });

  describe('Provider Status Reset', () => {
    test('should reset provider status', async () => {
      const provider = 'openai';
      
      // Simulate failures
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Error');
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Error');
      
      // Reset status
      healthMonitor.resetProviderStatus(provider);
      
      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.failureCount).toBe(0);
      expect(status!.consecutiveFailures).toBe(0);
      expect(status!.lastFailure).toBeNull();
      expect(status!.circuitBreakerState).toBe('CLOSED');
      expect(status!.isHealthy).toBe(true);
    });
  });

  describe('Data Management', () => {
    test('should clear all data', () => {
      healthMonitor.clearAllData();
      
      const statuses = healthMonitor.getAllProviderStatuses();
      const alerts = healthMonitor.getHealthAlerts();
      const events = healthMonitor.getFallbackEvents();
      
      expect(statuses.size).toBeGreaterThan(0); // Should reinitialize
      expect(alerts.length).toBe(0);
      expect(events.length).toBe(0);
    });
  });

  describe('Error Handling', () => {
    test('should handle health check errors gracefully', async () => {
      // Mock health check to throw error
      jest.spyOn(healthMonitor as any, 'performHealthCheck').mockRejectedValue(new Error('Health check failed'));

      const result = await healthMonitor.checkProviderHealth('openai');

      expect(result.isHealthy).toBe(false);
      expect(result.error).toBe('Health check failed');
    });

    test('should handle invalid provider names', async () => {
      const result = await healthMonitor.checkProviderHealth('nonexistent-provider');

      expect(result).toBeDefined();
      expect(result.provider).toBe('nonexistent-provider');
      expect(result.isHealthy).toBe(false);
    });
  });
});
