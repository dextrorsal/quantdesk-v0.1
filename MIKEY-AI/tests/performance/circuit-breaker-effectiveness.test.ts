/**
 * Performance tests for circuit breaker effectiveness
 */

import { ProviderHealthMonitor } from '../services/ProviderHealthMonitor';

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

describe('Circuit Breaker Effectiveness Tests', () => {
  let healthMonitor: ProviderHealthMonitor;

  beforeEach(() => {
    healthMonitor = new ProviderHealthMonitor();
  });

  afterEach(() => {
    healthMonitor.destroy();
    jest.clearAllMocks();
  });

  describe('Circuit Breaker State Transitions', () => {
    test('should transition from CLOSED to OPEN on threshold failures', async () => {
      const provider = 'openai';
      const threshold = 5;

      // Verify initial state is CLOSED
      let status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('CLOSED');

      // Simulate failures up to threshold
      for (let i = 0; i < threshold; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      // Should transition to OPEN
      status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');
      expect(status!.consecutiveFailures).toBe(threshold);
    });

    test('should transition from OPEN to HALF_OPEN after timeout', async () => {
      const provider = 'openai';

      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      let status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');

      // Mock time passage (70 seconds ago)
      if (status) {
        status.lastFailure = new Date(Date.now() - 70000);
      }

      // Trigger state update
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Test error');

      status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('HALF_OPEN');
    });

    test('should transition from HALF_OPEN to CLOSED on success', async () => {
      const provider = 'openai';

      // Move to HALF_OPEN state
      let status = healthMonitor.getProviderStatus(provider);
      if (status) {
        status.circuitBreakerState = 'HALF_OPEN';
        status.lastFailure = new Date(Date.now() - 70000);
      }

      // Successful request
      await healthMonitor.updateProviderStatus(provider, true, 1500);

      status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('CLOSED');
      expect(status!.consecutiveFailures).toBe(0);
    });

    test('should transition from HALF_OPEN to OPEN on failure', async () => {
      const provider = 'openai';

      // Move to HALF_OPEN state
      let status = healthMonitor.getProviderStatus(provider);
      if (status) {
        status.circuitBreakerState = 'HALF_OPEN';
        status.lastFailure = new Date(Date.now() - 70000);
      }

      // Failed request
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Still failing');

      status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');
    });
  });

  describe('Circuit Breaker Effectiveness Metrics', () => {
    test('should prevent cascading failures', async () => {
      const provider = 'openai';
      const totalRequests = 100;
      let blockedRequests = 0;

      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');

      // Simulate requests when circuit breaker is open
      for (let i = 0; i < totalRequests; i++) {
        const healthyProviders = await healthMonitor.getHealthyProviders();
        if (!healthyProviders.includes(provider)) {
          blockedRequests++;
        }
      }

      // Should block most requests when circuit breaker is open
      expect(blockedRequests).toBe(totalRequests);
    });

    test('should allow recovery attempts in HALF_OPEN state', async () => {
      const provider = 'openai';

      // Move to HALF_OPEN state
      let status = healthMonitor.getProviderStatus(provider);
      if (status) {
        status.circuitBreakerState = 'HALF_OPEN';
        status.lastFailure = new Date(Date.now() - 70000);
      }

      // Should allow some requests in HALF_OPEN state
      const healthyProviders = await healthMonitor.getHealthyProviders();
      expect(healthyProviders).toContain(provider);
    });

    test('should track circuit breaker metrics accurately', async () => {
      const provider = 'openai';
      const metrics = healthMonitor.getProviderHealthMetrics(provider);

      expect(metrics).toBeDefined();
      expect(metrics!.circuitBreakerState).toBe('CLOSED');
      expect(typeof metrics!.availability).toBe('number');
      expect(typeof metrics!.errorRate).toBe('number');
      expect(typeof metrics!.successRate).toBe('number');
    });
  });

  describe('Circuit Breaker Performance', () => {
    test('should detect failures quickly', async () => {
      const provider = 'openai';
      const startTime = Date.now();

      // Simulate rapid failures
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      const endTime = Date.now();
      const detectionTime = endTime - startTime;

      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');
      expect(detectionTime).toBeLessThan(100); // <100ms detection
    });

    test('should recover efficiently', async () => {
      const provider = 'openai';

      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      const status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');

      // Simulate timeout and recovery
      if (status) {
        status.lastFailure = new Date(Date.now() - 70000);
      }

      const startTime = Date.now();
      await healthMonitor.updateProviderStatus(provider, true, 1500);
      const endTime = Date.now();

      const recoveryTime = endTime - startTime;
      expect(recoveryTime).toBeLessThan(100); // <100ms recovery
    });

    test('should handle high frequency state changes', async () => {
      const provider = 'openai';
      const stateChanges = 100;
      const startTime = Date.now();

      for (let i = 0; i < stateChanges; i++) {
        // Alternate between success and failure
        const isSuccess = i % 2 === 0;
        await healthMonitor.updateProviderStatus(provider, isSuccess, 1500, `Test ${i}`);
      }

      const endTime = Date.now();
      const totalTime = endTime - startTime;

      expect(totalTime).toBeLessThan(1000); // Should complete within 1 second
    });
  });

  describe('Circuit Breaker Resilience', () => {
    test('should handle multiple providers independently', async () => {
      const providers = ['openai', 'google', 'mistral', 'cohere'];

      // Open circuit breaker for one provider
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus('openai', false, 5000, `Error ${i}`);
      }

      // Other providers should remain healthy
      const healthyProviders = await healthMonitor.getHealthyProviders();
      expect(healthyProviders).toContain('google');
      expect(healthyProviders).toContain('mistral');
      expect(healthyProviders).toContain('cohere');
      expect(healthyProviders).not.toContain('openai');
    });

    test('should handle concurrent circuit breaker operations', async () => {
      const provider = 'openai';
      const concurrentOperations = 50;
      const promises: Promise<any>[] = [];

      // Create concurrent operations
      for (let i = 0; i < concurrentOperations; i++) {
        promises.push(
          healthMonitor.updateProviderStatus(provider, i % 2 === 0, 1500, `Concurrent ${i}`)
        );
      }

      await Promise.allSettled(promises);

      // Should handle concurrent operations without issues
      const status = healthMonitor.getProviderStatus(provider);
      expect(status).toBeDefined();
    });

    test('should maintain circuit breaker state consistency', async () => {
      const provider = 'openai';

      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      // Check state consistency
      const status1 = healthMonitor.getProviderStatus(provider);
      const status2 = healthMonitor.getProviderStatus(provider);
      const metrics = healthMonitor.getProviderHealthMetrics(provider);

      expect(status1!.circuitBreakerState).toBe(status2!.circuitBreakerState);
      expect(status1!.circuitBreakerState).toBe(metrics!.circuitBreakerState);
      expect(status1!.circuitBreakerState).toBe('OPEN');
    });
  });

  describe('Circuit Breaker Monitoring', () => {
    test('should generate appropriate alerts', async () => {
      const provider = 'openai';

      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      const alerts = healthMonitor.getHealthAlerts();
      const circuitBreakerAlerts = alerts.filter(alert => 
        alert.alertType === 'circuit_breaker_open'
      );

      expect(circuitBreakerAlerts.length).toBeGreaterThan(0);
      expect(circuitBreakerAlerts[0].provider).toBe(provider);
      expect(circuitBreakerAlerts[0].severity).toBe('high');
    });

    test('should track circuit breaker events', async () => {
      const provider = 'openai';

      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      // Record fallback event
      healthMonitor.recordFallbackEvent({
        originalProvider: provider,
        fallbackProvider: 'google',
        reason: 'Circuit breaker open',
        timestamp: new Date(),
        success: true,
        responseTime: 2000,
        retryCount: 0
      });

      const events = healthMonitor.getFallbackEvents();
      const circuitBreakerEvents = events.filter(event => 
        event.reason.includes('Circuit breaker')
      );

      expect(circuitBreakerEvents.length).toBeGreaterThan(0);
    });

    test('should provide circuit breaker statistics', () => {
      const summary = healthMonitor.getProviderHealthSummary();

      expect(summary).toBeDefined();
      expect(typeof summary.circuitBreakerOpen).toBe('number');
      expect(typeof summary.overallHealthScore).toBe('number');
      expect(typeof summary.averageAvailability).toBe('number');
    });
  });

  describe('Circuit Breaker Edge Cases', () => {
    test('should handle rapid state transitions', async () => {
      const provider = 'openai';

      // Rapidly alternate between success and failure
      for (let i = 0; i < 20; i++) {
        const isSuccess = i % 2 === 0;
        await healthMonitor.updateProviderStatus(provider, isSuccess, 1500, `Rapid ${i}`);
      }

      const status = healthMonitor.getProviderStatus(provider);
      expect(status).toBeDefined();
      expect(['CLOSED', 'OPEN', 'HALF_OPEN']).toContain(status!.circuitBreakerState);
    });

    test('should handle timeout edge cases', async () => {
      const provider = 'openai';

      // Open circuit breaker
      for (let i = 0; i < 5; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      // Test exact timeout boundary
      const status = healthMonitor.getProviderStatus(provider);
      if (status) {
        status.lastFailure = new Date(Date.now() - 60000); // Exactly timeout
      }

      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Boundary test');

      const updatedStatus = healthMonitor.getProviderStatus(provider);
      expect(updatedStatus!.circuitBreakerState).toBe('HALF_OPEN');
    });

    test('should handle threshold edge cases', async () => {
      const provider = 'openai';

      // Test exact threshold
      for (let i = 0; i < 4; i++) {
        await healthMonitor.updateProviderStatus(provider, false, 5000, `Error ${i}`);
      }

      let status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('CLOSED');

      // One more failure should open circuit breaker
      await healthMonitor.updateProviderStatus(provider, false, 5000, 'Threshold error');

      status = healthMonitor.getProviderStatus(provider);
      expect(status!.circuitBreakerState).toBe('OPEN');
    });
  });
});
