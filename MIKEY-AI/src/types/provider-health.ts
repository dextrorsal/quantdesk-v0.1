/**
 * Provider Health Types
 * Type definitions for provider health monitoring and fallback mechanisms
 */

export interface ProviderStatus {
  provider: string;
  isHealthy: boolean;
  failureCount: number;
  lastFailure: Date | null;
  circuitBreakerState: CircuitBreakerState;
  responseTime: number;
  successCount: number;
  totalRequests: number;
  lastHealthCheck: Date;
  consecutiveFailures: number;
  averageResponseTime: number;
  errorRate: number;
}

export type CircuitBreakerState = 'CLOSED' | 'OPEN' | 'HALF_OPEN';

export interface FallbackConfig {
  maxRetries: number;
  retryDelay: number;
  circuitBreakerThreshold: number;
  circuitBreakerTimeout: number;
  healthCheckInterval: number;
  fallbackOrder: string[];
  enableCircuitBreaker: boolean;
  enableHealthMonitoring: boolean;
  retryBackoffMultiplier: number;
  maxRetryDelay: number;
}

export interface HealthCheckResult {
  provider: string;
  isHealthy: boolean;
  responseTime: number;
  error?: string;
  timestamp: Date;
}

export interface FallbackDecision {
  shouldFallback: boolean;
  reason: string;
  suggestedProvider: string;
  retryCount: number;
  maxRetries: number;
  estimatedDelay: number;
}

export interface ProviderHealthMetrics {
  provider: string;
  availability: number; // percentage
  averageResponseTime: number;
  errorRate: number;
  successRate: number;
  circuitBreakerState: CircuitBreakerState;
  lastUpdated: Date;
}

export interface FallbackStrategy {
  strategy: 'cost-first' | 'quality-first' | 'balanced' | 'availability-first';
  maxFallbackDepth: number;
  enableCascadingFallback: boolean;
  qualityThreshold: number;
  costThreshold: number;
}

export interface RetryConfig {
  maxRetries: number;
  baseDelay: number;
  maxDelay: number;
  backoffMultiplier: number;
  jitter: boolean;
  retryableErrors: string[];
}

export interface CircuitBreakerConfig {
  failureThreshold: number;
  timeout: number;
  halfOpenMaxCalls: number;
  enableAutoRecovery: boolean;
  recoveryCheckInterval: number;
}

export interface HealthMonitorConfig {
  checkInterval: number;
  timeout: number;
  enableContinuousMonitoring: boolean;
  alertThresholds: {
    errorRate: number;
    responseTime: number;
    availability: number;
  };
}

export interface FallbackMetrics {
  totalFallbacks: number;
  successfulFallbacks: number;
  failedFallbacks: number;
  averageFallbackTime: number;
  fallbackReasons: Record<string, number>;
  providerFallbackCounts: Record<string, number>;
}

export interface ProviderHealthSummary {
  totalProviders: number;
  healthyProviders: number;
  unhealthyProviders: number;
  circuitBreakerOpen: number;
  averageAvailability: number;
  overallHealthScore: number;
  lastUpdated: Date;
}

export interface FallbackEvent {
  eventId: string;
  originalProvider: string;
  fallbackProvider: string;
  reason: string;
  timestamp: Date;
  success: boolean;
  responseTime: number;
  retryCount: number;
}

export interface HealthAlert {
  alertId: string;
  provider: string;
  alertType: 'error_rate_high' | 'response_time_slow' | 'availability_low' | 'circuit_breaker_open';
  severity: 'low' | 'medium' | 'high' | 'critical';
  message: string;
  timestamp: Date;
  resolved: boolean;
  resolvedAt?: Date;
}
