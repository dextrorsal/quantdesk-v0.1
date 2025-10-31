/**
 * Fallback Configuration
 * Configuration management for advanced fallback mechanisms
 */

import { FallbackConfig, CircuitBreakerConfig, RetryConfig, FallbackStrategy, HealthMonitorConfig } from '../types/provider-health';

export class FallbackConfiguration {
  private static instance: FallbackConfiguration;
  private fallbackConfig: FallbackConfig;
  private circuitBreakerConfig: CircuitBreakerConfig;
  private retryConfig: RetryConfig;
  private fallbackStrategy: FallbackStrategy;
  private healthMonitorConfig: HealthMonitorConfig;

  private constructor() {
    this.fallbackConfig = this.loadFallbackConfig();
    this.circuitBreakerConfig = this.loadCircuitBreakerConfig();
    this.retryConfig = this.loadRetryConfig();
    this.fallbackStrategy = this.loadFallbackStrategy();
    this.healthMonitorConfig = this.loadHealthMonitorConfig();
  }

  public static getInstance(): FallbackConfiguration {
    if (!FallbackConfiguration.instance) {
      FallbackConfiguration.instance = new FallbackConfiguration();
    }
    return FallbackConfiguration.instance;
  }

  private loadFallbackConfig(): FallbackConfig {
    return {
      maxRetries: parseInt(process.env.FALLBACK_MAX_RETRIES || '3'),
      retryDelay: parseInt(process.env.FALLBACK_RETRY_DELAY || '1000'),
      circuitBreakerThreshold: parseInt(process.env.CIRCUIT_BREAKER_THRESHOLD || '5'),
      circuitBreakerTimeout: parseInt(process.env.CIRCUIT_BREAKER_TIMEOUT || '60000'),
      healthCheckInterval: parseInt(process.env.HEALTH_CHECK_INTERVAL || '30000'),
      fallbackOrder: (process.env.FALLBACK_ORDER || 'openai,google,mistral,cohere').split(','),
      enableCircuitBreaker: process.env.ENABLE_CIRCUIT_BREAKER !== 'false',
      enableHealthMonitoring: process.env.ENABLE_HEALTH_MONITORING !== 'false',
      retryBackoffMultiplier: parseFloat(process.env.RETRY_BACKOFF_MULTIPLIER || '2'),
      maxRetryDelay: parseInt(process.env.MAX_RETRY_DELAY || '10000')
    };
  }

  private loadCircuitBreakerConfig(): CircuitBreakerConfig {
    return {
      failureThreshold: parseInt(process.env.CIRCUIT_BREAKER_FAILURE_THRESHOLD || '5'),
      timeout: parseInt(process.env.CIRCUIT_BREAKER_TIMEOUT || '60000'),
      halfOpenMaxCalls: parseInt(process.env.CIRCUIT_BREAKER_HALF_OPEN_MAX_CALLS || '3'),
      enableAutoRecovery: process.env.CIRCUIT_BREAKER_AUTO_RECOVERY !== 'false',
      recoveryCheckInterval: parseInt(process.env.CIRCUIT_BREAKER_RECOVERY_CHECK_INTERVAL || '30000')
    };
  }

  private loadRetryConfig(): RetryConfig {
    return {
      maxRetries: parseInt(process.env.RETRY_MAX_RETRIES || '3'),
      baseDelay: parseInt(process.env.RETRY_BASE_DELAY || '1000'),
      maxDelay: parseInt(process.env.RETRY_MAX_DELAY || '10000'),
      backoffMultiplier: parseFloat(process.env.RETRY_BACKOFF_MULTIPLIER || '2'),
      jitter: process.env.RETRY_JITTER !== 'false',
      retryableErrors: (process.env.RETRYABLE_ERRORS || 'timeout,network,rate_limit').split(',')
    };
  }

  private loadFallbackStrategy(): FallbackStrategy {
    const strategy = process.env.FALLBACK_STRATEGY || 'balanced';
    return {
      strategy: strategy as 'cost-first' | 'quality-first' | 'balanced' | 'availability-first',
      maxFallbackDepth: parseInt(process.env.MAX_FALLBACK_DEPTH || '3'),
      enableCascadingFallback: process.env.ENABLE_CASCADING_FALLBACK !== 'false',
      qualityThreshold: parseFloat(process.env.FALLBACK_QUALITY_THRESHOLD || '0.7'),
      costThreshold: parseFloat(process.env.FALLBACK_COST_THRESHOLD || '0.1')
    };
  }

  private loadHealthMonitorConfig(): HealthMonitorConfig {
    return {
      checkInterval: parseInt(process.env.HEALTH_MONITOR_CHECK_INTERVAL || '30000'),
      timeout: parseInt(process.env.HEALTH_MONITOR_TIMEOUT || '5000'),
      enableContinuousMonitoring: process.env.ENABLE_CONTINUOUS_MONITORING !== 'false',
      alertThresholds: {
        errorRate: parseFloat(process.env.ALERT_ERROR_RATE_THRESHOLD || '0.1'),
        responseTime: parseInt(process.env.ALERT_RESPONSE_TIME_THRESHOLD || '5000'),
        availability: parseFloat(process.env.ALERT_AVAILABILITY_THRESHOLD || '0.95')
      }
    };
  }

  // Getters for configurations
  public getFallbackConfig(): FallbackConfig {
    return { ...this.fallbackConfig };
  }

  public getCircuitBreakerConfig(): CircuitBreakerConfig {
    return { ...this.circuitBreakerConfig };
  }

  public getRetryConfig(): RetryConfig {
    return { ...this.retryConfig };
  }

  public getFallbackStrategy(): FallbackStrategy {
    return { ...this.fallbackStrategy };
  }

  public getHealthMonitorConfig(): HealthMonitorConfig {
    return { ...this.healthMonitorConfig };
  }

  // Update methods
  public updateFallbackConfig(updates: Partial<FallbackConfig>): void {
    this.fallbackConfig = { ...this.fallbackConfig, ...updates };
  }

  public updateCircuitBreakerConfig(updates: Partial<CircuitBreakerConfig>): void {
    this.circuitBreakerConfig = { ...this.circuitBreakerConfig, ...updates };
  }

  public updateRetryConfig(updates: Partial<RetryConfig>): void {
    this.retryConfig = { ...this.retryConfig, ...updates };
  }

  public updateFallbackStrategy(updates: Partial<FallbackStrategy>): void {
    this.fallbackStrategy = { ...this.fallbackStrategy, ...updates };
  }

  public updateHealthMonitorConfig(updates: Partial<HealthMonitorConfig>): void {
    this.healthMonitorConfig = { ...this.healthMonitorConfig, ...updates };
  }

  // Validation methods
  public validateConfiguration(): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate fallback config
    if (this.fallbackConfig.maxRetries < 0) {
      errors.push('Max retries must be non-negative');
    }

    if (this.fallbackConfig.retryDelay < 0) {
      errors.push('Retry delay must be non-negative');
    }

    if (this.fallbackConfig.circuitBreakerThreshold < 1) {
      errors.push('Circuit breaker threshold must be at least 1');
    }

    if (this.fallbackConfig.circuitBreakerTimeout < 1000) {
      errors.push('Circuit breaker timeout must be at least 1000ms');
    }

    if (this.fallbackConfig.healthCheckInterval < 5000) {
      errors.push('Health check interval must be at least 5000ms');
    }

    // Validate circuit breaker config
    if (this.circuitBreakerConfig.failureThreshold < 1) {
      errors.push('Circuit breaker failure threshold must be at least 1');
    }

    if (this.circuitBreakerConfig.timeout < 1000) {
      errors.push('Circuit breaker timeout must be at least 1000ms');
    }

    if (this.circuitBreakerConfig.halfOpenMaxCalls < 1) {
      errors.push('Half-open max calls must be at least 1');
    }

    // Validate retry config
    if (this.retryConfig.maxRetries < 0) {
      errors.push('Retry max retries must be non-negative');
    }

    if (this.retryConfig.baseDelay < 0) {
      errors.push('Retry base delay must be non-negative');
    }

    if (this.retryConfig.maxDelay < this.retryConfig.baseDelay) {
      errors.push('Retry max delay must be greater than or equal to base delay');
    }

    if (this.retryConfig.backoffMultiplier < 1) {
      errors.push('Retry backoff multiplier must be at least 1');
    }

    // Validate fallback strategy
    if (this.fallbackStrategy.maxFallbackDepth < 1) {
      errors.push('Max fallback depth must be at least 1');
    }

    if (this.fallbackStrategy.qualityThreshold < 0 || this.fallbackStrategy.qualityThreshold > 1) {
      errors.push('Quality threshold must be between 0 and 1');
    }

    if (this.fallbackStrategy.costThreshold < 0) {
      errors.push('Cost threshold must be non-negative');
    }

    // Validate health monitor config
    if (this.healthMonitorConfig.checkInterval < 1000) {
      errors.push('Health monitor check interval must be at least 1000ms');
    }

    if (this.healthMonitorConfig.timeout < 100) {
      errors.push('Health monitor timeout must be at least 100ms');
    }

    if (this.healthMonitorConfig.alertThresholds.errorRate < 0 || this.healthMonitorConfig.alertThresholds.errorRate > 1) {
      errors.push('Alert error rate threshold must be between 0 and 1');
    }

    if (this.healthMonitorConfig.alertThresholds.responseTime < 0) {
      errors.push('Alert response time threshold must be non-negative');
    }

    if (this.healthMonitorConfig.alertThresholds.availability < 0 || this.healthMonitorConfig.alertThresholds.availability > 1) {
      errors.push('Alert availability threshold must be between 0 and 1');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  // Utility methods
  public isCircuitBreakerEnabled(): boolean {
    return this.fallbackConfig.enableCircuitBreaker;
  }

  public isHealthMonitoringEnabled(): boolean {
    return this.fallbackConfig.enableHealthMonitoring;
  }

  public getFallbackOrder(): string[] {
    return [...this.fallbackConfig.fallbackOrder];
  }

  public getMaxRetries(): number {
    return this.fallbackConfig.maxRetries;
  }

  public getRetryDelay(): number {
    return this.fallbackConfig.retryDelay;
  }

  public getCircuitBreakerThreshold(): number {
    return this.fallbackConfig.circuitBreakerThreshold;
  }

  public getCircuitBreakerTimeout(): number {
    return this.fallbackConfig.circuitBreakerTimeout;
  }

  public getHealthCheckInterval(): number {
    return this.fallbackConfig.healthCheckInterval;
  }

  public getRetryBackoffMultiplier(): number {
    return this.fallbackConfig.retryBackoffMultiplier;
  }

  public getMaxRetryDelay(): number {
    return this.fallbackConfig.maxRetryDelay;
  }

  public getFallbackStrategyType(): string {
    return this.fallbackStrategy.strategy;
  }

  public getMaxFallbackDepth(): number {
    return this.fallbackStrategy.maxFallbackDepth;
  }

  public isCascadingFallbackEnabled(): boolean {
    return this.fallbackStrategy.enableCascadingFallback;
  }

  public getQualityThreshold(): number {
    return this.fallbackStrategy.qualityThreshold;
  }

  public getCostThreshold(): number {
    return this.fallbackStrategy.costThreshold;
  }
}
