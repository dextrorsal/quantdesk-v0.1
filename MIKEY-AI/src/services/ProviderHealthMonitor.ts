/**
 * Provider Health Monitor
 * Advanced provider health monitoring with circuit breaker pattern and fallback mechanisms
 */

import { 
  ProviderStatus, 
  CircuitBreakerState, 
  HealthCheckResult, 
  ProviderHealthMetrics,
  ProviderHealthSummary,
  HealthAlert,
  FallbackEvent
} from '../types/provider-health';
import { FallbackConfiguration } from '../config/fallback-config';
import { systemLogger, errorLogger } from '../utils/logger';

export class ProviderHealthMonitor {
  private config: FallbackConfiguration;
  private providerStatuses: Map<string, ProviderStatus> = new Map();
  private healthCheckTimer?: NodeJS.Timeout;
  private alerts: HealthAlert[] = [];
  private fallbackEvents: FallbackEvent[] = [];
  private isMonitoring: boolean = false;

  constructor() {
    this.config = FallbackConfiguration.getInstance();
    this.initializeProviderStatuses();
    
    if (this.config.isHealthMonitoringEnabled()) {
      this.startHealthMonitoring();
    }
    
    systemLogger.startup('ProviderHealthMonitor', 'Initialized with circuit breaker pattern');
  }

  /**
   * Check provider health
   */
  public async checkProviderHealth(provider: string): Promise<HealthCheckResult> {
    const startTime = Date.now();
    
    try {
      // Simulate health check - in real implementation, this would make an actual request
      const isHealthy = await this.performHealthCheck(provider);
      const responseTime = Date.now() - startTime;
      
      const result: HealthCheckResult = {
        provider,
        isHealthy,
        responseTime,
        timestamp: new Date()
      };
      
      // Update provider status
      await this.updateProviderStatus(provider, isHealthy, responseTime);
      
      return result;
      
    } catch (error) {
      const responseTime = Date.now() - startTime;
      const result: HealthCheckResult = {
        provider,
        isHealthy: false,
        responseTime,
        error: error instanceof Error ? error.message : 'Unknown error',
        timestamp: new Date()
      };
      
      // Update provider status with failure
      await this.updateProviderStatus(provider, false, responseTime, error instanceof Error ? error.message : 'Unknown error');
      
      return result;
    }
  }

  /**
   * Update provider status
   */
  public async updateProviderStatus(
    provider: string, 
    isHealthy: boolean, 
    responseTime: number, 
    error?: string
  ): Promise<void> {
    const currentStatus = this.providerStatuses.get(provider) || this.createDefaultStatus(provider);
    
    // Update status
    currentStatus.isHealthy = isHealthy;
    currentStatus.responseTime = responseTime;
    currentStatus.lastHealthCheck = new Date();
    currentStatus.totalRequests++;
    
    if (isHealthy) {
      currentStatus.successCount++;
      currentStatus.consecutiveFailures = 0;
    } else {
      currentStatus.failureCount++;
      currentStatus.consecutiveFailures++;
      currentStatus.lastFailure = new Date();
      
      // Create health alert if needed
      await this.createHealthAlert(provider, error || 'Health check failed');
    }
    
    // Update circuit breaker state
    this.updateCircuitBreakerState(currentStatus);
    
    // Update average response time
    this.updateAverageResponseTime(currentStatus, responseTime);
    
    // Update error rate
    this.updateErrorRate(currentStatus);
    
    this.providerStatuses.set(provider, currentStatus);
    
    systemLogger.startup('ProviderHealthMonitor', 
      `Updated ${provider}: ${isHealthy ? 'healthy' : 'unhealthy'} (${responseTime}ms)`
    );
  }

  /**
   * Get healthy providers
   */
  public async getHealthyProviders(): Promise<string[]> {
    const healthyProviders: string[] = [];
    
    this.providerStatuses.forEach((status, provider) => {
      if (status.isHealthy && status.circuitBreakerState === 'CLOSED') {
        healthyProviders.push(provider);
      }
    });
    
    return healthyProviders;
  }

  /**
   * Get circuit breaker state
   */
  public async getCircuitBreakerState(provider: string): Promise<CircuitBreakerState> {
    const status = this.providerStatuses.get(provider);
    return status ? status.circuitBreakerState : 'CLOSED';
  }

  /**
   * Get provider status
   */
  public getProviderStatus(provider: string): ProviderStatus | null {
    return this.providerStatuses.get(provider) || null;
  }

  /**
   * Get all provider statuses
   */
  public getAllProviderStatuses(): Map<string, ProviderStatus> {
    return new Map(this.providerStatuses);
  }

  /**
   * Get provider health metrics
   */
  public getProviderHealthMetrics(provider: string): ProviderHealthMetrics | null {
    const status = this.providerStatuses.get(provider);
    if (!status) return null;
    
    return {
      provider,
      availability: this.calculateAvailability(status),
      averageResponseTime: status.averageResponseTime,
      errorRate: status.errorRate,
      successRate: status.totalRequests > 0 ? status.successCount / status.totalRequests : 0,
      circuitBreakerState: status.circuitBreakerState,
      lastUpdated: status.lastHealthCheck
    };
  }

  /**
   * Get provider health summary
   */
  public getProviderHealthSummary(): ProviderHealthSummary {
    const totalProviders = this.providerStatuses.size;
    let healthyProviders = 0;
    let unhealthyProviders = 0;
    let circuitBreakerOpen = 0;
    let totalAvailability = 0;
    
    this.providerStatuses.forEach((status) => {
      if (status.isHealthy) {
        healthyProviders++;
      } else {
        unhealthyProviders++;
      }
      
      if (status.circuitBreakerState === 'OPEN') {
        circuitBreakerOpen++;
      }
      
      totalAvailability += this.calculateAvailability(status);
    });
    
    const averageAvailability = totalProviders > 0 ? totalAvailability / totalProviders : 0;
    const overallHealthScore = this.calculateOverallHealthScore();
    
    return {
      totalProviders,
      healthyProviders,
      unhealthyProviders,
      circuitBreakerOpen,
      averageAvailability,
      overallHealthScore,
      lastUpdated: new Date()
    };
  }

  /**
   * Get health alerts
   */
  public getHealthAlerts(): HealthAlert[] {
    return [...this.alerts];
  }

  /**
   * Get unresolved health alerts
   */
  public getUnresolvedHealthAlerts(): HealthAlert[] {
    return this.alerts.filter(alert => !alert.resolved);
  }

  /**
   * Resolve health alert
   */
  public resolveHealthAlert(alertId: string): boolean {
    const alert = this.alerts.find(a => a.alertId === alertId);
    if (alert && !alert.resolved) {
      alert.resolved = true;
      alert.resolvedAt = new Date();
      return true;
    }
    return false;
  }

  /**
   * Get fallback events
   */
  public getFallbackEvents(): FallbackEvent[] {
    return [...this.fallbackEvents];
  }

  /**
   * Record fallback event
   */
  public recordFallbackEvent(event: Omit<FallbackEvent, 'eventId'>): void {
    const fallbackEvent: FallbackEvent = {
      ...event,
      eventId: `fallback-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
    };
    
    this.fallbackEvents.push(fallbackEvent);
    
    // Keep only last 1000 events
    if (this.fallbackEvents.length > 1000) {
      this.fallbackEvents = this.fallbackEvents.slice(-1000);
    }
    
    systemLogger.startup('ProviderHealthMonitor', 
      `Recorded fallback: ${event.originalProvider} → ${event.fallbackProvider} (${event.success ? 'success' : 'failed'})`
    );
  }

  /**
   * Start health monitoring
   */
  public startHealthMonitoring(): void {
    if (this.isMonitoring) {
      return;
    }
    
    this.isMonitoring = true;
    const interval = this.config.getHealthCheckInterval();
    
    this.healthCheckTimer = setInterval(async () => {
      await this.performHealthChecks();
    }, interval);
    
    systemLogger.startup('ProviderHealthMonitor', `Started health monitoring with ${interval}ms interval`);
  }

  /**
   * Stop health monitoring
   */
  public stopHealthMonitoring(): void {
    if (this.healthCheckTimer) {
      clearInterval(this.healthCheckTimer);
      this.healthCheckTimer = undefined;
    }
    
    this.isMonitoring = false;
    systemLogger.startup('ProviderHealthMonitor', 'Stopped health monitoring');
  }

  /**
   * Reset provider status
   */
  public resetProviderStatus(provider: string): void {
    const status = this.providerStatuses.get(provider);
    if (status) {
      status.failureCount = 0;
      status.consecutiveFailures = 0;
      status.lastFailure = null;
      status.circuitBreakerState = 'CLOSED';
      status.isHealthy = true;
      status.lastHealthCheck = new Date();
      
      this.providerStatuses.set(provider, status);
      
      systemLogger.startup('ProviderHealthMonitor', `Reset status for provider: ${provider}`);
    }
  }

  /**
   * Clear all data
   */
  public clearAllData(): void {
    this.providerStatuses.clear();
    this.alerts = [];
    this.fallbackEvents = [];
    this.initializeProviderStatuses();
    
    systemLogger.startup('ProviderHealthMonitor', 'Cleared all health monitoring data');
  }

  // Private helper methods
  private initializeProviderStatuses(): void {
    const providers = this.config.getFallbackOrder();
    
    providers.forEach(provider => {
      this.providerStatuses.set(provider, this.createDefaultStatus(provider));
    });
  }

  private createDefaultStatus(provider: string): ProviderStatus {
    return {
      provider,
      isHealthy: true,
      failureCount: 0,
      lastFailure: null,
      circuitBreakerState: 'CLOSED',
      responseTime: 0,
      successCount: 0,
      totalRequests: 0,
      lastHealthCheck: new Date(),
      consecutiveFailures: 0,
      averageResponseTime: 0,
      errorRate: 0
    };
  }

  private async performHealthCheck(provider: string): Promise<boolean> {
    // Simulate health check - in real implementation, this would make an actual request
    // For now, we'll simulate some providers being unhealthy occasionally
    
    const random = Math.random();
    
    // Simulate different providers having different reliability
    switch (provider.toLowerCase()) {
      case 'openai':
        return random > 0.05; // 95% success rate
      case 'google':
        return random > 0.08; // 92% success rate
      case 'mistral':
        return random > 0.10; // 90% success rate
      case 'cohere':
        return random > 0.12; // 88% success rate
      default:
        return random > 0.15; // 85% success rate
    }
  }

  private updateCircuitBreakerState(status: ProviderStatus): void {
    const threshold = this.config.getCircuitBreakerThreshold();
    const timeout = this.config.getCircuitBreakerTimeout();
    
    switch (status.circuitBreakerState) {
      case 'CLOSED':
        if (status.consecutiveFailures >= threshold) {
          status.circuitBreakerState = 'OPEN';
          status.lastFailure = new Date();
          
          this.createHealthAlert(status.provider, 'Circuit breaker opened due to consecutive failures');
        }
        break;
        
      case 'OPEN':
        if (status.lastFailure && Date.now() - status.lastFailure.getTime() >= timeout) {
          status.circuitBreakerState = 'HALF_OPEN';
          
          this.createHealthAlert(status.provider, 'Circuit breaker moved to half-open state');
        }
        break;
        
      case 'HALF_OPEN':
        if (status.isHealthy) {
          status.circuitBreakerState = 'CLOSED';
          status.consecutiveFailures = 0;
          
          this.createHealthAlert(status.provider, 'Circuit breaker closed - provider recovered');
        } else {
          status.circuitBreakerState = 'OPEN';
          status.lastFailure = new Date();
          
          this.createHealthAlert(status.provider, 'Circuit breaker reopened - provider still failing');
        }
        break;
    }
  }

  private updateAverageResponseTime(status: ProviderStatus, responseTime: number): void {
    if (status.totalRequests === 1) {
      status.averageResponseTime = responseTime;
    } else {
      status.averageResponseTime = (status.averageResponseTime * (status.totalRequests - 1) + responseTime) / status.totalRequests;
    }
  }

  private updateErrorRate(status: ProviderStatus): void {
    if (status.totalRequests > 0) {
      status.errorRate = status.failureCount / status.totalRequests;
    }
  }

  private calculateAvailability(status: ProviderStatus): number {
    if (status.totalRequests === 0) return 1;
    return status.successCount / status.totalRequests;
  }

  private calculateOverallHealthScore(): number {
    let totalScore = 0;
    let providerCount = 0;
    
    this.providerStatuses.forEach((status) => {
      const availability = this.calculateAvailability(status);
      const responseTimeScore = Math.max(0, 1 - (status.averageResponseTime / 10000)); // Normalize to 10s max
      const errorRateScore = Math.max(0, 1 - status.errorRate);
      
      const providerScore = (availability + responseTimeScore + errorRateScore) / 3;
      totalScore += providerScore;
      providerCount++;
    });
    
    return providerCount > 0 ? totalScore / providerCount : 0;
  }

  private async createHealthAlert(provider: string, message: string): Promise<void> {
    const alert: HealthAlert = {
      alertId: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      provider,
      alertType: this.determineAlertType(provider, message),
      severity: this.determineAlertSeverity(provider, message),
      message,
      timestamp: new Date(),
      resolved: false
    };
    
    this.alerts.push(alert);
    
    // Keep only last 1000 alerts
    if (this.alerts.length > 1000) {
      this.alerts = this.alerts.slice(-1000);
    }
    
    systemLogger.startup('ProviderHealthMonitor', 
      `Created ${alert.severity} alert for ${provider}: ${message}`
    );
  }

  private determineAlertType(provider: string, message: string): HealthAlert['alertType'] {
    if (message.includes('Circuit breaker opened')) return 'circuit_breaker_open';
    if (message.includes('response time')) return 'response_time_slow';
    if (message.includes('availability')) return 'availability_low';
    return 'error_rate_high';
  }

  private determineAlertSeverity(provider: string, message: string): HealthAlert['severity'] {
    if (message.includes('Circuit breaker opened')) return 'high';
    if (message.includes('recovered')) return 'low';
    if (message.includes('consecutive failures')) return 'medium';
    return 'medium';
  }

  private async performHealthChecks(): Promise<void> {
    const providers = Array.from(this.providerStatuses.keys());
    
    const healthCheckPromises = providers.map(async (provider) => {
      try {
        await this.checkProviderHealth(provider);
      } catch (error) {
        errorLogger.aiError(error as Error, `Health check for ${provider}`);
      }
    });
    
    await Promise.allSettled(healthCheckPromises);
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.stopHealthMonitoring();
  }
}
