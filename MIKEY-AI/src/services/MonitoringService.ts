/**
 * Monitoring Service
 * Comprehensive monitoring and alerting for cost optimization features
 */

import { 
  MonitoringMetrics, 
  PerformanceMetrics, 
  CostMetrics, 
  QualityMetrics,
  Alert,
  MonitoringStatus,
  MonitoringDashboard,
  ProviderStats,
  MonitoringEvent,
  MonitoringHealthCheck,
  MonitoringStats,
  AlertType,
  AlertSeverity
} from '../types/monitoring';
import { MonitoringConfigurationManager } from '../config/monitoring-config';
import { CostOptimizationEngine } from './CostOptimizationEngine';
import { QualityThresholdManager } from './QualityThresholdManager';
import { ProviderHealthMonitor } from './ProviderHealthMonitor';
import { AnalyticsCollector } from './AnalyticsCollector';
import { systemLogger, errorLogger } from '../utils/logger';

export class MonitoringService {
  private config: MonitoringConfigurationManager;
  private costEngine: CostOptimizationEngine;
  private qualityManager: QualityThresholdManager;
  private healthMonitor: ProviderHealthMonitor;
  private analyticsCollector: AnalyticsCollector;
  
  private metricsHistory: MonitoringMetrics[] = [];
  private performanceHistory: PerformanceMetrics[] = [];
  private costHistory: CostMetrics[] = [];
  private qualityHistory: QualityMetrics[] = [];
  private alerts: Alert[] = [];
  private events: MonitoringEvent[] = [];
  private healthChecks: MonitoringHealthCheck[] = [];
  
  private monitoringTimer?: NodeJS.Timeout;
  private alertCheckTimer?: NodeJS.Timeout;
  private isMonitoring: boolean = false;
  private startTime: Date = new Date();

  constructor() {
    this.config = MonitoringConfigurationManager.getInstance();
    this.costEngine = new CostOptimizationEngine();
    this.qualityManager = new QualityThresholdManager();
    this.healthMonitor = new ProviderHealthMonitor();
    this.analyticsCollector = new AnalyticsCollector();
    
    if (this.config.isMonitoringEnabled()) {
      this.startMonitoring();
    }
    
    systemLogger.startup('MonitoringService', 'Initialized comprehensive monitoring and alerting');
  }

  /**
   * Collect monitoring metrics
   */
  public async collectMetrics(metrics: Omit<MonitoringMetrics, 'timestamp'>): Promise<void> {
    try {
      if (!this.config.isMonitoringEnabled()) {
        return;
      }

      const fullMetrics: MonitoringMetrics = {
        ...metrics,
        timestamp: new Date()
      };

      this.metricsHistory.push(fullMetrics);
      
      // Keep only recent metrics based on retention period
      this.trimMetricsHistory();
      
      // Check for alerts
      await this.checkAlerts(fullMetrics);
      
      // Record event
      this.recordEvent('metric_collected', {
        requestId: metrics.requestId,
        provider: metrics.provider,
        cost: metrics.cost,
        responseTime: metrics.responseTime,
        qualityScore: metrics.qualityScore,
        success: metrics.success
      });

    } catch (error) {
      errorLogger.aiError(error as Error, 'Metrics collection');
    }
  }

  /**
   * Collect performance metrics
   */
  public async collectPerformanceMetrics(metrics: Omit<PerformanceMetrics, 'timestamp'>): Promise<void> {
    try {
      if (!this.config.isPerformanceTrackingEnabled()) {
        return;
      }

      const fullMetrics: PerformanceMetrics = {
        ...metrics,
        timestamp: new Date()
      };

      this.performanceHistory.push(fullMetrics);
      
      // Keep only recent performance metrics
      this.trimPerformanceHistory();
      
      // Check for performance alerts
      await this.checkPerformanceAlerts(fullMetrics);

    } catch (error) {
      errorLogger.aiError(error as Error, 'Performance metrics collection');
    }
  }

  /**
   * Collect cost metrics
   */
  public async collectCostMetrics(metrics: Omit<CostMetrics, 'timestamp'>): Promise<void> {
    try {
      if (!this.config.isCostTrackingEnabled()) {
        return;
      }

      const fullMetrics: CostMetrics = {
        ...metrics,
        timestamp: new Date()
      };

      this.costHistory.push(fullMetrics);
      
      // Keep only recent cost metrics
      this.trimCostHistory();
      
      // Check for cost alerts
      await this.checkCostAlerts(fullMetrics);

    } catch (error) {
      errorLogger.aiError(error as Error, 'Cost metrics collection');
    }
  }

  /**
   * Collect quality metrics
   */
  public async collectQualityMetrics(metrics: Omit<QualityMetrics, 'timestamp'>): Promise<void> {
    try {
      if (!this.config.isQualityTrackingEnabled()) {
        return;
      }

      const fullMetrics: QualityMetrics = {
        ...metrics,
        timestamp: new Date()
      };

      this.qualityHistory.push(fullMetrics);
      
      // Keep only recent quality metrics
      this.trimQualityHistory();
      
      // Check for quality alerts
      await this.checkQualityAlerts(fullMetrics);

    } catch (error) {
      errorLogger.aiError(error as Error, 'Quality metrics collection');
    }
  }

  /**
   * Get monitoring dashboard
   */
  public getMonitoringDashboard(): MonitoringDashboard {
    const status = this.getMonitoringStatus();
    const recentMetrics = this.getRecentMetrics(100);
    const activeAlerts = this.getActiveAlerts();
    const performanceTrends = this.getRecentPerformanceMetrics(50);
    const costTrends = this.getRecentCostMetrics(50);
    const qualityTrends = this.getRecentQualityMetrics(50);
    const providerStats = this.getProviderStats();

    return {
      status,
      recentMetrics,
      activeAlerts,
      performanceTrends,
      costTrends,
      qualityTrends,
      providerStats
    };
  }

  /**
   * Get monitoring status
   */
  public getMonitoringStatus(): MonitoringStatus {
    const activeAlerts = this.getActiveAlerts();
    const criticalAlerts = activeAlerts.filter(alert => alert.severity === 'critical');
    const uptime = Date.now() - this.startTime.getTime();
    const systemLoad = this.calculateSystemLoad();

    return {
      isHealthy: criticalAlerts.length === 0 && systemLoad < 0.8,
      lastUpdate: new Date(),
      activeAlerts: activeAlerts.length,
      criticalAlerts: criticalAlerts.length,
      metricsCollected: this.metricsHistory.length,
      performanceScore: this.calculatePerformanceScore(),
      uptime: uptime,
      systemLoad: systemLoad
    };
  }

  /**
   * Get active alerts
   */
  public getActiveAlerts(): Alert[] {
    return this.alerts.filter(alert => !alert.resolved);
  }

  /**
   * Get all alerts
   */
  public getAllAlerts(): Alert[] {
    return [...this.alerts];
  }

  /**
   * Resolve alert
   */
  public resolveAlert(alertId: string): boolean {
    const alert = this.alerts.find(a => a.alertId === alertId);
    if (alert && !alert.resolved) {
      alert.resolved = true;
      alert.resolvedAt = new Date();
      
      this.recordEvent('alert_resolved', {
        alertId: alertId,
        alertType: alert.alertType,
        severity: alert.severity
      });
      
      systemLogger.startup('MonitoringService', `Resolved alert: ${alertId}`);
      return true;
    }
    return false;
  }

  /**
   * Get recent metrics
   */
  public getRecentMetrics(count: number): MonitoringMetrics[] {
    return this.metricsHistory.slice(-count);
  }

  /**
   * Get recent performance metrics
   */
  public getRecentPerformanceMetrics(count: number): PerformanceMetrics[] {
    return this.performanceHistory.slice(-count);
  }

  /**
   * Get recent cost metrics
   */
  public getRecentCostMetrics(count: number): CostMetrics[] {
    return this.costHistory.slice(-count);
  }

  /**
   * Get recent quality metrics
   */
  public getRecentQualityMetrics(count: number): QualityMetrics[] {
    return this.qualityHistory.slice(-count);
  }

  /**
   * Get provider statistics
   */
  public getProviderStats(): Record<string, ProviderStats> {
    const providerStats: Record<string, ProviderStats> = {};
    
    // Get provider health information
    const healthSummary = this.healthMonitor.getProviderHealthSummary();
    const allStatuses = this.healthMonitor.getAllProviderStatuses();
    
    allStatuses.forEach((status, provider) => {
      const metrics = this.healthMonitor.getProviderHealthMetrics(provider);
      const recentMetrics = this.getRecentMetrics(1000).filter(m => m.provider === provider);
      
      if (recentMetrics.length > 0) {
        const totalRequests = recentMetrics.length;
        const successfulRequests = recentMetrics.filter(m => m.success).length;
        const averageResponseTime = recentMetrics.reduce((sum, m) => sum + m.responseTime, 0) / totalRequests;
        const averageCost = recentMetrics.reduce((sum, m) => sum + m.cost, 0) / totalRequests;
        const averageQualityScore = recentMetrics.reduce((sum, m) => sum + m.qualityScore, 0) / totalRequests;
        const fallbackCount = recentMetrics.filter(m => m.fallbackUsed).length;
        
        providerStats[provider] = {
          provider,
          totalRequests,
          successRate: successfulRequests / totalRequests,
          averageResponseTime,
          averageCost,
          averageQualityScore,
          circuitBreakerState: status.circuitBreakerState,
          lastHealthCheck: status.lastHealthCheck,
          errorRate: 1 - (successfulRequests / totalRequests),
          fallbackCount
        };
      }
    });
    
    return providerStats;
  }

  /**
   * Get monitoring statistics
   */
  public getMonitoringStats(): MonitoringStats {
    const recentMetrics = this.getRecentMetrics(1000);
    const activeAlerts = this.getActiveAlerts();
    const resolvedAlerts = this.alerts.filter(alert => alert.resolved);
    
    const totalRequests = recentMetrics.length;
    const averageResponseTime = totalRequests > 0 ? 
      recentMetrics.reduce((sum, m) => sum + m.responseTime, 0) / totalRequests : 0;
    const averageCost = totalRequests > 0 ? 
      recentMetrics.reduce((sum, m) => sum + m.cost, 0) / totalRequests : 0;
    const averageQualityScore = totalRequests > 0 ? 
      recentMetrics.reduce((sum, m) => sum + m.qualityScore, 0) / totalRequests : 0;
    
    const healthSummary = this.healthMonitor.getProviderHealthSummary();
    
    return {
      totalMetricsCollected: this.metricsHistory.length,
      totalAlertsGenerated: this.alerts.length,
      totalAlertsResolved: resolvedAlerts.length,
      averageResponseTime,
      averageCost,
      averageQualityScore,
      systemUptime: Date.now() - this.startTime.getTime(),
      lastHealthCheck: new Date(),
      activeProviders: healthSummary.totalProviders,
      healthyProviders: healthSummary.healthyProviders
    };
  }

  /**
   * Start monitoring
   */
  public startMonitoring(): void {
    if (this.isMonitoring) {
      return;
    }
    
    this.isMonitoring = true;
    
    // Start metrics collection timer
    const metricsInterval = this.config.getMetricsCollectionInterval();
    this.monitoringTimer = setInterval(async () => {
      await this.performHealthCheck();
    }, metricsInterval);
    
    // Start alert checking timer
    const alertInterval = this.config.getAlertCheckInterval();
    this.alertCheckTimer = setInterval(async () => {
      await this.checkAllAlerts();
    }, alertInterval);
    
    systemLogger.startup('MonitoringService', `Started monitoring with ${metricsInterval}ms interval`);
  }

  /**
   * Stop monitoring
   */
  public stopMonitoring(): void {
    if (this.monitoringTimer) {
      clearInterval(this.monitoringTimer);
      this.monitoringTimer = undefined;
    }
    
    if (this.alertCheckTimer) {
      clearInterval(this.alertCheckTimer);
      this.alertCheckTimer = undefined;
    }
    
    this.isMonitoring = false;
    systemLogger.startup('MonitoringService', 'Stopped monitoring');
  }

  /**
   * Clear all data
   */
  public clearAllData(): void {
    this.metricsHistory = [];
    this.performanceHistory = [];
    this.costHistory = [];
    this.qualityHistory = [];
    this.alerts = [];
    this.events = [];
    this.healthChecks = [];
    
    systemLogger.startup('MonitoringService', 'Cleared all monitoring data');
  }

  // Private helper methods
  private async checkAlerts(metrics: MonitoringMetrics): Promise<void> {
    try {
      const thresholds = this.config.getAlertThresholds();
      
      // Check cost anomaly
      if (metrics.cost > thresholds.costAnomalyThreshold) {
        await this.createAlert('cost_anomaly', 'high', 
          `Cost anomaly detected: ${metrics.cost} exceeds threshold ${thresholds.costAnomalyThreshold}`,
          { provider: metrics.provider, cost: metrics.cost, threshold: thresholds.costAnomalyThreshold }
        );
      }
      
      // Check quality degradation
      if (metrics.qualityScore < thresholds.qualityDegradationThreshold) {
        await this.createAlert('quality_degradation', 'high',
          `Quality degradation detected: ${metrics.qualityScore} below threshold ${thresholds.qualityDegradationThreshold}`,
          { provider: metrics.provider, qualityScore: metrics.qualityScore, threshold: thresholds.qualityDegradationThreshold }
        );
      }
      
      // Check response time
      if (metrics.responseTime > thresholds.responseTimeThreshold) {
        await this.createAlert('response_time_slow', 'medium',
          `Slow response time: ${metrics.responseTime}ms exceeds threshold ${thresholds.responseTimeThreshold}ms`,
          { provider: metrics.provider, responseTime: metrics.responseTime, threshold: thresholds.responseTimeThreshold }
        );
      }
      
      // Check error rate
      if (!metrics.success) {
        await this.checkErrorRateAlert(metrics.provider);
      }
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Alert checking');
    }
  }

  private async checkPerformanceAlerts(metrics: PerformanceMetrics): Promise<void> {
    try {
      const thresholds = this.config.getAlertThresholds();
      
      // Check routing decision time
      if (metrics.routingDecisionTime > 200) { // 200ms threshold
        await this.createAlert('performance_degradation', 'medium',
          `Slow routing decision: ${metrics.routingDecisionTime}ms`,
          { routingDecisionTime: metrics.routingDecisionTime }
        );
      }
      
      // Check total request time
      if (metrics.totalRequestTime > thresholds.responseTimeThreshold) {
        await this.createAlert('response_time_slow', 'medium',
          `Slow total request time: ${metrics.totalRequestTime}ms`,
          { totalRequestTime: metrics.totalRequestTime, threshold: thresholds.responseTimeThreshold }
        );
      }
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Performance alert checking');
    }
  }

  private async checkCostAlerts(metrics: CostMetrics): Promise<void> {
    try {
      const thresholds = this.config.getAlertThresholds();
      
      // Check budget utilization
      if (metrics.budgetUtilization >= thresholds.budgetLimitThreshold) {
        await this.createAlert('budget_limit', 'critical',
          `Budget limit reached: ${(metrics.budgetUtilization * 100).toFixed(1)}%`,
          { budgetUtilization: metrics.budgetUtilization, threshold: thresholds.budgetLimitThreshold }
        );
      }
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Cost alert checking');
    }
  }

  private async checkQualityAlerts(metrics: QualityMetrics): Promise<void> {
    try {
      const thresholds = this.config.getAlertThresholds();
      
      // Check quality degradation rate
      if (metrics.qualityDegradationRate > 0.1) { // 10% degradation rate
        await this.createAlert('quality_degradation', 'high',
          `Quality degradation rate: ${(metrics.qualityDegradationRate * 100).toFixed(1)}%`,
          { qualityDegradationRate: metrics.qualityDegradationRate }
        );
      }
      
      // Check escalation rate
      if (metrics.escalationRate > 0.2) { // 20% escalation rate
        await this.createAlert('quality_degradation', 'medium',
          `High escalation rate: ${(metrics.escalationRate * 100).toFixed(1)}%`,
          { escalationRate: metrics.escalationRate }
        );
      }
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Quality alert checking');
    }
  }

  private async checkErrorRateAlert(provider: string): Promise<void> {
    try {
      const recentMetrics = this.getRecentMetrics(100).filter(m => m.provider === provider);
      if (recentMetrics.length < 10) return; // Need minimum samples
      
      const errorRate = recentMetrics.filter(m => !m.success).length / recentMetrics.length;
      const thresholds = this.config.getAlertThresholds();
      
      if (errorRate > thresholds.errorRateThreshold) {
        await this.createAlert('error_rate_high', 'high',
          `High error rate for ${provider}: ${(errorRate * 100).toFixed(1)}%`,
          { provider, errorRate, threshold: thresholds.errorRateThreshold }
        );
      }
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Error rate alert checking');
    }
  }

  private async checkAllAlerts(): Promise<void> {
    try {
      // Check circuit breaker alerts
      const healthSummary = this.healthMonitor.getProviderHealthSummary();
      if (healthSummary.circuitBreakerOpen > 0) {
        await this.createAlert('circuit_breaker_open', 'high',
          `${healthSummary.circuitBreakerOpen} circuit breakers are open`,
          { circuitBreakerOpen: healthSummary.circuitBreakerOpen }
        );
      }
      
      // Check availability
      const thresholds = this.config.getAlertThresholds();
      if (healthSummary.averageAvailability < thresholds.availabilityThreshold) {
        await this.createAlert('availability_low', 'critical',
          `Low availability: ${(healthSummary.averageAvailability * 100).toFixed(1)}%`,
          { availability: healthSummary.averageAvailability, threshold: thresholds.availabilityThreshold }
        );
      }
      
      // Check fallback rate
      const recentMetrics = this.getRecentMetrics(100);
      if (recentMetrics.length > 0) {
        const fallbackRate = recentMetrics.filter(m => m.fallbackUsed).length / recentMetrics.length;
        if (fallbackRate > thresholds.fallbackRateThreshold) {
          await this.createAlert('fallback_rate_high', 'medium',
            `High fallback rate: ${(fallbackRate * 100).toFixed(1)}%`,
            { fallbackRate, threshold: thresholds.fallbackRateThreshold }
          );
        }
      }
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Comprehensive alert checking');
    }
  }

  private async createAlert(
    alertType: AlertType, 
    severity: AlertSeverity, 
    message: string, 
    metadata: Record<string, any>
  ): Promise<void> {
    try {
      // Check cooldown
      const recentAlerts = this.alerts.filter(alert => 
        alert.alertType === alertType && 
        !alert.resolved && 
        Date.now() - alert.timestamp.getTime() < this.config.getAlertCooldownMinutes() * 60 * 1000
      );
      
      if (recentAlerts.length > 0) {
        return; // Still in cooldown
      }
      
      // Check max alerts per hour
      const hourlyAlerts = this.alerts.filter(alert => 
        Date.now() - alert.timestamp.getTime() < 60 * 60 * 1000
      );
      
      if (hourlyAlerts.length >= this.config.getMaxAlertsPerHour()) {
        return; // Rate limited
      }
      
      const alert: Alert = {
        alertId: `alert-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        alertType,
        severity,
        message,
        timestamp: new Date(),
        resolved: false,
        metadata
      };
      
      this.alerts.push(alert);
      
      // Record event
      this.recordEvent('alert_triggered', {
        alertId: alert.alertId,
        alertType: alert.alertType,
        severity: alert.severity,
        message: alert.message
      });
      
      systemLogger.startup('MonitoringService', 
        `Created ${severity} alert: ${alertType} - ${message}`
      );
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Alert creation');
    }
  }

  private recordEvent(eventType: MonitoringEvent['eventType'], metadata: Record<string, any>): void {
    const event: MonitoringEvent = {
      eventId: `event-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
      eventType,
      timestamp: new Date(),
      metadata,
      severity: 'low' // Default severity
    };
    
    this.events.push(event);
    
    // Keep only recent events
    if (this.events.length > 1000) {
      this.events = this.events.slice(-1000);
    }
  }

  private async performHealthCheck(): Promise<void> {
    try {
      const healthCheck: MonitoringHealthCheck = {
        checkId: `health-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`,
        timestamp: new Date(),
        component: 'monitoring-service',
        status: 'healthy',
        responseTime: 0,
        metadata: {
          metricsCount: this.metricsHistory.length,
          alertsCount: this.alerts.length,
          eventsCount: this.events.length
        }
      };
      
      this.healthChecks.push(healthCheck);
      
      // Keep only recent health checks
      if (this.healthChecks.length > 100) {
        this.healthChecks = this.healthChecks.slice(-100);
      }
      
    } catch (error) {
      errorLogger.aiError(error as Error, 'Health check');
    }
  }

  private trimMetricsHistory(): void {
    const retentionDays = this.config.getDataRetentionDays();
    const cutoffTime = new Date(Date.now() - retentionDays * 24 * 60 * 60 * 1000);
    
    this.metricsHistory = this.metricsHistory.filter(metric => metric.timestamp > cutoffTime);
  }

  private trimPerformanceHistory(): void {
    const retentionDays = this.config.getDataRetentionDays();
    const cutoffTime = new Date(Date.now() - retentionDays * 24 * 60 * 60 * 1000);
    
    this.performanceHistory = this.performanceHistory.filter(metric => metric.timestamp > cutoffTime);
  }

  private trimCostHistory(): void {
    const retentionDays = this.config.getDataRetentionDays();
    const cutoffTime = new Date(Date.now() - retentionDays * 24 * 60 * 60 * 1000);
    
    this.costHistory = this.costHistory.filter(metric => metric.timestamp > cutoffTime);
  }

  private trimQualityHistory(): void {
    const retentionDays = this.config.getDataRetentionDays();
    const cutoffTime = new Date(Date.now() - retentionDays * 24 * 60 * 60 * 1000);
    
    this.qualityHistory = this.qualityHistory.filter(metric => metric.timestamp > cutoffTime);
  }

  private calculateSystemLoad(): number {
    // Simplified system load calculation
    const recentMetrics = this.getRecentMetrics(100);
    if (recentMetrics.length === 0) return 0;
    
    const averageResponseTime = recentMetrics.reduce((sum, m) => sum + m.responseTime, 0) / recentMetrics.length;
    const errorRate = recentMetrics.filter(m => !m.success).length / recentMetrics.length;
    
    // Normalize to 0-1 scale
    return Math.min(1, (averageResponseTime / 10000) + errorRate);
  }

  private calculatePerformanceScore(): number {
    const recentMetrics = this.getRecentMetrics(100);
    if (recentMetrics.length === 0) return 1;
    
    const successRate = recentMetrics.filter(m => m.success).length / recentMetrics.length;
    const averageQualityScore = recentMetrics.reduce((sum, m) => sum + m.qualityScore, 0) / recentMetrics.length;
    const averageResponseTime = recentMetrics.reduce((sum, m) => sum + m.responseTime, 0) / recentMetrics.length;
    
    // Calculate performance score (0-1)
    const responseTimeScore = Math.max(0, 1 - (averageResponseTime / 10000));
    return (successRate + averageQualityScore + responseTimeScore) / 3;
  }

  /**
   * Cleanup resources
   */
  public destroy(): void {
    this.stopMonitoring();
  }
}
