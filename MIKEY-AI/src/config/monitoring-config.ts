/**
 * Monitoring Configuration
 * Configuration management for cost optimization monitoring and alerting
 */

import { 
  MonitoringConfig, 
  AlertThresholds, 
  MonitoringConfiguration,
  AlertRule,
  MonitoringThreshold
} from '../types/monitoring';
import { systemLogger } from '../utils/logger';

export class MonitoringConfigurationManager {
  private static instance: MonitoringConfigurationManager;
  private monitoringConfig: MonitoringConfig;
  private monitoringConfiguration: MonitoringConfiguration;
  private alertRules: Map<string, AlertRule> = new Map();
  private monitoringThresholds: Map<string, MonitoringThreshold> = new Map();

  private constructor() {
    this.monitoringConfig = this.loadMonitoringConfig();
    this.monitoringConfiguration = this.loadMonitoringConfiguration();
    this.initializeAlertRules();
    this.initializeMonitoringThresholds();
    
    systemLogger.startup('MonitoringConfigurationManager', 'Initialized monitoring configuration');
  }

  public static getInstance(): MonitoringConfigurationManager {
    if (!MonitoringConfigurationManager.instance) {
      MonitoringConfigurationManager.instance = new MonitoringConfigurationManager();
    }
    return MonitoringConfigurationManager.instance;
  }

  private loadMonitoringConfig(): MonitoringConfig {
    return {
      metricsInterval: parseInt(process.env.MONITORING_METRICS_INTERVAL || '5000'),
      alertThresholds: this.loadAlertThresholds(),
      retentionPeriod: parseInt(process.env.MONITORING_RETENTION_PERIOD || '30'),
      enabledMetrics: (process.env.MONITORING_ENABLED_METRICS || 'cost,quality,performance,availability').split(','),
      enableRealTimeMonitoring: process.env.MONITORING_REAL_TIME_ENABLED !== 'false',
      enablePerformanceTracking: process.env.MONITORING_PERFORMANCE_ENABLED !== 'false',
      enableCostTracking: process.env.MONITORING_COST_ENABLED !== 'false',
      enableQualityTracking: process.env.MONITORING_QUALITY_ENABLED !== 'false'
    };
  }

  private loadAlertThresholds(): AlertThresholds {
    return {
      costAnomalyThreshold: parseFloat(process.env.ALERT_COST_ANOMALY_THRESHOLD || '0.1'),
      qualityDegradationThreshold: parseFloat(process.env.ALERT_QUALITY_DEGRADATION_THRESHOLD || '0.7'),
      budgetLimitThreshold: parseFloat(process.env.ALERT_BUDGET_LIMIT_THRESHOLD || '0.8'),
      responseTimeThreshold: parseInt(process.env.ALERT_RESPONSE_TIME_THRESHOLD || '5000'),
      errorRateThreshold: parseFloat(process.env.ALERT_ERROR_RATE_THRESHOLD || '0.1'),
      availabilityThreshold: parseFloat(process.env.ALERT_AVAILABILITY_THRESHOLD || '0.95'),
      circuitBreakerThreshold: parseInt(process.env.ALERT_CIRCUIT_BREAKER_THRESHOLD || '5'),
      fallbackRateThreshold: parseFloat(process.env.ALERT_FALLBACK_RATE_THRESHOLD || '0.2')
    };
  }

  private loadMonitoringConfiguration(): MonitoringConfiguration {
    return {
      enableMonitoring: process.env.MONITORING_ENABLED !== 'false',
      enableAlerting: process.env.MONITORING_ALERTING_ENABLED !== 'false',
      enablePerformanceTracking: process.env.MONITORING_PERFORMANCE_ENABLED !== 'false',
      enableCostTracking: process.env.MONITORING_COST_ENABLED !== 'false',
      enableQualityTracking: process.env.MONITORING_QUALITY_ENABLED !== 'false',
      metricsCollectionInterval: parseInt(process.env.MONITORING_COLLECTION_INTERVAL || '5000'),
      alertCheckInterval: parseInt(process.env.MONITORING_ALERT_CHECK_INTERVAL || '10000'),
      dataRetentionDays: parseInt(process.env.MONITORING_DATA_RETENTION_DAYS || '30'),
      maxAlertsPerHour: parseInt(process.env.MONITORING_MAX_ALERTS_PER_HOUR || '100'),
      alertCooldownMinutes: parseInt(process.env.MONITORING_ALERT_COOLDOWN_MINUTES || '15')
    };
  }

  private initializeAlertRules(): void {
    // Cost anomaly detection rule
    this.alertRules.set('cost_anomaly', {
      ruleId: 'cost_anomaly',
      name: 'Cost Anomaly Detection',
      description: 'Detects unusual cost patterns',
      alertType: 'cost_anomaly',
      condition: {
        metric: 'cost',
        operator: 'gt',
        value: this.monitoringConfig.alertThresholds.costAnomalyThreshold,
        timeWindow: 5,
        aggregation: 'avg'
      },
      threshold: this.monitoringConfig.alertThresholds.costAnomalyThreshold,
      severity: 'medium',
      enabled: true,
      cooldownMinutes: 15,
      notificationChannels: ['email', 'slack']
    });

    // Quality degradation rule
    this.alertRules.set('quality_degradation', {
      ruleId: 'quality_degradation',
      name: 'Quality Degradation Detection',
      description: 'Detects quality score degradation',
      alertType: 'quality_degradation',
      condition: {
        metric: 'qualityScore',
        operator: 'lt',
        value: this.monitoringConfig.alertThresholds.qualityDegradationThreshold,
        timeWindow: 10,
        aggregation: 'avg'
      },
      threshold: this.monitoringConfig.alertThresholds.qualityDegradationThreshold,
      severity: 'high',
      enabled: true,
      cooldownMinutes: 10,
      notificationChannels: ['email', 'slack', 'pagerduty']
    });

    // Budget limit rule
    this.alertRules.set('budget_limit', {
      ruleId: 'budget_limit',
      name: 'Budget Limit Warning',
      description: 'Warns when approaching budget limits',
      alertType: 'budget_limit',
      condition: {
        metric: 'budgetUtilization',
        operator: 'gte',
        value: this.monitoringConfig.alertThresholds.budgetLimitThreshold,
        timeWindow: 1,
        aggregation: 'max'
      },
      threshold: this.monitoringConfig.alertThresholds.budgetLimitThreshold,
      severity: 'critical',
      enabled: true,
      cooldownMinutes: 5,
      notificationChannels: ['email', 'slack', 'pagerduty', 'sms']
    });

    // Response time rule
    this.alertRules.set('response_time_slow', {
      ruleId: 'response_time_slow',
      name: 'Slow Response Time',
      description: 'Detects slow response times',
      alertType: 'response_time_slow',
      condition: {
        metric: 'responseTime',
        operator: 'gt',
        value: this.monitoringConfig.alertThresholds.responseTimeThreshold,
        timeWindow: 5,
        aggregation: 'avg'
      },
      threshold: this.monitoringConfig.alertThresholds.responseTimeThreshold,
      severity: 'medium',
      enabled: true,
      cooldownMinutes: 10,
      notificationChannels: ['email', 'slack']
    });

    // Error rate rule
    this.alertRules.set('error_rate_high', {
      ruleId: 'error_rate_high',
      name: 'High Error Rate',
      description: 'Detects high error rates',
      alertType: 'error_rate_high',
      condition: {
        metric: 'errorRate',
        operator: 'gt',
        value: this.monitoringConfig.alertThresholds.errorRateThreshold,
        timeWindow: 5,
        aggregation: 'avg'
      },
      threshold: this.monitoringConfig.alertThresholds.errorRateThreshold,
      severity: 'high',
      enabled: true,
      cooldownMinutes: 5,
      notificationChannels: ['email', 'slack', 'pagerduty']
    });

    // Availability rule
    this.alertRules.set('availability_low', {
      ruleId: 'availability_low',
      name: 'Low Availability',
      description: 'Detects low system availability',
      alertType: 'availability_low',
      condition: {
        metric: 'availability',
        operator: 'lt',
        value: this.monitoringConfig.alertThresholds.availabilityThreshold,
        timeWindow: 5,
        aggregation: 'avg'
      },
      threshold: this.monitoringConfig.alertThresholds.availabilityThreshold,
      severity: 'critical',
      enabled: true,
      cooldownMinutes: 5,
      notificationChannels: ['email', 'slack', 'pagerduty', 'sms']
    });

    // Circuit breaker rule
    this.alertRules.set('circuit_breaker_open', {
      ruleId: 'circuit_breaker_open',
      name: 'Circuit Breaker Open',
      description: 'Detects when circuit breakers are open',
      alertType: 'circuit_breaker_open',
      condition: {
        metric: 'circuitBreakerFailures',
        operator: 'gte',
        value: this.monitoringConfig.alertThresholds.circuitBreakerThreshold,
        timeWindow: 1,
        aggregation: 'count'
      },
      threshold: this.monitoringConfig.alertThresholds.circuitBreakerThreshold,
      severity: 'high',
      enabled: true,
      cooldownMinutes: 5,
      notificationChannels: ['email', 'slack', 'pagerduty']
    });

    // Fallback rate rule
    this.alertRules.set('fallback_rate_high', {
      ruleId: 'fallback_rate_high',
      name: 'High Fallback Rate',
      description: 'Detects high fallback rates',
      alertType: 'fallback_rate_high',
      condition: {
        metric: 'fallbackRate',
        operator: 'gt',
        value: this.monitoringConfig.alertThresholds.fallbackRateThreshold,
        timeWindow: 10,
        aggregation: 'avg'
      },
      threshold: this.monitoringConfig.alertThresholds.fallbackRateThreshold,
      severity: 'medium',
      enabled: true,
      cooldownMinutes: 15,
      notificationChannels: ['email', 'slack']
    });
  }

  private initializeMonitoringThresholds(): void {
    // Initialize monitoring thresholds based on alert rules
    this.alertRules.forEach((rule) => {
      const threshold: MonitoringThreshold = {
        thresholdId: rule.ruleId,
        metric: rule.condition.metric,
        threshold: rule.threshold,
        operator: rule.condition.operator as any, // Operator type mismatch in interface
        severity: rule.severity,
        enabled: rule.enabled,
        cooldownMinutes: rule.cooldownMinutes
      };
      
      this.monitoringThresholds.set(rule.ruleId, threshold);
    });
  }

  // Getters
  public getMonitoringConfig(): MonitoringConfig {
    return { ...this.monitoringConfig };
  }

  public getMonitoringConfiguration(): MonitoringConfiguration {
    return { ...this.monitoringConfiguration };
  }

  public getAlertThresholds(): AlertThresholds {
    return { ...this.monitoringConfig.alertThresholds };
  }

  public getAlertRules(): Map<string, AlertRule> {
    return new Map(this.alertRules);
  }

  public getMonitoringThresholds(): Map<string, MonitoringThreshold> {
    return new Map(this.monitoringThresholds);
  }

  public getAlertRule(ruleId: string): AlertRule | undefined {
    return this.alertRules.get(ruleId);
  }

  public getMonitoringThreshold(thresholdId: string): MonitoringThreshold | undefined {
    return this.monitoringThresholds.get(thresholdId);
  }

  // Update methods
  public updateMonitoringConfig(updates: Partial<MonitoringConfig>): void {
    this.monitoringConfig = { ...this.monitoringConfig, ...updates };
    systemLogger.startup('MonitoringConfigurationManager', 'Updated monitoring configuration');
  }

  public updateAlertThresholds(updates: Partial<AlertThresholds>): void {
    this.monitoringConfig.alertThresholds = { ...this.monitoringConfig.alertThresholds, ...updates };
    this.initializeAlertRules(); // Reinitialize with new thresholds
    systemLogger.startup('MonitoringConfigurationManager', 'Updated alert thresholds');
  }

  public updateMonitoringConfiguration(updates: Partial<MonitoringConfiguration>): void {
    this.monitoringConfiguration = { ...this.monitoringConfiguration, ...updates };
    systemLogger.startup('MonitoringConfigurationManager', 'Updated monitoring configuration settings');
  }

  public updateAlertRule(ruleId: string, updates: Partial<AlertRule>): void {
    const existingRule = this.alertRules.get(ruleId);
    if (existingRule) {
      const updatedRule = { ...existingRule, ...updates };
      this.alertRules.set(ruleId, updatedRule);
      
      // Update corresponding monitoring threshold
      const threshold = this.monitoringThresholds.get(ruleId);
      if (threshold) {
        threshold.threshold = updatedRule.threshold;
        threshold.severity = updatedRule.severity;
        threshold.enabled = updatedRule.enabled;
        threshold.cooldownMinutes = updatedRule.cooldownMinutes;
      }
      
      systemLogger.startup('MonitoringConfigurationManager', `Updated alert rule: ${ruleId}`);
    }
  }

  public enableAlertRule(ruleId: string): boolean {
    const rule = this.alertRules.get(ruleId);
    if (rule) {
      rule.enabled = true;
      const threshold = this.monitoringThresholds.get(ruleId);
      if (threshold) {
        threshold.enabled = true;
      }
      systemLogger.startup('MonitoringConfigurationManager', `Enabled alert rule: ${ruleId}`);
      return true;
    }
    return false;
  }

  public disableAlertRule(ruleId: string): boolean {
    const rule = this.alertRules.get(ruleId);
    if (rule) {
      rule.enabled = false;
      const threshold = this.monitoringThresholds.get(ruleId);
      if (threshold) {
        threshold.enabled = false;
      }
      systemLogger.startup('MonitoringConfigurationManager', `Disabled alert rule: ${ruleId}`);
      return true;
    }
    return false;
  }

  // Validation methods
  public validateConfiguration(): { isValid: boolean; errors: string[] } {
    const errors: string[] = [];

    // Validate monitoring config
    if (this.monitoringConfig.metricsInterval < 1000) {
      errors.push('Metrics interval must be at least 1000ms');
    }

    if (this.monitoringConfig.retentionPeriod < 1) {
      errors.push('Retention period must be at least 1 day');
    }

    // Validate alert thresholds
    if (this.monitoringConfig.alertThresholds.costAnomalyThreshold < 0) {
      errors.push('Cost anomaly threshold must be non-negative');
    }

    if (this.monitoringConfig.alertThresholds.qualityDegradationThreshold < 0 || 
        this.monitoringConfig.alertThresholds.qualityDegradationThreshold > 1) {
      errors.push('Quality degradation threshold must be between 0 and 1');
    }

    if (this.monitoringConfig.alertThresholds.budgetLimitThreshold < 0 || 
        this.monitoringConfig.alertThresholds.budgetLimitThreshold > 1) {
      errors.push('Budget limit threshold must be between 0 and 1');
    }

    if (this.monitoringConfig.alertThresholds.responseTimeThreshold < 0) {
      errors.push('Response time threshold must be non-negative');
    }

    if (this.monitoringConfig.alertThresholds.errorRateThreshold < 0 || 
        this.monitoringConfig.alertThresholds.errorRateThreshold > 1) {
      errors.push('Error rate threshold must be between 0 and 1');
    }

    if (this.monitoringConfig.alertThresholds.availabilityThreshold < 0 || 
        this.monitoringConfig.alertThresholds.availabilityThreshold > 1) {
      errors.push('Availability threshold must be between 0 and 1');
    }

    if (this.monitoringConfig.alertThresholds.circuitBreakerThreshold < 1) {
      errors.push('Circuit breaker threshold must be at least 1');
    }

    if (this.monitoringConfig.alertThresholds.fallbackRateThreshold < 0 || 
        this.monitoringConfig.alertThresholds.fallbackRateThreshold > 1) {
      errors.push('Fallback rate threshold must be between 0 and 1');
    }

    // Validate monitoring configuration
    if (this.monitoringConfiguration.metricsCollectionInterval < 1000) {
      errors.push('Metrics collection interval must be at least 1000ms');
    }

    if (this.monitoringConfiguration.alertCheckInterval < 1000) {
      errors.push('Alert check interval must be at least 1000ms');
    }

    if (this.monitoringConfiguration.dataRetentionDays < 1) {
      errors.push('Data retention days must be at least 1');
    }

    if (this.monitoringConfiguration.maxAlertsPerHour < 1) {
      errors.push('Max alerts per hour must be at least 1');
    }

    if (this.monitoringConfiguration.alertCooldownMinutes < 1) {
      errors.push('Alert cooldown minutes must be at least 1');
    }

    return {
      isValid: errors.length === 0,
      errors
    };
  }

  // Utility methods
  public isMonitoringEnabled(): boolean {
    return this.monitoringConfiguration.enableMonitoring;
  }

  public isAlertingEnabled(): boolean {
    return this.monitoringConfiguration.enableAlerting;
  }

  public isPerformanceTrackingEnabled(): boolean {
    return this.monitoringConfiguration.enablePerformanceTracking;
  }

  public isCostTrackingEnabled(): boolean {
    return this.monitoringConfiguration.enableCostTracking;
  }

  public isQualityTrackingEnabled(): boolean {
    return this.monitoringConfiguration.enableQualityTracking;
  }

  public getMetricsCollectionInterval(): number {
    return this.monitoringConfiguration.metricsCollectionInterval;
  }

  public getAlertCheckInterval(): number {
    return this.monitoringConfiguration.alertCheckInterval;
  }

  public getDataRetentionDays(): number {
    return this.monitoringConfiguration.dataRetentionDays;
  }

  public getMaxAlertsPerHour(): number {
    return this.monitoringConfiguration.maxAlertsPerHour;
  }

  public getAlertCooldownMinutes(): number {
    return this.monitoringConfiguration.alertCooldownMinutes;
  }

  public getEnabledMetrics(): string[] {
    return [...this.monitoringConfig.enabledMetrics];
  }

  public isMetricEnabled(metric: string): boolean {
    return this.monitoringConfig.enabledMetrics.includes(metric);
  }

  public getEnabledAlertRules(): AlertRule[] {
    return Array.from(this.alertRules.values()).filter(rule => rule.enabled);
  }

  public getEnabledMonitoringThresholds(): MonitoringThreshold[] {
    return Array.from(this.monitoringThresholds.values()).filter(threshold => threshold.enabled);
  }
}
