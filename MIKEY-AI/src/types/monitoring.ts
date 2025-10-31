/**
 * Monitoring Types
 * Type definitions for cost optimization monitoring and alerting
 */

export interface MonitoringConfig {
  metricsInterval: number;
  alertThresholds: AlertThresholds;
  retentionPeriod: number;
  enabledMetrics: string[];
  enableRealTimeMonitoring: boolean;
  enablePerformanceTracking: boolean;
  enableCostTracking: boolean;
  enableQualityTracking: boolean;
}

export interface AlertThresholds {
  costAnomalyThreshold: number;
  qualityDegradationThreshold: number;
  budgetLimitThreshold: number;
  responseTimeThreshold: number;
  errorRateThreshold: number;
  availabilityThreshold: number;
  circuitBreakerThreshold: number;
  fallbackRateThreshold: number;
}

export interface MonitoringMetrics {
  timestamp: Date;
  requestId: string;
  provider: string;
  cost: number;
  tokensUsed: number;
  responseTime: number;
  qualityScore: number;
  success: boolean;
  errorType?: string;
  fallbackUsed: boolean;
  escalationUsed: boolean;
  taskType: string;
  sessionId?: string;
}

export interface PerformanceMetrics {
  timestamp: Date;
  routingDecisionTime: number;
  providerSelectionTime: number;
  tokenEstimationTime: number;
  qualityEvaluationTime: number;
  fallbackDecisionTime: number;
  totalRequestTime: number;
  memoryUsage: number;
  cpuUsage: number;
}

export interface CostMetrics {
  timestamp: Date;
  totalCost: number;
  averageCost: number;
  costPerToken: number;
  costSavings: number;
  providerCostBreakdown: Record<string, number>;
  dailyCost: number;
  monthlyCost: number;
  budgetUtilization: number;
}

export interface QualityMetrics {
  timestamp: Date;
  averageQualityScore: number;
  qualityDegradationRate: number;
  escalationRate: number;
  userSatisfactionScore: number;
  providerQualityBreakdown: Record<string, number>;
  qualityThresholdViolations: number;
}

export interface Alert {
  alertId: string;
  alertType: AlertType;
  severity: AlertSeverity;
  message: string;
  timestamp: Date;
  resolved: boolean;
  resolvedAt?: Date;
  metadata: Record<string, any>;
  provider?: string;
  threshold?: number;
  actualValue?: number;
}

export type AlertType = 
  | 'cost_anomaly'
  | 'quality_degradation'
  | 'budget_limit'
  | 'response_time_slow'
  | 'error_rate_high'
  | 'availability_low'
  | 'circuit_breaker_open'
  | 'fallback_rate_high'
  | 'performance_degradation'
  | 'memory_usage_high'
  | 'cpu_usage_high';

export type AlertSeverity = 'low' | 'medium' | 'high' | 'critical';

export interface MonitoringStatus {
  isHealthy: boolean;
  lastUpdate: Date;
  activeAlerts: number;
  criticalAlerts: number;
  metricsCollected: number;
  performanceScore: number;
  uptime: number;
  systemLoad: number;
}

export interface MonitoringDashboard {
  status: MonitoringStatus;
  recentMetrics: MonitoringMetrics[];
  activeAlerts: Alert[];
  performanceTrends: PerformanceMetrics[];
  costTrends: CostMetrics[];
  qualityTrends: QualityMetrics[];
  providerStats: Record<string, ProviderStats>;
}

export interface ProviderStats {
  provider: string;
  totalRequests: number;
  successRate: number;
  averageResponseTime: number;
  averageCost: number;
  averageQualityScore: number;
  circuitBreakerState: string;
  lastHealthCheck: Date;
  errorRate: number;
  fallbackCount: number;
}

export interface MonitoringConfiguration {
  enableMonitoring: boolean;
  enableAlerting: boolean;
  enablePerformanceTracking: boolean;
  enableCostTracking: boolean;
  enableQualityTracking: boolean;
  metricsCollectionInterval: number;
  alertCheckInterval: number;
  dataRetentionDays: number;
  maxAlertsPerHour: number;
  alertCooldownMinutes: number;
}

export interface AlertRule {
  ruleId: string;
  name: string;
  description: string;
  alertType: AlertType;
  condition: AlertCondition;
  threshold: number;
  severity: AlertSeverity;
  enabled: boolean;
  cooldownMinutes: number;
  notificationChannels: string[];
}

export interface AlertCondition {
  metric: string;
  operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte' | 'ne';
  value: number;
  timeWindow: number; // in minutes
  aggregation: 'avg' | 'sum' | 'max' | 'min' | 'count';
}

export interface MonitoringReport {
  reportId: string;
  reportType: 'daily' | 'weekly' | 'monthly';
  generatedAt: Date;
  timeRange: {
    start: Date;
    end: Date;
  };
  summary: {
    totalRequests: number;
    totalCost: number;
    averageQualityScore: number;
    averageResponseTime: number;
    errorRate: number;
    availability: number;
  };
  trends: {
    costTrend: 'increasing' | 'decreasing' | 'stable';
    qualityTrend: 'improving' | 'degrading' | 'stable';
    performanceTrend: 'improving' | 'degrading' | 'stable';
  };
  alerts: Alert[];
  recommendations: string[];
}

export interface MonitoringEvent {
  eventId: string;
  eventType: 'metric_collected' | 'alert_triggered' | 'alert_resolved' | 'threshold_exceeded' | 'performance_degradation';
  timestamp: Date;
  provider?: string;
  metadata: Record<string, any>;
  severity: AlertSeverity;
}

export interface MonitoringHealthCheck {
  checkId: string;
  timestamp: Date;
  component: string;
  status: 'healthy' | 'degraded' | 'unhealthy';
  responseTime: number;
  error?: string;
  metadata: Record<string, any>;
}

export interface MonitoringThreshold {
  thresholdId: string;
  metric: string;
  threshold: number;
  operator: 'gt' | 'lt' | 'eq' | 'gte' | 'lte';
  severity: AlertSeverity;
  enabled: boolean;
  cooldownMinutes: number;
  lastTriggered?: Date;
}

export interface MonitoringStats {
  totalMetricsCollected: number;
  totalAlertsGenerated: number;
  totalAlertsResolved: number;
  averageResponseTime: number;
  averageCost: number;
  averageQualityScore: number;
  systemUptime: number;
  lastHealthCheck: Date;
  activeProviders: number;
  healthyProviders: number;
}
