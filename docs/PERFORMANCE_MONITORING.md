# QuantDesk Performance Monitoring Setup

## 📊 Monitoring Overview

This document outlines the performance monitoring setup for QuantDesk contract operations, including metrics collection, alerting, and dashboards.

## 🎯 Key Metrics

### Contract Performance Metrics
- **Transaction Success Rate**: Percentage of successful transactions
- **Transaction Latency**: Time from submission to confirmation
- **Gas Usage**: Compute units consumed per transaction
- **Error Rate**: Frequency of transaction failures
- **Throughput**: Transactions per second

### Oracle Performance Metrics
- **Price Update Frequency**: How often prices are updated
- **Price Staleness**: Age of price data
- **Oracle Availability**: Uptime of oracle feeds
- **Price Deviation**: Deviation from expected prices

### System Performance Metrics
- **API Response Time**: Backend API latency
- **Database Query Time**: Database operation latency
- **WebSocket Connection Count**: Active connections
- **Memory Usage**: System memory consumption
- **CPU Usage**: System CPU utilization

## 🔧 Monitoring Implementation

### Backend Monitoring Service

```typescript
// backend/src/services/monitoringService.ts
import { Logger } from '../utils/logger';
import { SupabaseDatabaseService } from './supabaseDatabase';

export interface PerformanceMetrics {
  timestamp: Date;
  service: string;
  operation: string;
  duration: number;
  success: boolean;
  error?: string;
  metadata?: Record<string, any>;
}

export class MonitoringService {
  private logger: Logger;
  private db: SupabaseDatabaseService;
  private metrics: PerformanceMetrics[] = [];

  constructor() {
    this.logger = new Logger();
    this.db = SupabaseDatabaseService.getInstance();
  }

  // Record performance metrics
  async recordMetric(metric: PerformanceMetrics): Promise<void> {
    try {
      // Store in memory for real-time access
      this.metrics.push(metric);
      
      // Store in database for historical analysis
      await this.db.insert('performance_metrics', {
        timestamp: metric.timestamp.toISOString(),
        service: metric.service,
        operation: metric.operation,
        duration: metric.duration,
        success: metric.success,
        error: metric.error,
        metadata: metric.metadata
      });

      // Log for immediate visibility
      if (metric.success) {
        this.logger.info(`Performance metric recorded`, {
          service: metric.service,
          operation: metric.operation,
          duration: metric.duration
        });
      } else {
        this.logger.error(`Performance metric recorded (failed)`, {
          service: metric.service,
          operation: metric.operation,
          duration: metric.duration,
          error: metric.error
        });
      }
    } catch (error) {
      this.logger.error('Failed to record performance metric', { error });
    }
  }

  // Get recent metrics
  getRecentMetrics(service?: string, limit: number = 100): PerformanceMetrics[] {
    let filtered = this.metrics;
    
    if (service) {
      filtered = filtered.filter(m => m.service === service);
    }
    
    return filtered
      .sort((a, b) => b.timestamp.getTime() - a.timestamp.getTime())
      .slice(0, limit);
  }

  // Calculate success rate
  calculateSuccessRate(service?: string, timeWindow: number = 3600000): number {
    const now = new Date();
    const cutoff = new Date(now.getTime() - timeWindow);
    
    const relevantMetrics = this.metrics.filter(m => {
      const matchesService = !service || m.service === service;
      const withinWindow = m.timestamp >= cutoff;
      return matchesService && withinWindow;
    });

    if (relevantMetrics.length === 0) return 100;

    const successful = relevantMetrics.filter(m => m.success).length;
    return (successful / relevantMetrics.length) * 100;
  }

  // Calculate average latency
  calculateAverageLatency(service?: string, timeWindow: number = 3600000): number {
    const now = new Date();
    const cutoff = new Date(now.getTime() - timeWindow);
    
    const relevantMetrics = this.metrics.filter(m => {
      const matchesService = !service || m.service === service;
      const withinWindow = m.timestamp >= cutoff;
      return matchesService && withinWindow;
    });

    if (relevantMetrics.length === 0) return 0;

    const totalDuration = relevantMetrics.reduce((sum, m) => sum + m.duration, 0);
    return totalDuration / relevantMetrics.length;
  }

  // Get performance summary
  async getPerformanceSummary(): Promise<{
    successRate: number;
    averageLatency: number;
    totalTransactions: number;
    errorRate: number;
  }> {
    const successRate = this.calculateSuccessRate();
    const averageLatency = this.calculateAverageLatency();
    const totalTransactions = this.metrics.length;
    const errorRate = 100 - successRate;

    return {
      successRate,
      averageLatency,
      totalTransactions,
      errorRate
    };
  }
}

export const monitoringService = new MonitoringService();
```

### Contract Operation Monitoring

```typescript
// backend/src/middleware/contractMonitoring.ts
import { Request, Response, NextFunction } from 'express';
import { monitoringService } from '../services/monitoringService';

export const contractMonitoringMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const startTime = Date.now();
  const originalSend = res.send;

  res.send = function(data) {
    const duration = Date.now() - startTime;
    const success = res.statusCode < 400;

    // Record contract operation metrics
    monitoringService.recordMetric({
      timestamp: new Date(),
      service: 'contract',
      operation: `${req.method} ${req.path}`,
      duration,
      success,
      error: success ? undefined : data,
      metadata: {
        statusCode: res.statusCode,
        userAgent: req.get('User-Agent'),
        ip: req.ip
      }
    });

    return originalSend.call(this, data);
  };

  next();
};
```

### Oracle Monitoring

```typescript
// backend/src/services/oracleMonitoringService.ts
import { monitoringService } from './monitoringService';
import { pythOracleService } from './pythOracleService';

export class OracleMonitoringService {
  private lastPriceUpdate: Map<string, number> = new Map();
  private priceUpdateCount: Map<string, number> = new Map();

  // Monitor oracle price updates
  async monitorPriceUpdates(): Promise<void> {
    try {
      const prices = await pythOracleService.getAllPrices();
      const now = Date.now();

      for (const [symbol, price] of Object.entries(prices)) {
        const lastUpdate = this.lastPriceUpdate.get(symbol);
        const updateCount = this.priceUpdateCount.get(symbol) || 0;

        // Record price update frequency
        if (lastUpdate) {
          const timeSinceLastUpdate = now - lastUpdate;
          
          monitoringService.recordMetric({
            timestamp: new Date(),
            service: 'oracle',
            operation: 'price_update',
            duration: timeSinceLastUpdate,
            success: true,
            metadata: {
              symbol,
              price,
              updateCount: updateCount + 1
            }
          });
        }

        this.lastPriceUpdate.set(symbol, now);
        this.priceUpdateCount.set(symbol, updateCount + 1);
      }
    } catch (error) {
      monitoringService.recordMetric({
        timestamp: new Date(),
        service: 'oracle',
        operation: 'price_update',
        duration: 0,
        success: false,
        error: error.message
      });
    }
  }

  // Check for stale prices
  checkStalePrices(): void {
    const now = Date.now();
    const stalenessThreshold = 300000; // 5 minutes

    for (const [symbol, lastUpdate] of this.lastPriceUpdate.entries()) {
      const age = now - lastUpdate;
      
      if (age > stalenessThreshold) {
        monitoringService.recordMetric({
          timestamp: new Date(),
          service: 'oracle',
          operation: 'stale_price_detected',
          duration: age,
          success: false,
          error: `Price for ${symbol} is stale (${age}ms old)`,
          metadata: {
            symbol,
            age,
            threshold: stalenessThreshold
          }
        });
      }
    }
  }

  // Get oracle performance summary
  getOraclePerformanceSummary(): {
    totalUpdates: number;
    averageUpdateInterval: number;
    stalePrices: string[];
  } {
    const totalUpdates = Array.from(this.priceUpdateCount.values()).reduce((sum, count) => sum + count, 0);
    const now = Date.now();
    const stalePrices: string[] = [];

    for (const [symbol, lastUpdate] of this.lastPriceUpdate.entries()) {
      const age = now - lastUpdate;
      if (age > 300000) { // 5 minutes
        stalePrices.push(symbol);
      }
    }

    // Calculate average update interval
    const intervals: number[] = [];
    for (const [symbol, lastUpdate] of this.lastPriceUpdate.entries()) {
      const count = this.priceUpdateCount.get(symbol) || 1;
      intervals.push((now - lastUpdate) / count);
    }

    const averageUpdateInterval = intervals.length > 0 
      ? intervals.reduce((sum, interval) => sum + interval, 0) / intervals.length 
      : 0;

    return {
      totalUpdates,
      averageUpdateInterval,
      stalePrices
    };
  }
}

export const oracleMonitoringService = new OracleMonitoringService();
```

### Performance Dashboard API

```typescript
// backend/src/routes/monitoring.ts
import { Router } from 'express';
import { monitoringService } from '../services/monitoringService';
import { oracleMonitoringService } from '../services/oracleMonitoringService';

const router = Router();

// Get performance metrics
router.get('/metrics', async (req, res) => {
  try {
    const summary = await monitoringService.getPerformanceSummary();
    const recentMetrics = monitoringService.getRecentMetrics(undefined, 50);
    
    res.json({
      success: true,
      data: {
        summary,
        recentMetrics
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get oracle performance
router.get('/oracle', async (req, res) => {
  try {
    const oracleSummary = oracleMonitoringService.getOraclePerformanceSummary();
    
    res.json({
      success: true,
      data: oracleSummary
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get service-specific metrics
router.get('/metrics/:service', async (req, res) => {
  try {
    const { service } = req.params;
    const limit = parseInt(req.query.limit as string) || 100;
    
    const metrics = monitoringService.getRecentMetrics(service, limit);
    const successRate = monitoringService.calculateSuccessRate(service);
    const averageLatency = monitoringService.calculateAverageLatency(service);
    
    res.json({
      success: true,
      data: {
        service,
        metrics,
        successRate,
        averageLatency
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

export default router;
```

## 🚨 Alerting System

### Alert Configuration

```typescript
// backend/src/services/alertingService.ts
import { Logger } from '../utils/logger';
import { monitoringService } from './monitoringService';

export interface AlertRule {
  id: string;
  name: string;
  condition: (metrics: any) => boolean;
  severity: 'low' | 'medium' | 'high' | 'critical';
  cooldown: number; // milliseconds
}

export class AlertingService {
  private logger: Logger;
  private alertRules: AlertRule[] = [];
  private lastAlertTime: Map<string, number> = new Map();

  constructor() {
    this.logger = new Logger();
    this.setupDefaultRules();
  }

  private setupDefaultRules(): void {
    // High error rate alert
    this.addRule({
      id: 'high_error_rate',
      name: 'High Error Rate',
      condition: (metrics) => metrics.errorRate > 10,
      severity: 'high',
      cooldown: 300000 // 5 minutes
    });

    // High latency alert
    this.addRule({
      id: 'high_latency',
      name: 'High Latency',
      condition: (metrics) => metrics.averageLatency > 5000,
      severity: 'medium',
      cooldown: 600000 // 10 minutes
    });

    // Low success rate alert
    this.addRule({
      id: 'low_success_rate',
      name: 'Low Success Rate',
      condition: (metrics) => metrics.successRate < 95,
      severity: 'critical',
      cooldown: 180000 // 3 minutes
    });
  }

  addRule(rule: AlertRule): void {
    this.alertRules.push(rule);
  }

  async checkAlerts(): Promise<void> {
    try {
      const summary = await monitoringService.getPerformanceSummary();
      
      for (const rule of this.alertRules) {
        if (rule.condition(summary)) {
          await this.triggerAlert(rule, summary);
        }
      }
    } catch (error) {
      this.logger.error('Failed to check alerts', { error });
    }
  }

  private async triggerAlert(rule: AlertRule, metrics: any): Promise<void> {
    const now = Date.now();
    const lastAlert = this.lastAlertTime.get(rule.id) || 0;
    
    if (now - lastAlert < rule.cooldown) {
      return; // Still in cooldown period
    }

    this.lastAlertTime.set(rule.id, now);

    const alertMessage = {
      rule: rule.name,
      severity: rule.severity,
      metrics,
      timestamp: new Date().toISOString()
    };

    // Log alert
    this.logger.error(`ALERT: ${rule.name}`, alertMessage);

    // Send to external monitoring service (e.g., PagerDuty, Slack)
    await this.sendExternalAlert(alertMessage);
  }

  private async sendExternalAlert(alert: any): Promise<void> {
    // Implement external alerting (Slack, email, etc.)
    console.log('External alert:', alert);
  }
}

export const alertingService = new AlertingService();
```

## 📊 Monitoring Dashboard

### Frontend Monitoring Component

```typescript
// frontend/src/components/MonitoringDashboard.tsx
import React, { useState, useEffect } from 'react';
import { Card, CardContent, CardHeader, CardTitle } from './ui/card';
import { Badge } from './ui/badge';

interface PerformanceMetrics {
  successRate: number;
  averageLatency: number;
  totalTransactions: number;
  errorRate: number;
}

interface OracleMetrics {
  totalUpdates: number;
  averageUpdateInterval: number;
  stalePrices: string[];
}

export const MonitoringDashboard: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetrics | null>(null);
  const [oracleMetrics, setOracleMetrics] = useState<OracleMetrics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetchMetrics = async () => {
      try {
        const [metricsResponse, oracleResponse] = await Promise.all([
          fetch('/api/monitoring/metrics'),
          fetch('/api/monitoring/oracle')
        ]);

        const metricsData = await metricsResponse.json();
        const oracleData = await oracleResponse.json();

        if (metricsData.success) {
          setMetrics(metricsData.data.summary);
        }

        if (oracleData.success) {
          setOracleMetrics(oracleData.data);
        }
      } catch (error) {
        console.error('Failed to fetch metrics:', error);
      } finally {
        setLoading(false);
      }
    };

    fetchMetrics();
    const interval = setInterval(fetchMetrics, 30000); // Update every 30 seconds

    return () => clearInterval(interval);
  }, []);

  if (loading) {
    return <div>Loading monitoring data...</div>;
  }

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
      {/* Success Rate */}
      <Card>
        <CardHeader>
          <CardTitle>Success Rate</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {metrics?.successRate.toFixed(2)}%
          </div>
          <Badge 
            variant={metrics?.successRate >= 95 ? 'default' : 'destructive'}
          >
            {metrics?.successRate >= 95 ? 'Healthy' : 'Warning'}
          </Badge>
        </CardContent>
      </Card>

      {/* Average Latency */}
      <Card>
        <CardHeader>
          <CardTitle>Average Latency</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {metrics?.averageLatency.toFixed(0)}ms
          </div>
          <Badge 
            variant={metrics?.averageLatency < 5000 ? 'default' : 'destructive'}
          >
            {metrics?.averageLatency < 5000 ? 'Good' : 'Slow'}
          </Badge>
        </CardContent>
      </Card>

      {/* Total Transactions */}
      <Card>
        <CardHeader>
          <CardTitle>Total Transactions</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {metrics?.totalTransactions.toLocaleString()}
          </div>
          <p className="text-sm text-muted-foreground">
            All time
          </p>
        </CardContent>
      </Card>

      {/* Error Rate */}
      <Card>
        <CardHeader>
          <CardTitle>Error Rate</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="text-2xl font-bold">
            {metrics?.errorRate.toFixed(2)}%
          </div>
          <Badge 
            variant={metrics?.errorRate < 5 ? 'default' : 'destructive'}
          >
            {metrics?.errorRate < 5 ? 'Low' : 'High'}
          </Badge>
        </CardContent>
      </Card>

      {/* Oracle Metrics */}
      <Card className="md:col-span-2">
        <CardHeader>
          <CardTitle>Oracle Performance</CardTitle>
        </CardHeader>
        <CardContent>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <div className="text-lg font-semibold">
                {oracleMetrics?.totalUpdates.toLocaleString()}
              </div>
              <p className="text-sm text-muted-foreground">Total Updates</p>
            </div>
            <div>
              <div className="text-lg font-semibold">
                {oracleMetrics?.averageUpdateInterval.toFixed(0)}ms
              </div>
              <p className="text-sm text-muted-foreground">Avg Interval</p>
            </div>
          </div>
          {oracleMetrics?.stalePrices.length > 0 && (
            <div className="mt-4">
              <p className="text-sm font-semibold text-red-600">
                Stale Prices: {oracleMetrics.stalePrices.join(', ')}
              </p>
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  );
};
```

## 🔄 Automated Monitoring

### Scheduled Monitoring Tasks

```typescript
// backend/src/services/scheduledMonitoring.ts
import { CronJob } from 'cron';
import { monitoringService } from './monitoringService';
import { oracleMonitoringService } from './oracleMonitoringService';
import { alertingService } from './alertingService';

export class ScheduledMonitoringService {
  private jobs: CronJob[] = [];

  start(): void {
    // Check alerts every minute
    this.jobs.push(new CronJob('0 * * * * *', async () => {
      await alertingService.checkAlerts();
    }));

    // Monitor oracle updates every 30 seconds
    this.jobs.push(new CronJob('0,30 * * * * *', async () => {
      await oracleMonitoringService.monitorPriceUpdates();
    }));

    // Check for stale prices every 5 minutes
    this.jobs.push(new CronJob('0 */5 * * * *', async () => {
      oracleMonitoringService.checkStalePrices();
    }));

    // Start all jobs
    this.jobs.forEach(job => job.start());
  }

  stop(): void {
    this.jobs.forEach(job => job.stop());
  }
}

export const scheduledMonitoringService = new ScheduledMonitoringService();
```

## 📈 Performance Benchmarks

### Target Performance Metrics
- **Success Rate**: > 99%
- **Average Latency**: < 2 seconds
- **Error Rate**: < 1%
- **Oracle Update Frequency**: < 1 second
- **Price Staleness**: < 5 minutes
- **API Response Time**: < 500ms
- **Database Query Time**: < 100ms

### Performance Testing

```typescript
// backend/src/tests/performanceTests.ts
import { describe, it, expect } from 'vitest';
import { monitoringService } from '../services/monitoringService';

describe('Performance Monitoring', () => {
  it('should record metrics correctly', async () => {
    const metric = {
      timestamp: new Date(),
      service: 'test',
      operation: 'test_operation',
      duration: 100,
      success: true
    };

    await monitoringService.recordMetric(metric);
    const recentMetrics = monitoringService.getRecentMetrics('test', 1);
    
    expect(recentMetrics).toHaveLength(1);
    expect(recentMetrics[0].service).toBe('test');
  });

  it('should calculate success rate correctly', () => {
    const successRate = monitoringService.calculateSuccessRate();
    expect(successRate).toBeGreaterThanOrEqual(0);
    expect(successRate).toBeLessThanOrEqual(100);
  });

  it('should calculate average latency correctly', () => {
    const averageLatency = monitoringService.calculateAverageLatency();
    expect(averageLatency).toBeGreaterThanOrEqual(0);
  });
});
```

## 🚀 Deployment

### Environment Variables
```bash
# Monitoring Configuration
MONITORING_ENABLED=true
MONITORING_RETENTION_DAYS=30
ALERTING_WEBHOOK_URL=https://hooks.slack.com/services/...
ALERTING_EMAIL=alerts@quantdesk.com

# Performance Thresholds
SUCCESS_RATE_THRESHOLD=95
LATENCY_THRESHOLD=5000
ERROR_RATE_THRESHOLD=5
```

### Database Schema
```sql
-- Performance metrics table
CREATE TABLE performance_metrics (
  id SERIAL PRIMARY KEY,
  timestamp TIMESTAMP WITH TIME ZONE NOT NULL,
  service VARCHAR(100) NOT NULL,
  operation VARCHAR(200) NOT NULL,
  duration INTEGER NOT NULL,
  success BOOLEAN NOT NULL,
  error TEXT,
  metadata JSONB,
  created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_performance_metrics_timestamp ON performance_metrics(timestamp);
CREATE INDEX idx_performance_metrics_service ON performance_metrics(service);
CREATE INDEX idx_performance_metrics_success ON performance_metrics(success);
```

---

This monitoring setup provides comprehensive visibility into QuantDesk's performance, enabling proactive issue detection and resolution.
