import { EventEmitter } from 'events';
import { Logger } from '../utils/logger';

export interface SystemMetrics {
  tps: number;                    // Transactions per second
  latency_p99: number;            // 99th percentile latency (ms)
  error_rate: number;             // Error rate in basis points
  memory_usage: number;           // Memory usage in MB
  cpu_usage: number;              // CPU usage percentage
  active_connections: number;     // Active WebSocket connections
  queue_size: number;             // Message queue size
  last_updated: Date;             // Last metrics update
}

export interface AlertThresholds {
  max_tps: number;
  max_latency_p99: number;
  max_error_rate: number;
  max_memory_usage: number;
  max_cpu_usage: number;
  max_queue_size: number;
}

export class SystemMonitor extends EventEmitter {
  private logger: Logger;
  private metrics: SystemMetrics;
  private thresholds: AlertThresholds;
  private isMonitoring: boolean = false;
  private monitoringInterval?: NodeJS.Timeout;
  private latencyHistory: number[] = [];
  private errorCount: number = 0;
  private requestCount: number = 0;
  private lastResetTime: Date = new Date();

  constructor(logger: Logger) {
    super();
    this.logger = logger;
    this.metrics = this.initializeMetrics();
    this.thresholds = this.initializeThresholds();
  }

  private initializeMetrics(): SystemMetrics {
    return {
      tps: 0,
      latency_p99: 0,
      error_rate: 0,
      memory_usage: 0,
      cpu_usage: 0,
      active_connections: 0,
      queue_size: 0,
      last_updated: new Date(),
    };
  }

  private initializeThresholds(): AlertThresholds {
    return {
      max_tps: 1000,              // Max 1000 TPS
      max_latency_p99: 1000,      // Max 1 second latency
      max_error_rate: 100,        // Max 1% error rate
      max_memory_usage: 1024,     // Max 1GB memory usage
      max_cpu_usage: 80,          // Max 80% CPU usage
      max_queue_size: 1000,       // Max 1000 queued messages
    };
  }

  public startMonitoring(intervalMs: number = 5000): void {
    if (this.isMonitoring) {
      this.logger.warn('System monitoring is already running');
      return;
    }

    this.isMonitoring = true;
    this.monitoringInterval = setInterval(() => {
      this.updateMetrics();
      this.checkThresholds();
    }, intervalMs);

    this.logger.info(`System monitoring started with ${intervalMs}ms interval`);
  }

  public stopMonitoring(): void {
    if (!this.isMonitoring) {
      return;
    }

    this.isMonitoring = false;
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = undefined;
    }

    this.logger.info('System monitoring stopped');
  }

  public recordRequest(latencyMs: number, isError: boolean = false): void {
    this.requestCount++;
    this.latencyHistory.push(latencyMs);
    
    if (isError) {
      this.errorCount++;
    }

    // Keep only last 1000 requests for latency calculation
    if (this.latencyHistory.length > 1000) {
      this.latencyHistory = this.latencyHistory.slice(-1000);
    }
  }

  public updateActiveConnections(count: number): void {
    this.metrics.active_connections = count;
  }

  public updateQueueSize(size: number): void {
    this.metrics.queue_size = size;
  }

  private updateMetrics(): void {
    const now = new Date();
    const timeDiff = (now.getTime() - this.lastResetTime.getTime()) / 1000; // seconds

    // Calculate TPS
    this.metrics.tps = timeDiff > 0 ? this.requestCount / timeDiff : 0;

    // Calculate error rate (in basis points)
    this.metrics.error_rate = this.requestCount > 0 
      ? (this.errorCount / this.requestCount) * 10000 
      : 0;

    // Calculate 99th percentile latency
    if (this.latencyHistory.length > 0) {
      const sortedLatencies = [...this.latencyHistory].sort((a, b) => a - b);
      const p99Index = Math.floor(sortedLatencies.length * 0.99);
      this.metrics.latency_p99 = sortedLatencies[p99Index] || 0;
    }

    // Get system metrics
    const memUsage = process.memoryUsage();
    this.metrics.memory_usage = Math.round(memUsage.heapUsed / 1024 / 1024); // MB

    // CPU usage calculation (simplified)
    const cpuUsage = process.cpuUsage();
    this.metrics.cpu_usage = Math.min(100, Math.round(cpuUsage.user / 1000000)); // Rough estimate

    this.metrics.last_updated = now;

    // Reset counters
    this.requestCount = 0;
    this.errorCount = 0;
    this.lastResetTime = now;

    // Emit metrics update event
    this.emit('metricsUpdate', this.metrics);
  }

  private checkThresholds(): void {
    const alerts: string[] = [];

    if (this.metrics.tps > this.thresholds.max_tps) {
      alerts.push(`High TPS: ${this.metrics.tps.toFixed(2)} (threshold: ${this.thresholds.max_tps})`);
    }

    if (this.metrics.latency_p99 > this.thresholds.max_latency_p99) {
      alerts.push(`High latency: ${this.metrics.latency_p99}ms (threshold: ${this.thresholds.max_latency_p99}ms)`);
    }

    if (this.metrics.error_rate > this.thresholds.max_error_rate) {
      alerts.push(`High error rate: ${this.metrics.error_rate.toFixed(2)}bps (threshold: ${this.thresholds.max_error_rate}bps)`);
    }

    if (this.metrics.memory_usage > this.thresholds.max_memory_usage) {
      alerts.push(`High memory usage: ${this.metrics.memory_usage}MB (threshold: ${this.thresholds.max_memory_usage}MB)`);
    }

    if (this.metrics.cpu_usage > this.thresholds.max_cpu_usage) {
      alerts.push(`High CPU usage: ${this.metrics.cpu_usage}% (threshold: ${this.thresholds.max_cpu_usage}%)`);
    }

    if (this.metrics.queue_size > this.thresholds.max_queue_size) {
      alerts.push(`Large queue size: ${this.metrics.queue_size} (threshold: ${this.thresholds.max_queue_size})`);
    }

    if (alerts.length > 0) {
      const alertMessage = `System alerts: ${alerts.join(', ')}`;
      this.logger.error(alertMessage);
      this.emit('alert', {
        type: 'system_threshold_exceeded',
        message: alertMessage,
        metrics: this.metrics,
        timestamp: new Date(),
      });
    }
  }

  public getMetrics(): SystemMetrics {
    return { ...this.metrics };
  }

  public getThresholds(): AlertThresholds {
    return { ...this.thresholds };
  }

  public updateThresholds(newThresholds: Partial<AlertThresholds>): void {
    this.thresholds = { ...this.thresholds, ...newThresholds };
    this.logger.info('Monitoring thresholds updated', { thresholds: this.thresholds });
  }

  public getHealthStatus(): 'healthy' | 'warning' | 'critical' {
    const criticalCount = [
      this.metrics.tps > this.thresholds.max_tps,
      this.metrics.latency_p99 > this.thresholds.max_latency_p99,
      this.metrics.error_rate > this.thresholds.max_error_rate,
      this.metrics.memory_usage > this.thresholds.max_memory_usage,
      this.metrics.cpu_usage > this.thresholds.max_cpu_usage,
    ].filter(Boolean).length;

    if (criticalCount >= 3) return 'critical';
    if (criticalCount >= 1) return 'warning';
    return 'healthy';
  }
}

// Singleton instance
let systemMonitor: SystemMonitor | null = null;

export function getSystemMonitor(logger?: Logger): SystemMonitor {
  if (!systemMonitor) {
    if (!logger) {
      throw new Error('Logger is required for first SystemMonitor initialization');
    }
    systemMonitor = new SystemMonitor(logger);
  }
  return systemMonitor;
}
