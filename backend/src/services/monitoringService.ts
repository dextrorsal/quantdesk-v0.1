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
