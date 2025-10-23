import { Logger } from '../utils/logger';
import { getSupabaseService } from './supabaseService';

const logger = new Logger();

export interface PerformanceTargets {
  orderPlacement: number; // ms
  orderExecution: number; // ms
  oracleCall: number; // ms
  databaseQuery: number; // ms
  websocketMessage: number; // ms
}

export interface PerformanceMetrics {
  operation: string;
  duration: number;
  timestamp: number;
  userId?: string;
  success: boolean;
  error?: string;
}

export interface PerformanceReport {
  operation: string;
  averageDuration: number;
  p95Duration: number;
  p99Duration: number;
  successRate: number;
  totalRequests: number;
  targetMet: boolean;
  violations: number;
}

export class PerformanceMonitoringService {
  private static instance: PerformanceMonitoringService;
  private readonly supabase: ReturnType<typeof getSupabaseService>;
  
  // Performance targets as defined in NFR assessment
  private readonly targets: PerformanceTargets = {
    orderPlacement: 200, // <200ms order placement
    orderExecution: 2000, // <2s execution
    oracleCall: 100, // <100ms Oracle calls
    databaseQuery: 50, // <50ms database queries
    websocketMessage: 10 // <10ms WebSocket messages
  };

  private constructor() {
    this.supabase = getSupabaseService();
  }

  public static getInstance(): PerformanceMonitoringService {
    if (!PerformanceMonitoringService.instance) {
      PerformanceMonitoringService.instance = new PerformanceMonitoringService();
    }
    return PerformanceMonitoringService.instance;
  }

  /**
   * Start performance monitoring for an operation
   */
  public startTimer(operation: string): (success?: boolean, error?: string, userId?: string) => void {
    const startTime = Date.now();
    
    return (success: boolean = true, error?: string, userId?: string) => {
      const duration = Date.now() - startTime;
      this.recordMetric({
        operation,
        duration,
        timestamp: Date.now(),
        userId,
        success,
        error
      });
    };
  }

  /**
   * Record a performance metric
   */
  public async recordMetric(metric: PerformanceMetrics): Promise<void> {
    try {
      // Log to database for analysis
      await this.supabase.getClient()
        .from('performance_metrics')
        .insert({
          operation: metric.operation,
          duration_ms: metric.duration,
          timestamp: new Date(metric.timestamp).toISOString(),
          user_id: metric.userId,
          success: metric.success,
          error_message: metric.error
        });

      // Log to console for immediate feedback
      const target = this.getTargetForOperation(metric.operation);
      const status = metric.duration <= target ? '✅' : '❌';
      
      logger.info(`${status} ${metric.operation}: ${metric.duration}ms (target: ${target}ms)`);

      // Alert if target is significantly exceeded
      if (metric.duration > target * 2) {
        logger.warn(`⚠️ Performance alert: ${metric.operation} took ${metric.duration}ms (${target * 2}ms threshold exceeded)`);
      }

    } catch (error) {
      logger.error('Error recording performance metric:', error);
    }
  }

  /**
   * Get performance report for an operation
   */
  public async getPerformanceReport(
    operation: string, 
    timeWindow: number = 24 * 60 * 60 * 1000 // 24 hours
  ): Promise<PerformanceReport | null> {
    try {
      const since = new Date(Date.now() - timeWindow).toISOString();
      
      const { data, error } = await this.supabase.getClient()
        .from('performance_metrics')
        .select('duration_ms, success')
        .eq('operation', operation)
        .gte('timestamp', since);

      if (error) {
        logger.error('Error getting performance metrics:', error);
        return null;
      }

      if (!data || data.length === 0) {
        return null;
      }

      const durations = data.map(r => r.duration_ms).sort((a, b) => a - b);
      const successes = data.filter(r => r.success).length;
      
      const averageDuration = durations.reduce((sum, d) => sum + d, 0) / durations.length;
      const p95Duration = durations[Math.floor(durations.length * 0.95)];
      const p99Duration = durations[Math.floor(durations.length * 0.99)];
      const successRate = successes / data.length;
      
      const target = this.getTargetForOperation(operation);
      const violations = durations.filter(d => d > target).length;
      const targetMet = averageDuration <= target;

      return {
        operation,
        averageDuration,
        p95Duration,
        p99Duration,
        successRate,
        totalRequests: data.length,
        targetMet,
        violations
      };

    } catch (error) {
      logger.error('Error getting performance report:', error);
      return null;
    }
  }

  /**
   * Get all performance targets
   */
  public getTargets(): PerformanceTargets {
    return { ...this.targets };
  }

  /**
   * Update performance targets
   */
  public updateTargets(newTargets: Partial<PerformanceTargets>): void {
    Object.assign(this.targets, newTargets);
    logger.info('Performance targets updated:', this.targets);
  }

  /**
   * Get target for specific operation
   */
  private getTargetForOperation(operation: string): number {
    const operationMap: Record<string, keyof PerformanceTargets> = {
      'order_placement': 'orderPlacement',
      'order_execution': 'orderExecution',
      'oracle_call': 'oracleCall',
      'database_query': 'databaseQuery',
      'websocket_message': 'websocketMessage'
    };

    const targetKey = operationMap[operation];
    return targetKey ? this.targets[targetKey] : 1000; // Default 1 second
  }

  /**
   * Performance decorator for methods
   */
  public performanceMonitor(operation: string) {
    return (target: any, propertyName: string, descriptor: PropertyDescriptor) => {
      const method = descriptor.value;

      descriptor.value = async function (...args: any[]) {
        const endTimer = PerformanceMonitoringService.getInstance().startTimer(operation);
        
        try {
          const result = await method.apply(this, args);
          endTimer(true);
          return result;
        } catch (error) {
          endTimer(false, error instanceof Error ? error.message : 'Unknown error');
          throw error;
        }
      };

      return descriptor;
    };
  }

  /**
   * Health check for performance monitoring
   */
  public async healthCheck(): Promise<{
    healthy: boolean;
    issues: string[];
    recommendations: string[];
  }> {
    const issues: string[] = [];
    const recommendations: string[] = [];

    try {
      // Check recent performance for each operation
      const operations = Object.keys(this.targets) as Array<keyof PerformanceTargets>;
      
      for (const operation of operations) {
        const report = await this.getPerformanceReport(operation, 60 * 60 * 1000); // Last hour
        
        if (report) {
          if (!report.targetMet) {
            issues.push(`${operation} average duration (${report.averageDuration}ms) exceeds target (${this.targets[operation]}ms)`);
            recommendations.push(`Optimize ${operation} performance`);
          }
          
          if (report.successRate < 0.95) {
            issues.push(`${operation} success rate (${(report.successRate * 100).toFixed(1)}%) is below 95%`);
            recommendations.push(`Investigate ${operation} error rate`);
          }
          
          if (report.violations > report.totalRequests * 0.1) {
            issues.push(`${operation} has high violation rate (${report.violations}/${report.totalRequests})`);
            recommendations.push(`Review ${operation} performance bottlenecks`);
          }
        }
      }

      return {
        healthy: issues.length === 0,
        issues,
        recommendations
      };

    } catch (error) {
      logger.error('Performance health check failed:', error);
      return {
        healthy: false,
        issues: ['Performance monitoring service error'],
        recommendations: ['Check performance monitoring service configuration']
      };
    }
  }
}

// Export singleton instance
export const performanceMonitoringService = PerformanceMonitoringService.getInstance();

// Performance monitoring decorators for common operations
export const monitorOrderPlacement = performanceMonitoringService.performanceMonitor('order_placement');
export const monitorOrderExecution = performanceMonitoringService.performanceMonitor('order_execution');
export const monitorOracleCall = performanceMonitoringService.performanceMonitor('oracle_call');
export const monitorDatabaseQuery = performanceMonitoringService.performanceMonitor('database_query');
export const monitorWebSocketMessage = performanceMonitoringService.performanceMonitor('websocket_message');
