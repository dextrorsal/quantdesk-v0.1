import { Logger } from '../utils/logger';

const logger = new Logger();

interface PerformanceMetrics {
  requestCount: number;
  averageResponseTime: number;
  slowRequests: number;
  errorRate: number;
  memoryUsage: NodeJS.MemoryUsage;
  cpuUsage: number;
  timestamp: number;
}

interface RequestMetrics {
  method: string;
  path: string;
  responseTime: number;
  statusCode: number;
  timestamp: number;
  memoryBefore: NodeJS.MemoryUsage;
  memoryAfter: NodeJS.MemoryUsage;
}

class PerformanceMonitoringService {
  private static instance: PerformanceMonitoringService;
  private metrics: PerformanceMetrics;
  private requestHistory: RequestMetrics[] = [];
  private slowRequestThreshold = 200; // ms
  private maxHistorySize = 1000;

  private constructor() {
    this.metrics = {
      requestCount: 0,
      averageResponseTime: 0,
      slowRequests: 0,
      errorRate: 0,
      memoryUsage: process.memoryUsage(),
      cpuUsage: 0,
      timestamp: Date.now()
    };
  }

  public static getInstance(): PerformanceMonitoringService {
    if (!PerformanceMonitoringService.instance) {
      PerformanceMonitoringService.instance = new PerformanceMonitoringService();
    }
    return PerformanceMonitoringService.instance;
  }

  /**
   * Record a request for performance monitoring
   */
  public recordRequest(
    method: string,
    path: string,
    responseTime: number,
    statusCode: number,
    memoryBefore: NodeJS.MemoryUsage,
    memoryAfter: NodeJS.MemoryUsage
  ): void {
    const requestMetric: RequestMetrics = {
      method,
      path,
      responseTime,
      statusCode,
      timestamp: Date.now(),
      memoryBefore,
      memoryAfter
    };

    // Add to history
    this.requestHistory.push(requestMetric);
    
    // Maintain history size
    if (this.requestHistory.length > this.maxHistorySize) {
      this.requestHistory = this.requestHistory.slice(-this.maxHistorySize);
    }

    // Update metrics
    this.updateMetrics(requestMetric);

    // Log slow requests
    if (responseTime > this.slowRequestThreshold) {
      logger.warn(`Slow request detected: ${method} ${path} - ${responseTime}ms`);
    }

    // Log high memory usage
    const memoryIncrease = memoryAfter.heapUsed - memoryBefore.heapUsed;
    if (memoryIncrease > 10 * 1024 * 1024) { // 10MB increase
      logger.warn(`High memory usage detected: ${method} ${path} - ${memoryIncrease / 1024 / 1024}MB increase`);
    }
  }

  /**
   * Update performance metrics
   */
  private updateMetrics(requestMetric: RequestMetrics): void {
    this.metrics.requestCount++;
    
    // Calculate average response time
    const totalTime = this.metrics.averageResponseTime * (this.metrics.requestCount - 1) + requestMetric.responseTime;
    this.metrics.averageResponseTime = totalTime / this.metrics.requestCount;

    // Count slow requests
    if (requestMetric.responseTime > this.slowRequestThreshold) {
      this.metrics.slowRequests++;
    }

    // Calculate error rate
    const errorCount = this.requestHistory.filter(r => r.statusCode >= 400).length;
    this.metrics.errorRate = (errorCount / this.metrics.requestCount) * 100;

    // Update memory usage
    this.metrics.memoryUsage = requestMetric.memoryAfter;
    this.metrics.timestamp = Date.now();
  }

  /**
   * Get current performance metrics
   */
  public getMetrics(): PerformanceMetrics {
    return { ...this.metrics };
  }

  /**
   * Get slow requests analysis
   */
  public getSlowRequestsAnalysis(): {
    slowestEndpoints: Array<{ path: string; avgTime: number; count: number }>;
    memoryLeaks: Array<{ path: string; avgMemoryIncrease: number; count: number }>;
  } {
    const endpointStats = new Map<string, { totalTime: number; count: number; memoryIncrease: number }>();

    this.requestHistory.forEach(request => {
      const key = `${request.method} ${request.path}`;
      const existing = endpointStats.get(key) || { totalTime: 0, count: 0, memoryIncrease: 0 };
      
      existing.totalTime += request.responseTime;
      existing.count++;
      existing.memoryIncrease += request.memoryAfter.heapUsed - request.memoryBefore.heapUsed;
      
      endpointStats.set(key, existing);
    });

    // Find slowest endpoints
    const slowestEndpoints = Array.from(endpointStats.entries())
      .map(([path, stats]) => ({
        path,
        avgTime: stats.totalTime / stats.count,
        count: stats.count
      }))
      .sort((a, b) => b.avgTime - a.avgTime)
      .slice(0, 10);

    // Find potential memory leaks
    const memoryLeaks = Array.from(endpointStats.entries())
      .map(([path, stats]) => ({
        path,
        avgMemoryIncrease: stats.memoryIncrease / stats.count,
        count: stats.count
      }))
      .filter(item => item.avgMemoryIncrease > 1024 * 1024) // > 1MB average increase
      .sort((a, b) => b.avgMemoryIncrease - a.avgMemoryIncrease)
      .slice(0, 10);

    return { slowestEndpoints, memoryLeaks };
  }

  /**
   * Get performance recommendations
   */
  public getPerformanceRecommendations(): string[] {
    const recommendations: string[] = [];
    const analysis = this.getSlowRequestsAnalysis();

    // Check for slow endpoints
    if (analysis.slowestEndpoints.length > 0) {
      const slowest = analysis.slowestEndpoints[0];
      if (slowest.avgTime > 500) {
        recommendations.push(`Consider optimizing ${slowest.path} - average response time: ${slowest.avgTime.toFixed(2)}ms`);
      }
    }

    // Check for memory leaks
    if (analysis.memoryLeaks.length > 0) {
      const worstLeak = analysis.memoryLeaks[0];
      recommendations.push(`Potential memory leak in ${worstLeak.path} - average memory increase: ${(worstLeak.avgMemoryIncrease / 1024 / 1024).toFixed(2)}MB`);
    }

    // Check error rate
    if (this.metrics.errorRate > 5) {
      recommendations.push(`High error rate detected: ${this.metrics.errorRate.toFixed(2)}% - investigate error handling`);
    }

    // Check memory usage
    const memoryUsageMB = this.metrics.memoryUsage.heapUsed / 1024 / 1024;
    if (memoryUsageMB > 500) {
      recommendations.push(`High memory usage: ${memoryUsageMB.toFixed(2)}MB - consider memory optimization`);
    }

    return recommendations;
  }

  /**
   * Reset metrics
   */
  public resetMetrics(): void {
    this.metrics = {
      requestCount: 0,
      averageResponseTime: 0,
      slowRequests: 0,
      errorRate: 0,
      memoryUsage: process.memoryUsage(),
      cpuUsage: 0,
      timestamp: Date.now()
    };
    this.requestHistory = [];
  }
}

export const performanceMonitoringService = PerformanceMonitoringService.getInstance();