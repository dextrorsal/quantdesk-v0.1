import { Request, Response, NextFunction } from 'express';
import { performanceMonitoringService } from '../services/performanceMonitoringService';
import { Logger } from '../utils/logger';

const logger = new Logger();

/**
 * Performance monitoring middleware
 * Tracks request performance, memory usage, and identifies bottlenecks
 */
export const performanceMonitoringMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const startTime = Date.now();
  const memoryBefore = process.memoryUsage();

  // Override res.end to capture response metrics
  const originalEnd = res.end;
  res.end = function(chunk?: any, encoding?: any): Response {
    const endTime = Date.now();
    const responseTime = endTime - startTime;
    const memoryAfter = process.memoryUsage();

    // Record the request metrics
    performanceMonitoringService.recordRequest(
      req.method,
      req.path,
      responseTime,
      res.statusCode,
      memoryBefore,
      memoryAfter
    );

    // Log performance issues
    if (responseTime > 200) {
      logger.warn(`Slow request: ${req.method} ${req.path} - ${responseTime}ms`);
    }

    // Call original end method
    return originalEnd.call(this, chunk, encoding);
  };

  next();
};

/**
 * Performance analysis endpoint middleware
 * Provides performance insights and recommendations
 */
export const performanceAnalysisMiddleware = (req: Request, res: Response, next: NextFunction) => {
  if (req.path === '/api/performance/analysis') {
    try {
      const metrics = performanceMonitoringService.getMetrics();
      const analysis = performanceMonitoringService.getSlowRequestsAnalysis();
      const recommendations = performanceMonitoringService.getPerformanceRecommendations();

      res.json({
        success: true,
        data: {
          metrics,
          analysis,
          recommendations,
          timestamp: new Date().toISOString()
        }
      });
    } catch (error) {
      logger.error('Error generating performance analysis:', error);
      res.status(500).json({
        success: false,
        error: 'Failed to generate performance analysis'
      });
    }
  } else {
    next();
  }
};

/**
 * Memory usage monitoring middleware
 * Tracks memory usage patterns and detects potential leaks
 */
export const memoryMonitoringMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const memoryBefore = process.memoryUsage();
  
  res.on('finish', () => {
    const memoryAfter = process.memoryUsage();
    const memoryIncrease = memoryAfter.heapUsed - memoryBefore.heapUsed;
    
    // Log significant memory increases
    if (memoryIncrease > 5 * 1024 * 1024) { // 5MB
      logger.warn(`High memory usage: ${req.method} ${req.path} - ${memoryIncrease / 1024 / 1024}MB increase`);
    }
  });

  next();
};
