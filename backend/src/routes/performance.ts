import { Router } from 'express';
import { performanceMonitoringService } from '../services/performanceMonitoringService';
import { performanceTestSuite } from '../services/performanceTestSuite';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

/**
 * Performance Monitoring Routes
 * Provides performance insights and optimization recommendations
 */

// Get current performance metrics
router.get('/metrics', async (req, res) => {
  try {
    const metrics = performanceMonitoringService.getMetrics();
    
    res.json({
      success: true,
      data: {
        ...metrics,
        memoryUsageMB: {
          rss: Math.round(metrics.memoryUsage.rss / 1024 / 1024),
          heapTotal: Math.round(metrics.memoryUsage.heapTotal / 1024 / 1024),
          heapUsed: Math.round(metrics.memoryUsage.heapUsed / 1024 / 1024),
          external: Math.round(metrics.memoryUsage.external / 1024 / 1024)
        }
      }
    });
  } catch (error) {
    logger.error('Error fetching performance metrics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch performance metrics'
    });
  }
});

// Get performance analysis
router.get('/analysis', async (req, res) => {
  try {
    const analysis = performanceMonitoringService.getSlowRequestsAnalysis();
    const recommendations = performanceMonitoringService.getPerformanceRecommendations();
    
    res.json({
      success: true,
      data: {
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
});

// Get performance recommendations
router.get('/recommendations', async (req, res) => {
  try {
    const recommendations = performanceMonitoringService.getPerformanceRecommendations();
    
    res.json({
      success: true,
      data: {
        recommendations,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    logger.error('Error fetching performance recommendations:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch performance recommendations'
    });
  }
});

// Reset performance metrics
router.post('/reset', async (req, res) => {
  try {
    performanceMonitoringService.resetMetrics();
    
    res.json({
      success: true,
      message: 'Performance metrics reset successfully'
    });
  } catch (error) {
    logger.error('Error resetting performance metrics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to reset performance metrics'
    });
  }
});

// Health check with performance data
router.get('/health', async (req, res) => {
  try {
    const metrics = performanceMonitoringService.getMetrics();
    const analysis = performanceMonitoringService.getSlowRequestsAnalysis();
    
    const isHealthy = 
      metrics.averageResponseTime < 200 &&
      metrics.errorRate < 5 &&
      metrics.memoryUsage.heapUsed < 500 * 1024 * 1024; // 500MB
    
    res.json({
      success: true,
      data: {
        healthy: isHealthy,
        metrics: {
          averageResponseTime: metrics.averageResponseTime,
          errorRate: metrics.errorRate,
          memoryUsageMB: Math.round(metrics.memoryUsage.heapUsed / 1024 / 1024),
          slowRequests: metrics.slowRequests,
          totalRequests: metrics.requestCount
        },
        issues: analysis.slowestEndpoints.slice(0, 3),
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    logger.error('Error checking performance health:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to check performance health'
    });
  }
});

// Run comprehensive performance tests
router.post('/test', async (req, res) => {
  try {
    logger.info('Starting comprehensive performance test suite...');
    
    const testResults = await performanceTestSuite.runAllTests();
    
    res.json({
      success: true,
      data: testResults
    });
  } catch (error) {
    logger.error('Error running performance tests:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to run performance tests'
    });
  }
});

// Get performance test results summary
router.get('/test/summary', async (req, res) => {
  try {
    const summary = performanceTestSuite.getTestResultsSummary();
    
    res.json({
      success: true,
      data: summary
    });
  } catch (error) {
    logger.error('Error getting performance test summary:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to get performance test summary'
    });
  }
});

export default router;
