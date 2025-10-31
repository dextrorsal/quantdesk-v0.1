/**
 * Analytics API Routes
 * RESTful API endpoints for comprehensive usage analytics
 */

import { Router, Request, Response } from 'express';
import type { Express } from 'express';
import { AnalyticsCollector } from '../services/AnalyticsCollector';
import { RequestMetrics, TimeRange, AnalyticsQuery } from '../types/analytics';
import { systemLogger, errorLogger } from '../utils/logger';

const router: Express = Router() as any;
const analyticsCollector = new AnalyticsCollector();

/**
 * GET /api/analytics/cost-report
 * Generate comprehensive cost report for specified time range
 */
router.get('/cost-report', async (req: Request, res: Response) => {
  try {
    const { start, end } = req.query;
    
    if (!start || !end) {
      return res.status(400).json({
        error: 'Missing required parameters',
        message: 'start and end date parameters are required',
        example: '/api/analytics/cost-report?start=2025-01-01&end=2025-01-31'
      });
    }

    const timeRange: TimeRange = {
      start: new Date(start as string),
      end: new Date(end as string)
    };

    // Validate date range
    if (timeRange.start >= timeRange.end) {
      return res.status(400).json({
        error: 'Invalid date range',
        message: 'Start date must be before end date'
      });
    }

    const costReport = await analyticsCollector.generateCostReport(timeRange);
    
    res.json({
      success: true,
      data: costReport,
      generatedAt: new Date().toISOString()
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'Cost report generation');
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to generate cost report'
    });
  }
});

/**
 * GET /api/analytics/provider-utilization
 * Get provider utilization metrics
 */
router.get('/provider-utilization', async (req: Request, res: Response) => {
  try {
    const utilization = await analyticsCollector.getProviderUtilization();
    
    res.json({
      success: true,
      data: utilization,
      generatedAt: new Date().toISOString()
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'Provider utilization calculation');
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to get provider utilization'
    });
  }
});

/**
 * GET /api/analytics/user-satisfaction
 * Get user satisfaction metrics
 */
router.get('/user-satisfaction', async (req: Request, res: Response) => {
  try {
    const satisfaction = await analyticsCollector.getUserSatisfactionMetrics();
    
    res.json({
      success: true,
      data: satisfaction,
      generatedAt: new Date().toISOString()
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'User satisfaction calculation');
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to get user satisfaction metrics'
    });
  }
});

/**
 * GET /api/analytics/cost-savings
 * Get cost savings report for specified time range
 */
router.get('/cost-savings', async (req: Request, res: Response) => {
  try {
    const { start, end } = req.query;
    
    if (!start || !end) {
      return res.status(400).json({
        error: 'Missing required parameters',
        message: 'start and end date parameters are required',
        example: '/api/analytics/cost-savings?start=2025-01-01&end=2025-01-31'
      });
    }

    const timeRange: TimeRange = {
      start: new Date(start as string),
      end: new Date(end as string)
    };

    const savingsReport = await analyticsCollector.generateCostSavingsReport(timeRange);
    
    res.json({
      success: true,
      data: savingsReport,
      generatedAt: new Date().toISOString()
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'Cost savings calculation');
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to generate cost savings report'
    });
  }
});

/**
 * GET /api/analytics/dashboard
 * Get comprehensive analytics dashboard
 */
router.get('/dashboard', async (req: Request, res: Response) => {
  try {
    const { start, end } = req.query;
    
    let timeRange: TimeRange | undefined;
    if (start && end) {
      timeRange = {
        start: new Date(start as string),
        end: new Date(end as string)
      };
    }

    const dashboard = await analyticsCollector.getAnalyticsDashboard(timeRange);
    
    res.json({
      success: true,
      data: dashboard,
      generatedAt: new Date().toISOString()
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'Dashboard generation');
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to generate analytics dashboard'
    });
  }
});

/**
 * GET /api/analytics/stats
 * Get basic analytics statistics
 */
router.get('/stats', async (req: Request, res: Response) => {
  try {
    const stats = analyticsCollector.getAnalyticsStats();
    
    res.json({
      success: true,
      data: stats,
      generatedAt: new Date().toISOString()
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'Analytics stats retrieval');
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to get analytics statistics'
    });
  }
});

/**
 * POST /api/analytics/track
 * Track request metrics (for internal use)
 */
router.post('/track', async (req: Request, res: Response) => {
  try {
    const metrics: RequestMetrics = req.body;
    
    // Validate required fields
    if (!metrics.requestId || !metrics.provider || !metrics.taskType) {
      return res.status(400).json({
        error: 'Missing required fields',
        message: 'requestId, provider, and taskType are required',
        required: ['requestId', 'provider', 'taskType', 'tokensUsed', 'cost', 'qualityScore', 'responseTime', 'timestamp']
      });
    }

    await analyticsCollector.trackRequestMetrics(metrics);
    
    res.json({
      success: true,
      message: 'Metrics tracked successfully',
      requestId: metrics.requestId
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'Metrics tracking');
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to track metrics'
    });
  }
});

/**
 * POST /api/analytics/export
 * Export analytics data with filters
 */
router.post('/export', async (req: Request, res: Response) => {
  try {
    const query: AnalyticsQuery = req.body;
    
    if (!query.filter) {
      return res.status(400).json({
        error: 'Missing filter',
        message: 'Filter object is required'
      });
    }

    const exportedData = await analyticsCollector.exportAnalyticsData(query);
    
    res.json({
      success: true,
      data: exportedData,
      count: exportedData.length,
      exportedAt: new Date().toISOString()
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'Analytics data export');
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to export analytics data'
    });
  }
});

/**
 * POST /api/analytics/cleanup
 * Trigger data cleanup based on retention policy
 */
router.post('/cleanup', async (req: Request, res: Response) => {
  try {
    await analyticsCollector.clearOldData();
    
    res.json({
      success: true,
      message: 'Data cleanup completed successfully',
      cleanedAt: new Date().toISOString()
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'Data cleanup');
    res.status(500).json({
      error: 'Internal server error',
      message: 'Failed to cleanup old data'
    });
  }
});

/**
 * GET /api/analytics/health
 * Health check endpoint for analytics service
 */
router.get('/health', async (req: Request, res: Response) => {
  try {
    const stats = analyticsCollector.getAnalyticsStats();
    
    res.json({
      success: true,
      status: 'healthy',
      data: {
        totalRequests: stats.totalRequests,
        dataPointsCollected: stats.dataPointsCollected,
        lastUpdated: stats.lastUpdated,
        uptime: process.uptime()
      },
      checkedAt: new Date().toISOString()
    });

  } catch (error) {
    errorLogger.aiError(error as Error, 'Analytics health check');
    res.status(500).json({
      success: false,
      status: 'unhealthy',
      error: 'Analytics service health check failed'
    });
  }
});

// Error handling middleware
router.use((error: Error, req: Request, res: Response, next: any) => {
  errorLogger.aiError(error, 'Analytics API error');
  res.status(500).json({
    error: 'Internal server error',
    message: 'An unexpected error occurred'
  });
});

export default router;
