import { Router, Request, Response } from 'express';
import { grafanaMetricsService } from '../services/grafanaMetrics';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

/**
 * GET /api/metrics/trading
 * Get current trading metrics
 */
router.get('/trading', async (req: Request, res: Response) => {
  try {
    const metrics = await grafanaMetricsService.collectTradingMetrics();
    
    res.json({
      success: true,
      data: metrics,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error fetching trading metrics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch trading metrics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/metrics/system
 * Get current system metrics
 */
router.get('/system', async (req: Request, res: Response) => {
  try {
    const metrics = await grafanaMetricsService.collectSystemMetrics();
    
    res.json({
      success: true,
      data: metrics,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error fetching system metrics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch system metrics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/metrics/trading/history
 * Get historical trading metrics
 */
router.get('/trading/history', async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const metrics = grafanaMetricsService.getRecentTradingMetrics(limit);
    
    res.json({
      success: true,
      data: metrics,
      count: metrics.length,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error fetching trading metrics history:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch trading metrics history',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/metrics/system/history
 * Get historical system metrics
 */
router.get('/system/history', async (req: Request, res: Response) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const metrics = grafanaMetricsService.getRecentSystemMetrics(limit);
    
    res.json({
      success: true,
      data: metrics,
      count: metrics.length,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error fetching system metrics history:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch system metrics history',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/metrics/grafana/dashboard-config
 * Get Grafana dashboard configuration for QuantDesk
 */
router.get('/grafana/dashboard-config', async (req: Request, res: Response) => {
  try {
    const dashboardConfig = {
      title: 'QuantDesk Trading Analytics',
      description: 'Real-time trading metrics and system performance for QuantDesk',
      panels: [
        {
          title: 'Trading Volume (24h)',
          type: 'stat',
          targets: [
            {
              expr: 'sum(quantdesk_trading_volume_24h)',
              legendFormat: 'Total Volume'
            }
          ]
        },
        {
          title: 'Active Traders',
          type: 'stat',
          targets: [
            {
              expr: 'sum(quantdesk_active_traders)',
              legendFormat: 'Active Traders'
            }
          ]
        },
        {
          title: 'Total Value Locked',
          type: 'stat',
          targets: [
            {
              expr: 'sum(quantdesk_total_value_locked)',
              legendFormat: 'TVL'
            }
          ]
        },
        {
          title: 'API Response Time',
          type: 'graph',
          targets: [
            {
              expr: 'quantdesk_api_response_time',
              legendFormat: 'Response Time (ms)'
            }
          ]
        },
        {
          title: 'Market Volume Distribution',
          type: 'piechart',
          targets: [
            {
              expr: 'quantdesk_market_volume_by_symbol',
              legendFormat: '{{symbol}}'
            }
          ]
        },
        {
          title: 'Leverage Distribution',
          type: 'piechart',
          targets: [
            {
              expr: 'quantdesk_leverage_distribution',
              legendFormat: '{{leverage_range}}'
            }
          ]
        }
      ],
      timeRange: {
        from: 'now-24h',
        to: 'now'
      },
      refresh: '5s'
    };

    res.json({
      success: true,
      data: dashboardConfig,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error generating Grafana dashboard config:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to generate dashboard configuration',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
