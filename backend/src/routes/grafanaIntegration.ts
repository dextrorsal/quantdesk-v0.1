import { Router, Request, Response } from 'express';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

/**
 * GET /api/grafana/metrics
 * Get metrics formatted for Grafana consumption
 */
router.get('/metrics', async (req: Request, res: Response) => {
  try {
    // This endpoint will be used by Grafana to fetch metrics
    // We'll format the data in a way that Grafana can consume
    
    const metrics = {
      quantdesk_trading_volume_24h: 300,
      quantdesk_active_traders: 2,
      quantdesk_total_positions: 2,
      quantdesk_total_value_locked: 3000,
      quantdesk_average_position_size: 1500,
      quantdesk_api_response_time: Math.random() * 50 + 10, // Mock response time
      quantdesk_memory_usage: process.memoryUsage().heapUsed / 1024 / 1024,
      quantdesk_cpu_usage: process.cpuUsage().user / 1000000,
      quantdesk_market_volume_by_symbol: [
        { symbol: 'BTC-PERP', volume: 100 },
        { symbol: 'ETH-PERP', volume: 200 },
        { symbol: 'SOL-PERP', volume: 0 }
      ],
      quantdesk_leverage_distribution: [
        { leverage_range: '1x-5x', count: 1 },
        { leverage_range: '5x-10x', count: 1 },
        { leverage_range: '10x-20x', count: 0 },
        { leverage_range: '20x+', count: 0 }
      ],
      quantdesk_long_short_ratio: [
        { symbol: 'BTC-PERP', ratio: 0.6 },
        { symbol: 'ETH-PERP', ratio: 0.4 },
        { symbol: 'SOL-PERP', ratio: 0.5 }
      ]
    };

    res.json({
      success: true,
      data: metrics,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error fetching Grafana metrics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch Grafana metrics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/grafana/dashboard-config
 * Get Grafana dashboard configuration
 */
router.get('/dashboard-config', async (req: Request, res: Response) => {
  try {
    const dashboardConfig = {
      title: 'QuantDesk Trading Analytics',
      description: 'Real-time trading metrics and system performance for QuantDesk',
      refresh: '5s',
      time: {
        from: 'now-24h',
        to: 'now'
      },
      panels: [
        {
          title: 'Trading Volume (24h)',
          type: 'stat',
          targets: [
            {
              expr: 'quantdesk_trading_volume_24h',
              legendFormat: 'Total Volume ($)'
            }
          ],
          fieldConfig: {
            unit: 'currencyUSD',
            thresholds: {
              steps: [
                { color: 'green', value: null },
                { color: 'yellow', value: 10000 },
                { color: 'red', value: 50000 }
              ]
            }
          }
        },
        {
          title: 'Active Traders',
          type: 'stat',
          targets: [
            {
              expr: 'quantdesk_active_traders',
              legendFormat: 'Active Traders'
            }
          ],
          fieldConfig: {
            unit: 'short',
            thresholds: {
              steps: [
                { color: 'green', value: null },
                { color: 'yellow', value: 10 },
                { color: 'red', value: 50 }
              ]
            }
          }
        },
        {
          title: 'Total Value Locked',
          type: 'stat',
          targets: [
            {
              expr: 'quantdesk_total_value_locked',
              legendFormat: 'TVL ($)'
            }
          ],
          fieldConfig: {
            unit: 'currencyUSD',
            thresholds: {
              steps: [
                { color: 'green', value: null },
                { color: 'yellow', value: 100000 },
                { color: 'red', value: 500000 }
              ]
            }
          }
        },
        {
          title: 'API Response Time',
          type: 'graph',
          targets: [
            {
              expr: 'quantdesk_api_response_time',
              legendFormat: 'Response Time (ms)'
            }
          ],
          yAxes: [
            {
              label: 'Response Time (ms)',
              unit: 'ms'
            }
          ]
        },
        {
          title: 'Memory Usage',
          type: 'graph',
          targets: [
            {
              expr: 'quantdesk_memory_usage',
              legendFormat: 'Memory Usage (MB)'
            }
          ],
          yAxes: [
            {
              label: 'Memory (MB)',
              unit: 'MB'
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
        },
        {
          title: 'Long/Short Ratio by Market',
          type: 'bargauge',
          targets: [
            {
              expr: 'quantdesk_long_short_ratio',
              legendFormat: '{{symbol}}'
            }
          ]
        }
      ]
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

/**
 * POST /api/grafana/webhook
 * Receive webhook data from Grafana (for future integrations)
 */
router.post('/webhook', async (req: Request, res: Response) => {
  try {
    const webhookData = req.body;
    logger.info('Received Grafana webhook:', webhookData);
    
    // Process webhook data here
    // This could be used for alerts, notifications, etc.
    
    res.json({
      success: true,
      message: 'Webhook received successfully',
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error processing Grafana webhook:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to process webhook',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
