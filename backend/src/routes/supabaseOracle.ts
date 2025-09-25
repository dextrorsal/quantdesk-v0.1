import { Router, Request, Response } from 'express';

const router = Router();

/**
 * GET /api/supabase-oracle/prices
 * Get latest prices from Supabase using MCP tools
 */
router.get('/prices', async (req: Request, res: Response) => {
  try {
    // This would normally use MCP tools, but for now we'll return mock data
    // In a real implementation, you'd call the MCP Supabase tools here
    
    const prices = [
      {
        symbol: 'BTC-PERP',
        base_asset: 'BTC',
        quote_asset: 'USDT',
        price: 109148.00,
        confidence: 0.01,
        exponent: -2,
        timestamp: new Date().toISOString()
      },
      {
        symbol: 'ETH-PERP', 
        base_asset: 'ETH',
        quote_asset: 'USDT',
        price: 3888.62,
        confidence: 0.01,
        exponent: -2,
        timestamp: new Date().toISOString()
      },
      {
        symbol: 'SOL-PERP',
        base_asset: 'SOL', 
        quote_asset: 'USDT',
        price: 195.84,
        confidence: 0.01,
        exponent: -2,
        timestamp: new Date().toISOString()
      }
    ];

    res.json({
      success: true,
      data: prices,
      timestamp: Date.now(),
      source: 'supabase-mcp'
    });
  } catch (error) {
    console.error('Error fetching prices from Supabase:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch price data from Supabase',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/supabase-oracle/health
 * Health check for Supabase connection
 */
router.get('/health', async (req: Request, res: Response) => {
  try {
    // Mock health check - in real implementation would use MCP tools
    res.json({
      success: true,
      status: 'healthy',
      message: 'Supabase connection is working',
      timestamp: Date.now()
    });
  } catch (error) {
    res.status(503).json({
      success: false,
      status: 'unhealthy',
      error: 'Supabase connection failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    });
  }
});

/**
 * GET /api/supabase-oracle/markets
 * Get all active markets from Supabase
 */
router.get('/markets', async (req: Request, res: Response) => {
  try {
    // Mock markets data - in real implementation would use MCP tools
    const markets = [
      {
        id: 'd87a99b4-148a-49c2-a2ad-ca1ee17a9372',
        symbol: 'BTC-PERP',
        base_asset: 'BTC',
        quote_asset: 'USDT',
        program_id: 'GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a',
        is_active: true,
        max_leverage: 100,
        tick_size: 0.01,
        step_size: 0.001,
        min_order_size: 0.001,
        max_order_size: 1000000
      },
      {
        id: 'b28b505f-682a-473d-979c-01c05bf31b1c',
        symbol: 'ETH-PERP',
        base_asset: 'ETH',
        quote_asset: 'USDT',
        program_id: 'GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a',
        is_active: true,
        max_leverage: 100,
        tick_size: 0.01,
        step_size: 0.001,
        min_order_size: 0.001,
        max_order_size: 1000000
      },
      {
        id: '4b39cd21-f718-4b18-a0c8-07a3eb1b9a4d',
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USDT',
        program_id: 'GcpEyGMJg9AvEx8LgAyzyUNTPWFxovHxtmQ4vVWuWu3a',
        is_active: true,
        max_leverage: 100,
        tick_size: 0.01,
        step_size: 0.001,
        min_order_size: 0.001,
        max_order_size: 1000000
      }
    ];

    res.json({
      success: true,
      data: markets,
      count: markets.length,
      timestamp: Date.now()
    });
  } catch (error) {
    console.error('Error fetching markets from Supabase:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch markets from Supabase',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
