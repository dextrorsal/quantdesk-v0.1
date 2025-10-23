import { Router, Request, Response } from 'express';
import { pythOracleService } from '../services/pythOracleService';
import { fallbackPriceService } from '../services/fallbackPriceService';

const router = Router();

/**
 * GET /api/oracle/prices
 * Get latest prices for all supported assets
 */
router.get('/prices', async (req: Request, res: Response) => {
  try {
    // Try Pyth first, fallback to CoinGecko if it fails
    let prices;
    let source = 'pyth-network';
    
    try {
      prices = await pythOracleService.getAllPrices();
    } catch (pythError) {
      console.log('Pyth failed, using CoinGecko fallback:', pythError);
      const fallbackPrices = await fallbackPriceService.getLatestPrices();
      prices = {};
      for (const [asset, priceData] of Object.entries(fallbackPrices)) {
        prices[asset] = priceData.price;
      }
      source = 'coingecko-fallback';
    }
    
    res.json({
      success: true,
      data: prices,
      timestamp: Date.now(),
      source: source
    });
  } catch (error) {
    console.error('Error fetching prices:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch price data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/oracle/price/:asset
 * Get price for a specific asset (BTC, ETH, SOL, USDC)
 */
router.get('/price/:asset', async (req: Request, res: Response) => {
  try {
    const { asset } = req.params;
    const upperAsset = asset.toUpperCase() as 'BTC' | 'ETH' | 'SOL' | 'USDC';
    
    if (!['BTC'].includes(upperAsset)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid asset',
        message: 'Supported assets: BTC (ETH, SOL, USDC coming soon)'
      });
    }

    const price = await pythOracleService.getAssetPrice(upperAsset as 'BTC');
    const confidence = await pythOracleService.getPriceConfidence(upperAsset as 'BTC');
    
    res.json({
      success: true,
      data: {
        asset: upperAsset,
        price,
        confidence,
        timestamp: Date.now(),
        source: 'pyth-network'
      }
    });
  } catch (error) {
    console.error(`Error fetching price for ${req.params.asset}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch price data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/oracle/raw
 * Get raw Pyth Network data
 */
router.get('/raw', async (req: Request, res: Response) => {
  try {
    const rawData = await pythOracleService.getLatestPrices();
    
    res.json({
      success: true,
      data: rawData,
      timestamp: Date.now(),
      source: 'pyth-network-raw'
    });
  } catch (error) {
    console.error('Error fetching raw price data:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch raw price data',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/oracle/health
 * Check oracle service health with comprehensive monitoring
 */
router.get('/health', async (req: Request, res: Response) => {
  try {
    const healthCheck = await pythOracleService.healthCheck();
    
    if (healthCheck.status === 'healthy') {
      res.json({
        success: true,
        status: 'healthy',
        timestamp: Date.now(),
        message: 'Oracle service is operational',
        details: healthCheck.details
      });
    } else {
      res.status(503).json({
        success: false,
        status: healthCheck.status,
        timestamp: Date.now(),
        message: `Oracle service is ${healthCheck.status}`,
        details: healthCheck.details
      });
    }
  } catch (error) {
    res.status(503).json({
      success: false,
      status: 'unhealthy',
      error: 'Oracle service health check failed',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    });
  }
});

/**
 * GET /api/oracle/stats
 * Get oracle service cache statistics
 */
router.get('/stats', async (req: Request, res: Response) => {
  try {
    const stats = pythOracleService.getCacheStats();
    
    res.json({
      success: true,
      data: stats,
      timestamp: Date.now(),
      message: 'Oracle service cache statistics'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to get cache statistics',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    });
  }
});

export default router;
