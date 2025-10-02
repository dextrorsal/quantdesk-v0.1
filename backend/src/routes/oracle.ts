import { Router, Request, Response } from 'express';
import { pythService } from '../services/pythService';
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
      prices = await pythService.getAllPrices();
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

    const price = await pythService.getAssetPrice(upperAsset as 'BTC');
    const confidence = await pythService.getPriceConfidence(upperAsset as 'BTC');
    
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
    const rawData = await pythService.getLatestPrices();
    
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
 * Check oracle service health
 */
router.get('/health', async (req: Request, res: Response) => {
  try {
    // Test fetching BTC price as health check
    const btcPrice = await pythService.getAssetPrice('BTC');
    
    res.json({
      success: true,
      status: 'healthy',
      btcPrice,
      timestamp: Date.now(),
      message: 'Oracle service is operational'
    });
  } catch (error) {
    res.status(503).json({
      success: false,
      status: 'unhealthy',
      error: 'Oracle service is down',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    });
  }
});

export default router;
