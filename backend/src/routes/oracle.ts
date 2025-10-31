import { Router, Request, Response } from 'express';
import { Connection, PublicKey } from '@solana/web3.js'
import { PythSolanaReceiver } from '@pythnetwork/pyth-solana-receiver'
import { pythOracleService } from '../services/pythOracleService';
import { redisCacheService } from '../services/redisCache';
import { fallbackPriceService } from '../services/fallbackPriceService';

const router = Router();

/**
 * GET /api/oracle/binance/:symbol
 * Proxy endpoint to fetch Binance candles (avoids CORS)
 */
router.get('/binance/:symbol', async (req: Request, res: Response) => {
  try {
    let { symbol } = req.params;
    const { interval = '1h', limit = '500' } = req.query;
    
    // Normalize symbol format (SOL-PERPUSDT -> SOLUSDT, ETH-PERPUSDT -> ETHUSDT)
    symbol = symbol.replace('-PERP', '').replace('USDTUSDT', 'USDT');
    
    // Add logging
    console.log(`📊 Binance proxy: ${req.params.symbol} -> ${symbol} (${interval})`);
    
    const response = await fetch(
      `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&limit=${limit}`
    );
    
    if (!response.ok) {
      const errorText = await response.text();
      console.error(`❌ Binance API error for ${symbol}: ${response.status} ${errorText}`);
      return res.status(response.status).json({ 
        error: 'Binance API error',
        status: response.status,
        message: errorText 
      });
    }
    
    const data = await response.json();
    const candles = data as any[];
    console.log(`✅ Binance data fetched for ${symbol}: ${candles.length} candles`);
    res.json(candles);
  } catch (error: any) {
    console.error('Binance proxy error:', error);
    res.status(500).json({ error: error.message });
  }
});

/**
 * GET /api/oracle/prices
 * Get latest prices for all supported assets
 */
router.get('/prices', async (req: Request, res: Response) => {
  try {
    // Toggleable cache-aside: try Redis first (1s TTL) if enabled
    const cachingEnabled = (process.env.CACHE_ENABLE ?? 'true').toLowerCase() !== 'false';
    const cacheKey = 'oracle:prices';
    if (cachingEnabled) {
      const cached = await redisCacheService.get<Record<string, number>>(cacheKey);
      if (cached) {
        return res.json({ success: true, data: cached, timestamp: Date.now(), source: 'cache' });
      }
    }

    // Try Pyth first, fallback to CoinGecko if it fails
    let prices: Record<string, number>;
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
    
    // Write to cache with 1s TTL (if enabled)
    if (cachingEnabled) {
      await redisCacheService.set(cacheKey, prices, 1);
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
 * GET /api/oracle/onchain/:asset
 * Read price from Pyth Solana Receiver accounts.
 * Requires env PYTH_SOLANA_RECEIVER_PROGRAM_ID and SOLANA RPC set.
 */
router.get('/onchain/:asset', async (req: Request, res: Response) => {
  try {
    const asset = req.params.asset.toUpperCase();
    const rpcUrl = process.env['VITE_SOLANA_RPC_URL'] || process.env['SOLANA_RPC_URL'] || 'https://api.devnet.solana.com';
    const receiverProgramId = process.env['PYTH_SOLANA_RECEIVER_PROGRAM_ID'];
    if (!receiverProgramId) {
      return res.status(503).json({ success: false, error: 'receiver_program_missing' });
    }
    // On-chain Pyth Receiver endpoint (disabled for now - using Hermes off-chain)
    // TODO: Implement proper PythSolanaReceiver initialization when needed
    return res.status(501).json({ success: false, error: 'onchain_pyth_not_implemented', message: 'Using Hermes (off-chain) for now' });
    
    // Placeholder for future on-chain implementation:
    // const feedId = (process.env[`PYTH_PRICE_FEED_${asset}`] || '').trim();
    // const conn = new Connection(rpcUrl, 'confirmed');
    // const receiver = new PythSolanaReceiver({ connection: conn, wallet: ..., ... });
    // const price = await receiver.getLatestPrice(...);
  } catch (e: any) {
    res.status(500).json({ success: false, error: e?.message || 'onchain_error' });
  }
});

/**
 * GET /api/oracle/price/:asset
 * Get price for a specific asset (BTC, ETH, SOL, USDC, etc.)
 * Uses Pyth Network with CoinGecko fallback
 */
router.get('/price/:asset', async (req: Request, res: Response) => {
  try {
    const { asset } = req.params;
    const upperAsset = asset.toUpperCase();
    
    const supportedAssets = ['BTC', 'ETH', 'SOL', 'USDC', 'USDT', 'AVAX', 'MATIC', 'DOGE', 'ADA', 'DOT', 'LINK'];
    
    let price: number | null = null;
    let confidence: number = 0;
    let source = 'pyth-network';

    // Freshness/confidence thresholds
    const STALE_MS = 60_000;           // 60s
    const MAX_CONF_RATIO = 0.10;       // 10%

    try {
      // Prefer detailed price data for freshness/confidence checks
      const priceData = await pythOracleService.getPrice(upperAsset);
      if (priceData && priceData.price) {
        const ageMs = Date.now() - priceData.timestamp;
        const confRatio = priceData.price > 0 ? Math.abs(priceData.confidence) / Math.abs(priceData.price) : 1;
        
        if (ageMs <= STALE_MS && confRatio <= MAX_CONF_RATIO) {
          price = priceData.price;
          confidence = priceData.confidence;
        } else {
          throw new Error(`Stale or low-confidence price: ageMs=${ageMs}, confRatio=${confRatio.toFixed(4)}`);
        }
      } else {
        throw new Error('No Pyth price data');
      }
    } catch (pythError) {
      console.log(`⚠️ Pyth detailed price failed for ${upperAsset}, trying aggregated/fallback:`, pythError);
      // Try compatibility getters (may lack freshness metadata)
      try {
        const aggPrice = await pythOracleService.getAssetPrice(upperAsset);
        confidence = await pythOracleService.getPriceConfidence(upperAsset);
        if (aggPrice && aggPrice > 0) {
          price = aggPrice;
        } else {
          throw new Error('Aggregated price invalid');
        }
      } catch (aggError) {
        console.log(`⚠️ Aggregated Pyth failed for ${upperAsset}, trying CoinGecko fallback:`, aggError);
        // Fallback to CoinGecko
        try {
          const fallbackPrices = await fallbackPriceService.getLatestPrices();
          const priceData = fallbackPrices[upperAsset as keyof typeof fallbackPrices];
          if (priceData && priceData.price) {
            price = priceData.price;
            source = 'coingecko-fallback';
          } else {
            throw new Error(`Price not found in CoinGecko fallback for ${upperAsset}`);
          }
        } catch (fallbackError) {
          console.error(`❌ Both Pyth and CoinGecko failed for ${upperAsset}:`, fallbackError);
          return res.status(404).json({
            success: false,
            error: 'Price not found',
            message: `Unable to fetch price for ${upperAsset}. Supported assets: ${supportedAssets.join(', ')}`
          });
        }
      }
    }
    
    res.json({
      success: true,
      price: price,
      value: price,
      data: {
        asset: upperAsset,
        price,
        confidence,
        timestamp: Date.now(),
        source
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
