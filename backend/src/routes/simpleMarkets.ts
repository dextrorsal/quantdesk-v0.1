import express, { Request, Response } from 'express';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandler';

const router = express.Router();
const logger = new Logger();

// Simple markets data - this will be replaced with MCP Supabase calls
const MARKETS = [
  {
    id: '1',
    symbol: 'BTC-PERP',
    baseAsset: 'BTC',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '2',
    symbol: 'ETH-PERP',
    baseAsset: 'ETH',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '3',
    symbol: 'SOL-PERP',
    baseAsset: 'SOL',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '4',
    symbol: 'AVAX-PERP',
    baseAsset: 'AVAX',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '5',
    symbol: 'MATIC-PERP',
    baseAsset: 'MATIC',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '6',
    symbol: 'ARB-PERP',
    baseAsset: 'ARB',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '7',
    symbol: 'OP-PERP',
    baseAsset: 'OP',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '8',
    symbol: 'DOGE-PERP',
    baseAsset: 'DOGE',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '9',
    symbol: 'ADA-PERP',
    baseAsset: 'ADA',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '10',
    symbol: 'DOT-PERP',
    baseAsset: 'DOT',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  },
  {
    id: '11',
    symbol: 'LINK-PERP',
    baseAsset: 'LINK',
    quoteAsset: 'USDC',
    isActive: true,
    maxLeverage: 100,
    initialMarginRatio: 0.01,
    maintenanceMarginRatio: 0.005,
    tickSize: 0.01,
    stepSize: 0.0001,
    minOrderSize: 0.001,
    maxOrderSize: 1000,
    fundingInterval: 3600,
    currentFundingRate: 0.0001,
    createdAt: new Date().toISOString()
  }
];

// Get all markets
router.get('/', asyncHandler(async (_req: Request, res: Response) => {
  try {
    // Add mock price data - match frontend interface
    const marketData = MARKETS.map(market => ({
      ...market,
      price: Math.random() * 100000 + 1000, // Mock price
      change24h: Math.random() * 10 - 5, // Mock 24h change
      volume24h: Math.random() * 1000000, // Mock volume
      openInterest: Math.random() * 10000000, // Mock open interest
      fundingRate: market.currentFundingRate,
      timestamp: new Date().toISOString(),
    }));

    res.json({
      success: true,
      markets: marketData
    });

  } catch (error) {
    logger.error('Error fetching markets:', error);
    res.status(500).json({
      error: 'Failed to fetch markets',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get market by symbol
router.get('/:symbol', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const market = MARKETS.find(m => m.symbol === symbol);
    
    if (!market) {
      return res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    res.json({
      success: true,
      market: {
        ...market,
        price: Math.random() * 100000 + 1000,
        change24h: Math.random() * 10 - 5,
        volume24h: Math.random() * 1000000,
        tradesCount24h: Math.floor(Math.random() * 1000),
        openInterest: Math.random() * 10000000,
        fundingRate: market.currentFundingRate,
        timestamp: new Date().toISOString(),
      }
    });

  } catch (error) {
    logger.error(`Error fetching market ${symbol}:`, error);
    res.status(500).json({
      error: 'Failed to fetch market',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get market price
router.get('/:symbol/price', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const market = MARKETS.find(m => m.symbol === symbol);
    
    if (!market) {
      return res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    res.json({
      success: true,
      price: Math.random() * 100000 + 1000,
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error(`Error fetching price for ${symbol}:`, error);
    res.status(500).json({
      error: 'Failed to fetch price',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get price history
router.get('/:symbol/price-history', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;
  const { hours = 24 } = req.query;

  try {
    const market = MARKETS.find(m => m.symbol === symbol);
    
    if (!market) {
      return res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    // Generate mock price history
    const dataPoints = parseInt(hours as string) || 24;
    const priceHistory = [];
    const basePrice = Math.random() * 100000 + 1000;
    
    for (let i = 0; i < dataPoints; i++) {
      const timestamp = new Date(Date.now() - (dataPoints - i) * 60 * 60 * 1000);
      const price = basePrice + (Math.random() - 0.5) * basePrice * 0.1; // ±5% variation
      
      priceHistory.push({
        price: Math.round(price * 100) / 100,
        timestamp: timestamp.toISOString()
      });
    }

    res.json({
      success: true,
      symbol,
      hours: parseInt(hours as string),
      data: priceHistory
    });

  } catch (error) {
    logger.error(`Error fetching price history for ${symbol}:`, error);
    res.status(500).json({
      error: 'Failed to fetch price history',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get funding rate
router.get('/:symbol/funding', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const market = MARKETS.find(m => m.symbol === symbol);
    
    if (!market) {
      return res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    res.json({
      success: true,
      symbol,
      fundingRate: market.currentFundingRate,
      nextFundingTime: new Date(Date.now() + 3600000).toISOString(), // 1 hour from now
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error(`Error fetching funding rate for ${symbol}:`, error);
    res.status(500).json({
      error: 'Failed to fetch funding rate',
      code: 'FETCH_ERROR'
    });
  }
}));

export default router;
