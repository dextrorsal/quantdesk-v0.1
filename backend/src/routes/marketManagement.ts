import express, { Request, Response } from 'express';
import { SupabaseDatabaseService } from '../services/supabaseDatabase';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';

const router = express.Router();
const logger = new Logger();
const db = SupabaseDatabaseService.getInstance();

// Get all markets with categories and metadata
router.get('/', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const markets = await db.getMarkets();
    
    // Get current prices for all markets
    const marketData = await Promise.all(
      markets.map(async (market) => {
        const symbol = market.base_asset;
        const oraclePrice = await db.getOraclePrice(symbol);
        
        return {
          id: market.id,
          symbol: market.symbol,
          baseAsset: market.base_asset,
          quoteAsset: market.quote_asset,
          isActive: market.is_active,
          maxLeverage: market.max_leverage,
          initialMarginRatio: market.initial_margin_ratio,
          maintenanceMarginRatio: market.maintenance_margin_ratio,
          tickSize: market.tick_size,
          stepSize: market.step_size,
          minOrderSize: market.min_order_size,
          maxOrderSize: market.max_order_size,
          fundingInterval: market.funding_interval,
          currentFundingRate: market.current_funding_rate,
          currentPrice: oraclePrice || 0,
          priceChange24h: 0, // Calculate from historical data
          volume24h: 0, // Calculate from trades
          openInterest: 0, // Calculate from positions
          category: market.metadata?.category || 'other',
          description: market.metadata?.description || '',
          logoUrl: market.metadata?.logo_url || '',
          createdAt: market.created_at
        };
      })
    );

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

// Get markets by category
router.get('/category/:category', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { category } = req.params;
    const markets = await db.getMarketsByCategory(category);
    
    res.json({
      success: true,
      category,
      markets
    });

  } catch (error) {
    logger.error('Error fetching markets by category:', error);
    res.status(500).json({
      error: 'Failed to fetch markets by category',
      code: 'FETCH_ERROR'
    });
  }
}));

// Search markets
router.get('/search', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { q } = req.query;
    if (!q || typeof q !== 'string') {
      res.status(400).json({
        error: 'Search query is required',
        code: 'MISSING_QUERY'
      });
      return;
    }

    const markets = await db.searchMarkets(q);
    
    res.json({
      success: true,
      query: q,
      markets
    });

  } catch (error) {
    logger.error('Error searching markets:', error);
    res.status(500).json({
      error: 'Failed to search markets',
      code: 'SEARCH_ERROR'
    });
  }
}));

// Get single market details
router.get('/:symbol', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { symbol } = req.params;
    const market = await db.getMarketBySymbol(symbol);
    
    if (!market) {
      res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
      return;
    }

    const oraclePrice = await db.getOraclePrice(market.base_asset);
    
    res.json({
      success: true,
      market: {
        id: market.id,
        symbol: market.symbol,
        baseAsset: market.base_asset,
        quoteAsset: market.quote_asset,
        isActive: market.is_active,
        maxLeverage: market.max_leverage,
        initialMarginRatio: market.initial_margin_ratio,
        maintenanceMarginRatio: market.maintenance_margin_ratio,
        tickSize: market.tick_size,
        stepSize: market.step_size,
        minOrderSize: market.min_order_size,
        maxOrderSize: market.max_order_size,
        fundingInterval: market.funding_interval,
        currentFundingRate: market.current_funding_rate,
        currentPrice: oraclePrice || 0,
        priceChange24h: 0,
        volume24h: 0,
        openInterest: 0,
        category: market.metadata?.category || 'other',
        description: market.metadata?.description || '',
        logoUrl: market.metadata?.logo_url || '',
        createdAt: market.created_at
      }
    });

  } catch (error) {
    logger.error('Error fetching market:', error);
    res.status(500).json({
      error: 'Failed to fetch market',
      code: 'FETCH_ERROR'
    });
  }
}));

// Create new market (admin only)
router.post('/', asyncHandler(async (req: Request, res: Response) => {
  try {
    const {
      symbol,
      baseAsset,
      quoteAsset,
      maxLeverage = 100,
      initialMarginRatio = 500,
      maintenanceMarginRatio = 300,
      category = 'other',
      description = '',
      logoUrl = '',
      pythPriceFeedId = ''
    } = req.body;

    if (!symbol || !baseAsset || !quoteAsset) {
      res.status(400).json({
        error: 'Missing required fields: symbol, baseAsset, quoteAsset',
        code: 'MISSING_FIELDS'
      });
      return;
    }

    const marketData = {
      symbol,
      base_asset: baseAsset,
      quote_asset: quoteAsset,
      program_id: 'HsSzXVuSwiqGQvocT2zMX8zc86KQ9TYfZFZcfmmejzso', // Default program ID
      market_account: '', // Will be generated
      oracle_account: pythPriceFeedId,
      max_leverage: maxLeverage,
      initial_margin_ratio: initialMarginRatio,
      maintenance_margin_ratio: maintenanceMarginRatio,
      metadata: {
        category,
        description,
        logo_url: logoUrl,
        volume_24h: 0,
        price_change_24h: 0,
        market_cap: 0
      }
    };

    const market = await db.createMarket(marketData);
    
    res.status(201).json({
      success: true,
      market: {
        id: market.id,
        symbol: market.symbol,
        baseAsset: market.base_asset,
        quoteAsset: market.quote_asset,
        category: market.metadata?.category,
        description: market.metadata?.description,
        logoUrl: market.metadata?.logo_url
      }
    });

  } catch (error) {
    logger.error('Error creating market:', error);
    res.status(500).json({
      error: 'Failed to create market',
      code: 'CREATE_ERROR'
    });
  }
}));

// Update market (admin only)
router.put('/:id', asyncHandler(async (req: Request, res: Response) => {
  try {
    const { id } = req.params;
    const updates = req.body;

    const market = await db.updateMarket(id, updates);
    
    res.json({
      success: true,
      market
    });

  } catch (error) {
    logger.error('Error updating market:', error);
    res.status(500).json({
      error: 'Failed to update market',
      code: 'UPDATE_ERROR'
    });
  }
}));

// Get market categories
router.get('/meta/categories', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const categories = await db.getMarketCategories();
    
    res.json({
      success: true,
      categories
    });

  } catch (error) {
    logger.error('Error fetching categories:', error);
    res.status(500).json({
      error: 'Failed to fetch categories',
      code: 'FETCH_ERROR'
    });
  }
}));

export default router;
