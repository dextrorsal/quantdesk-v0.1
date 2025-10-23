import express, { Request, Response } from 'express';
import { mcpSupabaseService } from '../services/mcpSupabaseService';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';

const router = express.Router();
const logger = new Logger();

// Get all markets with real data from Supabase
router.get('/', asyncHandler(async (_req: Request, res: Response) => {
  try {
    // Get markets and prices from Supabase using public methods
    const [markets, oraclePrices] = await Promise.all([
      mcpSupabaseService.getMarkets(),
      mcpSupabaseService.getLatestOraclePrices()
    ]);

    if (!markets || markets.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'No markets found'
      });
    }

    // Create a map of market_id to latest price
    const priceMap = new Map();
    oraclePrices.forEach(price => {
      priceMap.set(price.market_id, price);
    });

    // Transform the data to match frontend interface
    const marketData = markets.map((market) => {
      const oraclePrice = priceMap.get(market.id);
      const price = oraclePrice ? oraclePrice.price : 0;
      const confidence = oraclePrice ? oraclePrice.confidence : 0;
      
      // Generate some mock volume and change data for now
      // In production, this would come from trades table
      const volume24h = Math.random() * 1000000 + 100000;
      const change24h = (Math.random() - 0.5) * price * 0.1; // ±5% change
      const openInterest = Math.random() * 10000000 + 1000000;

      return {
        id: market.id,
        symbol: market.symbol,
        baseAsset: market.base_asset,
        quoteAsset: market.quote_asset,
        isActive: market.is_active,
        maxLeverage: market.max_leverage,
        initialMarginRatio: market.initial_margin_ratio / 10000, // Convert from basis points
        maintenanceMarginRatio: market.maintenance_margin_ratio / 10000,
        tickSize: market.tick_size,
        stepSize: market.step_size,
        minOrderSize: market.min_order_size,
        maxOrderSize: market.max_order_size,
        fundingInterval: market.funding_interval,
        currentFundingRate: market.current_funding_rate,
        createdAt: market.created_at,
        // Real-time data
        price: price,
        change24h: change24h,
        volume24h: volume24h,
        openInterest: openInterest,
        fundingRate: market.current_funding_rate,
        timestamp: oraclePrice ? oraclePrice.created_at.toISOString() : new Date().toISOString(),
        confidence: confidence
      };
    });

    res.json({
      success: true,
      markets: marketData
    });

  } catch (error) {
    logger.error('Error fetching markets from Supabase:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch markets',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get specific market
router.get('/:symbol', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const market = await mcpSupabaseService.getMarketBySymbol(symbol);
    
    if (!market) {
      return res.status(404).json({
        success: false,
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    // Get latest price for this market
    const oraclePrices = await mcpSupabaseService.getLatestOraclePrices();
    const oraclePrice = oraclePrices.find(p => p.market_id === market.id);
    
    const price = oraclePrice ? oraclePrice.price : 0;
    const confidence = oraclePrice ? oraclePrice.confidence : 0;
    
    // Generate mock volume and change data
    const volume24h = Math.random() * 1000000 + 100000;
    const change24h = (Math.random() - 0.5) * price * 0.1;
    const openInterest = Math.random() * 10000000 + 1000000;

    const marketData = {
      id: market.id,
      symbol: market.symbol,
      baseAsset: market.base_asset,
      quoteAsset: market.quote_asset,
      isActive: market.is_active,
      maxLeverage: market.max_leverage,
      initialMarginRatio: market.initial_margin_ratio / 10000,
      maintenanceMarginRatio: market.maintenance_margin_ratio / 10000,
      tickSize: market.tick_size,
      stepSize: market.step_size,
      minOrderSize: market.min_order_size,
      maxOrderSize: market.max_order_size,
      fundingInterval: market.funding_interval,
      currentFundingRate: market.current_funding_rate,
      createdAt: market.created_at,
      price: price,
      change24h: change24h,
      volume24h: volume24h,
      openInterest: openInterest,
      fundingRate: market.current_funding_rate,
      timestamp: oraclePrice ? oraclePrice.created_at.toISOString() : new Date().toISOString(),
      confidence: confidence
    };

    res.json({
      success: true,
      market: marketData
    });

  } catch (error) {
    logger.error(`Error fetching market ${symbol} from Supabase:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch market',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get market price
router.get('/:symbol/price', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const priceResult = await mcpSupabaseService.executeQuery(`
      SELECT 
        op.price,
        op.confidence,
        op.created_at
      FROM markets m
      LEFT JOIN oracle_prices op ON m.id = op.market_id
      WHERE m.symbol = $1 AND m.is_active = true
      ORDER BY op.created_at DESC
      LIMIT 1
    `, [symbol]);

    if (!priceResult || priceResult.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Price not found',
        code: 'PRICE_NOT_FOUND'
      });
    }

    const priceData = priceResult[0];
    const price = priceData.price ? parseFloat(priceData.price) : 0;

    res.json({
      success: true,
      price: price,
      confidence: priceData.confidence ? parseFloat(priceData.confidence) : 0,
      timestamp: priceData.created_at || new Date().toISOString()
    });

  } catch (error) {
    logger.error(`Error fetching price for ${symbol}:`, error);
    res.status(500).json({
      success: false,
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
    const hoursInt = parseInt(hours as string) || 24;
    const startTime = new Date(Date.now() - hoursInt * 60 * 60 * 1000);

    const historyResult = await mcpSupabaseService.executeQuery(`
      SELECT 
        op.price,
        op.created_at
      FROM markets m
      LEFT JOIN oracle_prices op ON m.id = op.market_id
      WHERE m.symbol = $1 
        AND m.is_active = true 
        AND op.created_at >= $2
      ORDER BY op.created_at ASC
    `, [symbol, startTime.toISOString()]);

    if (!historyResult || historyResult.length === 0) {
      // Generate mock price history if no real data
      const dataPoints = hoursInt;
      const priceHistory = [];
      const basePrice = Math.random() * 100000 + 1000;
      
      for (let i = 0; i < dataPoints; i++) {
        const timestamp = new Date(Date.now() - (dataPoints - i) * 60 * 60 * 1000);
        const price = basePrice + (Math.random() - 0.5) * basePrice * 0.1;
        
        priceHistory.push({
          price: Math.round(price * 100) / 100,
          timestamp: timestamp.toISOString()
        });
      }

      return res.json({
        success: true,
        symbol,
        hours: hoursInt,
        data: priceHistory
      });
    }

    const priceHistory = historyResult.map((row: any) => ({
      price: parseFloat(row.price),
      timestamp: row.created_at
    }));

    res.json({
      success: true,
      symbol,
      hours: hoursInt,
      data: priceHistory
    });

  } catch (error) {
    logger.error(`Error fetching price history for ${symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch price history',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get funding rate
router.get('/:symbol/funding', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const fundingResult = await mcpSupabaseService.executeQuery(`
      SELECT 
        m.current_funding_rate,
        m.funding_interval,
        m.last_funding_time
      FROM markets m
      WHERE m.symbol = $1 AND m.is_active = true
    `, [symbol]);

    if (!fundingResult || fundingResult.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    const funding = fundingResult[0];
    const nextFundingTime = funding.last_funding_time 
      ? new Date(new Date(funding.last_funding_time).getTime() + funding.funding_interval * 1000)
      : new Date(Date.now() + funding.funding_interval * 1000);

    res.json({
      success: true,
      symbol,
      fundingRate: parseFloat(funding.current_funding_rate),
      nextFundingTime: nextFundingTime.toISOString(),
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error(`Error fetching funding rate for ${symbol}:`, error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch funding rate',
      code: 'FETCH_ERROR'
    });
  }
}));

export default router;
