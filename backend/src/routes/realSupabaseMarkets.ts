import express, { Request, Response } from 'express';
import { databaseService } from '../services/supabaseDatabase';
import { SupabaseDatabaseService } from '../services/supabaseDatabase';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';

const router = express.Router();
const logger = new Logger();

// Get all markets with real data from Supabase
router.get('/', asyncHandler(async (_req: Request, res: Response) => {
  try {
    // Get markets from Supabase using the standard client
    const { data: markets, error: marketsError } = await databaseService.getClient()
      .from('markets')
      .select('*')
      .eq('is_active', true)
      .order('created_at', { ascending: true });

    if (marketsError) {
      logger.error('Error fetching markets:', marketsError);
      return res.status(500).json({
        success: false,
        error: 'Failed to fetch markets',
        code: 'FETCH_ERROR'
      });
    }

    if (!markets || markets.length === 0) {
      return res.status(404).json({
        success: false,
        error: 'No markets found'
      });
    }

    // Get latest oracle prices
    const { data: oraclePrices, error: pricesError } = await databaseService.getClient()
      .from('oracle_prices')
      .select('*')
      .order('created_at', { ascending: false });

    if (pricesError) {
      logger.warn('Error fetching oracle prices:', pricesError);
    }

    // Create a map of market_id to latest price
    const priceMap = new Map();
    if (oraclePrices) {
      oraclePrices.forEach(price => {
        if (!priceMap.has(price.market_id)) {
          priceMap.set(price.market_id, price);
        }
      });
    }

    // Transform the data to match frontend interface
    const marketData = markets.map((market) => {
      const oraclePrice = priceMap.get(market.id);
      const price = oraclePrice ? parseFloat(oraclePrice.price) : 0;
      const confidence = oraclePrice ? parseFloat(oraclePrice.confidence) : 0;
      
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
        tickSize: parseFloat(market.tick_size),
        stepSize: parseFloat(market.step_size),
        minOrderSize: parseFloat(market.min_order_size),
        maxOrderSize: parseFloat(market.max_order_size),
        fundingInterval: market.funding_interval,
        // Add metadata fields
        category: market.metadata?.category || 'other',
        description: market.metadata?.description || '',
        logoUrl: market.metadata?.logo_url || '',
        currentFundingRate: parseFloat(market.current_funding_rate),
        createdAt: market.created_at,
        // Real-time data
        price: price,
        change24h: change24h,
        volume24h: volume24h,
        openInterest: openInterest,
        fundingRate: parseFloat(market.current_funding_rate),
        timestamp: oraclePrice ? oraclePrice.created_at : new Date().toISOString(),
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
    const { data: market, error: marketError } = await databaseService.getClient()
      .from('markets')
      .select('*')
      .eq('symbol', symbol)
      .eq('is_active', true)
      .single();

    if (marketError || !market) {
      return res.status(404).json({
        success: false,
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    // Get latest price for this market
    const { data: oraclePrice } = await databaseService.getClient()
      .from('oracle_prices')
      .select('*')
      .eq('market_id', market.id)
      .order('created_at', { ascending: false })
      .limit(1)
      .single();
    
    const price = oraclePrice ? parseFloat(oraclePrice.price) : 0;
    const confidence = oraclePrice ? parseFloat(oraclePrice.confidence) : 0;
    
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
      tickSize: parseFloat(market.tick_size),
      stepSize: parseFloat(market.step_size),
      minOrderSize: parseFloat(market.min_order_size),
      maxOrderSize: parseFloat(market.max_order_size),
      fundingInterval: market.funding_interval,
      currentFundingRate: parseFloat(market.current_funding_rate),
      createdAt: market.created_at,
      price: price,
      change24h: change24h,
      volume24h: volume24h,
      openInterest: openInterest,
      fundingRate: parseFloat(market.current_funding_rate),
      timestamp: oraclePrice ? oraclePrice.created_at : new Date().toISOString(),
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
    // First get the market
    const { data: market } = await databaseService.getClient()
      .from('markets')
      .select('id')
      .eq('symbol', symbol)
      .eq('is_active', true)
      .single();

    if (!market) {
      return res.status(404).json({
        success: false,
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    // Get latest price
    const { data: oraclePrice } = await databaseService.getClient()
      .from('oracle_prices')
      .select('price, confidence, created_at')
      .eq('market_id', market.id)
      .order('created_at', { ascending: false })
      .limit(1)
      .single();

    if (!oraclePrice) {
      return res.status(404).json({
        success: false,
        error: 'Price not found',
        code: 'PRICE_NOT_FOUND'
      });
    }

    res.json({
      success: true,
      price: parseFloat(oraclePrice.price),
      confidence: parseFloat(oraclePrice.confidence),
      timestamp: oraclePrice.created_at
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

    // First get the market
    const { data: market } = await databaseService.getClient()
      .from('markets')
      .select('id')
      .eq('symbol', symbol)
      .eq('is_active', true)
      .single();

    if (!market) {
      return res.status(404).json({
        success: false,
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    // Get price history
    const { data: history, error } = await databaseService.getClient()
      .from('oracle_prices')
      .select('price, created_at')
      .eq('market_id', market.id)
      .gte('created_at', startTime.toISOString())
      .order('created_at', { ascending: true });

    if (error) {
      logger.error('Error fetching price history:', error);
      return res.status(500).json({
        success: false,
        error: 'Failed to fetch price history',
        code: 'FETCH_ERROR'
      });
    }

    if (!history || history.length === 0) {
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

    const priceHistory = history.map(row => ({
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
    const { data: market, error } = await databaseService.getClient()
      .from('markets')
      .select('current_funding_rate, funding_interval, last_funding_time')
      .eq('symbol', symbol)
      .eq('is_active', true)
      .single();

    if (error || !market) {
      return res.status(404).json({
        success: false,
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    const nextFundingTime = market.last_funding_time 
      ? new Date(new Date(market.last_funding_time).getTime() + market.funding_interval * 1000)
      : new Date(Date.now() + market.funding_interval * 1000);

    res.json({
      success: true,
      symbol,
      fundingRate: parseFloat(market.current_funding_rate),
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

// Get market order book
router.get('/:symbol/orderbook', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const db = SupabaseDatabaseService.getInstance();
    const market = await db.getMarketBySymbol(symbol);

    if (!market) {
      return res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    // Get pending orders
    const { data: orders, error: ordersError } = await db.getClient()
      .from('orders')
      .select('side, price, remaining_size')
      .eq('market_id', market.id)
      .eq('status', 'pending')
      .order('price', { ascending: false });

    if (ordersError) {
      logger.error('Error fetching orders:', ordersError);
      return res.status(500).json({
        success: false,
        error: 'Failed to fetch orders',
        code: 'FETCH_ERROR'
      });
    }

    // Build order book
    const bids: [number, number][] = [];
    const asks: [number, number][] = [];

    for (const order of orders || []) {
      const price = parseFloat(order.price);
      const size = parseFloat(order.remaining_size);

      if (order.side === 'long') {
        bids.push([price, size]);
      } else {
        asks.push([price, size]);
      }
    }

    // Calculate spread
    const bestBid = bids.length > 0 ? bids[0][0] : 0;
    const bestAsk = asks.length > 0 ? asks[0][0] : 0;
    const spread = bestAsk - bestBid;

    res.json({
      success: true,
      orderbook: {
        symbol,
        bids: bids.slice(0, 20), // Top 20 bids
        asks: asks.slice(0, 20), // Top 20 asks
        spread,
        timestamp: Date.now()
      }
    });

  } catch (error) {
    logger.error(`Error fetching order book for ${symbol}:`, error);
    res.status(500).json({
      error: 'Failed to fetch order book',
      code: 'FETCH_ERROR'
    });
  }
}));

export default router;
