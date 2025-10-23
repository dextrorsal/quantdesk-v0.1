import express, { Request, Response } from 'express';
import { SupabaseDatabaseService } from '../services/supabaseDatabase';
import { pythOracleService } from '../services/pythOracleService';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';

const router = express.Router();
const logger = new Logger();
const db = SupabaseDatabaseService.getInstance();
const oracle = pythOracleService;

// Get all markets
router.get('/', asyncHandler(async (_req: Request, res: Response) => {
  try {
    const markets = await db.getMarkets();
    
    // Get current prices for all markets
    const marketData = await Promise.all(
      markets.map(async (market) => {
        const symbol = market.base_asset;
        const oraclePrice = await oracle.getPrice(symbol);
        
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
          currentPrice: oraclePrice?.price || 0,
          priceChange24h: 0, // Calculate from historical data
          volume24h: 0, // Calculate from trades
          openInterest: 0, // Calculate from positions
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

// Get market by symbol
router.get('/:symbol', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const market = await db.getMarketBySymbol(symbol);
    
    if (!market) {
      return res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    const oraclePrice = await oracle.getPrice(market.base_asset);
    
    // Get market statistics using fluent API
    const trades24h = await db.select('trades', 'size, price', {
      market_id: market.id
    });
    
    // Filter trades from last 24 hours
    const now = new Date();
    const yesterday = new Date(now.getTime() - 24 * 60 * 60 * 1000);
    const recentTrades = trades24h.filter(trade => new Date(trade.created_at) >= yesterday);
    
    const volume24h = recentTrades.reduce((sum, trade) => sum + (trade.size * trade.price), 0);
    const tradesCount = recentTrades.length;

    // Get open interest using fluent API
    const positions = await db.select('positions', 'size, entry_price', {
      market_id: market.id
    });
    
    const activePositions = positions.filter(pos => pos.size > 0 && !pos.is_liquidated);
    const oi = activePositions.reduce((sum, pos) => sum + (pos.size * pos.entry_price), 0);

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
        currentPrice: oraclePrice?.price || 0,
        priceChange24h: 0,
        volume24h,
        tradesCount24h: tradesCount,
        openInterest: oi,
        createdAt: market.created_at
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

// Get market order book
router.get('/:symbol/orderbook', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const market = await db.getMarketBySymbol(symbol);
    
    if (!market) {
      return res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    // Get pending orders using fluent API
    let orders;
    try {
      const orderData = await db.select('orders', 'side, price, remaining_size', {
        market_id: market.id,
        status: 'pending'
      });
      
      // Sort by price descending
      orders = { rows: orderData.sort((a, b) => b.price - a.price) };
    } catch (dbError) {
      logger.warn(`Database query failed for orderbook ${symbol}, returning empty orderbook:`, dbError);
      // Return empty orderbook if database query fails
      return res.json({
        success: true,
        orderbook: {
          symbol,
          bids: [],
          asks: [],
          spread: 0,
          timestamp: Date.now()
        }
      });
    }

    // Build order book
    const bids: [number, number][] = [];
    const asks: [number, number][] = [];

    // Handle case where orders.rows might be undefined
    const orderRows = orders?.rows || [];
    
    for (const order of orderRows) {
      const price = parseFloat(order.price);
      const size = parseFloat(order.remaining_size);

      // Skip invalid orders
      if (isNaN(price) || isNaN(size) || price <= 0 || size <= 0) {
        continue;
      }

      if (order.side === 'long') {
        bids.push([price, size]);
      } else {
        asks.push([price, size]);
      }
    }

    // Sort bids (highest first) and asks (lowest first)
    bids.sort((a, b) => b[0] - a[0]);
    asks.sort((a, b) => a[0] - b[0]);

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

// Get market trades
router.get('/:symbol/trades', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;
  const limit = parseInt(req.query.limit as string) || 100;

  try {
    const market = await db.getMarketBySymbol(symbol);
    
    if (!market) {
      return res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    // Get trades using fluent API
    const trades = await db.select('trades', '*', {
      market_id: market.id
    });
    
    // Sort by created_at descending and limit
    const sortedTrades = trades
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, limit);
    
    // Get user wallet addresses for trades
    const userIds = [...new Set(sortedTrades.map(trade => trade.user_id))];
    const users = await Promise.all(
      userIds.map(userId => db.select('users', 'id, wallet_pubkey', { id: userId }))
    );
    
    const userMap = new Map();
    users.forEach(userList => {
      if (userList.length > 0) {
        const user = userList[0];
        userMap.set(user.id, user.wallet_pubkey);
      }
    });

    res.json({
      success: true,
      trades: sortedTrades.map((trade: any) => ({
        id: trade.id,
        side: trade.side,
        size: trade.size,
        price: trade.price,
        value: trade.value,
        fees: trade.fees,
        pnl: trade.pnl,
        timestamp: trade.created_at,
        user: userMap.get(trade.user_id)?.substring(0, 8) + '...' || 'Unknown'
      }))
    });

  } catch (error) {
    logger.error(`Error fetching trades for ${symbol}:`, error);
    res.status(500).json({
      error: 'Failed to fetch trades',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get market funding rates
router.get('/:symbol/funding', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;
  const limit = parseInt(req.query.limit as string) || 24;

  try {
    const market = await db.getMarketBySymbol(symbol);
    
    if (!market) {
      return res.status(404).json({
        error: 'Market not found',
        code: 'MARKET_NOT_FOUND'
      });
    }

    // Get funding rates using fluent API
    const fundingRates = await db.select('funding_rates', '*', {
      market_id: market.id
    });
    
    // Sort by created_at descending and limit
    const sortedRates = fundingRates
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(0, limit);

    res.json({
      success: true,
      fundingRates: sortedRates.map((rate: any) => ({
        fundingRate: rate.funding_rate,
        premiumIndex: rate.premium_index,
        oraclePrice: rate.oracle_price,
        markPrice: rate.mark_price,
        totalFunding: rate.total_funding,
        timestamp: rate.created_at
      }))
    });

  } catch (error) {
    logger.error(`Error fetching funding rates for ${symbol}:`, error);
    res.status(500).json({
      error: 'Failed to fetch funding rates',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get real-time price for a market
router.get('/:symbol/price', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;

  try {
    const price = await pythOracleService.getLatestPrice(symbol);
    
    if (price === null) {
      return res.status(404).json({
        error: 'Price not found for market',
        code: 'PRICE_NOT_FOUND'
      });
    }

    res.json({
      success: true,
      symbol,
      price,
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

// Get price history for charts
router.get('/:symbol/price-history', asyncHandler(async (req: Request, res: Response) => {
  const { symbol } = req.params;
  const hours = parseInt(req.query.hours as string) || 24;

  try {
    const priceHistory = await pythOracleService.getPriceHistory(symbol, hours);
    
    res.json({
      success: true,
      symbol,
      hours,
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

export default router;
