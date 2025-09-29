import express, { Request, Response } from 'express';
import { DatabaseService } from '../services/database';
import { OracleService } from '../services/oracle';
import { pythOracleService } from '../services/pythOracleService';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandler';

const router = express.Router();
const logger = new Logger();
const db = DatabaseService.getInstance();
const oracle = OracleService.getInstance();

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
    
    // Get market statistics
    const stats = await db.query(
      `SELECT 
         COUNT(*) as trades_count,
         SUM(size * price) as volume_usd,
         AVG(price) as avg_price
       FROM trades 
       WHERE market_id = $1 
       AND created_at >= NOW() - INTERVAL '24 hours'`,
      [market.id]
    );

    const row = stats.rows[0] || {};
    const volume24h = row.volume_usd != null ? parseFloat(row.volume_usd) : 0;
    const tradesCount = row.trades_count != null ? parseInt(row.trades_count) : 0;

    // Get open interest
    const openInterest = await db.query(
      `SELECT SUM(size * entry_price) as open_interest
       FROM positions 
       WHERE market_id = $1 AND size > 0 AND NOT is_liquidated`,
      [market.id]
    );

    const oi = parseFloat(openInterest.rows[0]?.open_interest) || 0;

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

    // Get pending orders
    const orders = await db.query(
      `SELECT side, price, remaining_size
       FROM orders 
       WHERE market_id = $1 AND status = 'pending'
       ORDER BY price DESC`,
      [market.id]
    );

    // Build order book
    const bids: [number, number][] = [];
    const asks: [number, number][] = [];

    for (const order of orders.rows) {
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

    const trades = await db.query(
      `SELECT t.*, u.wallet_address
       FROM trades t
       JOIN users u ON t.user_id = u.id
       WHERE t.market_id = $1
       ORDER BY t.created_at DESC
       LIMIT $2`,
      [market.id, limit]
    );

    res.json({
      success: true,
      trades: trades.rows.map((trade: any) => ({
        id: trade.id,
        side: trade.side,
        size: trade.size,
        price: trade.price,
        value: trade.value,
        fees: trade.fees,
        pnl: trade.pnl,
        timestamp: trade.created_at,
        user: trade.wallet_address.substring(0, 8) + '...' // Partial wallet for privacy
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

    const fundingRates = await db.query(
      `SELECT funding_rate, premium_index, oracle_price, mark_price, total_funding, created_at
       FROM funding_rates
       WHERE market_id = $1
       ORDER BY created_at DESC
       LIMIT $2`,
      [market.id, limit]
    );

    res.json({
      success: true,
      fundingRates: fundingRates.rows.map((rate: any) => ({
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
