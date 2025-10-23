/**
 * QuantDesk Backend API Service Examples
 * 
 * This file demonstrates reusable API service patterns for building trading backends.
 * These services are open source and can be used by the community.
 */

import express, { Request, Response, NextFunction } from 'express';
import { z } from 'zod';
import { DatabaseService } from './services/database';
import { OracleService } from './services/oracle';
import { RateLimiter } from './middleware/rateLimiter';
import { AuthMiddleware } from './middleware/auth';

// Example: Market Data Service
export class MarketDataService {
  constructor(
    private database: DatabaseService,
    private oracle: OracleService
  ) {}

  async getPrice(symbol: string): Promise<number> {
    try {
      // Get latest price from oracle
      const price = await this.oracle.getPrice(symbol);
      
      // Store in database for historical tracking
      await this.database.storePriceUpdate(symbol, price, new Date());
      
      return price;
    } catch (error) {
      console.error(`Failed to get price for ${symbol}:`, error);
      throw new Error('Price data unavailable');
    }
  }

  async getPriceHistory(symbol: string, timeframe: string): Promise<Array<{
    timestamp: Date;
    price: number;
    volume: number;
  }>> {
    return await this.database.getPriceHistory(symbol, timeframe);
  }

  async getMarketSummary(): Promise<{
    totalVolume: number;
    activeMarkets: number;
    priceChanges: Array<{ symbol: string; change24h: number }>;
  }> {
    const markets = await this.database.getActiveMarkets();
    const summary = await Promise.all(
      markets.map(async (market) => {
        const currentPrice = await this.getPrice(market.symbol);
        const previousPrice = await this.database.getPriceAtTime(
          market.symbol, 
          new Date(Date.now() - 24 * 60 * 60 * 1000)
        );
        
        return {
          symbol: market.symbol,
          change24h: previousPrice ? 
            ((currentPrice - previousPrice) / previousPrice) * 100 : 0
        };
      })
    );

    return {
      totalVolume: markets.reduce((sum, m) => sum + m.volume24h, 0),
      activeMarkets: markets.length,
      priceChanges: summary
    };
  }
}

// Example: Order Management Service
export class OrderService {
  constructor(private database: DatabaseService) {}

  async createOrder(userId: string, orderData: {
    symbol: string;
    side: 'buy' | 'sell';
    amount: number;
    price?: number;
    type: 'market' | 'limit';
  }): Promise<string> {
    // Validate order data
    const orderSchema = z.object({
      symbol: z.string().min(1),
      side: z.enum(['buy', 'sell']),
      amount: z.number().positive(),
      price: z.number().positive().optional(),
      type: z.enum(['market', 'limit'])
    });

    const validatedData = orderSchema.parse(orderData);

    // Check user balance for sell orders
    if (validatedData.side === 'sell') {
      const balance = await this.database.getUserBalance(userId, validatedData.symbol);
      if (balance < validatedData.amount) {
        throw new Error('Insufficient balance');
      }
    }

    // Create order
    const orderId = await this.database.createOrder({
      userId,
      ...validatedData,
      status: 'pending',
      createdAt: new Date()
    });

    return orderId;
  }

  async getOrders(userId: string, status?: string): Promise<Array<{
    id: string;
    symbol: string;
    side: string;
    amount: number;
    price?: number;
    status: string;
    createdAt: Date;
  }>> {
    return await this.database.getUserOrders(userId, status);
  }

  async cancelOrder(userId: string, orderId: string): Promise<void> {
    const order = await this.database.getOrder(orderId);
    
    if (!order) {
      throw new Error('Order not found');
    }
    
    if (order.userId !== userId) {
      throw new Error('Unauthorized');
    }
    
    if (order.status !== 'pending') {
      throw new Error('Cannot cancel filled order');
    }

    await this.database.updateOrderStatus(orderId, 'cancelled');
  }
}

// Example: Portfolio Service
export class PortfolioService {
  constructor(private database: DatabaseService) {}

  async getPortfolio(userId: string): Promise<{
    totalValue: number;
    positions: Array<{
      symbol: string;
      amount: number;
      value: number;
      pnl: number;
    }>;
  }> {
    const positions = await this.database.getUserPositions(userId);
    const balances = await this.database.getUserBalances(userId);
    
    const portfolioPositions = positions.map(position => {
      const balance = balances.find(b => b.symbol === position.symbol);
      const currentValue = position.amount * (balance?.price || 0);
      const pnl = currentValue - position.costBasis;
      
      return {
        symbol: position.symbol,
        amount: position.amount,
        value: currentValue,
        pnl
      };
    });

    const totalValue = portfolioPositions.reduce((sum, pos) => sum + pos.value, 0);

    return {
      totalValue,
      positions: portfolioPositions
    };
  }

  async getBalance(userId: string, symbol: string): Promise<number> {
    const balance = await this.database.getUserBalance(userId, symbol);
    return balance || 0;
  }
}

// Example: API Routes
export const createApiRoutes = (
  marketDataService: MarketDataService,
  orderService: OrderService,
  portfolioService: PortfolioService
) => {
  const router = express.Router();

  // Market data routes
  router.get('/market-data/:symbol', async (req: Request, res: Response) => {
    try {
      const { symbol } = req.params;
      const price = await marketDataService.getPrice(symbol);
      res.json({ symbol, price, timestamp: new Date() });
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch market data' });
    }
  });

  router.get('/market-data/:symbol/history', async (req: Request, res: Response) => {
    try {
      const { symbol } = req.params;
      const { timeframe = '1h' } = req.query;
      const history = await marketDataService.getPriceHistory(symbol, timeframe as string);
      res.json(history);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch price history' });
    }
  });

  router.get('/market-summary', async (req: Request, res: Response) => {
    try {
      const summary = await marketDataService.getMarketSummary();
      res.json(summary);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch market summary' });
    }
  });

  // Order routes (protected)
  router.post('/orders', AuthMiddleware, async (req: Request, res: Response) => {
    try {
      const userId = req.user?.id;
      const orderId = await orderService.createOrder(userId, req.body);
      res.json({ orderId, status: 'created' });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  });

  router.get('/orders', AuthMiddleware, async (req: Request, res: Response) => {
    try {
      const userId = req.user?.id;
      const { status } = req.query;
      const orders = await orderService.getOrders(userId, status as string);
      res.json(orders);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch orders' });
    }
  });

  router.delete('/orders/:orderId', AuthMiddleware, async (req: Request, res: Response) => {
    try {
      const userId = req.user?.id;
      const { orderId } = req.params;
      await orderService.cancelOrder(userId, orderId);
      res.json({ status: 'cancelled' });
    } catch (error) {
      res.status(400).json({ error: error.message });
    }
  });

  // Portfolio routes (protected)
  router.get('/portfolio', AuthMiddleware, async (req: Request, res: Response) => {
    try {
      const userId = req.user?.id;
      const portfolio = await portfolioService.getPortfolio(userId);
      res.json(portfolio);
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch portfolio' });
    }
  });

  router.get('/balance/:symbol', AuthMiddleware, async (req: Request, res: Response) => {
    try {
      const userId = req.user?.id;
      const { symbol } = req.params;
      const balance = await portfolioService.getBalance(userId, symbol);
      res.json({ symbol, balance });
    } catch (error) {
      res.status(500).json({ error: 'Failed to fetch balance' });
    }
  });

  return router;
};

// Example: Error handling middleware
export const errorHandler = (
  error: Error,
  req: Request,
  res: Response,
  next: NextFunction
) => {
  console.error('API Error:', error);
  
  if (error.name === 'ValidationError') {
    return res.status(400).json({ error: 'Invalid request data' });
  }
  
  if (error.name === 'UnauthorizedError') {
    return res.status(401).json({ error: 'Unauthorized' });
  }
  
  res.status(500).json({ error: 'Internal server error' });
};

// Example: Rate limiting middleware
export const createRateLimiter = () => {
  return RateLimiter({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // limit each IP to 100 requests per windowMs
    message: 'Too many requests from this IP'
  });
};

export default {
  MarketDataService,
  OrderService,
  PortfolioService,
  createApiRoutes,
  errorHandler,
  createRateLimiter
};
