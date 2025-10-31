import { Server as SocketIOServer } from 'socket.io';
import { pythOracleService } from './pythOracleService';
import { Logger } from '../utils/logger';
import { config } from '../config/environment';
import { SupabaseDatabaseService } from './supabaseDatabase';
import { portfolioCalculationService } from './portfolioCalculationService';
import { portfolioBackgroundService } from './portfolioBackgroundService';

const logger = new Logger();

export interface WebSocketMessage {
  type: string;
  data: any;
  timestamp: number;
}

export interface MarketDataUpdate {
  symbol: string;
  price: number;
  change24h: number;
  volume24h: number;
  openInterest: number;
  fundingRate: number;
}

export interface OrderBookUpdate {
  symbol: string;
  bids: [number, number][];
  asks: [number, number][];
  spread: number;
}

export interface PositionUpdate {
  userId: string;
  positionId: string;
  symbol?: string;
  side?: 'long' | 'short';
  size?: number;
  entryPrice?: number;
  leverage?: number;
  status?: string;
  unrealizedPnl?: number;
  marginRatio?: number;
  healthFactor?: number;
  timestamp?: number;
}

export interface TradeUpdate {
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  price: number;
  timestamp: number;
}

export interface PortfolioUpdate {
  userId: string;
  totalValue: number;
  totalUnrealizedPnl: number;
  totalRealizedPnl: number;
  marginRatio: number;
  healthFactor: number;
  positions: Array<{
    id: string;
    symbol: string;
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnl: number;
    unrealizedPnlPercent: number;
    margin: number;
    leverage: number;
  }>;
  timestamp: number;
}

export class WebSocketService {
  private static instance: WebSocketService;
  public static current: WebSocketService | undefined;
  private io: SocketIOServer;
  private db: SupabaseDatabaseService;
  private oracle: typeof pythOracleService;
  private connectedClients: Map<string, Set<string>> = new Map(); // userId -> Set of socketIds
  private marketSubscriptions: Map<string, Set<string>> = new Map(); // symbol -> Set of socketIds
  private portfolioSubscriptions: Map<string, Set<string>> = new Map(); // userId -> Set of socketIds
  private isRunning: boolean = false;
  public broadcast?: (type: string, data: any) => void;

  private constructor(io: SocketIOServer) {
    this.io = io;
    this.db = SupabaseDatabaseService.getInstance();
    this.oracle = pythOracleService;
  }

  public static getInstance(io: SocketIOServer): WebSocketService {
    if (!WebSocketService.instance) {
      WebSocketService.instance = new WebSocketService(io);
    }
    WebSocketService.current = WebSocketService.instance;
    return WebSocketService.instance;
  }

  public initialize(): void {
    this.setupSocketHandlers();
    this.startDataBroadcasting();
    this.isRunning = true;
    logger.info('WebSocket service initialized');

    // Start portfolio background service
    portfolioBackgroundService.start();

    // Simple broadcast helper
    this.broadcast = (type: string, data: any) => {
      const payload = { type, data, timestamp: Date.now() };
      if (data?.symbol) {
        this.io.to(`market:${data.symbol}`).emit('message', payload);
      }
      this.io.emit('message', payload);
    };
    WebSocketService.current = this;
  }

  private setupSocketHandlers(): void {
    this.io.on('connection', (socket) => {
      logger.info(`Client connected: ${socket.id}`);

      // Handle authentication
      socket.on('authenticate', async (data: { token: string }) => {
        try {
          const user = await this.authenticateUser(data.token);
          if (user) {
            socket.data.userId = user.id;
            socket.data.walletAddress = user.wallet_address;
            
            // Track connected client
            if (!this.connectedClients.has(user.id)) {
              this.connectedClients.set(user.id, new Set());
            }
            this.connectedClients.get(user.id)!.add(socket.id);

            socket.emit('authenticated', { userId: user.id });
            logger.info(`User ${user.wallet_address} authenticated`);
          } else {
            socket.emit('auth_error', { message: 'Invalid token' });
          }
        } catch (error) {
          logger.error('Authentication error:', error);
          socket.emit('auth_error', { message: 'Authentication failed' });
        }
      });

      // Handle market data subscription (no auth required for public data)
      socket.on('subscribe_market_data', () => {
        logger.info(`Client ${socket.id} subscribed to market data`);
        
        // Subscribe to all market symbols
        const symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'ARB', 'OP', 'DOGE', 'ADA', 'DOT', 'LINK'];
        symbols.forEach(symbol => {
          const symbolKey = `${symbol}-PERP`;
          if (!this.marketSubscriptions.has(symbolKey)) {
            this.marketSubscriptions.set(symbolKey, new Set());
          }
          this.marketSubscriptions.get(symbolKey)!.add(socket.id);
          socket.join(`market:${symbolKey}`);
        });
        
        logger.info(`Client ${socket.id} subscribed to all market data`);
      });

      // Handle market subscriptions
      socket.on('subscribe_market', (data: { symbol: string }) => {
        if (!socket.data.userId) {
          socket.emit('error', { message: 'Not authenticated' });
          return;
        }

        const symbol = data.symbol.toUpperCase();
        if (!this.marketSubscriptions.has(symbol)) {
          this.marketSubscriptions.set(symbol, new Set());
        }
        this.marketSubscriptions.get(symbol)!.add(socket.id);

        socket.join(`market:${symbol}`);
        logger.info(`User ${socket.data.walletAddress} subscribed to ${symbol}`);
      });

      socket.on('unsubscribe_market', (data: { symbol: string }) => {
        const symbol = data.symbol.toUpperCase();
        this.marketSubscriptions.get(symbol)?.delete(socket.id);
        socket.leave(`market:${symbol}`);
        logger.info(`User ${socket.data.walletAddress} unsubscribed from ${symbol}`);
      });

      // Handle position updates subscription
      socket.on('subscribe_positions', () => {
        if (!socket.data.userId) {
          socket.emit('error', { message: 'Not authenticated' });
          return;
        }

        socket.join(`positions:${socket.data.userId}`);
        logger.info(`User ${socket.data.walletAddress} subscribed to position updates`);
      });

      // Handle order updates subscription
      socket.on('subscribe_orders', () => {
        if (!socket.data.userId) {
          socket.emit('error', { message: 'Not authenticated' });
          return;
        }

        socket.join(`orders:${socket.data.userId}`);
        logger.info(`User ${socket.data.walletAddress} subscribed to order updates`);
      });

      socket.on('unsubscribe_orders', () => {
        if (!socket.data.userId) {
          socket.emit('error', { message: 'Not authenticated' });
          return;
        }

        socket.leave(`orders:${socket.data.userId}`);
        logger.info(`User ${socket.data.walletAddress} unsubscribed from order updates`);
      });

      // Handle portfolio updates subscription
      socket.on('subscribe_portfolio', () => {
        if (!socket.data.userId) {
          socket.emit('error', { message: 'Not authenticated' });
          return;
        }

        const userId = socket.data.userId;
        
        // Add to portfolio subscriptions
        if (!this.portfolioSubscriptions.has(userId)) {
          this.portfolioSubscriptions.set(userId, new Set());
        }
        this.portfolioSubscriptions.get(userId)!.add(socket.id);
        
        // Join user-specific portfolio room
        socket.join(`portfolio:${userId}`);
        
        logger.info(`User ${socket.data.walletAddress} subscribed to portfolio updates`);
        
        // Send initial portfolio data
        this.sendPortfolioUpdate(userId);
      });

      socket.on('unsubscribe_portfolio', () => {
        if (!socket.data.userId) {
          socket.emit('error', { message: 'Not authenticated' });
          return;
        }

        const userId = socket.data.userId;
        this.portfolioSubscriptions.get(userId)?.delete(socket.id);
        socket.leave(`portfolio:${userId}`);
        logger.info(`User ${socket.data.walletAddress} unsubscribed from portfolio updates`);
      });


      // Handle disconnect
      socket.on('disconnect', () => {
        this.handleDisconnect(socket);
      });
    });
  }

  private async authenticateUser(token: string): Promise<any> {
    try {
      const jwt = require('jsonwebtoken');
      const decoded = jwt.verify(token, config.JWT_SECRET);
      
      if (!decoded.walletAddress) {
        return null;
      }

      const user = await this.db.getUserByWallet(decoded.walletAddress);
      return user;
    } catch (error) {
      logger.error('Token verification failed:', error);
      return null;
    }
  }

  private handleDisconnect(socket: any): void {
    logger.info(`Client disconnected: ${socket.id}`);

    // Remove from connected clients
    if (socket.data.userId) {
      const userClients = this.connectedClients.get(socket.data.userId);
      if (userClients) {
        userClients.delete(socket.id);
        if (userClients.size === 0) {
          this.connectedClients.delete(socket.data.userId);
        }
      }
    }

    // Remove from market subscriptions
    for (const [symbol, subscribers] of this.marketSubscriptions) {
      subscribers.delete(socket.id);
      if (subscribers.size === 0) {
        this.marketSubscriptions.delete(symbol);
      }
    }

    // Remove from portfolio subscriptions
    if (socket.data.userId) {
      const portfolioSubscribers = this.portfolioSubscriptions.get(socket.data.userId);
      if (portfolioSubscribers) {
        portfolioSubscribers.delete(socket.id);
        if (portfolioSubscribers.size === 0) {
          this.portfolioSubscriptions.delete(socket.data.userId);
        }
      }
    }
  }

  private startDataBroadcasting(): void {
    // Broadcast market data every 1 second for live updates
    setInterval(async () => {
      try {
        await this.broadcastMarketData();
      } catch (error) {
        logger.error('Error broadcasting market data:', error);
      }
    }, 1000);

    // Broadcast order book updates every 500ms for real-time trading
    setInterval(async () => {
      try {
        await this.broadcastOrderBookUpdates();
      } catch (error) {
        logger.error('Error broadcasting order book updates:', error);
      }
    }, 500);

    // Broadcast trade updates in real-time
    this.setupTradeBroadcasting();

    // Portfolio updates are now handled by the dedicated background service
    // No need to call startPortfolioUpdates() here
  }

  private async broadcastMarketData(): Promise<void> {
    const symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'ARB', 'OP', 'DOGE', 'ADA', 'DOT', 'LINK'];
    
    for (const symbol of symbols) {
      const subscribers = this.marketSubscriptions.get(`${symbol}-PERP`);
      if (!subscribers || subscribers.size === 0) {
        continue;
      }

      try {
        const marketData = await this.getMarketData(symbol);
        if (marketData) {
          const message: WebSocketMessage = {
            type: 'market_data',
            data: marketData,
            timestamp: Date.now()
          };

          this.io.to(`market:${symbol}-PERP`).emit('market_update', message);
          
          // Also broadcast to general market data subscribers
          this.io.emit('market_data_update', message);
        }
      } catch (error) {
        logger.error(`Error broadcasting market data for ${symbol}:`, error);
      }
    }
  }

  private async getMarketData(symbol: string): Promise<MarketDataUpdate | null> {
    try {
      // Get price directly from Pyth Oracle service
      const oraclePrice = await this.oracle.getPrice(symbol);
      if (!oraclePrice) {
        logger.warn(`No price data available for ${symbol}`);
        return null;
      }

      // Calculate 24h change (simplified - in production, use historical data)
      const change24h = (Math.random() - 0.5) * oraclePrice.price * 0.1;

      logger.info(`📊 Broadcasting market data for ${symbol}: $${oraclePrice.price.toFixed(2)}`);

      return {
        symbol: `${symbol}-PERP`,
        price: oraclePrice.price,
        change24h: change24h,
        volume24h: Math.random() * 1000000,
        openInterest: Math.random() * 50000000,
        fundingRate: (Math.random() - 0.5) * 0.01
      };
    } catch (error) {
      logger.error(`Error getting market data for ${symbol}:`, error);
      return null;
    }
  }

  private async broadcastOrderBookUpdates(): Promise<void> {
    const symbols = ['BTC', 'ETH', 'SOL'];
    
    for (const symbol of symbols) {
      const subscribers = this.marketSubscriptions.get(`${symbol}-PERP`);
      if (!subscribers || subscribers.size === 0) {
        continue;
      }

      try {
        const orderBook = await this.getOrderBook(symbol);
        if (orderBook) {
          const message: WebSocketMessage = {
            type: 'order_book',
            data: orderBook,
            timestamp: Date.now()
          };

          this.io.to(`market:${symbol}-PERP`).emit('order_book_update', message);
        }
      } catch (error) {
        logger.error(`Error broadcasting order book for ${symbol}:`, error);
      }
    }
  }

  private async getOrderBook(symbol: string): Promise<OrderBookUpdate | null> {
    try {
      const market = await this.db.getMarketBySymbol(`${symbol}-PERP`);
      if (!market) {
        return null;
      }

      // Get pending orders using fluent API
      const orderData = await this.db.select('orders', 'side, price, remaining_size', {
        market_id: market.id,
        status: 'pending'
      });
      
      // Sort by price descending
      const orders = { rows: orderData.sort((a, b) => b.price - a.price) };

      // Build order book (simplified)
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

      return {
        symbol: `${symbol}-PERP`,
        bids: bids.slice(0, 20), // Top 20 bids
        asks: asks.slice(0, 20), // Top 20 asks
        spread
      };
    } catch (error) {
      logger.error(`Error getting order book for ${symbol}:`, error);
      return null;
    }
  }

  private setupTradeBroadcasting(): void {
    // In production, this would listen to blockchain events
    // For now, we'll simulate trade updates
    setInterval(() => {
      this.simulateTradeUpdates();
    }, 5000);
  }

  private simulateTradeUpdates(): void {
    const symbols = ['BTC', 'ETH', 'SOL'];
    
    symbols.forEach(symbol => {
      const subscribers = this.marketSubscriptions.get(`${symbol}-PERP`);
      if (!subscribers || subscribers.size === 0) {
        return;
      }

      // Simulate a random trade
      const side = Math.random() > 0.5 ? 'buy' : 'sell';
      const price = 50000 + Math.random() * 10000; // Random price
      const size = 0.1 + Math.random() * 0.9; // Random size

      const trade: TradeUpdate = {
        symbol: `${symbol}-PERP`,
        side,
        size,
        price,
        timestamp: Date.now()
      };

      const message: WebSocketMessage = {
        type: 'trade',
        data: trade,
        timestamp: Date.now()
      };

      this.io.to(`market:${symbol}-PERP`).emit('trade_update', message);
    });
  }

  // Public methods for broadcasting updates
  public broadcastPositionUpdate(userId: string, positionUpdate: PositionUpdate): void {
    const message: WebSocketMessage = {
      type: 'position_update',
      data: positionUpdate,
      timestamp: Date.now()
    };

    this.io.to(`positions:${userId}`).emit('position_update', message);
  }

  public broadcastOrderUpdate(userId: string, orderUpdate: any): void {
    const message: WebSocketMessage = {
      type: 'order_update',
      data: orderUpdate,
      timestamp: Date.now()
    };

    this.io.to(`orders:${userId}`).emit('order_update', message);
  }

  public broadcastTradeUpdate(tradeUpdate: TradeUpdate): void {
    const message: WebSocketMessage = {
      type: 'trade',
      data: tradeUpdate,
      timestamp: Date.now()
    };

    this.io.to(`market:${tradeUpdate.symbol}`).emit('trade_update', message);
  }

  public getConnectedClientsCount(): number {
    return this.connectedClients.size;
  }

  public getMarketSubscribersCount(symbol: string): number {
    return this.marketSubscriptions.get(symbol)?.size || 0;
  }

  public getPortfolioSubscribersCount(userId: string): number {
    return this.portfolioSubscriptions.get(userId)?.size || 0;
  }

  /**
   * Send portfolio update to a specific user
   */
  public async sendPortfolioUpdate(userId: string): Promise<void> {
    try {
      const portfolioData = await portfolioCalculationService.calculatePortfolio(userId);
      if (portfolioData) {
        this.broadcastPortfolioUpdate(userId, portfolioData);
      }
    } catch (error) {
      logger.error(`Error sending portfolio update for user ${userId}:`, error);
    }
  }

  /**
   * Broadcast portfolio update to all subscribers of a user
   */
  public broadcastPortfolioUpdate(userId: string, portfolioData: any): void {
    const message: WebSocketMessage = {
      type: 'portfolio_update',
      data: portfolioData,
      timestamp: Date.now()
    };

    this.io.to(`portfolio:${userId}`).emit('portfolio_update', message);
  }

  /**
   * Broadcast message to a specific user
   */
  public broadcastToUser(userId: string, event: string, data: any): void {
    if (!this.io) return;
    
    this.io.to(`orders:${userId}`).emit(event, data);
    logger.debug(`Broadcasted ${event} to user ${userId}`);
  }

  public stop(): void {
    this.isRunning = false;
    
    // Stop portfolio background service
    portfolioBackgroundService.stop();
    
    logger.info('WebSocket service stopped');
  }
}
