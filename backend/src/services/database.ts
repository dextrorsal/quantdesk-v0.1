import { Pool, PoolClient } from 'pg';
import { supabaseService } from './supabaseService';
import { Logger } from '../utils/logger';

const logger = new Logger();

export interface User {
  id: string;
  wallet_address: string;
  username?: string;
  email?: string;
  created_at: Date;
  updated_at: Date;
  last_login?: Date;
  is_active: boolean;
  kyc_status: string;
  risk_level: string;
  total_volume: number;
  total_trades: number;
  metadata: any;
}

export interface Market {
  id: string;
  symbol: string;
  base_asset: string;
  quote_asset: string;
  program_id: string;
  market_account: string;
  oracle_account: string;
  is_active: boolean;
  max_leverage: number;
  initial_margin_ratio: number;
  maintenance_margin_ratio: number;
  tick_size: number;
  step_size: number;
  min_order_size: number;
  max_order_size: number;
  funding_interval: number;
  last_funding_time?: Date;
  current_funding_rate: number;
  created_at: Date;
  updated_at: Date;
  metadata: any;
}

export interface Position {
  id: string;
  user_id: string;
  market_id: string;
  position_account: string;
  side: 'long' | 'short';
  size: number;
  entry_price: number;
  current_price?: number;
  margin: number;
  leverage: number;
  unrealized_pnl: number;
  realized_pnl: number;
  funding_fees: number;
  is_liquidated: boolean;
  liquidation_price?: number;
  health_factor?: number;
  created_at: Date;
  updated_at: Date;
  closed_at?: Date;
}

export interface Order {
  id: string;
  user_id: string;
  market_id: string;
  order_account: string;
  order_type: 'market' | 'limit' | 'stop_loss' | 'take_profit' | 'trailing_stop' | 'post_only' | 'ioc' | 'fok';
  side: 'long' | 'short';
  size: number;
  price?: number;
  stop_price?: number;
  trailing_distance?: number;
  leverage: number;
  status: 'pending' | 'filled' | 'cancelled' | 'expired' | 'partially_filled';
  filled_size: number;
  average_fill_price?: number;
  created_at: Date;
  updated_at: Date;
  expires_at?: Date;
  filled_at?: Date;
  cancelled_at?: Date;
}

export interface Trade {
  id: string;
  user_id: string;
  market_id: string;
  position_id?: string;
  order_id?: string;
  trade_account: string;
  side: 'buy' | 'sell';
  size: number;
  price: number;
  fees: number;
  pnl?: number;
  created_at: Date;
  metadata: any;
}

export class DatabaseService {
  private static instance: DatabaseService;
  private pool: Pool;

  private constructor() {
    // Force IPv4 lookup to avoid ENETUNREACH on some networks/providers
    const ipv4Lookup: any = (hostname: string, _opts: any, cb: any) => {
      try {
        const dns = require('dns');
        dns.lookup(hostname, { family: 4, all: false }, cb);
      } catch (e) {
        cb(e, null, 4);
      }
    };

    this.pool = new Pool({
      connectionString: process.env['DATABASE_URL'],
      ssl: process.env['NODE_ENV'] === 'production' ? { 
        rejectUnauthorized: true,
        // Add CA certificate if available
        ...(process.env['DATABASE_CA_CERT'] && { ca: process.env['DATABASE_CA_CERT'] })
      } : false,
      max: 20,
      idleTimeoutMillis: 30000,
      connectionTimeoutMillis: 5000,
      statement_timeout: 15000,
      query_timeout: 15000,
      keepAlive: true,
      keepAliveInitialDelayMillis: 10000,
      application_name: 'quantdesk-backend',
      // @ts-expect-error: lookup is supported by pg ClientConfig
      lookup: ipv4Lookup,
    } as any);

    this.pool.on('error', (err) => {
      logger.error('Database pool error:', err);
    });

    this.pool.on('connect', () => {
      logger.info('Database connection established');
    });

    this.pool.on('remove', () => {
      logger.info('Database connection removed');
    });
  }

  public static getInstance(): DatabaseService {
    if (!DatabaseService.instance) {
      DatabaseService.instance = new DatabaseService();
    }
    return DatabaseService.instance;
  }

  public async connect(): Promise<void> {
    try {
      await this.pool.connect();
      logger.info('Database connected successfully');
    } catch (error) {
      logger.error('Database connection failed:', error);
      throw error;
    }
  }

  public async disconnect(): Promise<void> {
    try {
      await this.pool.end();
      logger.info('Database disconnected successfully');
    } catch (error) {
      logger.error('Database disconnection failed:', error);
      throw error;
    }
  }

  public async query(text: string, params?: any[]): Promise<any> {
    try {
      const start = Date.now();
      const result = await this.pool.query(text, params);
      const duration = Date.now() - start;
      
      if (duration > 1000) {
        logger.warn(`Slow query detected: ${duration}ms - ${text.substring(0, 100)}...`);
      }
      
      return result;
    } catch (error) {
      logger.error('Database query error:', error);
      throw error;
    }
  }

  public async transaction<T>(callback: (client: PoolClient) => Promise<T>): Promise<T> {
    const client = await this.pool.connect();
    try {
      await client.query('BEGIN');
      const result = await callback(client);
      await client.query('COMMIT');
      return result;
    } catch (error) {
      await client.query('ROLLBACK');
      throw error;
    } finally {
      client.release();
    }
  }

  // User operations
  public async getUserByWalletAddress(walletAddress: string): Promise<User | null> {
    const result = await this.query(
      'SELECT * FROM users WHERE wallet_address = $1',
      [walletAddress]
    );
    return result.rows[0] || null;
  }

  public async getUserById(id: string): Promise<User | null> {
    const result = await this.query(
      'SELECT * FROM users WHERE id = $1',
      [id]
    );
    return result.rows[0] || null;
  }

  public async createUser(walletAddress: string, username?: string, email?: string): Promise<User> {
    const result = await this.query(
      `INSERT INTO users (wallet_address, username, email) 
       VALUES ($1, $2, $3) 
       RETURNING *`,
      [walletAddress, username, email]
    );
    return result.rows[0];
  }

  public async updateUser(id: string, updates: Partial<User>): Promise<User> {
    const fields = Object.keys(updates).filter(key => key !== 'id');
    const values = fields.map((field, index) => `${field} = $${index + 2}`);
    
    const result = await this.query(
      `UPDATE users SET ${values.join(', ')}, updated_at = NOW() 
       WHERE id = $1 RETURNING *`,
      [id, ...Object.values(updates)]
    );
    return result.rows[0];
  }

  // Market operations
  public async getMarkets(): Promise<Market[]> {
    const result = await this.query('SELECT * FROM markets WHERE is_active = true ORDER BY symbol');
    return result.rows;
  }

  public async getMarketBySymbol(symbol: string): Promise<Market | null> {
    const result = await this.query(
      'SELECT * FROM markets WHERE symbol = $1',
      [symbol]
    );
    return result.rows[0] || null;
  }

  public async getMarketById(id: string): Promise<Market | null> {
    const result = await this.query(
      'SELECT * FROM markets WHERE id = $1',
      [id]
    );
    return result.rows[0] || null;
  }

  // Position operations
  public async getUserPositions(userId: string): Promise<Position[]> {
    const result = await this.query(
      `SELECT p.*, m.symbol, m.base_asset, m.quote_asset 
       FROM positions p 
       JOIN markets m ON p.market_id = m.id 
       WHERE p.user_id = $1 AND p.size > 0 AND NOT p.is_liquidated
       ORDER BY p.created_at DESC`,
      [userId]
    );
    return result.rows;
  }

  public async getPositionById(id: string): Promise<Position | null> {
    const result = await this.query(
      'SELECT * FROM positions WHERE id = $1',
      [id]
    );
    return result.rows[0] || null;
  }

  public async createPosition(position: Omit<Position, 'id' | 'created_at' | 'updated_at'>): Promise<Position> {
    const result = await this.query(
      `INSERT INTO positions (user_id, market_id, position_account, side, size, entry_price, 
       current_price, margin, leverage, unrealized_pnl, realized_pnl, funding_fees, 
       is_liquidated, liquidation_price, health_factor, closed_at) 
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16) 
       RETURNING *`,
      [
        position.user_id, position.market_id, position.position_account, position.side,
        position.size, position.entry_price, position.current_price, position.margin,
        position.leverage, position.unrealized_pnl, position.realized_pnl, position.funding_fees,
        position.is_liquidated, position.liquidation_price, position.health_factor, position.closed_at
      ]
    );
    return result.rows[0];
  }

  public async updatePosition(id: string, updates: Partial<Position>): Promise<Position> {
    const fields = Object.keys(updates).filter(key => key !== 'id');
    const values = fields.map((field, index) => `${field} = $${index + 2}`);
    
    const result = await this.query(
      `UPDATE positions SET ${values.join(', ')}, updated_at = NOW() 
       WHERE id = $1 RETURNING *`,
      [id, ...Object.values(updates)]
    );
    return result.rows[0];
  }

  // Order operations
  public async getUserOrders(userId: string, status?: string): Promise<Order[]> {
    let query = `SELECT o.*, m.symbol, m.base_asset, m.quote_asset 
                 FROM orders o 
                 JOIN markets m ON o.market_id = m.id 
                 WHERE o.user_id = $1`;
    const params = [userId];
    
    if (status) {
      query += ' AND o.status = $2';
      params.push(status);
    }
    
    query += ' ORDER BY o.created_at DESC';
    
    const result = await this.query(query, params);
    return result.rows;
  }

  public async getOrderById(id: string): Promise<Order | null> {
    const result = await this.query(
      'SELECT * FROM orders WHERE id = $1',
      [id]
    );
    return result.rows[0] || null;
  }

  public async createOrder(order: Omit<Order, 'id' | 'created_at' | 'updated_at'>): Promise<Order> {
    const result = await this.query(
      `INSERT INTO orders (user_id, market_id, order_account, order_type, side, size, 
       price, stop_price, trailing_distance, leverage, status, filled_size, 
       average_fill_price, expires_at, filled_at, cancelled_at) 
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16) 
       RETURNING *`,
      [
        order.user_id, order.market_id, order.order_account, order.order_type, order.side,
        order.size, order.price, order.stop_price, order.trailing_distance, order.leverage,
        order.status, order.filled_size, order.average_fill_price, order.expires_at,
        order.filled_at, order.cancelled_at
      ]
    );
    return result.rows[0];
  }

  public async updateOrder(id: string, updates: Partial<Order>): Promise<Order> {
    const fields = Object.keys(updates).filter(key => key !== 'id');
    const values = fields.map((field, index) => `${field} = $${index + 2}`);
    
    const result = await this.query(
      `UPDATE orders SET ${values.join(', ')}, updated_at = NOW() 
       WHERE id = $1 RETURNING *`,
      [id, ...Object.values(updates)]
    );
    return result.rows[0];
  }

  // Trade operations
  public async getUserTrades(userId: string, limit: number = 100): Promise<Trade[]> {
    const result = await this.query(
      `SELECT t.*, m.symbol, m.base_asset, m.quote_asset 
       FROM trades t 
       JOIN markets m ON t.market_id = m.id 
       WHERE t.user_id = $1 
       ORDER BY t.created_at DESC 
       LIMIT $2`,
      [userId, limit]
    );
    return result.rows;
  }

  public async createTrade(trade: Omit<Trade, 'id' | 'created_at'>): Promise<Trade> {
    const result = await this.query(
      `INSERT INTO trades (user_id, market_id, position_id, order_id, trade_account, 
       side, size, price, fees, pnl, metadata) 
       VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11) 
       RETURNING *`,
      [
        trade.user_id, trade.market_id, trade.position_id, trade.order_id, trade.trade_account,
        trade.side, trade.size, trade.price, trade.fees, trade.pnl, trade.metadata
      ]
    );
    return result.rows[0];
  }

  // Health check
  public async healthCheck(): Promise<boolean> {
    try {
      await this.query('SELECT 1');
      return true;
    } catch (error) {
      logger.error('Database health check failed:', error);
      return false;
    }
  }
}
