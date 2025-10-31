import { getSupabaseService } from './supabaseService';
import { Logger } from '../utils/logger';

const logger = new Logger();

/**
 * Consolidated Database Service for QuantDesk
 * 
 * This service provides a unified interface for all database operations.
 * All database access should go through this service rather than direct Supabase calls.
 * 
 * @example
 * ```typescript
 * import { databaseService } from '../services/supabaseDatabase';
 * 
 * // Get all active users
 * const users = await databaseService.select('users', '*', { is_active: true });
 * 
 * // Insert new market
 * const market = await databaseService.insert('markets', {
 *   symbol: 'BTC-PERP',
 *   base_asset: 'BTC',
 *   quote_asset: 'USDT'
 * });
 * ```
 */

export interface User {
  id: string;
  wallet_address: string;
  wallet_pubkey?: string; // Add wallet_pubkey for compatibility
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
  role?: string; // Add role property for chat functionality
  referrer_pubkey?: string; // Add referrer_pubkey for referral system
  is_activated?: boolean; // Add is_activated for referral system
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
  unrealized_pnl?: number;
  realized_pnl?: number;
  funding_fees?: number;
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
  filled_size?: number;
  remaining_size?: number;
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
  value?: number;
  fees: number;
  pnl?: number;
  created_at: Date;
  metadata: any;
}

export interface FundingRate {
  id: string;
  market_id: string;
  funding_rate: number;
  premium_index: number;
  oracle_price: number;
  mark_price: number;
  total_funding: number;
  created_at: Date;
}

export interface Liquidation {
  id: string;
  user_id: string;
  market_id: string;
  position_id: string;
  liquidator_address: string;
  liquidation_type: 'market' | 'backstop';
  liquidated_size: number;
  liquidation_price: number;
  liquidation_fee: number;
  remaining_margin?: number;
  created_at: Date;
  metadata: any;
}

export interface OraclePrice {
  id: string;
  market_id: string;
  price: number;
  confidence: number;
  exponent: number;
  created_at: Date;
}

export interface UserBalance {
  id: string;
  user_id: string;
  asset: string;
  balance: number;
  locked_balance: number;
  available_balance?: number;
  created_at: Date;
  updated_at: Date;
}

/**
 * SupabaseDatabaseService - Singleton database service
 * 
 * Provides consolidated database operations for the QuantDesk application.
 * All database access should use this service instead of direct Supabase calls.
 */
export class SupabaseDatabaseService {
  private static instance: SupabaseDatabaseService;

  private constructor() {
    // Supabase connection is handled by getSupabaseService()
  }

  public static getInstance(): SupabaseDatabaseService {
    if (!SupabaseDatabaseService.instance) {
      SupabaseDatabaseService.instance = new SupabaseDatabaseService();
    }
    return SupabaseDatabaseService.instance;
  }

  public async connect(): Promise<void> {
    try {
      const isHealthy = await getSupabaseService().healthCheck();
      if (isHealthy) {
        logger.info('Supabase database connected successfully');
      } else {
        throw new Error('Supabase health check failed');
      }
    } catch (error) {
      logger.error('Supabase connection failed:', error);
      throw error;
    }
  }

  public async disconnect(): Promise<void> {
    // Supabase handles connection management automatically
    logger.info('Supabase database disconnected successfully');
  }

  /**
   * @deprecated Use fluent API methods instead for better type safety and security
   * This method is kept for backward compatibility but should be replaced with fluent APIs
   */
  public async query(text: string, params?: any[]): Promise<any> {
    logger.warn('⚠️  Using deprecated query() method. Consider using fluent API methods instead.');
    try {
      const start = Date.now();
      const result = await getSupabaseService().query(text, params);
      const duration = Date.now() - start;
      
      // Log slow queries for performance monitoring
      if (duration > 1000) {
        logger.warn(`Slow query detected: ${duration}ms - ${text.substring(0, 100)}...`);
        // Log to database for monitoring (non-blocking)
        this.logSlowQuery(text, duration).catch(() => {
          // Ignore logging errors to prevent breaking main operations
        });
      }
      
      return result;
    } catch (error) {
      logger.error('Database query error:', error);
      throw error;
    }
  }

  /**
   * Execute a complex query with custom SQL (for cases where fluent API is insufficient)
   * @deprecated Use fluent API methods when possible. This method is for complex queries only.
   * @param sql - SQL query string
   * @param params - Query parameters
   * @returns Promise<any> - Query result
   */
  public async complexQuery(sql: string, params: any[] = []): Promise<any> {
    console.warn('⚠️  Using complexQuery() method. Consider using fluent API methods when possible.');
    try {
      const start = Date.now();
      const result = await getSupabaseService().query(sql, params);
      const duration = Date.now() - start;
      
      // Log slow queries for performance monitoring
      if (duration > 1000) {
        logger.warn(`Slow complex query detected: ${duration}ms - SQL: ${sql.substring(0, 100)}...`);
        this.logSlowQuery(sql, duration).catch(() => {});
      }
      
      return result;
    } catch (error) {
      logger.error(`Database complex query error:`, error);
      throw error;
    }
  }
  public async count(table: string, filters?: any): Promise<number> {
    try {
      const start = Date.now();
      let query = getSupabaseService().getClient().from(table).select('*', { count: 'exact', head: true });
      
      if (filters) {
        Object.keys(filters).forEach(key => {
          const value = filters[key];
          
          // Handle different filter types
          if (value === null) {
            query = query.is(key, null);
          } else if (value === undefined) {
            // Skip undefined values
          } else if (Array.isArray(value)) {
            // Handle IN clauses
            query = query.in(key, value);
          } else if (typeof value === 'object' && value !== null) {
            // Handle operators like { gt: 0 }, { lt: 100 }, etc.
            if (value.gt !== undefined) query = query.gt(key, value.gt);
            if (value.gte !== undefined) query = query.gte(key, value.gte);
            if (value.lt !== undefined) query = query.lt(key, value.lt);
            if (value.lte !== undefined) query = query.lte(key, value.lte);
            if (value.neq !== undefined) query = query.neq(key, value.neq);
            if (value.like !== undefined) query = query.like(key, value.like);
            if (value.ilike !== undefined) query = query.ilike(key, value.ilike);
            if (value.is !== undefined) query = query.is(key, value.is);
          } else {
            // Simple equality
            query = query.eq(key, value);
          }
        });
      }
      
      const { count, error } = await query;
      const duration = Date.now() - start;
      
      // Log slow queries for performance monitoring
      if (duration > 1000) {
        logger.warn(`Slow count query detected: ${duration}ms - table: ${table}`);
        this.logSlowQuery(`SELECT COUNT(*) FROM ${table}`, duration).catch(() => {});
      }
      
      if (error) throw error;
      return count || 0;
    } catch (error) {
      logger.error(`Database count error for table ${table}:`, error);
      throw error;
    }
  }

  /**
   * Log slow queries to database for performance monitoring
   */
  private async logSlowQuery(queryText: string, executionTimeMs: number): Promise<void> {
    try {
      await getSupabaseService().getClient()
        .from('performance_logs')
        .insert({
          query_text: queryText.substring(0, 1000), // Limit text length
          execution_time_ms: executionTimeMs,
          user_id: null, // Will be set by RLS policy
          created_at: new Date().toISOString()
        });
    } catch (error) {
      // Ignore logging errors to prevent breaking main operations
      logger.debug('Failed to log slow query:', error);
    }
  }

  // Generic Supabase methods for compatibility
  /**
   * Select data from a table with optional filters
   * @param table - Table name to query
   * @param columns - Columns to select (default: '*')
   * @param filters - Optional filters to apply
   * @returns Promise<any[]> - Array of matching records
   * @example
   * const users = await databaseService.select('users', '*', { is_active: true });
   * const markets = await databaseService.select('markets', 'id, symbol, base_asset');
   */
  public async select(table: string, columns: string = '*', filters?: any): Promise<any[]> {
    try {
      const start = Date.now();
      let query = getSupabaseService().getClient().from(table).select(columns);
      
      if (filters) {
        Object.keys(filters).forEach(key => {
          const value = filters[key];
          
          // Handle different filter types
          if (value === null) {
            query = query.is(key, null);
          } else if (value === undefined) {
            // Skip undefined values
          } else if (Array.isArray(value)) {
            // Handle IN clauses
            query = query.in(key, value);
          } else if (typeof value === 'object' && value !== null) {
            // Handle operators like { gt: 0 }, { lt: 100 }, etc.
            if (value.gt !== undefined) query = query.gt(key, value.gt);
            if (value.gte !== undefined) query = query.gte(key, value.gte);
            if (value.lt !== undefined) query = query.lt(key, value.lt);
            if (value.lte !== undefined) query = query.lte(key, value.lte);
            if (value.neq !== undefined) query = query.neq(key, value.neq);
            if (value.like !== undefined) query = query.like(key, value.like);
            if (value.ilike !== undefined) query = query.ilike(key, value.ilike);
            if (value.is !== undefined) query = query.is(key, value.is);
          } else {
            // Simple equality
            query = query.eq(key, value);
          }
        });
      }
      
      const { data, error } = await query;
      const duration = Date.now() - start;
      
      // Log slow queries for performance monitoring
      if (duration > 1000) {
        logger.warn(`Slow select query detected: ${duration}ms - table: ${table}`);
        this.logSlowQuery(`SELECT ${columns} FROM ${table}`, duration).catch(() => {});
      }
      
      if (error) throw error;
      return data || [];
    } catch (error) {
      logger.error(`Database select error for table ${table}:`, error);
      throw error;
    }
  }

  /**
   * Insert data into a table
   * @param table - Table name to insert into
   * @param data - Data object to insert
   * @returns Promise<any> - Inserted record
   * @example
   * const user = await databaseService.insert('users', {
   *   wallet_address: '0x123...',
   *   username: 'trader1',
   *   is_active: true
   * });
   */
  public async insert(table: string, data: any): Promise<any> {
    try {
      const { data: result, error } = await getSupabaseService().getClient()
        .from(table)
        .insert(data)
        .select()
        .single();
      
      if (error) throw error;
      return result;
    } catch (error) {
      logger.error(`Database insert error for table ${table}:`, error);
      throw error;
    }
  }

  public async update(table: string, data: any, filters: any): Promise<any> {
    try {
      let query = getSupabaseService().getClient().from(table).update(data);
      
      Object.keys(filters).forEach(key => {
        query = query.eq(key, filters[key]);
      });
      
      const { data: result, error } = await query.select().single();
      if (error) throw error;
      return result;
    } catch (error) {
      logger.error(`Database update error for table ${table}:`, error);
      throw error;
    }
  }

  public async upsert(table: string, data: any): Promise<any> {
    try {
      const { data: result, error } = await getSupabaseService().getClient()
        .from(table)
        .upsert(data)
        .select()
        .single();
      
      if (error) throw error;
      return result;
    } catch (error) {
      logger.error(`Database upsert error for table ${table}:`, error);
      throw error;
    }
  }

  public getClient() {
    return getSupabaseService().getClient();
  }

  // Chat-related methods
  public async getChannels(): Promise<any[]> {
    try {
      const { data, error } = await getSupabaseService().getClient()
        .from('chat_channels')
        .select('*')
        .order('created_at', { ascending: false });
      
      if (error) throw error;
      return data || [];
    } catch (error) {
      logger.error('Error getting channels:', error);
      throw error;
    }
  }

  public async createChannel(name: string, description: string, isPrivate: boolean, userId: string): Promise<any> {
    try {
      const channelData = {
        name,
        description,
        is_private: isPrivate,
        created_by: userId,
        created_at: new Date().toISOString()
      };
      
      const { data, error } = await getSupabaseService().getClient()
        .from('chat_channels')
        .insert(channelData)
        .select()
        .single();
      
      if (error) throw error;
      return data;
    } catch (error) {
      logger.error('Error creating channel:', error);
      throw error;
    }
  }

  public async getMessages(channelId: string, limit: number = 50): Promise<any[]> {
    try {
      const { data, error } = await getSupabaseService().getClient()
        .from('chat_messages')
        .select('*, users(username, wallet_address)')
        .eq('channel_id', channelId)
        .order('created_at', { ascending: false })
        .limit(limit);
      
      if (error) throw error;
      return data || [];
    } catch (error) {
      logger.error('Error getting messages:', error);
      throw error;
    }
  }

  public async sendMessage(channelId: string, walletPubkey: string, message: string, mentions: string[] = []): Promise<any> {
    try {
      const messageData = {
        channel_id: channelId,
        wallet_pubkey: walletPubkey,
        message,
        mentions: mentions.length > 0 ? mentions : null,
        created_at: new Date().toISOString()
      };
      
      const { data, error } = await getSupabaseService().getClient()
        .from('chat_messages')
        .insert(messageData)
        .select('*, users(username, wallet_address)')
        .single();
      
      if (error) throw error;
      return { data: [data], error: null };
    } catch (error) {
      logger.error('Error sending message:', error);
      throw error;
    }
  }

  public async transaction<T>(callback: (client: any) => Promise<T>): Promise<T> {
    // Supabase handles transactions differently, but we can simulate it
    try {
      const result = await callback(getSupabaseService());
      return result;
    } catch (error) {
      throw error;
    }
  }

  // Market methods
  public async getMarkets(): Promise<Market[]> {
    try {
      const markets = await getSupabaseService().getMarkets();
      return markets.map(market => ({
        ...market,
        created_at: new Date(market.created_at),
        updated_at: new Date(market.updated_at),
        last_funding_time: market.last_funding_time ? new Date(market.last_funding_time) : undefined
      }));
    } catch (error) {
      logger.error('Error getting markets:', error);
      throw error;
    }
  }

  public async getMarketBySymbol(symbol: string): Promise<Market | null> {
    try {
      const markets = await getSupabaseService().select('markets', '*', { symbol, is_active: true });
      const market = markets?.[0];
      if (!market) return null;
      
      return {
        ...market,
        created_at: new Date(market.created_at),
        updated_at: new Date(market.updated_at),
        last_funding_time: market.last_funding_time ? new Date(market.last_funding_time) : undefined
      };
    } catch (error) {
      logger.error('Error getting market by symbol:', error);
      throw error;
    }
  }

  // User methods
  public async getUserByWallet(walletAddress: string): Promise<User | null> {
    try {
      const user = await getSupabaseService().getUserByWallet(walletAddress);
      if (!user) return null;
      
      return {
        ...user,
        created_at: new Date(user.created_at),
        updated_at: new Date(user.updated_at),
        last_login: user.last_login ? new Date(user.last_login) : undefined
      };
    } catch (error) {
      logger.error('Error getting user by wallet:', error);
      throw error;
    }
  }

  public async createUser(userData: Partial<User>): Promise<User> {
    try {
      const { data: user, error } = await getSupabaseService().getClient()
        .from('users')
        .insert(userData)
        .select()
        .single();
      
      if (error) throw error;
      
      return {
        ...user,
        created_at: new Date(user.created_at),
        updated_at: new Date(user.updated_at),
        last_login: user.last_login ? new Date(user.last_login) : undefined,
      };
    } catch (error) {
      logger.error('Error creating user:', error);
      throw error;
    }
  }

  public async upsertUser(userData: Partial<User>): Promise<User> {
    try {
      const { data: user, error } = await getSupabaseService().getClient()
        .from('users')
        .upsert(userData)
        .select()
        .single();
      
      if (error) throw error;
      
      return {
        ...user,
        created_at: new Date(user.created_at),
        updated_at: new Date(user.updated_at),
        last_login: user.last_login ? new Date(user.last_login) : undefined,
      };
    } catch (error) {
      logger.error('Error upserting user:', error);
      throw error;
    }
  }

  public async updateUser(userId: string, updates: Partial<User>): Promise<User> {
    try {
      const user = await getSupabaseService().update('users', updates, { id: userId });
      return {
        ...user,
        created_at: new Date(user.created_at),
        updated_at: new Date(user.updated_at),
        last_login: user.last_login ? new Date(user.last_login) : undefined
      };
    } catch (error) {
      logger.error('Error updating user:', error);
      throw error;
    }
  }

  public async healthCheck(): Promise<boolean> {
    try {
      return await getSupabaseService().healthCheck();
    } catch (error) {
      logger.error('Health check failed:', error);
      return false;
    }
  }

  // Oracle price methods
  public async storeOraclePrice(marketId: string, price: number, confidence: number, exponent: number): Promise<OraclePrice> {
    try {
      const oraclePrice = await getSupabaseService().storeOraclePrice(marketId, price, confidence, exponent);
      return {
        ...oraclePrice,
        created_at: new Date(oraclePrice.created_at)
      };
    } catch (error) {
      logger.error('Error storing oracle price:', error);
      throw error;
    }
  }

  public async getLatestOraclePrices(): Promise<OraclePrice[]> {
    try {
      const prices = await getSupabaseService().getLatestOraclePrices();
      return prices.map(price => ({
        ...price,
        created_at: new Date(price.created_at)
      }));
    } catch (error) {
      logger.error('Error getting latest oracle prices:', error);
      throw error;
    }
  }

  // Position methods
  public async getPositionsByUser(userId: string): Promise<Position[]> {
    try {
      const positions = await getSupabaseService().select('positions', '*', { user_id: userId });
      return positions.map(position => ({
        ...position,
        created_at: new Date(position.created_at),
        updated_at: new Date(position.updated_at),
        closed_at: position.closed_at ? new Date(position.closed_at) : undefined
      }));
    } catch (error) {
      logger.error('Error getting positions by user:', error);
      throw error;
    }
  }

  public async getUserPositions(userId: string): Promise<Position[]> {
    try {
      const positions = await getSupabaseService().select('positions', '*', { user_id: userId });
      return positions.map(position => ({
        ...position,
        created_at: new Date(position.created_at),
        updated_at: new Date(position.updated_at),
        closed_at: position.closed_at ? new Date(position.closed_at) : undefined
      }));
    } catch (error) {
      logger.error('Error getting user positions:', error);
      throw error;
    }
  }

  public async getPositionById(positionId: string): Promise<Position | null> {
    try {
      const positions = await getSupabaseService().select('positions', '*', { id: positionId });
      const position = positions?.[0];
      if (!position) return null;
      
      return {
        ...position,
        created_at: new Date(position.created_at),
        updated_at: new Date(position.updated_at),
        closed_at: position.closed_at ? new Date(position.closed_at) : undefined
      };
    } catch (error) {
      logger.error('Error getting position by ID:', error);
      throw error;
    }
  }

  public async createPosition(positionData: Partial<Position>): Promise<Position> {
    try {
      const position = await getSupabaseService().insert('positions', positionData);
      return {
        ...position,
        created_at: new Date(position.created_at),
        updated_at: new Date(position.updated_at),
        closed_at: position.closed_at ? new Date(position.closed_at) : undefined
      };
    } catch (error) {
      logger.error('Error creating position:', error);
      throw error;
    }
  }

  public async updatePosition(positionId: string, updates: Partial<Position>): Promise<Position> {
    try {
      const position = await getSupabaseService().update('positions', updates, { id: positionId });
      return {
        ...position,
        created_at: new Date(position.created_at),
        updated_at: new Date(position.updated_at),
        closed_at: position.closed_at ? new Date(position.closed_at) : undefined
      };
    } catch (error) {
      logger.error('Error updating position:', error);
      throw error;
    }
  }

  // Order methods
  public async getOrdersByUser(userId: string): Promise<Order[]> {
    try {
      const orders = await getSupabaseService().select('orders', '*', { user_id: userId });
      return orders.map(order => ({
        ...order,
        created_at: new Date(order.created_at),
        updated_at: new Date(order.updated_at),
        expires_at: order.expires_at ? new Date(order.expires_at) : undefined,
        filled_at: order.filled_at ? new Date(order.filled_at) : undefined,
        cancelled_at: order.cancelled_at ? new Date(order.cancelled_at) : undefined
      }));
    } catch (error) {
      logger.error('Error getting orders by user:', error);
      throw error;
    }
  }

  public async getUserOrders(userId: string, status?: string): Promise<Order[]> {
    try {
      const filters: any = { user_id: userId };
      if (status) {
        filters.status = status;
      }
      const orders = await getSupabaseService().select('orders', '*', filters);
      return orders.map(order => ({
        ...order,
        created_at: new Date(order.created_at),
        updated_at: new Date(order.updated_at),
        expires_at: order.expires_at ? new Date(order.expires_at) : undefined,
        filled_at: order.filled_at ? new Date(order.filled_at) : undefined,
        cancelled_at: order.cancelled_at ? new Date(order.cancelled_at) : undefined
      }));
    } catch (error) {
      logger.error('Error getting user orders:', error);
      throw error;
    }
  }

  public async getOrderById(orderId: string): Promise<Order | null> {
    try {
      const orders = await getSupabaseService().select('orders', '*', { id: orderId });
      const order = orders?.[0];
      if (!order) return null;
      
      return {
        ...order,
        created_at: new Date(order.created_at),
        updated_at: new Date(order.updated_at),
        expires_at: order.expires_at ? new Date(order.expires_at) : undefined,
        filled_at: order.filled_at ? new Date(order.filled_at) : undefined,
        cancelled_at: order.cancelled_at ? new Date(order.cancelled_at) : undefined
      };
    } catch (error) {
      logger.error('Error getting order by ID:', error);
      throw error;
    }
  }

  public async updateOrder(orderId: string, updates: Partial<Order>): Promise<Order> {
    try {
      const order = await getSupabaseService().update('orders', updates, { id: orderId });
      return {
        ...order,
        created_at: new Date(order.created_at),
        updated_at: new Date(order.updated_at),
        expires_at: order.expires_at ? new Date(order.expires_at) : undefined,
        filled_at: order.filled_at ? new Date(order.filled_at) : undefined,
        cancelled_at: order.cancelled_at ? new Date(order.cancelled_at) : undefined
      };
    } catch (error) {
      logger.error('Error updating order:', error);
      throw error;
    }
  }

  public async createOrder(orderData: Partial<Order>): Promise<Order> {
    try {
      const order = await getSupabaseService().insert('orders', orderData);
      return {
        ...order,
        created_at: new Date(order.created_at),
        updated_at: new Date(order.updated_at),
        expires_at: order.expires_at ? new Date(order.expires_at) : undefined,
        filled_at: order.filled_at ? new Date(order.filled_at) : undefined,
        cancelled_at: order.cancelled_at ? new Date(order.cancelled_at) : undefined
      };
    } catch (error) {
      logger.error('Error creating order:', error);
      throw error;
    }
  }

  // Trade methods
  public async createTrade(tradeData: Partial<Trade>): Promise<Trade> {
    try {
      const trade = await getSupabaseService().insert('trades', tradeData);
      return {
        ...trade,
        created_at: new Date(trade.created_at)
      };
    } catch (error) {
      logger.error('Error creating trade:', error);
      throw error;
    }
  }

  public async getUserTrades(userId: string, limit?: number): Promise<Trade[]> {
    try {
      const filters: any = { user_id: userId };
      const trades = await getSupabaseService().select('trades', '*', filters);
      
      // Sort by created_at descending and apply limit
      const sortedTrades = trades.sort((a: any, b: any) => 
        new Date(b.created_at).getTime() - new Date(a.created_at).getTime()
      );
      
      const limitedTrades = limit ? sortedTrades.slice(0, limit) : sortedTrades;
      
      return limitedTrades.map((trade: any) => ({
        ...trade,
        created_at: new Date(trade.created_at)
      }));
    } catch (error) {
      logger.error('Error getting user trades:', error);
      throw error;
    }
  }

  // Funding rate methods
  public async createFundingRate(fundingData: Partial<FundingRate>): Promise<FundingRate> {
    try {
      const funding = await getSupabaseService().insert('funding_rates', fundingData);
      return {
        ...funding,
        created_at: new Date(funding.created_at)
      };
    } catch (error) {
      logger.error('Error creating funding rate:', error);
      throw error;
    }
  }

  // Liquidation methods
  public async createLiquidation(liquidationData: Partial<Liquidation>): Promise<Liquidation> {
    try {
      const liquidation = await getSupabaseService().insert('liquidations', liquidationData);
      return {
        ...liquidation,
        created_at: new Date(liquidation.created_at)
      };
    } catch (error) {
      logger.error('Error creating liquidation:', error);
      throw error;
    }
  }

  // User balance methods
  public async getUserBalances(userId: string): Promise<UserBalance[]> {
    try {
      const balances = await getSupabaseService().select('user_balances', '*', { user_id: userId });
      return balances.map(balance => ({
        ...balance,
        created_at: new Date(balance.created_at),
        updated_at: new Date(balance.updated_at)
      }));
    } catch (error) {
      logger.error('Error getting user balances:', error);
      throw error;
    }
  }

  public async updateUserBalance(userId: string, asset: string, updates: Partial<UserBalance>): Promise<UserBalance> {
    try {
      const balance = await getSupabaseService().update('user_balances', updates, { user_id: userId, asset });
      return {
        ...balance,
        created_at: new Date(balance.created_at),
        updated_at: new Date(balance.updated_at)
      };
    } catch (error) {
      logger.error('Error updating user balance:', error);
      throw error;
    }
  }

  // Market Management Methods
  public async getOraclePrice(symbol: string): Promise<number | null> {
    try {
      // Use fluent API instead of executeQuery for security
      const { data: market, error: marketError } = await getSupabaseService().getClient()
        .from('markets')
        .select('id')
        .eq('base_asset', symbol)
        .single();

      if (marketError) {
        logger.error('Error getting market for oracle price:', marketError);
        return null;
      }

      const { data: priceData, error: priceError } = await getSupabaseService().getClient()
        .from('oracle_prices')
        .select('price')
        .eq('market_id', market.id)
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (priceError) {
        logger.error('Error getting oracle price:', priceError);
        return null;
      }

      return priceData?.price || null;
    } catch (error) {
      logger.error('Error getting oracle price:', error);
      return null;
    }
  }

  public async getMarketsByCategory(category: string): Promise<Market[]> {
    try {
      const markets = await getSupabaseService().select('markets', '*', { category, is_active: true });
      return markets.map(market => ({
        ...market,
        created_at: new Date(market.created_at),
        updated_at: new Date(market.updated_at),
        last_funding_time: market.last_funding_time ? new Date(market.last_funding_time) : undefined
      }));
    } catch (error) {
      logger.error('Error getting markets by category:', error);
      throw error;
    }
  }

  public async searchMarkets(query: string): Promise<Market[]> {
    try {
      // Use fluent API instead of executeQuery for security
      const { data: markets, error } = await getSupabaseService().getClient()
        .from('markets')
        .select('*')
        .or(`symbol.ilike.%${query}%,base_asset.ilike.%${query}%,quote_asset.ilike.%${query}%`)
        .eq('is_active', true)
        .order('symbol');

      if (error) {
        logger.error('Error searching markets:', error);
        return [];
      }

      return markets?.map((market: any) => ({
        ...market,
        created_at: new Date(market.created_at),
        updated_at: new Date(market.updated_at),
        last_funding_time: market.last_funding_time ? new Date(market.last_funding_time) : undefined
      })) || [];
    } catch (error) {
      logger.error('Error searching markets:', error);
      return [];
    }
  }

  public async getMarketCategories(): Promise<string[]> {
    try {
      // Use fluent API instead of executeQuery for security
      const { data: categories, error } = await getSupabaseService().getClient()
        .from('markets')
        .select('category')
        .not('category', 'is', null)
        .eq('is_active', true)
        .order('category');

      if (error) {
        logger.error('Error getting market categories:', error);
        return [];
      }

      // Extract unique categories
      const uniqueCategories = [...new Set(categories?.map((row: any) => row.category).filter(Boolean) || [])];
      return uniqueCategories;
    } catch (error) {
      logger.error('Error getting market categories:', error);
      return [];
    }
  }

  public async createMarket(marketData: Partial<Market>): Promise<Market> {
    try {
      const market = await getSupabaseService().insert('markets', marketData);
      return {
        ...market,
        created_at: new Date(market.created_at),
        updated_at: new Date(market.updated_at),
        last_funding_time: market.last_funding_time ? new Date(market.last_funding_time) : undefined
      };
    } catch (error) {
      logger.error('Error creating market:', error);
      throw error;
    }
  }

  public async updateMarket(marketId: string, updates: Partial<Market>): Promise<Market> {
    try {
      const market = await getSupabaseService().update('markets', updates, { id: marketId });
      return {
        ...market,
        created_at: new Date(market.created_at),
        updated_at: new Date(market.updated_at),
        last_funding_time: market.last_funding_time ? new Date(market.last_funding_time) : undefined
      };
    } catch (error) {
      logger.error('Error updating market:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const databaseService = SupabaseDatabaseService.getInstance();
