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

export class SupabaseDatabaseService {
  private static instance: SupabaseDatabaseService;

  private constructor() {
    // Supabase connection is handled by supabaseService
  }

  public static getInstance(): SupabaseDatabaseService {
    if (!SupabaseDatabaseService.instance) {
      SupabaseDatabaseService.instance = new SupabaseDatabaseService();
    }
    return SupabaseDatabaseService.instance;
  }

  public async connect(): Promise<void> {
    try {
      const isHealthy = await supabaseService.healthCheck();
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

  public async query(text: string, params?: any[]): Promise<any> {
    try {
      const start = Date.now();
      const result = await supabaseService.query(text, params);
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

  public async transaction<T>(callback: (client: any) => Promise<T>): Promise<T> {
    // Supabase handles transactions differently, but we can simulate it
    try {
      const result = await callback(supabaseService);
      return result;
    } catch (error) {
      throw error;
    }
  }

  // Market methods
  public async getMarkets(): Promise<Market[]> {
    try {
      const markets = await supabaseService.getMarkets();
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
      const markets = await supabaseService.select('markets', '*', { symbol, is_active: true });
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
      const user = await supabaseService.getUserByWallet(walletAddress);
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
      const user = await supabaseService.upsertUser(userData);
      return {
        ...user,
        created_at: new Date(user.created_at),
        updated_at: new Date(user.updated_at),
        last_login: user.last_login ? new Date(user.last_login) : undefined
      };
    } catch (error) {
      logger.error('Error creating user:', error);
      throw error;
    }
  }

  // Oracle price methods
  public async storeOraclePrice(marketId: string, price: number, confidence: number, exponent: number): Promise<OraclePrice> {
    try {
      const oraclePrice = await supabaseService.storeOraclePrice(marketId, price, confidence, exponent);
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
      const prices = await supabaseService.getLatestOraclePrices();
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
      const positions = await supabaseService.select('positions', '*', { user_id: userId });
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

  public async createPosition(positionData: Partial<Position>): Promise<Position> {
    try {
      const position = await supabaseService.insert('positions', positionData);
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
      const position = await supabaseService.update('positions', updates, { id: positionId });
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
      const orders = await supabaseService.select('orders', '*', { user_id: userId });
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

  public async createOrder(orderData: Partial<Order>): Promise<Order> {
    try {
      const order = await supabaseService.insert('orders', orderData);
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
      const trade = await supabaseService.insert('trades', tradeData);
      return {
        ...trade,
        created_at: new Date(trade.created_at)
      };
    } catch (error) {
      logger.error('Error creating trade:', error);
      throw error;
    }
  }

  // Funding rate methods
  public async createFundingRate(fundingData: Partial<FundingRate>): Promise<FundingRate> {
    try {
      const funding = await supabaseService.insert('funding_rates', fundingData);
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
      const liquidation = await supabaseService.insert('liquidations', liquidationData);
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
      const balances = await supabaseService.select('user_balances', '*', { user_id: userId });
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
      const balance = await supabaseService.update('user_balances', updates, { user_id: userId, asset });
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
}

// Export singleton instance
export const databaseService = SupabaseDatabaseService.getInstance();
