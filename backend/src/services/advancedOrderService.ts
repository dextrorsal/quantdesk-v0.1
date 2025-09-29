import { Logger } from '../utils/logger';
import { mcpSupabaseService } from './mcpSupabaseService';

const logger = new Logger();

export interface AdvancedOrder {
  id: string;
  user_id: string;
  market_id: string;
  order_type: OrderType;
  side: PositionSide;
  size: number;
  price: number;
  stop_price?: number;
  trailing_distance?: number;
  leverage: number;
  status: OrderStatus;
  created_at: Date;
  expires_at?: Date;
  filled_size: number;
  // Advanced order fields
  hidden_size?: number;
  display_size?: number;
  time_in_force: TimeInForce;
  target_price?: number;
  parent_order_id?: string;
  twap_duration?: number;
  twap_interval?: number;
  // Additional fields
  current_price?: number;
  unrealized_pnl?: number;
  execution_price?: number;
  execution_time?: Date;
}

export enum OrderType {
  MARKET = 'market',
  LIMIT = 'limit',
  STOP_LOSS = 'stop_loss',
  TAKE_PROFIT = 'take_profit',
  TRAILING_STOP = 'trailing_stop',
  POST_ONLY = 'post_only',
  IOC = 'ioc', // Immediate or Cancel
  FOK = 'fok', // Fill or Kill
  ICEBERG = 'iceberg',
  TWAP = 'twap',
  STOP_LIMIT = 'stop_limit',
  BRACKET = 'bracket'
}

export enum PositionSide {
  LONG = 'long',
  SHORT = 'short'
}

export enum OrderStatus {
  PENDING = 'pending',
  FILLED = 'filled',
  CANCELLED = 'cancelled',
  EXPIRED = 'expired',
  PARTIALLY_FILLED = 'partially_filled',
  REJECTED = 'rejected'
}

export enum TimeInForce {
  GTC = 'gtc', // Good Till Cancelled
  IOC = 'ioc', // Immediate or Cancel
  FOK = 'fok', // Fill or Kill
  GTD = 'gtd' // Good Till Date
}

export interface OrderExecutionResult {
  success: boolean;
  order_id: string;
  execution_price: number;
  execution_size: number;
  execution_time: Date;
  remaining_size: number;
  error?: string;
}

export class AdvancedOrderService {
  private static instance: AdvancedOrderService;
  private db: typeof mcpSupabaseService;
  private logger: Logger;

  private constructor() {
    this.db = mcpSupabaseService;
    this.logger = new Logger();
  }

  public static getInstance(): AdvancedOrderService {
    if (!AdvancedOrderService.instance) {
      AdvancedOrderService.instance = new AdvancedOrderService();
    }
    return AdvancedOrderService.instance;
  }

  /**
   * Place an advanced order with comprehensive validation
   */
  public async placeOrder(orderData: Partial<AdvancedOrder>): Promise<AdvancedOrder> {
    try {
      // Validate order data
      this.validateOrderData(orderData);

      // Get current market price
      const marketPrice = await this.getCurrentMarketPrice(orderData.market_id!);
      
      // Calculate required margin
      const requiredMargin = this.calculateRequiredMargin(
        orderData.size!,
        marketPrice,
        orderData.leverage!
      );

      // Check user balance
      await this.validateUserBalance(orderData.user_id!, requiredMargin);

      // Create order in database
      const order = await this.createOrderInDatabase({
        ...orderData,
        status: OrderStatus.PENDING,
        created_at: new Date(),
        current_price: marketPrice,
        filled_size: 0
      } as AdvancedOrder);

      this.logger.info(`✅ Advanced order placed: ${order.id} (${order.order_type})`);
      return order;

    } catch (error) {
      this.logger.error('❌ Error placing advanced order:', error);
      throw error;
    }
  }

  /**
   * Execute conditional orders (stop-loss, take-profit, trailing stops)
   */
  public async executeConditionalOrders(marketId: string, currentPrice: number): Promise<OrderExecutionResult[]> {
    try {
      const conditionalOrders = await this.getConditionalOrders(marketId);
      const results: OrderExecutionResult[] = [];

      for (const order of conditionalOrders) {
        const shouldExecute = this.shouldExecuteConditionalOrder(order, currentPrice);
        
        if (shouldExecute) {
          try {
            const result = await this.executeOrder(order, currentPrice);
            results.push(result);
            this.logger.info(`🎯 Conditional order executed: ${order.id} at $${currentPrice}`);
          } catch (error) {
            this.logger.error(`❌ Error executing conditional order ${order.id}:`, error);
            results.push({
              success: false,
              order_id: order.id,
              execution_price: 0,
              execution_size: 0,
              execution_time: new Date(),
              remaining_size: order.size,
              error: error instanceof Error ? error.message : 'Unknown error'
            });
          }
        }
      }

      return results;

    } catch (error) {
      this.logger.error('❌ Error executing conditional orders:', error);
      throw error;
    }
  }

  /**
   * Execute TWAP orders
   */
  public async executeTWAPOrders(): Promise<OrderExecutionResult[]> {
    try {
      const twapOrders = await this.getTWAPOrders();
      const results: OrderExecutionResult[] = [];

      for (const order of twapOrders) {
        try {
          const result = await this.executeTWAPOrder(order);
          if (result.success) {
            results.push(result);
            this.logger.info(`⏰ TWAP order executed: ${order.id}`);
          }
        } catch (error) {
          this.logger.error(`❌ Error executing TWAP order ${order.id}:`, error);
        }
      }

      return results;

    } catch (error) {
      this.logger.error('❌ Error executing TWAP orders:', error);
      throw error;
    }
  }

  /**
   * Cancel an order
   */
  public async cancelOrder(orderId: string, userId: string): Promise<boolean> {
    try {
      const order = await this.getOrderById(orderId);
      
      if (!order) {
        throw new Error('Order not found');
      }

      if (order.user_id !== userId) {
        throw new Error('Unauthorized to cancel this order');
      }

      if (order.status !== OrderStatus.PENDING) {
        throw new Error('Order cannot be cancelled');
      }

      await this.updateOrderStatus(orderId, OrderStatus.CANCELLED);
      this.logger.info(`❌ Order cancelled: ${orderId}`);
      
      return true;

    } catch (error) {
      this.logger.error('❌ Error cancelling order:', error);
      throw error;
    }
  }

  /**
   * Get user's orders
   */
  public async getUserOrders(userId: string, status?: OrderStatus): Promise<AdvancedOrder[]> {
    try {
      let query = `
        SELECT * FROM advanced_orders 
        WHERE user_id = $1
      `;
      const params: any[] = [userId];

      if (status) {
        query += ` AND status = $2`;
        params.push(status);
      }

      query += ` ORDER BY created_at DESC`;

      const orders = await this.db.query(query, params);
      return orders.map(this.mapDbOrderToAdvancedOrder);

    } catch (error) {
      this.logger.error('❌ Error fetching user orders:', error);
      throw error;
    }
  }

  /**
   * Get order by ID
   */
  public async getOrderById(orderId: string): Promise<AdvancedOrder | null> {
    try {
      const orders = await this.db.query(
        'SELECT * FROM advanced_orders WHERE id = $1',
        [orderId]
      );

      if (orders.length === 0) {
        return null;
      }

      return this.mapDbOrderToAdvancedOrder(orders[0]);

    } catch (error) {
      this.logger.error('❌ Error fetching order by ID:', error);
      throw error;
    }
  }

  // Private helper methods

  private validateOrderData(orderData: Partial<AdvancedOrder>): void {
    if (!orderData.user_id) throw new Error('User ID is required');
    if (!orderData.market_id) throw new Error('Market ID is required');
    if (!orderData.order_type) throw new Error('Order type is required');
    if (!orderData.side) throw new Error('Side is required');
    if (!orderData.size || orderData.size <= 0) throw new Error('Size must be positive');
    if (!orderData.leverage || orderData.leverage <= 0) throw new Error('Leverage must be positive');

    // Validate order type specific requirements
    switch (orderData.order_type) {
      case OrderType.STOP_LOSS:
      case OrderType.TAKE_PROFIT:
        if (!orderData.stop_price || orderData.stop_price <= 0) {
          throw new Error('Stop price is required for stop-loss/take-profit orders');
        }
        break;
      case OrderType.TRAILING_STOP:
        if (!orderData.trailing_distance || orderData.trailing_distance <= 0) {
          throw new Error('Trailing distance is required for trailing stop orders');
        }
        break;
      case OrderType.ICEBERG:
        if (!orderData.hidden_size || !orderData.display_size) {
          throw new Error('Hidden size and display size are required for iceberg orders');
        }
        break;
      case OrderType.TWAP:
        if (!orderData.twap_duration || !orderData.twap_interval) {
          throw new Error('TWAP duration and interval are required for TWAP orders');
        }
        break;
      case OrderType.BRACKET:
        if (!orderData.target_price || !orderData.stop_price) {
          throw new Error('Target price and stop price are required for bracket orders');
        }
        break;
    }
  }

  private async getCurrentMarketPrice(marketId: string): Promise<number> {
    try {
      const prices = await this.db.query(`
        SELECT price FROM oracle_prices 
        WHERE market_id = $1 
        ORDER BY created_at DESC 
        LIMIT 1
      `, [marketId]);

      if (prices.length === 0) {
        throw new Error('No price data available for market');
      }

      return parseFloat(prices[0].price);

    } catch (error) {
      this.logger.error('❌ Error fetching market price:', error);
      throw error;
    }
  }

  private calculateRequiredMargin(size: number, price: number, leverage: number): number {
    return (size * price) / leverage;
  }

  private async validateUserBalance(userId: string, requiredMargin: number): Promise<void> {
    try {
      const user = await this.db.query(
        'SELECT balance FROM users WHERE id = $1',
        [userId]
      );

      if (user.length === 0) {
        throw new Error('User not found');
      }

      const balance = parseFloat(user[0].balance);
      if (balance < requiredMargin) {
        throw new Error('Insufficient balance');
      }

    } catch (error) {
      this.logger.error('❌ Error validating user balance:', error);
      throw error;
    }
  }

  private async createOrderInDatabase(order: AdvancedOrder): Promise<AdvancedOrder> {
    try {
      const result = await this.db.query(`
        INSERT INTO advanced_orders (
          user_id, market_id, order_type, side, size, price, stop_price,
          trailing_distance, leverage, status, created_at, expires_at,
          hidden_size, display_size, time_in_force, target_price,
          parent_order_id, twap_duration, twap_interval, current_price
        ) VALUES (
          $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17, $18, $19, $20
        ) RETURNING *
      `, [
        order.user_id, order.market_id, order.order_type, order.side,
        order.size, order.price, order.stop_price, order.trailing_distance,
        order.leverage, order.status, order.created_at, order.expires_at,
        order.hidden_size, order.display_size, order.time_in_force,
        order.target_price, order.parent_order_id, order.twap_duration,
        order.twap_interval, order.current_price
      ]);

      return this.mapDbOrderToAdvancedOrder(result[0]);

    } catch (error) {
      this.logger.error('❌ Error creating order in database:', error);
      throw error;
    }
  }

  private async getConditionalOrders(marketId: string): Promise<AdvancedOrder[]> {
    try {
      const orders = await this.db.query(`
        SELECT * FROM advanced_orders 
        WHERE market_id = $1 
        AND status = 'pending'
        AND order_type IN ('stop_loss', 'take_profit', 'trailing_stop')
      `, [marketId]);

      return orders.map(this.mapDbOrderToAdvancedOrder);

    } catch (error) {
      this.logger.error('❌ Error fetching conditional orders:', error);
      throw error;
    }
  }

  private shouldExecuteConditionalOrder(order: AdvancedOrder, currentPrice: number): boolean {
    switch (order.order_type) {
      case OrderType.STOP_LOSS:
        return order.side === PositionSide.LONG 
          ? currentPrice <= order.stop_price!
          : currentPrice >= order.stop_price!;
      
      case OrderType.TAKE_PROFIT:
        return order.side === PositionSide.LONG 
          ? currentPrice >= order.stop_price!
          : currentPrice <= order.stop_price!;
      
      case OrderType.TRAILING_STOP:
        // Simplified trailing stop logic
        return Math.abs(currentPrice - order.current_price!) >= order.trailing_distance!;
      
      default:
        return false;
    }
  }

  private async executeOrder(order: AdvancedOrder, executionPrice: number): Promise<OrderExecutionResult> {
    try {
      // Update order status
      await this.updateOrderStatus(order.id, OrderStatus.FILLED);
      
      // Update order with execution details
      await this.db.query(`
        UPDATE advanced_orders 
        SET execution_price = $1, execution_time = NOW(), filled_size = size
        WHERE id = $2
      `, [executionPrice, order.id]);

      return {
        success: true,
        order_id: order.id,
        execution_price: executionPrice,
        execution_size: order.size,
        execution_time: new Date(),
        remaining_size: 0
      };

    } catch (error) {
      this.logger.error('❌ Error executing order:', error);
      throw error;
    }
  }

  private async getTWAPOrders(): Promise<AdvancedOrder[]> {
    try {
      const orders = await this.db.query(`
        SELECT * FROM advanced_orders 
        WHERE order_type = 'twap' 
        AND status = 'pending'
        AND NOW() < (created_at + INTERVAL '1 second' * twap_duration)
      `);

      return orders.map(this.mapDbOrderToAdvancedOrder);

    } catch (error) {
      this.logger.error('❌ Error fetching TWAP orders:', error);
      throw error;
    }
  }

  private async executeTWAPOrder(order: AdvancedOrder): Promise<OrderExecutionResult> {
    try {
      // Simplified TWAP execution - in production, this would be more sophisticated
      const timeElapsed = Date.now() - order.created_at.getTime();
      const progress = timeElapsed / (order.twap_duration! * 1000);
      
      if (progress >= 1) {
        // TWAP order completed
        await this.updateOrderStatus(order.id, OrderStatus.FILLED);
        return {
          success: true,
          order_id: order.id,
          execution_price: order.current_price!,
          execution_size: order.size,
          execution_time: new Date(),
          remaining_size: 0
        };
      }

      return {
        success: false,
        order_id: order.id,
        execution_price: 0,
        execution_size: 0,
        execution_time: new Date(),
        remaining_size: order.size
      };

    } catch (error) {
      this.logger.error('❌ Error executing TWAP order:', error);
      throw error;
    }
  }

  private async updateOrderStatus(orderId: string, status: OrderStatus): Promise<void> {
    try {
      await this.db.query(
        'UPDATE advanced_orders SET status = $1 WHERE id = $2',
        [status, orderId]
      );
    } catch (error) {
      this.logger.error('❌ Error updating order status:', error);
      throw error;
    }
  }

  private mapDbOrderToAdvancedOrder(dbOrder: any): AdvancedOrder {
    return {
      id: dbOrder.id,
      user_id: dbOrder.user_id,
      market_id: dbOrder.market_id,
      order_type: dbOrder.order_type,
      side: dbOrder.side,
      size: parseFloat(dbOrder.size),
      price: parseFloat(dbOrder.price),
      stop_price: dbOrder.stop_price ? parseFloat(dbOrder.stop_price) : undefined,
      trailing_distance: dbOrder.trailing_distance ? parseFloat(dbOrder.trailing_distance) : undefined,
      leverage: dbOrder.leverage,
      status: dbOrder.status,
      created_at: new Date(dbOrder.created_at),
      expires_at: dbOrder.expires_at ? new Date(dbOrder.expires_at) : undefined,
      filled_size: parseFloat(dbOrder.filled_size),
      hidden_size: dbOrder.hidden_size ? parseFloat(dbOrder.hidden_size) : undefined,
      display_size: dbOrder.display_size ? parseFloat(dbOrder.display_size) : undefined,
      time_in_force: dbOrder.time_in_force,
      target_price: dbOrder.target_price ? parseFloat(dbOrder.target_price) : undefined,
      parent_order_id: dbOrder.parent_order_id,
      twap_duration: dbOrder.twap_duration,
      twap_interval: dbOrder.twap_interval,
      current_price: dbOrder.current_price ? parseFloat(dbOrder.current_price) : undefined,
      unrealized_pnl: dbOrder.unrealized_pnl ? parseFloat(dbOrder.unrealized_pnl) : undefined,
      execution_price: dbOrder.execution_price ? parseFloat(dbOrder.execution_price) : undefined,
      execution_time: dbOrder.execution_time ? new Date(dbOrder.execution_time) : undefined
    };
  }
}

export const advancedOrderService = AdvancedOrderService.getInstance();
