import { SupabaseDatabaseService } from './supabaseDatabase';
import { pythOracleService } from './pythOracleService';
import { WebSocketService } from './websocket';
import { Logger } from '../utils/logger';
import { v4 as uuidv4 } from 'uuid';
import { smartContractService } from './smartContractService';
import { matchingService } from './matching';

const logger = new Logger();

interface ConditionalOrder {
  id: string;
  user_id: string;
  market_id: string;
  order_type: 'stop_loss' | 'take_profit' | 'trailing_stop' | 'post_only';
  side: 'long' | 'short';
  size: number;
  price?: number;
  stop_price?: number;
  trailing_distance?: number;
  leverage: number;
  status: 'pending' | 'filled' | 'cancelled' | 'expired' | 'partially_filled';
  filled_size?: number;
  remaining_size?: number;
  average_fill_price?: number; // added for update
  filled_at?: string; // ISO string used in update
  metadata?: any; // added for error logging update
}

interface Metrics {
  processedOrders: number;
  triggersFired: number;
  errors: number;
  lastExecutionLatency: number;
  executionLatencies: number[];
  startTime: number;
}

/**
 * Conditional Order Monitor Service
 * 
 * Monitors conditional orders (stop-loss, take-profit, trailing stop) and executes them
 * when price triggers are met. Runs at 1s cadence.
 */
export class ConditionalOrderMonitor {
  private static instance: ConditionalOrderMonitor;
  private db: SupabaseDatabaseService;
  private isRunning: boolean = false;
  private loopInterval: NodeJS.Timeout | null = null;
  private processedOrderIds: Set<string> = new Set(); // Idempotency guard
  private metrics: Metrics = {
    processedOrders: 0,
    triggersFired: 0,
    errors: 0,
    lastExecutionLatency: 0,
    executionLatencies: [],
    startTime: Date.now()
  };
  private readonly LOOP_CADENCE_MS = 1000; // 1 second
  private readonly CADENCE_TOLERANCE_MS = 200; // ±200ms tolerance
  private wsService: WebSocketService | null = null;
  private lastPrices: Record<string, number> | null = null;
  private circuitBreakerActive: boolean = false;
  private circuitBreakerReason: string | null = null;
  private readonly DEVIATION_TRIP_THRESHOLD = 0.05; // 5%
  private readonly DEVIATION_RESET_THRESHOLD = 0.02; // 2%

  private constructor() {
    this.db = SupabaseDatabaseService.getInstance();
  }

  /**
   * Get singleton instance
   */
  public static getInstance(): ConditionalOrderMonitor {
    if (!ConditionalOrderMonitor.instance) {
      ConditionalOrderMonitor.instance = new ConditionalOrderMonitor();
    }
    return ConditionalOrderMonitor.instance;
  }

  /**
   * Set WebSocket service for notifications
   */
  public setWebSocketService(wsService: WebSocketService): void {
    this.wsService = wsService;
  }

  /**
   * Start the monitoring loop
   * Protected with single instance guard
   */
  public start(): void {
    if (this.isRunning) {
      logger.warn('⚠️ Conditional Order Monitor already running');
      return;
    }

    this.isRunning = true;
    this.metrics.startTime = Date.now();
    logger.info('🔍 Starting Conditional Order Monitor (1s cadence)');

    // Start the monitoring loop
    this.runLoop();
  }

  /**
   * Stop the monitoring loop
   */
  public stop(): void {
    if (!this.isRunning) {
      return;
    }

    this.isRunning = false;
    if (this.loopInterval) {
      clearInterval(this.loopInterval);
      this.loopInterval = null;
    }

    logger.info('🛑 Conditional Order Monitor stopped');
  }

  /**
   * Main monitoring loop - runs every 1 second
   */
  private async runLoop(): Promise<void> {
    if (!this.isRunning) {
      return;
    }

    const loopStartTime = Date.now();

    try {
      // Clear idempotency set for new loop iteration
      this.processedOrderIds.clear();

      // Get current prices for all markets
      const prices = await pythOracleService.getAllPrices();
      if (!prices || Object.keys(prices).length === 0) {
        logger.warn('⚠️ No prices available from oracle, skipping this cycle');
        this.scheduleNextLoop(loopStartTime);
        return;
      }

      // Circuit breaker check based on price deviation
      if (this.lastPrices) {
        const { maxDeviation, symbol } = this.computeMaxDeviation(this.lastPrices, prices);
        if (!this.circuitBreakerActive && maxDeviation > this.DEVIATION_TRIP_THRESHOLD) {
          this.circuitBreakerActive = true;
          this.circuitBreakerReason = `Oracle deviation ${Math.round(maxDeviation * 100)}% on ${symbol}`;
          logger.warn(`🛑 Circuit breaker TRIPPED: ${this.circuitBreakerReason}`);
        }
        if (this.circuitBreakerActive && maxDeviation < this.DEVIATION_RESET_THRESHOLD) {
          logger.info('✅ Circuit breaker RESET: deviation back within threshold');
          this.circuitBreakerActive = false;
          this.circuitBreakerReason = null;
        }
      }
      this.lastPrices = prices;

      if (this.circuitBreakerActive) {
        logger.warn('⏸️ Circuit breaker active - skipping order evaluation this cycle');
        this.scheduleNextLoop(loopStartTime);
        return;
      }

      // Fetch all active conditional orders
      const orders = await this.fetchActiveConditionalOrders();
      this.metrics.processedOrders += orders.length;

      logger.debug(`📊 Evaluating ${orders.length} active conditional orders`);

      // Evaluate each order
      const evaluationPromises = orders.map(order => this.evaluateOrder(order, prices));
      await Promise.allSettled(evaluationPromises);

    } catch (error) {
      this.metrics.errors++;
      logger.error('❌ Error in monitoring loop:', error);
    } finally {
      // Schedule next loop iteration with cadence control
      this.scheduleNextLoop(loopStartTime);
    }
  }

  /**
   * Compute maximum percentage deviation between two price maps
   */
  private computeMaxDeviation(prev: Record<string, number>, curr: Record<string, number>): { maxDeviation: number; symbol: string } {
    let maxDeviation = 0;
    let maxSymbol = '';
    for (const [symbol, price] of Object.entries(curr)) {
      const prevPrice = prev[symbol];
      if (prevPrice && prevPrice > 0 && price > 0) {
        const deviation = Math.abs(price - prevPrice) / prevPrice;
        if (deviation > maxDeviation) {
          maxDeviation = deviation;
          maxSymbol = symbol;
        }
      }
    }
    return { maxDeviation, symbol: maxSymbol };
  }

  /**
   * Schedule next loop iteration with cadence control
   */
  private scheduleNextLoop(lastLoopStartTime: number): void {
    if (!this.isRunning) {
      return;
    }

    const elapsed = Date.now() - lastLoopStartTime;
    const delay = Math.max(0, this.LOOP_CADENCE_MS - elapsed);

    // Check if we're exceeding tolerance
    if (elapsed > this.LOOP_CADENCE_MS + this.CADENCE_TOLERANCE_MS) {
      logger.warn(`⚠️ Loop cadence exceeded: ${elapsed}ms (target: ${this.LOOP_CADENCE_MS}ms)`);
    }

    this.loopInterval = setTimeout(() => {
      this.runLoop();
    }, delay);
  }

  /**
   * Fetch active conditional orders from database
   */
  private async fetchActiveConditionalOrders(): Promise<ConditionalOrder[]> {
    try {
      const { data, error } = await this.db.getClient()
        .from('orders')
        .select('*')
        .in('order_type', ['stop_loss', 'take_profit', 'trailing_stop'])
        .in('status', ['pending', 'partially_filled'])
        .is('filled_at', null)
        .is('cancelled_at', null)
        .order('created_at', { ascending: true });

      if (error) {
        throw error;
      }

      return (data || []) as ConditionalOrder[];
    } catch (error) {
      logger.error('❌ Error fetching active conditional orders:', error);
      return [];
    }
  }

  /**
   * Evaluate a single order against current prices
   */
  private async evaluateOrder(order: ConditionalOrder, prices: Record<string, number>): Promise<void> {
    // Idempotency check - prevent duplicate processing in same loop
    if (this.processedOrderIds.has(order.id)) {
      logger.debug(`⏭️ Skipping already processed order: ${order.id}`);
      return;
    }

    this.processedOrderIds.add(order.id);

    try {
      // Get market symbol to fetch price
      const market = await this.getMarketForOrder(order.market_id);
      if (!market) {
        logger.warn(`⚠️ Market not found for order ${order.id}`);
        return;
      }

      const currentPrice = prices[market.symbol];
      if (!currentPrice || currentPrice <= 0) {
        logger.warn(`⚠️ No valid price for ${market.symbol}, order ${order.id}`);
        return;
      }

      logger.debug(`🔍 Evaluating order ${order.id}: type=${order.order_type}, current=${currentPrice}, trigger=${order.stop_price || 'N/A'}`);

      // Evaluate based on order type
      let shouldExecute = false;
      let updatedOrder: Partial<ConditionalOrder> | null = null;

      switch (order.order_type) {
        case 'stop_loss':
          shouldExecute = this.evaluateStopLoss(order, currentPrice);
          break;

        case 'take_profit':
          shouldExecute = this.evaluateTakeProfit(order, currentPrice);
          break;

        case 'trailing_stop':
          const result = this.evaluateTrailingStop(order, currentPrice);
          shouldExecute = result.shouldExecute;
          updatedOrder = result.updatedOrder;
          break;

        default:
          logger.warn(`⚠️ Unsupported order type: ${order.order_type}`);
          return;
      }

      // Execute or update order
      if (shouldExecute) {
        await this.executeOrder(order, currentPrice);
      } else if (updatedOrder) {
        await this.updateOrder(order.id, updatedOrder);
      }

    } catch (error) {
      this.metrics.errors++;
      const correlationId = uuidv4();
      logger.error(`❌ Error evaluating order ${order.id} (correlation: ${correlationId}):`, error);
      
      // Log error to database with correlation ID
      await this.logError(order.id, error, correlationId);
    }
  }

  /**
   * Evaluate stop-loss order
   * Trigger when price crosses stop_price (unfavorable direction)
   */
  private evaluateStopLoss(order: ConditionalOrder, currentPrice: number): boolean {
    if (!order.stop_price) {
      return false;
    }

    // For long positions: trigger when price falls below stop_price
    // For short positions: trigger when price rises above stop_price
    if (order.side === 'long') {
      return currentPrice <= order.stop_price;
    } else {
      return currentPrice >= order.stop_price;
    }
  }

  /**
   * Evaluate take-profit order
   * Trigger when price crosses trigger price (favorable direction)
   */
  private evaluateTakeProfit(order: ConditionalOrder, currentPrice: number): boolean {
    if (!order.stop_price) { // stop_price is used as take-profit target
      return false;
    }

    // For long positions: trigger when price rises above target
    // For short positions: trigger when price falls below target
    if (order.side === 'long') {
      return currentPrice >= order.stop_price;
    } else {
      return currentPrice <= order.stop_price;
    }
  }

  /**
   * Evaluate trailing stop order
   * Updates trigger price as market moves favorably
   */
  private evaluateTrailingStop(
    order: ConditionalOrder,
    currentPrice: number
  ): { shouldExecute: boolean; updatedOrder: Partial<ConditionalOrder> | null } {
    if (!order.stop_price || !order.trailing_distance) {
      return { shouldExecute: false, updatedOrder: null };
    }

    const trailingPercent = order.trailing_distance;
    let newStopPrice = order.stop_price;
    let shouldExecute = false;

    if (order.side === 'long') {
      // Long: trailing stop moves up as price rises
      const maxPrice = Math.max(currentPrice, order.stop_price + (order.stop_price * trailingPercent / 100));
      newStopPrice = maxPrice - (maxPrice * trailingPercent / 100);
      
      // Trigger if price falls below trailing stop
      shouldExecute = currentPrice <= newStopPrice;
    } else {
      // Short: trailing stop moves down as price falls
      const minPrice = Math.min(currentPrice, order.stop_price - (order.stop_price * trailingPercent / 100));
      newStopPrice = minPrice + (minPrice * trailingPercent / 100);
      
      // Trigger if price rises above trailing stop
      shouldExecute = currentPrice >= newStopPrice;
    }

    // If stop price needs updating
    const needsUpdate = Math.abs(newStopPrice - order.stop_price) > 0.01; // 1 cent tolerance

    if (needsUpdate && !shouldExecute) {
      return {
        shouldExecute: false,
        updatedOrder: { stop_price: newStopPrice }
      };
    }

    return { shouldExecute, updatedOrder: null };
  }

  /**
   * Execute an order (trigger was met)
   */
  private async executeOrder(order: ConditionalOrder, currentPrice: number): Promise<void> {
    const executionStartTime = Date.now();
    const correlationId = uuidv4();

    logger.info(`🎯 Trigger fired for order ${order.id} (type: ${order.order_type}, price: ${currentPrice})`);

    try {
      // Get market details
      const market = await this.getMarketForOrder(order.market_id);
      if (!market) {
        throw new Error(`Market not found for order ${order.id}`);
      }

      // Update order status to processing
      await this.updateOrder(order.id, {
        status: 'partially_filled' // Will be updated to 'filled' after execution
      });

      // Execute via matching service or smart contract
      // For now, we'll integrate with existing order execution path
      const executionResult = await this.callOrderExecution(order, market.symbol, currentPrice);

      // Calculate execution latency
      const executionLatency = Date.now() - executionStartTime;
      this.metrics.lastExecutionLatency = executionLatency;
      this.metrics.executionLatencies.push(executionLatency);
      
      // Keep only last 1000 latencies for p95 calculation
      if (this.metrics.executionLatencies.length > 1000) {
        this.metrics.executionLatencies.shift();
      }

      // Update order as filled
      await this.updateOrder(order.id, {
        status: 'filled',
        filled_size: order.size,
        remaining_size: 0,
        average_fill_price: currentPrice,
        filled_at: new Date().toISOString()
      });

      this.metrics.triggersFired++;

      // Send WebSocket notification
      this.notifyOrderExecuted(order, currentPrice, correlationId);

      logger.info(`✅ Order ${order.id} executed successfully (latency: ${executionLatency}ms)`);

    } catch (error) {
      this.metrics.errors++;
      logger.error(`❌ Error executing order ${order.id} (correlation: ${correlationId}):`, error);
      
      // Mark order as failed or keep pending for retry
      await this.updateOrder(order.id, {
        status: 'pending', // Keep pending for retry
        metadata: { last_error: error instanceof Error ? error.message : 'Unknown error', correlation_id: correlationId }
      });

      await this.logError(order.id, error, correlationId);
      throw error;
    }
  }

  /**
   * Call order execution (integrate with existing execution path)
   */
  private async callOrderExecution(
    order: ConditionalOrder,
    symbol: string,
    executionPrice: number
  ): Promise<any> {
    // TODO: Integrate with smartContractService or matchingService
    // For now, this is a placeholder that should call the actual execution logic
    
    // Convert order side to matching service format
    const side = order.side === 'long' ? 'buy' : 'sell';
    
    // Call matching service to execute order
    // For conditional orders, we execute at market price when trigger is met
    try {
      // Calculate remaining size (order.remaining_size is generated column, use remaining_size field or size - filled_size)
      const remainingSize = order.remaining_size !== undefined 
        ? order.remaining_size 
        : (order.size - (order.filled_size || 0));
      
      // For conditional orders, we execute at market price
      const result = await matchingService.placeOrder({
        userId: order.user_id,
        symbol,
        side,
        size: remainingSize,
        orderType: 'market',
        leverage: order.leverage
      });

      return result;
    } catch (error) {
      logger.error(`❌ Error in order execution call for ${order.id}:`, error);
      throw error;
    }
  }

  /**
   * Update order in database
   */
  private async updateOrder(orderId: string, updates: Partial<ConditionalOrder>): Promise<void> {
    try {
      const { error } = await this.db.getClient()
        .from('orders')
        .update({
          ...updates,
          updated_at: new Date().toISOString()
        })
        .eq('id', orderId);

      if (error) {
        throw error;
      }

      logger.debug(`✅ Updated order ${orderId}`);
    } catch (error) {
      logger.error(`❌ Error updating order ${orderId}:`, error);
      throw error;
    }
  }

  /**
   * Get market details for an order
   */
  private async getMarketForOrder(marketId: string): Promise<{ symbol: string } | null> {
    try {
      const { data, error } = await this.db.getClient()
        .from('markets')
        .select('symbol')
        .eq('id', marketId)
        .single();

      if (error || !data) {
        return null;
      }

      return data as { symbol: string };
    } catch (error) {
      logger.error(`❌ Error fetching market ${marketId}:`, error);
      return null;
    }
  }

  /**
   * Send WebSocket notification for order execution
   */
  private notifyOrderExecuted(
    order: ConditionalOrder,
    executionPrice: number,
    correlationId: string
  ): void {
    if (!this.wsService) {
      return;
    }

    try {
      const message = {
        type: 'order_executed',
        data: {
          orderId: order.id,
          userId: order.user_id,
          orderType: order.order_type,
          side: order.side,
          size: order.size,
          executionPrice,
          timestamp: Date.now(),
          correlationId
        },
        timestamp: Date.now()
      };

      // Broadcast to user's room
      if (this.wsService.broadcast) {
        this.wsService.broadcast('order_executed', message.data);
      }

      logger.debug(`📡 Sent WebSocket notification for order ${order.id}`);
    } catch (error) {
      logger.error(`❌ Error sending WebSocket notification for order ${order.id}:`, error);
    }
  }

  /**
   * Log error to database
   */
  private async logError(orderId: string, error: any, correlationId: string): Promise<void> {
    try {
      // You could create an error_logs table or append to order metadata
      await this.db.getClient()
        .from('orders')
        .update({
          metadata: {
            last_error: error instanceof Error ? error.message : 'Unknown error',
            error_correlation_id: correlationId,
            error_timestamp: new Date().toISOString()
          }
        })
        .eq('id', orderId);
    } catch (logError) {
      logger.error(`❌ Failed to log error for order ${orderId}:`, logError);
    }
  }

  /**
   * Get current metrics
   */
  public getMetrics(): Metrics & {
    processedOrdersPerSec: number;
    triggersFiredPerSec: number;
    errorsPerSec: number;
    executionLatencyP95: number;
    uptimeSeconds: number;
    circuitBreakerActive: boolean;
    circuitBreakerReason: string | null;
  } {
    const uptimeSeconds = (Date.now() - this.metrics.startTime) / 1000;
    const executionLatencies = [...this.metrics.executionLatencies].sort((a, b) => a - b);
    const p95Index = Math.floor(executionLatencies.length * 0.95);
    const executionLatencyP95 = executionLatencies[p95Index] || 0;

    return {
      ...this.metrics,
      processedOrdersPerSec: this.metrics.processedOrders / Math.max(uptimeSeconds, 1),
      triggersFiredPerSec: this.metrics.triggersFired / Math.max(uptimeSeconds, 1),
      errorsPerSec: this.metrics.errors / Math.max(uptimeSeconds, 1),
      executionLatencyP95,
      uptimeSeconds: Math.floor(uptimeSeconds),
      circuitBreakerActive: this.circuitBreakerActive,
      circuitBreakerReason: this.circuitBreakerReason
    };
  }

  /**
   * Public state snapshot
   */
  public getState(): { circuitBreakerActive: boolean; circuitBreakerReason: string | null } {
    return {
      circuitBreakerActive: this.circuitBreakerActive,
      circuitBreakerReason: this.circuitBreakerReason
    };
  }

  /**
   * Reset metrics
   */
  public resetMetrics(): void {
    this.metrics = {
      processedOrders: 0,
      triggersFired: 0,
      errors: 0,
      lastExecutionLatency: 0,
      executionLatencies: [],
      startTime: Date.now()
    };
  }
}

export const conditionalOrderMonitor = ConditionalOrderMonitor.getInstance();

