import { SupabaseDatabaseService } from './supabaseDatabase'
import { pythOracleService } from './pythOracleService'
import { WebSocketService } from './websocket'
import { referralService } from './referralService';
import { smartContractService } from './smartContractService';
import { getSupabaseService } from './supabaseService';
import { orderAuthorizationService } from './orderAuthorizationService';
import { performanceMonitoringService, monitorOrderPlacement, monitorOrderExecution } from './performanceMonitoringService';
import { errorHandlingService } from './errorHandlingService';
import { auditTrailService } from './auditTrailService';

export type OrderSide = 'buy' | 'sell'
export type PerpSide = 'long' | 'short'
export type OrderType = 'market' | 'limit'

export interface PlaceOrderInput {
  userId: string
  symbol: string // e.g. BTC-PERP
  side: OrderSide
  size: number
  orderType: OrderType
  price?: number
  leverage?: number
}

export class MatchingService {
  private static instance: MatchingService
  private readonly db: SupabaseDatabaseService
  private readonly supabase: ReturnType<typeof getSupabaseService>
  private readonly priceCache = new Map<string, { price: number; timestamp: number }>()
  private readonly CACHE_TTL = 5000 // 5 seconds cache TTL

  private constructor() {
    this.db = SupabaseDatabaseService.getInstance()
    this.supabase = getSupabaseService()
  }

  public static getInstance(): MatchingService {
    if (!MatchingService.instance) {
      MatchingService.instance = new MatchingService()
    }
    return MatchingService.instance
  }

  /**
   * Get cached price or fetch from Oracle with caching
   */
  private async getCachedPrice(symbol: string): Promise<number> {
    const now = Date.now()
    const cached = this.priceCache.get(symbol)
    
    if (cached && (now - cached.timestamp) < this.CACHE_TTL) {
      console.log(`📊 Using cached price for ${symbol}: ${cached.price}`);
      return cached.price
    }
    
    console.log(`📊 Fetching fresh price for ${symbol} from Oracle...`);
    const price = await pythOracleService.getLatestPrice(symbol)
    
    if (price != null) {
      this.priceCache.set(symbol, { price, timestamp: now })
      console.log(`📊 Cached price for ${symbol}: ${price}`);
    }
    
    return price
  }

  @monitorOrderPlacement
  public async placeOrder(input: PlaceOrderInput): Promise<{ orderId: string; filled: boolean; fills: Array<{ price: number; size: number }>; averageFillPrice?: number }>{
    const { userId, symbol, side, size, orderType } = input
    if (size <= 0) throw new Error('Size must be positive')
    if (orderType === 'limit' && (input.price == null || input.price <= 0)) throw new Error('Limit orders require positive price')

    let orderResult: any = null;
    
    try {

    // 🔐 SECURITY: Comprehensive order authorization check
    const authorizationContext = {
      userId,
      symbol,
      side,
      size,
      price: input.price,
      orderType,
      leverage: input.leverage || 1
    };

    const authorizationResult = await orderAuthorizationService.authorizeOrder(authorizationContext);
    
    if (!authorizationResult.authorized) {
      // Log the authorization failure for audit trail
      await orderAuthorizationService.logAuthorizationAttempt(authorizationContext, authorizationResult);
      
      throw new Error(`Order authorization failed: ${authorizationResult.reason} (Code: ${authorizationResult.code})`);
    }

    // Log successful authorization
    await orderAuthorizationService.logAuthorizationAttempt(authorizationContext, authorizationResult);

    console.log(`🔐 Order authorized for user ${userId} (Risk Level: ${authorizationResult.riskLevel})`);

    // Resolve market using fluent API
    console.log(`🔍 Looking up market for symbol: ${symbol}`);
    const market = await this.supabase.getMarketBySymbol(symbol);
    console.log(`🔍 Market found:`, market);
    const marketId = market.id as string

    // Create order row (pending by default)
    const orderTypeEnum = orderType === 'market' ? 'market' : 'limit'
    const perpSide: PerpSide = side === 'buy' ? 'long' : 'short'
    const price = input.price ?? null
    const leverage = input.leverage ?? 1

    const orderData = {
      user_id: userId,
      market_id: marketId,
      order_account: 'OFFCHAIN',
      order_type: orderTypeEnum,
      side: perpSide,
      size: size,
      price: price,
      leverage: leverage,
      status: 'pending'
    }

      console.log('🔍 Inserting order with data:', orderData);
      orderResult = await this.supabase.insertOrder(orderData)
      console.log('🔍 Order insertion result:', orderResult);
      const orderId = orderResult.id as string

      // 📝 AUDIT: Log order placement
      await auditTrailService.logOrderPlacement(
        userId,
        orderId,
        { ...orderData, marketId },
        undefined, // IP address would come from request context
        undefined, // User agent would come from request context
        undefined  // Session ID would come from request context
      );

    // Matching logic
    const fills: Array<{ price: number; size: number; makerOrderId?: string }> = []
    let remaining = size

    // Opposite side in DB schema uses 'long'|'short'
    const opposite = perpSide === 'long' ? 'short' : 'long'

    // Try match against resting orders using fluent API
    const priceCondition = orderType === 'limit' 
      ? { operator: side === 'buy' ? '<=' : '>=', value: input.price! }
      : undefined

    const pendingOrders = await this.supabase.getPendingOrders(marketId, opposite, priceCondition)

    for (const order of pendingOrders) {
      if (remaining <= 0) break
      const levelSize = parseFloat(order.remaining_size)
      const take = Math.min(remaining, levelSize)
      const levelPrice = parseFloat(order.price)
      fills.push({ price: levelPrice, size: take, makerOrderId: order.id })
      remaining -= take

      // Update resting order filled size using fluent API
      await this.supabase.updateOrderFill(order.id, take).catch(() => {})
    }

    // If no liquidity fully filled the order, use cached oracle price for the remainder if market order
    if (remaining > 0 && orderType === 'market') {
      console.log(`🔍 Getting cached oracle price for remaining ${remaining} size of market order`);
      const oraclePrice = await this.getCachedPrice(symbol)
      console.log(`🔍 Oracle price result: ${oraclePrice}`);
      if (oraclePrice == null) throw new Error('Price unavailable')
      fills.push({ price: oraclePrice, size: remaining })
      remaining = 0
    }

    const filled = remaining === 0
    const totalFilled = fills.reduce((s, f) => s + f.size, 0)
    const avg = totalFilled > 0 ? (fills.reduce((s, f) => s + f.price * f.size, 0) / totalFilled) : undefined

    // If not fully filled and limit order, keep pending; else mark filled
    if (filled) {
      // Update order status using fluent API
      await this.supabase.getClient()
        .from('orders')
        .update({
          status: 'filled',
          filled_size: totalFilled,
          average_fill_price: avg ?? null,
          filled_at: new Date().toISOString(),
          updated_at: new Date().toISOString()
        })
        .eq('id', orderId)

      // 🚀 Execute order on smart contract with atomic position creation
      const endExecutionTimer = performanceMonitoringService.startTimer('order_execution');
      
      try {
        console.log(`🚀 Executing order ${orderId} on smart contract with atomic position creation...`);
        
        const smartContractResult = await smartContractService.executeOrder({
          orderId,
          userId,
          marketSymbol: symbol,
          side: perpSide,
          size: totalFilled,
          price: avg || 0,
          leverage: leverage,
          orderType: orderTypeEnum
        });

        if (smartContractResult.success) {
          console.log(`✅ Order ${orderId} executed with atomic position creation:`, smartContractResult.transactionSignature);
          
          // Update order with smart contract details
          await this.supabase.getClient()
            .from('orders')
            .update({
              smart_contract_tx: smartContractResult.transactionSignature,
              smart_contract_position_id: smartContractResult.positionId,
              updated_at: new Date().toISOString()
            })
            .eq('id', orderId);
            
            endExecutionTimer();
            
            // 📝 AUDIT: Log successful order execution with atomic position creation
            await auditTrailService.logOrderExecution(
              userId,
              orderId,
              {
                filled: true,
                filledSize: totalFilled,
                averageFillPrice: avg,
                transactionSignature: smartContractResult.transactionSignature,
                positionId: smartContractResult.positionId,
                atomicPositionCreation: true
              }
            );

            // 🚀 BROADCAST: Send real-time updates via WebSocket
            try {
              const wsService = WebSocketService.current;
              if (wsService) {
                // Broadcast order update
                wsService.broadcastOrderUpdate(userId, {
                  orderId,
                  status: 'filled',
                  filledSize: totalFilled,
                  averageFillPrice: avg,
                  transactionSignature: smartContractResult.transactionSignature,
                  positionId: smartContractResult.positionId,
                  timestamp: Date.now()
                });

                // Broadcast position update (new position created)
                wsService.broadcastPositionUpdate(userId, {
                  userId,
                  positionId: smartContractResult.positionId,
                  symbol,
                  side: perpSide,
                  size: totalFilled,
                  entryPrice: avg,
                  leverage,
                  status: 'open',
                  timestamp: Date.now()
                });

                console.log(`📡 WebSocket broadcasts sent for order ${orderId} and position ${smartContractResult.positionId}`);
              }
            } catch (wsError) {
              console.error('❌ WebSocket broadcast failed:', wsError);
              // Don't fail the order execution if WebSocket fails
            }
          } else {
            console.error(`❌ Smart contract execution failed for order ${orderId}:`, smartContractResult.error);
            
            // Rollback order status if smart contract execution fails
            await this.supabase.getClient()
              .from('orders')
              .update({
                status: 'failed',
                error_message: smartContractResult.error,
                updated_at: new Date().toISOString()
              })
              .eq('id', orderId);
            
            endExecutionTimer();
            
            // 📝 AUDIT: Log failed order execution
            await auditTrailService.logOrderExecution(
              userId,
              orderId,
              {
                filled: false,
                filledSize: totalFilled,
                averageFillPrice: avg,
                error: smartContractResult.error,
                atomicPositionCreation: false
              }
            );

            // 🚀 BROADCAST: Send real-time updates via WebSocket for failed order
            try {
              const wsService = WebSocketService.current;
              if (wsService) {
                // Broadcast order update (failed)
                wsService.broadcastOrderUpdate(userId, {
                  orderId,
                  status: 'failed',
                  error: smartContractResult.error,
                  timestamp: Date.now()
                });

                console.log(`📡 WebSocket broadcast sent for failed order ${orderId}`);
              }
            } catch (wsError) {
              console.error('❌ WebSocket broadcast failed:', wsError);
              // Don't fail the order execution if WebSocket fails
            }
            
            throw new Error(`Smart contract execution failed: ${smartContractResult.error}`);
          }
      } catch (error) {
        console.error(`❌ Error executing order ${orderId} on smart contract:`, error);
        
        // Rollback order status and throw error to prevent inconsistent state
        await this.supabase.getClient()
          .from('orders')
          .update({
            status: 'failed',
            error_message: error instanceof Error ? error.message : 'Unknown error',
            updated_at: new Date().toISOString()
          })
          .eq('id', orderId);
        
        endExecutionTimer();
        throw error;
      }
    } else {
      // leave as pending with partial fill
      await this.supabase.getClient()
        .from('orders')
        .update({
          status: 'partially_filled',
          filled_size: totalFilled,
          average_fill_price: avg ?? null,
          updated_at: new Date().toISOString()
        })
        .eq('id', orderId)
    }

    // Write trade rows and update position for the taker (user)
    for (const f of fills) {
      const isMaker = f.makerOrderId === orderId; // If our order is the maker order, then it's a maker fill
      const fees = await this.calculateTradeFees(f.size, f.price, isMaker, userId);

      // Insert trade using fluent API
      await this.supabase.getClient()
        .from('trades')
        .insert({
          user_id: userId,
          market_id: marketId,
          position_id: null,
          order_id: orderId,
          trade_account: 'OFFCHAIN',
          side: side === 'buy' ? 'buy' : 'sell',
          size: f.size,
          price: f.price,
          fees: fees,
          created_at: new Date().toISOString()
        });

      await this.updatePositionOnFill({ 
        userId, 
        marketId, 
        side: perpSide, 
        price: f.price, 
        size: f.size,
        symbol,
        leverage
      });
      WebSocketService.current?.broadcast?.('trade', { symbol, price: f.price, size: f.size });
    }

    // Broadcast order update to user-specific room with smart contract status
    let smartContractTx = null;
    if (filled) {
      const { data, error } = await this.supabase.getClient()
        .from('orders')
        .select('smart_contract_tx, smart_contract_position_id')
        .eq('id', orderId)
        .single();
      
      if (!error && data) {
        smartContractTx = data;
      }
    }

    WebSocketService.current?.broadcastToUser?.(userId, 'order_update', { 
      symbol, 
      orderId, 
      status: filled ? 'filled' : 'partially_filled',
      filledSize: totalFilled,
      averageFillPrice: avg,
      userId,
      timestamp: Date.now(),
      smartContractTx: smartContractTx?.smart_contract_tx,
      smartContractPositionId: smartContractTx?.smart_contract_position_id,
      atomicPositionCreation: filled // Indicates if position was created atomically
    })

    // Also broadcast general order update for market data
    WebSocketService.current?.broadcast?.('order_update', { 
      symbol, 
      orderId, 
      status: filled ? 'filled' : 'partially_filled',
      filledSize: totalFilled,
      averageFillPrice: avg,
      userId,
      timestamp: Date.now(),
      smartContractTx: smartContractTx?.smart_contract_tx,
      atomicPositionCreation: filled
    })
    return { orderId, filled, fills, averageFillPrice: avg }

    } catch (error) {
      // 🚨 COMPREHENSIVE ERROR HANDLING
      const errorContext = {
        operation: 'order_placement',
        userId,
        orderId: orderResult?.id,
        metadata: { symbol, side, size, orderType, price: input.price, leverage: input.leverage }
      };

      const recoveryPlan = await errorHandlingService.handleError(error as Error, errorContext);
      
      console.error(`🚨 Order placement failed for user ${userId}:`, error);
      console.log(`🔄 Recovery plan:`, recoveryPlan);

      // Re-throw the original error for upstream handling
      throw error;
    }
  }

  private async updatePositionOnFill(params: { userId: string; marketId: string; side: PerpSide; size: number; price: number; symbol?: string; leverage?: number }): Promise<void> {
    const { userId, marketId, side, size, price, symbol, leverage } = params

    // Get or create position using fluent API
    const position = await this.supabase.getOrCreatePosition(userId, marketId, side)
    const positionId = position.id

    // Update position size and average price using fluent API
    await this.supabase.updatePosition(positionId, size, price)

    // Calculate and update health factor
    try {
      const healthFactor = await this.supabase.calculatePositionHealth(positionId)
      await this.supabase.updatePositionHealth(positionId, healthFactor)
      
      // Broadcast position update to user-specific room
      WebSocketService.current?.broadcastToUser?.(userId, 'position_update', { 
        positionId, 
        healthFactor,
        symbol,
        side,
        size: position.size,
        entryPrice: position.entry_price,
        currentPrice: price,
        unrealizedPnl: position.unrealized_pnl,
        timestamp: Date.now()
      });
      
      // Also broadcast general position update
      WebSocketService.current?.broadcast?.('position', { positionId, healthFactor })
    } catch (error) {
      console.error('Failed to update position health:', error)
    }

    // Position creation is now handled atomically in the smart contract execution
    // No separate position creation needed - it's part of the atomic transaction
  }

  private async updateHealthFactor(positionId: string): Promise<void> {
    try {
      const healthFactor = await this.supabase.calculatePositionHealth(positionId)
      await this.supabase.updatePositionHealth(positionId, healthFactor)
      WebSocketService.current?.broadcast?.('position', { positionId, healthFactor })
    } catch (error) {
      console.error('Failed to update position health:', error)
    }
  }

  private async calculateTradeFees(size: number, price: number, isMaker: boolean, userId: string): Promise<number> {
    const takerFeeRate = parseFloat(process.env.TAKER_FEE_RATE || '0.0005'); // 0.05%
    const makerRebateRate = parseFloat(process.env.MAKER_REBATE_RATE || '0.0002'); // 0.02% rebate

    const notionalValue = size * price;
    if (isMaker) {
      return -notionalValue * makerRebateRate; // Negative for rebate
    } else {
      let finalTakerFeeRate = takerFeeRate;
      const discount = await referralService.getRefereeFeeDiscount(userId);
      if (discount > 0) {
        finalTakerFeeRate = takerFeeRate * (1 - discount);
      }
      return notionalValue * finalTakerFeeRate;
    }
  }
}

export const matchingService = MatchingService.getInstance()


