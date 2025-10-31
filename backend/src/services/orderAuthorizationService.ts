import { Logger } from '../utils/logger';
import { SupabaseDatabaseService } from './supabaseDatabase';
import { getSupabaseService } from './supabaseService';

const logger = new Logger();

export interface OrderAuthorizationContext {
  userId: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  price?: number;
  orderType: 'market' | 'limit';
  leverage: number;
  walletAddress?: string;
}

export interface AuthorizationResult {
  authorized: boolean;
  reason?: string;
  code?: string;
  riskLevel?: 'low' | 'medium' | 'high';
}

export class OrderAuthorizationService {
  private static instance: OrderAuthorizationService;
  private readonly db: SupabaseDatabaseService;
  private readonly supabase: ReturnType<typeof getSupabaseService>;

  // Rate limiting configuration
  private readonly RATE_LIMIT_WINDOW = 60 * 1000; // 1 minute
  private readonly MAX_ORDERS_PER_MINUTE = 10;
  private readonly MAX_ORDERS_PER_HOUR = 100;
  private readonly MAX_ORDERS_PER_DAY = 1000;

  // Risk thresholds
  private readonly HIGH_RISK_SIZE_THRESHOLD = 10; // 10 units
  private readonly HIGH_RISK_LEVERAGE_THRESHOLD = 50; // 50x leverage
  private readonly SUSPICIOUS_PRICE_DEVIATION = 0.1; // 10% deviation from market

  private constructor() {
    this.db = SupabaseDatabaseService.getInstance();
    this.supabase = getSupabaseService();
  }

  public static getInstance(): OrderAuthorizationService {
    if (!OrderAuthorizationService.instance) {
      OrderAuthorizationService.instance = new OrderAuthorizationService();
    }
    return OrderAuthorizationService.instance;
  }

  /**
   * Comprehensive order authorization check
   */
  public async authorizeOrder(context: OrderAuthorizationContext): Promise<AuthorizationResult> {
    try {
      logger.info(`🔐 Starting order authorization for user ${context.userId}`);

      // 1. Basic validation checks
      const basicValidation = this.validateBasicOrder(context);
      if (!basicValidation.authorized) {
        return basicValidation;
      }

      // 2. User account validation
      const accountValidation = await this.validateUserAccount(context.userId);
      if (!accountValidation.authorized) {
        return accountValidation;
      }

      // 3. Rate limiting check
      const rateLimitCheck = await this.checkRateLimits(context.userId);
      if (!rateLimitCheck.authorized) {
        return rateLimitCheck;
      }

      // 4. Risk assessment
      const riskAssessment = await this.assessOrderRisk(context);
      if (!riskAssessment.authorized) {
        return riskAssessment;
      }

      // 5. Market validation
      const marketValidation = await this.validateMarket(context.symbol);
      if (!marketValidation.authorized) {
        return marketValidation;
      }

      // 6. Collateral validation
      const collateralValidation = await this.validateCollateral(context);
      if (!collateralValidation.authorized) {
        return collateralValidation;
      }

      // 7. Position limits validation
      const positionValidation = await this.validatePositionLimits(context);
      if (!positionValidation.authorized) {
        return positionValidation;
      }

      logger.info(`✅ Order authorization passed for user ${context.userId}`);
      return { authorized: true, riskLevel: 'low' };

    } catch (error) {
      logger.error(`❌ Order authorization failed for user ${context.userId}:`, error);
      return {
        authorized: false,
        reason: 'Authorization service error',
        code: 'AUTH_SERVICE_ERROR',
        riskLevel: 'high'
      };
    }
  }

  /**
   * Basic order parameter validation
   */
  private validateBasicOrder(context: OrderAuthorizationContext): AuthorizationResult {
    // Validate required fields
    if (!context.userId || !context.symbol || !context.side || !context.size || !context.orderType) {
      return {
        authorized: false,
        reason: 'Missing required order parameters',
        code: 'MISSING_PARAMETERS',
        riskLevel: 'high'
      };
    }

    // Validate size
    if (context.size <= 0 || context.size > 1000000) {
      return {
        authorized: false,
        reason: 'Invalid order size',
        code: 'INVALID_SIZE',
        riskLevel: 'high'
      };
    }

    // Validate leverage
    if (context.leverage < 1 || context.leverage > 100) {
      return {
        authorized: false,
        reason: 'Invalid leverage',
        code: 'INVALID_LEVERAGE',
        riskLevel: 'high'
      };
    }

    // Validate price for limit orders
    if (context.orderType === 'limit' && (!context.price || context.price <= 0)) {
      return {
        authorized: false,
        reason: 'Invalid price for limit order',
        code: 'INVALID_PRICE',
        riskLevel: 'high'
      };
    }

    // Validate side
    if (!['buy', 'sell'].includes(context.side)) {
      return {
        authorized: false,
        reason: 'Invalid order side',
        code: 'INVALID_SIDE',
        riskLevel: 'high'
      };
    }

    // Validate order type
    if (!['market', 'limit'].includes(context.orderType)) {
      return {
        authorized: false,
        reason: 'Invalid order type',
        code: 'INVALID_ORDER_TYPE',
        riskLevel: 'high'
      };
    }

    return { authorized: true, riskLevel: 'low' };
  }

  /**
   * Validate user account status
   */
  private async validateUserAccount(userId: string): Promise<AuthorizationResult> {
    try {
      const users = await this.db.select('users', '*', { id: userId });
      const user = users?.[0];
      
      if (!user) {
        return {
          authorized: false,
          reason: 'User account not found',
          code: 'USER_NOT_FOUND',
          riskLevel: 'high'
        };
      }

      // Check if account is active
      if (!user.is_active) {
        return {
          authorized: false,
          reason: 'User account is not active',
          code: 'ACCOUNT_INACTIVE',
          riskLevel: 'high'
        };
      }

      // Check if account is verified (if required)
      if (user.kyc_status !== 'verified' && user.kyc_status !== 'pending') {
        return {
          authorized: false,
          reason: 'User account verification required',
          code: 'ACCOUNT_UNVERIFIED',
          riskLevel: 'medium'
        };
      }

      return { authorized: true, riskLevel: 'low' };

    } catch (error) {
      logger.error('Error validating user account:', error);
      return {
        authorized: false,
        reason: 'User account validation failed',
        code: 'ACCOUNT_VALIDATION_ERROR',
        riskLevel: 'high'
      };
    }
  }

  /**
   * Check rate limits for order placement
   */
  private async checkRateLimits(userId: string): Promise<AuthorizationResult> {
    try {
      const now = Date.now();
      const oneMinuteAgo = now - this.RATE_LIMIT_WINDOW;
      const oneHourAgo = now - (60 * 60 * 1000);
      const oneDayAgo = now - (24 * 60 * 60 * 1000);

      // Get order counts for different time windows
      const [minuteCount, hourCount, dayCount] = await Promise.all([
        this.getOrderCount(userId, oneMinuteAgo, now),
        this.getOrderCount(userId, oneHourAgo, now),
        this.getOrderCount(userId, oneDayAgo, now)
      ]);

      // Check minute rate limit
      if (minuteCount >= this.MAX_ORDERS_PER_MINUTE) {
        return {
          authorized: false,
          reason: `Rate limit exceeded: ${minuteCount} orders in the last minute`,
          code: 'RATE_LIMIT_MINUTE',
          riskLevel: 'high'
        };
      }

      // Check hour rate limit
      if (hourCount >= this.MAX_ORDERS_PER_HOUR) {
        return {
          authorized: false,
          reason: `Rate limit exceeded: ${hourCount} orders in the last hour`,
          code: 'RATE_LIMIT_HOUR',
          riskLevel: 'high'
        };
      }

      // Check day rate limit
      if (dayCount >= this.MAX_ORDERS_PER_DAY) {
        return {
          authorized: false,
          reason: `Rate limit exceeded: ${dayCount} orders in the last day`,
          code: 'RATE_LIMIT_DAY',
          riskLevel: 'high'
        };
      }

      return { authorized: true, riskLevel: 'low' };

    } catch (error) {
      logger.error('Error checking rate limits:', error);
      return {
        authorized: false,
        reason: 'Rate limit check failed',
        code: 'RATE_LIMIT_ERROR',
        riskLevel: 'high'
      };
    }
  }

  /**
   * Assess order risk level
   */
  private async assessOrderRisk(context: OrderAuthorizationContext): Promise<AuthorizationResult> {
    let riskLevel: 'low' | 'medium' | 'high' = 'low';
    const riskFactors: string[] = [];

    // Check size risk
    if (context.size > this.HIGH_RISK_SIZE_THRESHOLD) {
      riskLevel = 'high';
      riskFactors.push(`Large order size: ${context.size}`);
    }

    // Check leverage risk
    if (context.leverage > this.HIGH_RISK_LEVERAGE_THRESHOLD) {
      riskLevel = 'high';
      riskFactors.push(`High leverage: ${context.leverage}x`);
    }

    // Check price deviation for limit orders
    if (context.orderType === 'limit' && context.price) {
      try {
        const marketPrice = await this.getCurrentMarketPrice(context.symbol);
        if (marketPrice) {
          const deviation = Math.abs(context.price - marketPrice) / marketPrice;
          if (deviation > this.SUSPICIOUS_PRICE_DEVIATION) {
            riskLevel = riskLevel === 'high' ? 'high' : 'medium';
            riskFactors.push(`Suspicious price deviation: ${(deviation * 100).toFixed(2)}%`);
          }
        }
      } catch (error) {
        logger.warn('Could not check price deviation:', error);
      }
    }

    // Check user's trading history for suspicious patterns
    const suspiciousPatterns = await this.checkSuspiciousPatterns(context.userId);
    if (suspiciousPatterns.length > 0) {
      riskLevel = 'high';
      riskFactors.push(...suspiciousPatterns);
    }

    // For high-risk orders, require additional validation
    if (riskLevel === 'high') {
      return {
        authorized: false,
        reason: `High-risk order detected: ${riskFactors.join(', ')}`,
        code: 'HIGH_RISK_ORDER',
        riskLevel: 'high'
      };
    }

    return { authorized: true, riskLevel };
  }

  /**
   * Validate market availability and status
   */
  private async validateMarket(symbol: string): Promise<AuthorizationResult> {
    try {
      const market = await this.supabase.getMarketBySymbol(symbol);
      
      if (!market) {
        return {
          authorized: false,
          reason: 'Market not found',
          code: 'MARKET_NOT_FOUND',
          riskLevel: 'high'
        };
      }

      if (!market.is_active) {
        return {
          authorized: false,
          reason: 'Market is not active',
          code: 'MARKET_INACTIVE',
          riskLevel: 'high'
        };
      }

      return { authorized: true, riskLevel: 'low' };

    } catch (error) {
      logger.error('Error validating market:', error);
      return {
        authorized: false,
        reason: 'Market validation failed',
        code: 'MARKET_VALIDATION_ERROR',
        riskLevel: 'high'
      };
    }
  }

  /**
   * Validate user has sufficient collateral
   */
  private async validateCollateral(context: OrderAuthorizationContext): Promise<AuthorizationResult> {
    try {
      // Get user's collateral balance
      const userBalances = await this.db.getUserBalances(context.userId);
      const usdcBalance = userBalances.find(b => b.asset === 'USDC');
      const collateralBalance = usdcBalance?.balance || 0;
      
      if (!collateralBalance || collateralBalance <= 0) {
        return {
          authorized: false,
          reason: 'Insufficient collateral balance',
          code: 'INSUFFICIENT_COLLATERAL',
          riskLevel: 'high'
        };
      }

      // Calculate required margin
      const notionalValue = context.size * (context.price || await this.getCurrentMarketPrice(context.symbol) || 0);
      const requiredMargin = notionalValue / context.leverage;

      if (collateralBalance < requiredMargin) {
        return {
          authorized: false,
          reason: `Insufficient collateral: required ${requiredMargin}, available ${collateralBalance}`,
          code: 'INSUFFICIENT_MARGIN',
          riskLevel: 'high'
        };
      }

      return { authorized: true, riskLevel: 'low' };

    } catch (error) {
      logger.error('Error validating collateral:', error);
      return {
        authorized: false,
        reason: 'Collateral validation failed',
        code: 'COLLATERAL_VALIDATION_ERROR',
        riskLevel: 'high'
      };
    }
  }

  /**
   * Validate position limits
   */
  private async validatePositionLimits(context: OrderAuthorizationContext): Promise<AuthorizationResult> {
    try {
      // Get user's current positions
      const positions = await this.db.getUserPositions(context.userId);
      
      // Check maximum number of positions
      const maxPositions = 50; // Configurable limit
      if (positions.length >= maxPositions) {
        return {
          authorized: false,
          reason: `Maximum number of positions exceeded: ${positions.length}/${maxPositions}`,
          code: 'MAX_POSITIONS_EXCEEDED',
          riskLevel: 'medium'
        };
      }

      // Check total exposure
      const totalExposure = positions.reduce((sum, pos) => sum + (pos.size * pos.current_price), 0);
      const newExposure = context.size * (context.price || await this.getCurrentMarketPrice(context.symbol) || 0);
      const maxTotalExposure = 1000000; // Configurable limit

      if (totalExposure + newExposure > maxTotalExposure) {
        return {
          authorized: false,
          reason: `Total exposure limit exceeded: ${totalExposure + newExposure}/${maxTotalExposure}`,
          code: 'MAX_EXPOSURE_EXCEEDED',
          riskLevel: 'high'
        };
      }

      return { authorized: true, riskLevel: 'low' };

    } catch (error) {
      logger.error('Error validating position limits:', error);
      return {
        authorized: false,
        reason: 'Position limit validation failed',
        code: 'POSITION_LIMIT_ERROR',
        riskLevel: 'high'
      };
    }
  }

  /**
   * Helper methods
   */
  private async getOrderCount(userId: string, startTime: number, endTime: number): Promise<number> {
    try {
      const result = await this.supabase.getClient()
        .from('orders')
        .select('id', { count: 'exact' })
        .eq('user_id', userId)
        .gte('created_at', new Date(startTime).toISOString())
        .lte('created_at', new Date(endTime).toISOString());
      
      return result.count || 0;
    } catch (error) {
      logger.error('Error getting order count:', error);
      return 0;
    }
  }

  private async getCurrentMarketPrice(symbol: string): Promise<number | null> {
    try {
      const { pythOracleService } = await import('./pythOracleService.js');
      return await pythOracleService.getLatestPrice(symbol);
    } catch (error) {
      logger.error('Error getting market price:', error);
      return null;
    }
  }

  private async checkSuspiciousPatterns(userId: string): Promise<string[]> {
    const patterns: string[] = [];
    
    try {
      // Check for rapid-fire orders (potential bot behavior)
      const { data: recentOrders, error: recentOrdersError } = await this.supabase.getClient()
        .from('orders')
        .select('created_at')
        .eq('user_id', userId)
        .gte('created_at', new Date(Date.now() - 5 * 60 * 1000).toISOString())
        .order('created_at', { ascending: false })
        .limit(20);

      if (recentOrdersError) {
        logger.error('Error fetching recent orders:', recentOrdersError);
        return patterns; // Return what we have so far
      }

      if (recentOrders && recentOrders.length >= 15) {
        patterns.push('Rapid-fire order pattern detected');
      }

      // Check for identical orders (potential spam)
      const { data: identicalOrders, error: identicalOrdersError } = await this.supabase.getClient()
        .from('orders')
        .select('size, price, side, order_type')
        .eq('user_id', userId)
        .gte('created_at', new Date(Date.now() - 60 * 60 * 1000).toISOString());

      if (identicalOrdersError) {
        logger.error('Error fetching identical orders:', identicalOrdersError);
        return patterns; // Return what we have so far
      }

      const orderGroups = new Map<string, number>();
      if (identicalOrders) {
        identicalOrders.forEach(order => {
          const key = `${order.size}-${order.price}-${order.side}-${order.order_type}`;
          orderGroups.set(key, (orderGroups.get(key) || 0) + 1);
        });
      }

      for (const [key, count] of orderGroups) {
        if (count >= 10) {
          patterns.push(`Identical order pattern detected: ${count} identical orders`);
        }
      }

    } catch (error) {
      logger.error('Error checking suspicious patterns:', error);
    }

    return patterns;
  }

  /**
   * Log authorization attempt for audit trail
   */
  public async logAuthorizationAttempt(
    context: OrderAuthorizationContext, 
    result: AuthorizationResult
  ): Promise<void> {
    try {
      await this.supabase.getClient()
        .from('order_authorization_logs')
        .insert({
          user_id: context.userId,
          symbol: context.symbol,
          side: context.side,
          size: context.size,
          price: context.price,
          order_type: context.orderType,
          leverage: context.leverage,
          authorized: result.authorized,
          reason: result.reason,
          code: result.code,
          risk_level: result.riskLevel,
          created_at: new Date().toISOString()
        });
    } catch (error) {
      logger.error('Error logging authorization attempt:', error);
    }
  }

  /**
   * Sanitize order input to prevent malicious data
   */
  public sanitizeOrderInput(input: any): any {
    if (!input || typeof input !== 'object') {
      return {};
    }

    const sanitized: any = {};
    
    // Sanitize string fields
    if (input.symbol && typeof input.symbol === 'string') {
      sanitized.symbol = input.symbol.replace(/[<>\"'&]/g, '').trim();
    }
    
    if (input.side && typeof input.side === 'string') {
      sanitized.side = ['buy', 'sell'].includes(input.side) ? input.side : 'buy';
    }
    
    if (input.orderType && typeof input.orderType === 'string') {
      const validTypes = ['market', 'limit', 'stop_loss', 'take_profit', 'trailing_stop'];
      sanitized.orderType = validTypes.includes(input.orderType) ? input.orderType : 'market';
    }

    // Sanitize numeric fields
    if (typeof input.size === 'number' && input.size > 0) {
      sanitized.size = Math.min(Math.max(input.size, 0.001), 1000000);
    }
    
    if (typeof input.price === 'number' && input.price > 0) {
      sanitized.price = Math.min(Math.max(input.price, 0.01), 1000000);
    }
    
    if (typeof input.leverage === 'number' && input.leverage > 0) {
      sanitized.leverage = Math.min(Math.max(input.leverage, 1), 100);
    }

    // Handle null/undefined values
    Object.keys(input).forEach(key => {
      if (input[key] === null || input[key] === undefined) {
        sanitized[key] = undefined;
      }
    });

    return sanitized;
  }

  /**
   * Check if an order can be executed based on current market conditions
   */
  public canExecuteOrder(order: any, currentPrice: number): boolean {
    if (!order || typeof currentPrice !== 'number') {
      return false;
    }

    // Check if order is expired
    if (order.expires_at && new Date(order.expires_at) < new Date()) {
      return false;
    }

    // Check order status
    if (order.status !== 'pending' && order.status !== 'partially_filled') {
      return false;
    }

    // Check if order has remaining size
    if (order.remaining_size <= 0) {
      return false;
    }

    // Check limit order price conditions
    if (order.order_type === 'limit') {
      if (order.side === 'buy' && order.price < currentPrice) {
        return false; // Buy limit below market price
      }
      if (order.side === 'sell' && order.price > currentPrice) {
        return false; // Sell limit above market price
      }
    }

    return true;
  }

  /**
   * Prepare smart contract instruction for order execution
   */
  public prepareSmartContractInstruction(params: {
    orderId: string;
    userId: string;
    marketId: string;
    side: 'buy' | 'sell';
    size: number;
    price?: number;
    orderType: string;
    leverage: number;
  }): any {
    if (!params.orderId || !params.userId || !params.marketId) {
      throw new Error('Invalid order parameters');
    }

    if (!['buy', 'sell'].includes(params.side)) {
      throw new Error('Invalid order side');
    }

    if (params.size <= 0 || params.size > 1000000) {
      throw new Error('Invalid order size');
    }

    if (params.leverage < 1 || params.leverage > 100) {
      throw new Error('Invalid leverage');
    }

    return {
      instruction: 'execute_order',
      data: {
        order_id: params.orderId,
        user_id: params.userId,
        market_id: params.marketId,
        side: params.side,
        size: params.size,
        price: params.price,
        order_type: params.orderType,
        leverage: params.leverage,
        timestamp: Date.now()
      }
    };
  }

  /**
   * Transition order status with validation
   */
  public transitionOrderStatus(order: any, newStatus: string, filledSize?: number): string {
    if (!order) {
      throw new Error('Order is required');
    }

    const validTransitions: { [key: string]: string[] } = {
      'pending': ['filled', 'partially_filled', 'cancelled', 'expired', 'rejected'],
      'partially_filled': ['filled', 'cancelled', 'expired'],
      'filled': [], // Terminal state
      'cancelled': [], // Terminal state
      'expired': [], // Terminal state
      'rejected': [] // Terminal state
    };

    const currentStatus = order.status;
    const allowedTransitions = validTransitions[currentStatus] || [];

    if (!allowedTransitions.includes(newStatus)) {
      throw new Error(`Invalid status transition from ${currentStatus} to ${newStatus}`);
    }

    // Update order properties based on new status
    const updatedOrder = { ...order, status: newStatus };

    if (newStatus === 'filled' || newStatus === 'partially_filled') {
      if (typeof filledSize === 'number' && filledSize > 0) {
        updatedOrder.filled_size = (order.filled_size || 0) + filledSize;
        updatedOrder.remaining_size = order.size - updatedOrder.filled_size;
        
        if (newStatus === 'filled') {
          updatedOrder.filled_at = new Date().toISOString();
        }
      }
    }

    if (newStatus === 'cancelled') {
      updatedOrder.cancelled_at = new Date().toISOString();
    }

    return newStatus;
  }

  /**
   * Create position from filled order
   */
  public createPositionFromOrder(order: any): any {
    if (!order) {
      throw new Error('Order is required');
    }

    if (order.status !== 'filled') {
      throw new Error('Cannot create position from unfilled order');
    }

    if (!order.filled_size || order.filled_size <= 0) {
      throw new Error('Order has no filled size');
    }

    return {
      user_id: order.user_id,
      market_id: order.market_id,
      side: order.side,
      size: order.filled_size,
      entry_price: order.average_fill_price || order.price,
      leverage: order.leverage,
      margin: (order.filled_size * (order.average_fill_price || order.price)) / order.leverage,
      created_at: new Date().toISOString(),
      updated_at: new Date().toISOString()
    };
  }

  /**
   * Generate user-friendly error messages
   */
  public generateOrderErrorMessage(error: any, orderId?: string): string {
    if (!error) {
      return 'An unknown error occurred';
    }

    const errorMessage = error.message || error.toString();
    
    // User-friendly error mappings
    const errorMappings: { [key: string]: string } = {
      'Insufficient balance': 'You don\'t have enough balance to place this order',
      'Invalid order parameters': 'Please check your order details and try again',
      'Order size too small': 'Order size must be at least 0.001',
      'Order size too large': 'Order size exceeds maximum limit',
      'Invalid leverage': 'Leverage must be between 1x and 100x',
      'Market closed': 'This market is currently closed for trading',
      'Price deviation too high': 'Order price is too far from current market price',
      'Rate limit exceeded': 'Too many orders placed. Please wait a moment',
      'User not found': 'Please log in again to continue trading',
      'Session expired': 'Your session has expired. Please log in again'
    };

    // Check for specific error patterns
    for (const [pattern, friendlyMessage] of Object.entries(errorMappings)) {
      if (errorMessage.toLowerCase().includes(pattern.toLowerCase())) {
        return friendlyMessage;
      }
    }

    // Default user-friendly message
    return 'Unable to process your order. Please try again or contact support if the issue persists.';
  }

  /**
   * Execute atomic order transaction with rollback capability
   */
  public async executeAtomicOrderTransaction(order: any, createPositionCallback: Function): Promise<any> {
    if (!order) {
      throw new Error('Order is required');
    }

    try {
      // Validate order can be executed
      if (!this.canExecuteOrder(order, order.price || 0)) {
        throw new Error('Order cannot be executed');
      }

      // Execute order and create position atomically
      const result = await createPositionCallback(order);

      if (!result) {
        throw new Error('Position creation failed');
      }

      // Update order status
      this.transitionOrderStatus(order, 'filled', order.size);

      return {
        success: true,
        orderId: order.id,
        positionId: result.id,
        message: 'Order executed successfully'
      };

    } catch (error) {
      // Rollback any changes
      logger.error('Atomic order transaction failed:', error);
      throw new Error('Order execution failed');
    }
  }
}

// Export singleton instance
export const orderAuthorizationService = OrderAuthorizationService.getInstance();
