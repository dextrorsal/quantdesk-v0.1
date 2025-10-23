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
      const user = await this.db.getUserByWallet(userId);
      
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
      const { pythOracleService } = await import('./pythOracleService');
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
}

// Export singleton instance
export const orderAuthorizationService = OrderAuthorizationService.getInstance();
