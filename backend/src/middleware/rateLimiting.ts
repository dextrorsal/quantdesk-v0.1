import { Request, Response, NextFunction } from 'express';
import { Logger } from '../utils/logger';
import { getSupabaseService } from '../services/supabaseService';

const logger = new Logger();

export interface RateLimitConfig {
  windowMs: number;
  maxRequests: number;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: Request) => string;
  onLimitReached?: (req: Request, res: Response) => void;
}

export interface RateLimitResult {
  allowed: boolean;
  remaining: number;
  resetTime: number;
  totalHits: number;
}

export class RateLimitService {
  private static instance: RateLimitService;
  private readonly supabase: ReturnType<typeof getSupabaseService>;
  private readonly defaultConfig: RateLimitConfig;

  private constructor() {
    this.supabase = getSupabaseService();
    this.defaultConfig = {
      windowMs: 60 * 1000, // 1 minute
      maxRequests: 10,
      skipSuccessfulRequests: false,
      skipFailedRequests: false,
      keyGenerator: (req: Request) => {
        // Use user ID if authenticated, otherwise IP address
        const userId = (req as any).user?.id;
        return userId ? `user:${userId}` : `ip:${req.ip}`;
      }
    };
  }

  public static getInstance(): RateLimitService {
    if (!RateLimitService.instance) {
      RateLimitService.instance = new RateLimitService();
    }
    return RateLimitService.instance;
  }

  /**
   * Create rate limiting middleware
   */
  public createRateLimit(config: Partial<RateLimitConfig> = {}) {
    const finalConfig = { ...this.defaultConfig, ...config };

    return async (req: Request, res: Response, next: NextFunction) => {
      try {
        const key = finalConfig.keyGenerator!(req);
        const now = Date.now();
        const windowStart = now - finalConfig.windowMs;

        // Get current request count for this key
        const result = await this.checkRateLimit(key, windowStart, now, finalConfig.maxRequests);

        // Add rate limit headers
        res.set({
          'X-RateLimit-Limit': finalConfig.maxRequests.toString(),
          'X-RateLimit-Remaining': result.remaining.toString(),
          'X-RateLimit-Reset': new Date(result.resetTime).toISOString(),
          'X-RateLimit-Total-Hits': result.totalHits.toString()
        });

        if (!result.allowed) {
          logger.warn(`Rate limit exceeded for key: ${key}, hits: ${result.totalHits}/${finalConfig.maxRequests}`);
          
          if (finalConfig.onLimitReached) {
            finalConfig.onLimitReached(req, res);
          } else {
            res.status(429).json({
              success: false,
              error: 'Rate limit exceeded',
              message: `Too many requests. Limit: ${finalConfig.maxRequests} per ${finalConfig.windowMs / 1000} seconds`,
              retryAfter: Math.ceil((result.resetTime - now) / 1000)
            });
          }
          return;
        }

        // Record this request
        await this.recordRequest(key, now);

        next();

      } catch (error) {
        logger.error('Rate limiting error:', error);
        // On error, allow the request to proceed (fail open)
        next();
      }
    };
  }

  /**
   * Check if request is within rate limit
   */
  private async checkRateLimit(
    key: string, 
    windowStart: number, 
    now: number, 
    maxRequests: number
  ): Promise<RateLimitResult> {
    try {
      // Get request count from database
      const { data, error } = await this.supabase.getClient()
        .from('rate_limit_logs')
        .select('created_at')
        .eq('key', key)
        .gte('created_at', new Date(windowStart).toISOString())
        .lte('created_at', new Date(now).toISOString());

      if (error) {
        logger.error('Error checking rate limit:', error);
        // On error, allow the request (fail open)
        return {
          allowed: true,
          remaining: maxRequests,
          resetTime: now + 60 * 1000,
          totalHits: 0
        };
      }

      const totalHits = data?.length || 0;
      const allowed = totalHits < maxRequests;
      const remaining = Math.max(0, maxRequests - totalHits);
      const resetTime = now + (windowStart + 60 * 1000 - now);

      return {
        allowed,
        remaining,
        resetTime,
        totalHits
      };

    } catch (error) {
      logger.error('Error checking rate limit:', error);
      // On error, allow the request (fail open)
      return {
        allowed: true,
        remaining: maxRequests,
        resetTime: now + 60 * 1000,
        totalHits: 0
      };
    }
  }

  /**
   * Record a request for rate limiting
   */
  private async recordRequest(key: string, timestamp: number): Promise<void> {
    try {
      await this.supabase.getClient()
        .from('rate_limit_logs')
        .insert({
          key,
          created_at: new Date(timestamp).toISOString()
        });

      // Clean up old records (older than 1 hour)
      const oneHourAgo = timestamp - (60 * 60 * 1000);
      await this.supabase.getClient()
        .from('rate_limit_logs')
        .delete()
        .lt('created_at', new Date(oneHourAgo).toISOString());

    } catch (error) {
      logger.error('Error recording rate limit request:', error);
    }
  }

  /**
   * Get rate limit status for a key
   */
  public async getRateLimitStatus(key: string): Promise<RateLimitResult | null> {
    try {
      const now = Date.now();
      const windowStart = now - this.defaultConfig.windowMs;

      return await this.checkRateLimit(key, windowStart, now, this.defaultConfig.maxRequests);

    } catch (error) {
      logger.error('Error getting rate limit status:', error);
      return null;
    }
  }

  /**
   * Reset rate limit for a key (admin function)
   */
  public async resetRateLimit(key: string): Promise<void> {
    try {
      await this.supabase.getClient()
        .from('rate_limit_logs')
        .delete()
        .eq('key', key);

      logger.info(`Rate limit reset for key: ${key}`);

    } catch (error) {
      logger.error('Error resetting rate limit:', error);
    }
  }
}

// Export singleton instance
export const rateLimitService = RateLimitService.getInstance();

// Pre-configured rate limiters for different endpoints
export const orderRateLimit = rateLimitService.createRateLimit({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 10, // 10 orders per minute
  onLimitReached: (req: Request, res: Response) => {
    res.status(429).json({
      success: false,
      error: 'Order rate limit exceeded',
      message: 'Too many orders placed. Maximum 10 orders per minute.',
      retryAfter: 60
    });
  }
});

export const apiRateLimit = rateLimitService.createRateLimit({
  windowMs: 60 * 1000, // 1 minute
  maxRequests: 100, // 100 API calls per minute
  onLimitReached: (req: Request, res: Response) => {
    res.status(429).json({
      success: false,
      error: 'API rate limit exceeded',
      message: 'Too many API requests. Maximum 100 requests per minute.',
      retryAfter: 60
    });
  }
});

export const authRateLimit = rateLimitService.createRateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  maxRequests: 5, // 5 auth attempts per 15 minutes
  onLimitReached: (req: Request, res: Response) => {
    res.status(429).json({
      success: false,
      error: 'Authentication rate limit exceeded',
      message: 'Too many authentication attempts. Please try again later.',
      retryAfter: 900
    });
  }
});

// Export rate limiters object for server.ts
export const rateLimiters = {
  admin: rateLimitService.createRateLimit({
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 50, // 50 admin requests per minute
  }),
  auth: authRateLimit,
  trading: orderRateLimit,
  webhook: rateLimitService.createRateLimit({
    windowMs: 60 * 1000, // 1 minute
    maxRequests: 20, // 20 webhook calls per minute
  }),
  public: apiRateLimit
};