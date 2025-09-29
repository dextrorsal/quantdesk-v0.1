import { Request, Response, NextFunction } from 'express';
import { Logger } from '../utils/logger';

const logger = new Logger();

// Rate limiting configuration
interface RateLimitConfig {
  windowMs: number;        // Time window in milliseconds
  maxRequests: number;    // Maximum requests per window
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: Request) => string;
  onLimitReached?: (req: Request, res: Response) => void;
}

// Default rate limit configurations
export const RATE_LIMITS = {
  // Public endpoints (price feeds, market data)
  PUBLIC: {
    windowMs: 60 * 1000,      // 1 minute
    maxRequests: 100,          // 100 requests per minute
    skipSuccessfulRequests: false,
    skipFailedRequests: false,
  },
  
  // Trading endpoints (orders, positions)
  TRADING: {
    windowMs: 60 * 1000,      // 1 minute
    maxRequests: 30,           // 30 requests per minute
    skipSuccessfulRequests: false,
    skipFailedRequests: false,
  },
  
  // Authentication endpoints
  AUTH: {
    windowMs: 15 * 60 * 1000, // 15 minutes
    maxRequests: 20,           // 20 requests per 15 minutes (more permissive for admin)
    skipSuccessfulRequests: true,
    skipFailedRequests: false,
  },
  
  // Admin endpoints
  ADMIN: {
    windowMs: 60 * 1000,      // 1 minute
    maxRequests: 100,          // 100 requests per minute
    skipSuccessfulRequests: false,
    skipFailedRequests: false,
  },
  
  // Webhook endpoints
  WEBHOOK: {
    windowMs: 60 * 1000,      // 1 minute
    maxRequests: 50,           // 50 requests per minute
    skipSuccessfulRequests: false,
    skipFailedRequests: false,
  }
};

// In-memory store for rate limiting (in production, use Redis)
class RateLimitStore {
  private store: Map<string, { count: number; resetTime: number }> = new Map();
  
  public get(key: string): { count: number; resetTime: number } | undefined {
    const entry = this.store.get(key);
    if (entry && Date.now() > entry.resetTime) {
      this.store.delete(key);
      return undefined;
    }
    return entry;
  }
  
  public set(key: string, count: number, windowMs: number): void {
    this.store.set(key, {
      count,
      resetTime: Date.now() + windowMs
    });
  }
  
  public increment(key: string, windowMs: number): { count: number; resetTime: number } {
    const entry = this.get(key);
    if (!entry) {
      this.set(key, 1, windowMs);
      return { count: 1, resetTime: Date.now() + windowMs };
    }
    
    const newCount = entry.count + 1;
    this.set(key, newCount, windowMs);
    return { count: newCount, resetTime: entry.resetTime };
  }
  
  public reset(key: string): void {
    this.store.delete(key);
  }
  
  public cleanup(): void {
    const now = Date.now();
    for (const [key, entry] of this.store.entries()) {
      if (now > entry.resetTime) {
        this.store.delete(key);
      }
    }
  }
}

const rateLimitStore = new RateLimitStore();

// Cleanup expired entries every 5 minutes
setInterval(() => {
  rateLimitStore.cleanup();
}, 5 * 60 * 1000);

// Generate rate limit key based on IP and user ID
function generateKey(req: Request): string {
  // Check for internal API key to bypass rate limiting
  const internalKey = req.headers['x-internal-key'];
  const expectedKey = process.env.INTERNAL_API_KEY;
  
  if (internalKey === expectedKey) {
    // Use a special key for internal services to bypass rate limiting
    return `internal:${req.ip || 'localhost'}`;
  }
  
  const ip = req.ip || req.connection.remoteAddress || 'unknown';
  const userId = (req as any).user?.id || 'anonymous';
  return `${ip}:${userId}`;
}

// Rate limiting middleware factory
export function createRateLimit(config: RateLimitConfig) {
  return (req: Request, res: Response, next: NextFunction) => {
    try {
      const key = config.keyGenerator ? config.keyGenerator(req) : generateKey(req);
      
      // Skip rate limiting for internal services
      if (key.startsWith('internal:')) {
        res.set({
          'X-RateLimit-Limit': 'unlimited',
          'X-RateLimit-Remaining': 'unlimited',
          'X-RateLimit-Reset': 'N/A',
          'X-RateLimit-Window': 'N/A'
        });
        return next();
      }
      
      const { count, resetTime } = rateLimitStore.increment(key, config.windowMs);
      
      // Set rate limit headers
      res.set({
        'X-RateLimit-Limit': config.maxRequests.toString(),
        'X-RateLimit-Remaining': Math.max(0, config.maxRequests - count).toString(),
        'X-RateLimit-Reset': new Date(resetTime).toISOString(),
        'X-RateLimit-Window': config.windowMs.toString()
      });
      
      // Check if limit exceeded
      if (count > config.maxRequests) {
        logger.warn(`Rate limit exceeded for ${key}: ${count}/${config.maxRequests} requests`);
        
        if (config.onLimitReached) {
          config.onLimitReached(req, res);
        }
        
        return res.status(429).json({
          success: false,
          error: 'Rate limit exceeded',
          message: `Too many requests. Limit: ${config.maxRequests} per ${config.windowMs / 1000} seconds`,
          retryAfter: Math.ceil((resetTime - Date.now()) / 1000),
          timestamp: Date.now()
        });
      }
      
      // Log rate limit status for monitoring
      if (count > config.maxRequests * 0.8) {
        logger.warn(`Rate limit warning for ${key}: ${count}/${config.maxRequests} requests`);
      }
      
      next();
    } catch (error) {
      logger.error('Rate limiting error:', error);
      next(); // Continue on error to avoid breaking the API
    }
  };
}

// Pre-configured rate limiters
export const rateLimiters = {
  public: createRateLimit(RATE_LIMITS.PUBLIC),
  trading: createRateLimit(RATE_LIMITS.TRADING),
  auth: createRateLimit(RATE_LIMITS.AUTH),
  admin: createRateLimit(RATE_LIMITS.ADMIN),
  webhook: createRateLimit(RATE_LIMITS.WEBHOOK)
};

// Advanced rate limiting for different user tiers
export function createTieredRateLimit() {
  return (req: Request, res: Response, next: NextFunction) => {
    const user = (req as any).user;
    
    // Determine user tier
    let config: RateLimitConfig;
    if (user?.tier === 'enterprise') {
      config = {
        windowMs: 60 * 1000,
        maxRequests: 1000,  // Enterprise: 1000 requests per minute
        skipSuccessfulRequests: false,
        skipFailedRequests: false,
      };
    } else if (user?.tier === 'professional') {
      config = {
        windowMs: 60 * 1000,
        maxRequests: 500,   // Professional: 500 requests per minute
        skipSuccessfulRequests: false,
        skipFailedRequests: false,
      };
    } else if (user?.tier === 'premium') {
      config = {
        windowMs: 60 * 1000,
        maxRequests: 200,   // Premium: 200 requests per minute
        skipSuccessfulRequests: false,
        skipFailedRequests: false,
      };
    } else {
      config = RATE_LIMITS.PUBLIC; // Free tier: 100 requests per minute
    }
    
    const rateLimiter = createRateLimit(config);
    rateLimiter(req, res, next);
  };
}

// Rate limit bypass for internal services
export function createInternalRateLimit() {
  return (req: Request, res: Response, next: NextFunction) => {
    const internalKey = req.headers['x-internal-key'];
    const expectedKey = process.env.INTERNAL_API_KEY;
    
    if (internalKey === expectedKey) {
      // Bypass rate limiting for internal services
      next();
    } else {
      // Apply normal rate limiting
      rateLimiters.public(req, res, next);
    }
  };
}

// Rate limit status endpoint
export function getRateLimitStatus(req: Request, res: Response) {
  const key = generateKey(req);
  const entry = rateLimitStore.get(key);
  
  if (!entry) {
    return res.json({
      success: true,
      data: {
        key,
        count: 0,
        limit: RATE_LIMITS.PUBLIC.maxRequests,
        remaining: RATE_LIMITS.PUBLIC.maxRequests,
        resetTime: null,
        windowMs: RATE_LIMITS.PUBLIC.windowMs
      }
    });
  }
  
  res.json({
    success: true,
    data: {
      key,
      count: entry.count,
      limit: RATE_LIMITS.PUBLIC.maxRequests,
      remaining: Math.max(0, RATE_LIMITS.PUBLIC.maxRequests - entry.count),
      resetTime: new Date(entry.resetTime).toISOString(),
      windowMs: RATE_LIMITS.PUBLIC.windowMs
    }
  });
}

export default {
  createRateLimit,
  rateLimiters,
  createTieredRateLimit,
  createInternalRateLimit,
  getRateLimitStatus,
  RATE_LIMITS
};
