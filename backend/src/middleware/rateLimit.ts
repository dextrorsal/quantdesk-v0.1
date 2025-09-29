import { Request, Response, NextFunction } from 'express';
import rateLimit from 'express-rate-limit';
import { Logger } from '../utils/logger';

const logger = new Logger();

interface RateLimitOptions {
  windowMs: number;
  max: number;
  skipSuccessfulRequests?: boolean;
  skipFailedRequests?: boolean;
  keyGenerator?: (req: Request) => string;
}

export const rateLimitMiddleware = (options: RateLimitOptions) => {
  return rateLimit({
    windowMs: options.windowMs,
    max: options.max,
    skipSuccessfulRequests: options.skipSuccessfulRequests || false,
    skipFailedRequests: options.skipFailedRequests || false,
    keyGenerator: options.keyGenerator || ((req: Request) => {
      // Use user ID if authenticated, otherwise IP
      const authHeader = req.headers.authorization;
      if (authHeader && authHeader.startsWith('Bearer ')) {
        try {
          const token = authHeader.substring(7);
          const decoded = require('jsonwebtoken').verify(token, process.env['JWT_SECRET']!);
          return `user:${decoded.walletAddress}`;
        } catch (error) {
          // Fall back to IP if token is invalid
          return req.ip || req.connection.remoteAddress || 'unknown';
        }
      }
      return req.ip || req.connection.remoteAddress || 'unknown';
    }),
    message: {
      error: 'Too many requests, please try again later.',
      retryAfter: Math.ceil(options.windowMs / 1000)
    },
    standardHeaders: true,
    legacyHeaders: false,
    handler: (req: Request, res: Response) => {
      logger.warn(`Rate limit exceeded for ${req.ip} on ${req.path}`);
      res.status(429).json({
        error: 'Too many requests, please try again later.',
        retryAfter: Math.ceil(options.windowMs / 1000),
        limit: options.max,
        windowMs: options.windowMs
      });
    }
  });
};

// Specific rate limits for different endpoints
export const tradingRateLimit = rateLimitMiddleware({
  windowMs: 60 * 1000, // 1 minute
  max: 60, // 60 requests per minute
  skipSuccessfulRequests: true
});

export const orderRateLimit = rateLimitMiddleware({
  windowMs: 60 * 1000, // 1 minute
  max: 30, // 30 orders per minute
  skipSuccessfulRequests: true
});

export const authRateLimit = rateLimitMiddleware({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 10, // 10 auth attempts per 15 minutes
  skipSuccessfulRequests: true
});

export const apiRateLimit = rateLimitMiddleware({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // 1000 requests per 15 minutes
  skipSuccessfulRequests: false
});

// Dynamic rate limiting based on user tier
export const dynamicRateLimit = (req: Request, res: Response, next: NextFunction) => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    // Use standard rate limit for unauthenticated users
    return apiRateLimit(req, res, next);
  }

  try {
    const token = authHeader.substring(7);
    const decoded = require('jsonwebtoken').verify(token, process.env['JWT_SECRET']!);
    
    // Get user tier from token or database
    const userTier = decoded.tier || 'standard';
    
    let maxRequests: number;
    let windowMs: number;
    
    switch (userTier) {
      case 'premium':
        maxRequests = 5000;
        windowMs = 15 * 60 * 1000; // 15 minutes
        break;
      case 'pro':
        maxRequests = 10000;
        windowMs = 15 * 60 * 1000; // 15 minutes
        break;
      case 'standard':
      default:
        maxRequests = 1000;
        windowMs = 15 * 60 * 1000; // 15 minutes
        break;
    }
    
    const dynamicLimiter = rateLimitMiddleware({
      windowMs,
      max: maxRequests,
      keyGenerator: () => `user:${decoded.walletAddress}`
    });
    
    dynamicLimiter(req, res, next);
  } catch (error) {
    // Fall back to standard rate limit if token is invalid
    return apiRateLimit(req, res, next);
  }
};
