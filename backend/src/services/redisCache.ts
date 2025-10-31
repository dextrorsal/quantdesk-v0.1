import { getRedis, pingRedis } from './redisClient';
import { Logger } from '../utils/logger';

const logger = new Logger();

/**
 * Redis Cache Service
 * 
 * Provides caching layer with configurable TTLs per architecture:
 * - Prices: 1s TTL
 * - Portfolio: 5s TTL
 * - Order status: 2s TTL
 */
export class RedisCacheService {
  private static instance: RedisCacheService;
  private cacheStats = {
    hits: 0,
    misses: 0,
    errors: 0,
    sets: 0
  };

  // TTLs per architecture spec
  private readonly TTL_PRICES = 1; // 1 second
  private readonly TTL_PORTFOLIO = 5; // 5 seconds
  private readonly TTL_ORDER_STATUS = 2; // 2 seconds

  private constructor() {}

  public static getInstance(): RedisCacheService {
    if (!RedisCacheService.instance) {
      RedisCacheService.instance = new RedisCacheService();
    }
    return RedisCacheService.instance;
  }

  /**
   * Get value from cache
   */
  public async get<T>(key: string): Promise<T | null> {
    try {
      const redis = getRedis();
      if (!redis) {
        this.cacheStats.misses++;
        return null;
      }

      const value = await redis.get(key);
      if (value) {
        this.cacheStats.hits++;
        return JSON.parse(value) as T;
      }

      this.cacheStats.misses++;
      return null;
    } catch (error) {
      this.cacheStats.errors++;
      logger.error(`Redis cache get error for key ${key}:`, error);
      return null;
    }
  }

  /**
   * Set value in cache with TTL
   */
  public async set(key: string, value: any, ttlSeconds: number): Promise<boolean> {
    try {
      const redis = getRedis();
      if (!redis) {
        return false;
      }

      await redis.set(key, JSON.stringify(value), { EX: ttlSeconds });
      this.cacheStats.sets++;
      return true;
    } catch (error) {
      this.cacheStats.errors++;
      logger.error(`Redis cache set error for key ${key}:`, error);
      return false;
    }
  }

  /**
   * Cache price data (1s TTL)
   */
  public async cachePrice(symbol: string, priceData: any): Promise<boolean> {
    const key = `price:${symbol}`;
    return this.set(key, priceData, this.TTL_PRICES);
  }

  /**
   * Get cached price data
   */
  public async getCachedPrice(symbol: string): Promise<any | null> {
    const key = `price:${symbol}`;
    return this.get(key);
  }

  /**
   * Cache portfolio data (5s TTL)
   */
  public async cachePortfolio(userId: string, portfolioData: any): Promise<boolean> {
    const key = `portfolio:${userId}`;
    return this.set(key, portfolioData, this.TTL_PORTFOLIO);
  }

  /**
   * Get cached portfolio data
   */
  public async getCachedPortfolio(userId: string): Promise<any | null> {
    const key = `portfolio:${userId}`;
    return this.get(key);
  }

  /**
   * Cache order status (2s TTL)
   */
  public async cacheOrderStatus(orderId: string, orderData: any): Promise<boolean> {
    const key = `order:${orderId}`;
    return this.set(key, orderData, this.TTL_ORDER_STATUS);
  }

  /**
   * Get cached order status
   */
  public async getCachedOrderStatus(orderId: string): Promise<any | null> {
    const key = `order:${orderId}`;
    return this.get(key);
  }

  /**
   * Delete cache key
   */
  public async delete(key: string): Promise<boolean> {
    try {
      const redis = getRedis();
      if (!redis) {
        return false;
      }

      await redis.del(key);
      return true;
    } catch (error) {
      this.cacheStats.errors++;
      logger.error(`Redis cache delete error for key ${key}:`, error);
      return false;
    }
  }

  /**
   * Clear all cache (use with caution)
   */
  public async clear(pattern: string = '*'): Promise<number> {
    try {
      const redis = getRedis();
      if (!redis) {
        return 0;
      }

      // Environment-aware key pattern
      const envName = process.env.ENV_NAME || process.env.NODE_ENV || 'dev';
      const fullPattern = `qd:${envName}:${pattern}`;
      const keys = await redis.keys(fullPattern);
      
      if (keys.length === 0) {
        return 0;
      }

      await redis.del(keys);
      return keys.length;
    } catch (error) {
      this.cacheStats.errors++;
      logger.error(`Redis cache clear error for pattern ${pattern}:`, error);
      return 0;
    }
  }

  /**
   * Get cache statistics
   */
  public getStats() {
    const total = this.cacheStats.hits + this.cacheStats.misses;
    const hitRatio = total > 0 ? (this.cacheStats.hits / total) * 100 : 0;

    return {
      hits: this.cacheStats.hits,
      misses: this.cacheStats.misses,
      errors: this.cacheStats.errors,
      sets: this.cacheStats.sets,
      hitRatio: hitRatio.toFixed(2),
      totalRequests: total
    };
  }

  /**
   * Reset statistics
   */
  public resetStats(): void {
    this.cacheStats = {
      hits: 0,
      misses: 0,
      errors: 0,
      sets: 0
    };
  }

  /**
   * Check if Redis is available
   */
  public async isAvailable(): Promise<boolean> {
    const pingResult = await pingRedis();
    return pingResult.ok;
  }

  /**
   * Get Redis health status with latency
   */
  public async getHealthStatus(): Promise<{
    status: 'healthy' | 'unhealthy' | 'disabled';
    latency: number;
    available: boolean;
    stats: any;
  }> {
    const startTime = Date.now();
    const pingResult = await pingRedis();
    const latency = Date.now() - startTime;

    if (!pingResult.ok) {
      return {
        status: 'unhealthy',
        latency,
        available: false,
        stats: this.getStats()
      };
    }

    const redis = getRedis();
    const available = !!redis && (redis as any).isOpen;

    return {
      status: available ? 'healthy' : 'unhealthy',
      latency,
      available,
      stats: this.getStats()
    };
  }
}

export const redisCacheService = RedisCacheService.getInstance();

