import { Logger } from '../utils/logger';
import { SupabaseDatabaseService } from './supabaseDatabase';
import { pythOracleService } from './pythOracleService';
import { pnlCalculationService } from './pnlCalculationService';
import { getRedis } from './redisClient';

const logger = new Logger();

export interface PortfolioCalculationResult {
  userId: string;
  totalValue: number;
  totalUnrealizedPnl: number;
  totalRealizedPnl: number;
  marginRatio: number;
  healthFactor: number;
  totalCollateral: number;
  usedMargin: number;
  availableMargin: number;
  positions: Array<{
    id: string;
    symbol: string;
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnl: number;
    unrealizedPnlPercent: number;
    margin: number;
    leverage: number;
    side: 'long' | 'short';
  }>;
  timestamp: number;
}

export interface PortfolioCacheData {
  data: PortfolioCalculationResult;
  lastUpdated: number;
  ttl: number;
}

export class PortfolioCalculationService {
  private static instance: PortfolioCalculationService;
  private db: SupabaseDatabaseService;
  private cache: Map<string, PortfolioCacheData> = new Map();
  private readonly CACHE_TTL = 5000; // 5 seconds
  private readonly REDIS_TTL = 10; // 10 seconds for Redis

  private constructor() {
    this.db = SupabaseDatabaseService.getInstance();
  }

  public static getInstance(): PortfolioCalculationService {
    if (!PortfolioCalculationService.instance) {
      PortfolioCalculationService.instance = new PortfolioCalculationService();
    }
    return PortfolioCalculationService.instance;
  }

  /**
   * Calculate portfolio for a user with caching
   */
  public async calculatePortfolio(userId: string, forceRefresh: boolean = false): Promise<PortfolioCalculationResult | null> {
    try {
      // Check cache first (unless force refresh)
      if (!forceRefresh) {
        const cached = await this.getCachedPortfolio(userId);
        if (cached) {
          logger.debug(`📊 Portfolio cache hit for user ${userId}`);
          return cached;
        }
      }

      logger.debug(`📊 Calculating fresh portfolio for user ${userId}`);

      // Get user data
      const user = await this.db.getUserByWallet(userId);
      if (!user) {
        logger.warn(`User not found: ${userId}`);
        return null;
      }

      // Use optimized database function to get portfolio summary
      const portfolioSummary = await this.getOptimizedPortfolioSummary(userId);
      if (!portfolioSummary) {
        logger.warn(`Portfolio summary not found for user: ${userId}`);
        return null;
      }

      // Get detailed positions with current prices using optimized function
      const positions = await this.getOptimizedUserPositions(userId);

      if (positions.length === 0) {
        // Return empty portfolio for users with no positions
        const emptyPortfolio: PortfolioCalculationResult = {
          userId,
          totalValue: portfolioSummary.total_collateral,
          totalUnrealizedPnl: 0,
          totalRealizedPnl: 0,
          marginRatio: 100,
          healthFactor: 100,
          totalCollateral: portfolioSummary.total_collateral,
          usedMargin: 0,
          availableMargin: portfolioSummary.total_collateral,
          positions: [],
          timestamp: Date.now()
        };

        await this.cachePortfolio(userId, emptyPortfolio);
        return emptyPortfolio;
      }

      // Transform positions to PnL calculation format
      const positionPnlData = positions.map(position => ({
        id: position.position_id,
        symbol: position.symbol,
        side: position.side,
        size: position.size,
        entryPrice: position.entry_price,
        currentPrice: position.current_price,
        leverage: position.leverage,
        margin: position.margin
      }));

      // Calculate portfolio P&L using centralized service
      const portfolioPnl = pnlCalculationService.calculatePortfolioPnl(positionPnlData);

      // Calculate portfolio metrics
      const totalValue = portfolioSummary.total_collateral + portfolioPnl.totalUnrealizedPnl;
      const usedMargin = portfolioPnl.totalMargin;
      const availableMargin = Math.max(0, portfolioSummary.total_collateral - usedMargin);
      
      // Calculate margin ratio
      const marginRatio = portfolioSummary.total_collateral > 0 ? (totalValue / portfolioSummary.total_collateral) * 100 : 100;
      
      // Calculate health factor
      const healthFactor = portfolioSummary.total_collateral > 0 
        ? Math.max(0, Math.min(100, ((portfolioSummary.total_collateral + portfolioPnl.totalUnrealizedPnl) / portfolioSummary.total_collateral) * 100)) 
        : 100;

      // Transform positions to include P&L data
      const positionsWithPnl = portfolioPnl.positions.map(pos => ({
        id: pos.id,
        symbol: pos.symbol,
        size: pos.size,
        entryPrice: pos.entryPrice,
        currentPrice: pos.currentPrice,
        unrealizedPnl: pos.unrealizedPnl,
        unrealizedPnlPercent: pos.unrealizedPnlPercent,
        margin: pos.margin,
        leverage: pos.leverage,
        side: pos.side
      }));

      const result: PortfolioCalculationResult = {
        userId,
        totalValue,
        totalUnrealizedPnl: portfolioPnl.totalUnrealizedPnl,
        totalRealizedPnl: 0, // Would calculate from closed positions
        marginRatio,
        healthFactor,
        totalCollateral: portfolioSummary.total_collateral,
        usedMargin,
        availableMargin,
        positions: positionsWithPnl,
        timestamp: Date.now()
      };

      // Cache the result
      await this.cachePortfolio(userId, result);

      logger.debug(`📊 Portfolio calculated for user ${userId}: $${totalValue.toFixed(2)} (${positions.length} positions)`);
      return result;

    } catch (error) {
      logger.error(`Error calculating portfolio for user ${userId}:`, error);
      return null;
    }
  }

  /**
   * Get optimized portfolio summary using materialized view
   */
  private async getOptimizedPortfolioSummary(userId: string): Promise<any> {
    try {
      // Use RPC call for stored procedure
      const { data: result, error } = await this.db.getClient()
        .rpc('get_user_portfolio_data', { user_id: userId });
      
      if (error) {
        logger.error(`Error calling get_user_portfolio_data for user ${userId}:`, error);
        return null;
      }
      
      return result?.[0] || null;
    } catch (error) {
      logger.error(`Error getting optimized portfolio summary for user ${userId}:`, error);
      return null;
    }
  }

  /**
   * Get optimized user positions with current prices
   */
  private async getOptimizedUserPositions(userId: string): Promise<any[]> {
    try {
      // Use RPC call for stored procedure
      const { data: result, error } = await this.db.getClient()
        .rpc('get_user_positions_with_prices', { user_id: userId });
      
      if (error) {
        logger.error(`Error calling get_user_positions_with_prices for user ${userId}:`, error);
        return [];
      }
      
      return result || [];
    } catch (error) {
      logger.error(`Error getting optimized user positions for user ${userId}:`, error);
      return [];
    }
  }

  /**
   * Get cached portfolio data
   */
  private async getCachedPortfolio(userId: string): Promise<PortfolioCalculationResult | null> {
    try {
      // Check Redis cache first
      const redisKey = `portfolio:${userId}`;
      const redis = getRedis();
      const redisData = redis ? await redis.get(redisKey) : null;
      
      if (redisData) {
        const parsed = JSON.parse(redisData) as PortfolioCacheData;
        if (this.isCacheValid(parsed)) {
          logger.debug(`📊 Redis cache hit for user ${userId}`);
          return parsed.data;
        }
      }

      // Check in-memory cache
      const cached = this.cache.get(userId);
      if (cached && this.isCacheValid(cached)) {
        logger.debug(`📊 Memory cache hit for user ${userId}`);
        return cached.data;
      }

      return null;
    } catch (error) {
      logger.error(`Error getting cached portfolio for user ${userId}:`, error);
      return null;
    }
  }

  /**
   * Cache portfolio data
   */
  private async cachePortfolio(userId: string, data: PortfolioCalculationResult): Promise<void> {
    try {
      const cacheData: PortfolioCacheData = {
        data,
        lastUpdated: Date.now(),
        ttl: this.CACHE_TTL
      };

      // Cache in memory
      this.cache.set(userId, cacheData);

      // Cache in Redis
      const redisKey = `portfolio:${userId}`;
      const redis = getRedis();
      if (redis) {
        await redis.setex(redisKey, this.REDIS_TTL, JSON.stringify(cacheData));
      }

      logger.debug(`📊 Portfolio cached for user ${userId}`);
    } catch (error) {
      logger.error(`Error caching portfolio for user ${userId}:`, error);
    }
  }

  /**
   * Check if cache is still valid
   */
  private isCacheValid(cached: PortfolioCacheData): boolean {
    return (Date.now() - cached.lastUpdated) < cached.ttl;
  }

  /**
   * Invalidate portfolio cache for a user
   */
  public async invalidatePortfolioCache(userId: string): Promise<void> {
    try {
      // Remove from memory cache
      this.cache.delete(userId);

      // Remove from Redis cache
      const redisKey = `portfolio:${userId}`;
      const redis = getRedis();
      if (redis) {
        await redis.del(redisKey);
      }

      logger.debug(`📊 Portfolio cache invalidated for user ${userId}`);
    } catch (error) {
      logger.error(`Error invalidating portfolio cache for user ${userId}:`, error);
    }
  }

  /**
   * Invalidate all portfolio caches (called when prices are updated)
   */
  public async invalidateAllPortfolioCaches(): Promise<void> {
    try {
      // Clear memory cache
      this.cache.clear();

      // Clear Redis cache (get all portfolio keys and delete them)
      const redis = getRedis();
      let clearedKeys = 0;
      if (redis) {
        const keys = await redis.keys('portfolio:*');
        if (keys.length > 0) {
          await redis.del(...keys);
          clearedKeys = keys.length;
        }
      }

      logger.debug(`📊 All portfolio caches invalidated (${clearedKeys} Redis keys cleared)`);
    } catch (error) {
      logger.error('Error invalidating all portfolio caches:', error);
    }
  }

  /**
   * Get portfolio summary for multiple users (for admin/analytics)
   */
  public async getPortfolioSummaries(userIds: string[]): Promise<Map<string, PortfolioCalculationResult>> {
    const results = new Map<string, PortfolioCalculationResult>();
    
    const promises = userIds.map(async (userId) => {
      try {
        const portfolio = await this.calculatePortfolio(userId);
        if (portfolio) {
          results.set(userId, portfolio);
        }
      } catch (error) {
        logger.error(`Error getting portfolio summary for user ${userId}:`, error);
      }
    });

    await Promise.all(promises);
    return results;
  }

  /**
   * Update positions with current market prices
   */
  public async updatePositionsWithCurrentPrices(positions: any[]): Promise<any[]> {
    try {
      if (!positions || positions.length === 0) {
        return [];
      }

      // Get current prices for all symbols
      const symbols = [...new Set(positions.map(p => p.symbol))];
      const prices = await pythOracleService.getAllPrices();
      
      // Update positions with current prices
      const updatedPositions = positions.map(position => {
        const currentPrice = prices[position.symbol];
        if (currentPrice && typeof currentPrice === 'number') {
          return {
            ...position,
            current_price: currentPrice,
            unrealized_pnl: this.calculateUnrealizedPnl(position, currentPrice),
            updated_at: new Date().toISOString()
          };
        }
        return position;
      });

      logger.debug(`📊 Updated ${updatedPositions.length} positions with current prices`);
      return updatedPositions;

    } catch (error) {
      logger.error('Error updating positions with current prices:', error);
      return positions; // Return original positions on error
    }
  }

  /**
   * Calculate unrealized P&L for a position
   */
  private calculateUnrealizedPnl(position: any, currentPrice: number): number {
    if (!position.entry_price || !position.size) {
      return 0;
    }

    const priceDiff = currentPrice - position.entry_price;
    const pnl = position.side === 'long' ? priceDiff : -priceDiff;
    return pnl * position.size;
  }

  /**
   * Get cache statistics
   */
  public getCacheStats(): { memoryCacheSize: number; memoryCacheKeys: string[] } {
    return {
      memoryCacheSize: this.cache.size,
      memoryCacheKeys: Array.from(this.cache.keys())
    };
  }
}

// Export singleton instance
export const portfolioCalculationService = PortfolioCalculationService.getInstance();
