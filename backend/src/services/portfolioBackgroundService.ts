import { Logger } from '../utils/logger';
import { portfolioCalculationService } from './portfolioCalculationService';
import { WebSocketService } from './websocket';

const logger = new Logger();

export interface PortfolioUpdateStats {
  totalUpdates: number;
  successfulUpdates: number;
  failedUpdates: number;
  averageUpdateTime: number;
  lastUpdateTime: number;
  activeSubscribers: number;
}

export class PortfolioBackgroundService {
  private static instance: PortfolioBackgroundService;
  private isRunning: boolean = false;
  private updateInterval: NodeJS.Timeout | null = null;
  private stats: PortfolioUpdateStats = {
    totalUpdates: 0,
    successfulUpdates: 0,
    failedUpdates: 0,
    averageUpdateTime: 0,
    lastUpdateTime: 0,
    activeSubscribers: 0
  };
  private updateTimes: number[] = [];
  private readonly MAX_UPDATE_TIMES = 100; // Keep last 100 update times for average calculation

  private constructor() {}

  public static getInstance(): PortfolioBackgroundService {
    if (!PortfolioBackgroundService.instance) {
      PortfolioBackgroundService.instance = new PortfolioBackgroundService();
    }
    return PortfolioBackgroundService.instance;
  }

  /**
   * Start the background portfolio update service
   */
  public start(): void {
    if (this.isRunning) {
      logger.warn('Portfolio background service is already running');
      return;
    }

    this.isRunning = true;
    logger.info('🚀 Starting portfolio background update service');

    // Start the update interval
    this.updateInterval = setInterval(async () => {
      await this.performPortfolioUpdates();
    }, 5000); // Every 5 seconds

    logger.info('✅ Portfolio background service started (5-second intervals)');
  }

  /**
   * Stop the background portfolio update service
   */
  public stop(): void {
    if (!this.isRunning) {
      logger.warn('Portfolio background service is not running');
      return;
    }

    this.isRunning = false;
    
    if (this.updateInterval) {
      clearInterval(this.updateInterval);
      this.updateInterval = null;
    }

    logger.info('🛑 Portfolio background service stopped');
  }

  /**
   * Perform portfolio updates for all active subscribers
   */
  private async performPortfolioUpdates(): Promise<void> {
    const startTime = Date.now();
    
    try {
      // Get WebSocket service instance
      const wsService = WebSocketService.current;
      if (!wsService) {
        logger.warn('WebSocket service not available for portfolio updates');
        return;
      }

      // Get all users with portfolio subscriptions
      const portfolioSubscribers = Array.from(wsService['portfolioSubscriptions'].keys());
      this.stats.activeSubscribers = portfolioSubscribers.length;

      if (portfolioSubscribers.length === 0) {
        logger.debug('📊 No active portfolio subscribers, skipping update cycle');
        return;
      }

      logger.debug(`📊 Starting portfolio update cycle for ${portfolioSubscribers.length} users`);

      // Process updates in batches to avoid overwhelming the system
      const batchSize = 10;
      const batches = this.chunkArray(portfolioSubscribers, batchSize);

      for (const batch of batches) {
        await this.processBatch(batch);
        
        // Small delay between batches to prevent system overload
        if (batches.length > 1) {
          await this.sleep(100); // 100ms delay between batches
        }
      }

      // Update statistics
      const updateTime = Date.now() - startTime;
      this.updateStats(updateTime, true);

      logger.debug(`📊 Portfolio update cycle completed in ${updateTime}ms for ${portfolioSubscribers.length} users`);

    } catch (error) {
      const updateTime = Date.now() - startTime;
      this.updateStats(updateTime, false);
      logger.error('Error in portfolio update cycle:', error);
    }
  }

  /**
   * Process a batch of user portfolio updates
   */
  private async processBatch(userIds: string[]): Promise<void> {
    const promises = userIds.map(async (userId) => {
      try {
        // Check if user still has active subscriptions
        const wsService = WebSocketService.current;
        if (!wsService || wsService.getPortfolioSubscribersCount(userId) === 0) {
          return; // Skip if no active subscribers
        }

        // Send portfolio update
        await wsService.sendPortfolioUpdate(userId);
        
      } catch (error) {
        logger.error(`Error updating portfolio for user ${userId}:`, error);
        this.stats.failedUpdates++;
      }
    });

    await Promise.allSettled(promises);
  }

  /**
   * Update statistics
   */
  private updateStats(updateTime: number, success: boolean): void {
    this.stats.totalUpdates++;
    this.stats.lastUpdateTime = Date.now();
    
    if (success) {
      this.stats.successfulUpdates++;
    } else {
      this.stats.failedUpdates++;
    }

    // Track update times for average calculation
    this.updateTimes.push(updateTime);
    if (this.updateTimes.length > this.MAX_UPDATE_TIMES) {
      this.updateTimes.shift(); // Remove oldest time
    }

    // Calculate average update time
    this.stats.averageUpdateTime = this.updateTimes.reduce((sum, time) => sum + time, 0) / this.updateTimes.length;
  }

  /**
   * Split array into chunks
   */
  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }

  /**
   * Sleep utility
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  /**
   * Get current statistics
   */
  public getStats(): PortfolioUpdateStats {
    return { ...this.stats };
  }

  /**
   * Get service status
   */
  public isServiceRunning(): boolean {
    return this.isRunning;
  }

  /**
   * Force update for a specific user (bypasses normal cycle)
   */
  public async forceUpdateForUser(userId: string): Promise<void> {
    try {
      const wsService = WebSocketService.current;
      if (!wsService) {
        throw new Error('WebSocket service not available');
      }

      if (wsService.getPortfolioSubscribersCount(userId) === 0) {
        logger.warn(`No active portfolio subscribers for user ${userId}`);
        return;
      }

      await wsService.sendPortfolioUpdate(userId);
      logger.debug(`📊 Forced portfolio update for user ${userId}`);
      
    } catch (error) {
      logger.error(`Error forcing portfolio update for user ${userId}:`, error);
      throw error;
    }
  }

  /**
   * Get health status
   */
  public getHealthStatus(): {
    isRunning: boolean;
    stats: PortfolioUpdateStats;
    healthScore: number;
  } {
    const healthScore = this.calculateHealthScore();
    
    return {
      isRunning: this.isRunning,
      stats: this.getStats(),
      healthScore
    };
  }

  /**
   * Calculate health score based on success rate and performance
   */
  private calculateHealthScore(): number {
    if (this.stats.totalUpdates === 0) {
      return 100; // No updates yet, consider healthy
    }

    const successRate = this.stats.successfulUpdates / this.stats.totalUpdates;
    const avgTimeScore = Math.max(0, 100 - (this.stats.averageUpdateTime / 100)); // Penalize slow updates
    
    return Math.round((successRate * 70) + (avgTimeScore * 0.3)); // Weight success rate more heavily
  }
}

// Export singleton instance
export const portfolioBackgroundService = PortfolioBackgroundService.getInstance();
