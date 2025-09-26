import { advancedOrderService } from './advancedOrderService';
import { Logger } from '../utils/logger';
import axios from 'axios';

const logger = new Logger();

export class OrderScheduler {
  private static instance: OrderScheduler;
  private twapInterval: NodeJS.Timeout | null = null;
  private conditionalInterval: NodeJS.Timeout | null = null;
  private isRunning: boolean = false;

  private constructor() {}

  public static getInstance(): OrderScheduler {
    if (!OrderScheduler.instance) {
      OrderScheduler.instance = new OrderScheduler();
    }
    return OrderScheduler.instance;
  }

  /**
   * Start the order scheduler
   */
  public start(): void {
    if (this.isRunning) {
      logger.warn('Order scheduler is already running');
      return;
    }

    this.isRunning = true;
    logger.info('🚀 Starting order scheduler...');

    // Execute TWAP orders every 30 seconds
    this.twapInterval = setInterval(async () => {
      try {
        await this.executeTWAPOrders();
      } catch (error) {
        logger.error('❌ Error executing TWAP orders:', error);
      }
    }, 30000);

    // Execute conditional orders every 5 seconds
    this.conditionalInterval = setInterval(async () => {
      try {
        await this.executeConditionalOrders();
      } catch (error) {
        logger.error('❌ Error executing conditional orders:', error);
      }
    }, 5000);

    logger.info('✅ Order scheduler started successfully');
  }

  /**
   * Stop the order scheduler
   */
  public stop(): void {
    if (!this.isRunning) {
      logger.warn('Order scheduler is not running');
      return;
    }

    this.isRunning = false;

    if (this.twapInterval) {
      clearInterval(this.twapInterval);
      this.twapInterval = null;
    }

    if (this.conditionalInterval) {
      clearInterval(this.conditionalInterval);
      this.conditionalInterval = null;
    }

    logger.info('🛑 Order scheduler stopped');
  }

  /**
   * Execute TWAP orders
   */
  private async executeTWAPOrders(): Promise<void> {
    try {
      const results = await advancedOrderService.executeTWAPOrders();
      
      if (results.length > 0) {
        logger.info(`⏰ Executed ${results.length} TWAP orders`);
        
        // Log successful executions
        const successful = results.filter(r => r.success);
        if (successful.length > 0) {
          logger.info(`✅ ${successful.length} TWAP orders executed successfully`);
        }
      }

    } catch (error) {
      logger.error('❌ Error in TWAP execution:', error);
    }
  }

  /**
   * Execute conditional orders for all active markets
   */
  private async executeConditionalOrders(): Promise<void> {
    try {
      // Get all active markets
      const markets = await this.getActiveMarkets();
      
      for (const market of markets) {
        try {
          // Get current price for the market
          const currentPrice = await this.getCurrentMarketPrice(market.id);
          
          if (currentPrice > 0) {
            // Execute conditional orders for this market
            const results = await advancedOrderService.executeConditionalOrders(market.id, currentPrice);
            
            if (results.length > 0) {
              logger.info(`🎯 Executed ${results.length} conditional orders for ${market.symbol}`);
              
              // Log successful executions
              const successful = results.filter(r => r.success);
              if (successful.length > 0) {
                logger.info(`✅ ${successful.length} conditional orders executed successfully for ${market.symbol}`);
              }
            }
          }
        } catch (error) {
          logger.error(`❌ Error executing conditional orders for market ${market.symbol}:`, error);
        }
      }

    } catch (error) {
      logger.error('❌ Error in conditional order execution:', error);
    }
  }

  /**
   * Get active markets
   */
  private async getActiveMarkets(): Promise<any[]> {
    try {
      const response = await axios.get('http://localhost:3002/api/supabase-oracle/markets');
      if (response.data.success) {
        return response.data.data;
      }
      return [];
    } catch (error) {
      logger.error('❌ Error fetching active markets:', error);
      return [];
    }
  }

  /**
   * Get current market price
   */
  private async getCurrentMarketPrice(marketId: string): Promise<number> {
    try {
      const response = await axios.get(`http://localhost:3002/api/supabase-oracle/prices`);
      if (response.data.success) {
        const prices = response.data.data;
        const marketPrice = prices.find((p: any) => p.market_id === marketId);
        return marketPrice ? marketPrice.price : 0;
      }
      return 0;
    } catch (error) {
      logger.error('❌ Error fetching market price:', error);
      return 0;
    }
  }

  /**
   * Get scheduler status
   */
  public getStatus(): { isRunning: boolean; intervals: { twap: boolean; conditional: boolean } } {
    return {
      isRunning: this.isRunning,
      intervals: {
        twap: this.twapInterval !== null,
        conditional: this.conditionalInterval !== null
      }
    };
  }

  /**
   * Manually trigger TWAP execution
   */
  public async triggerTWAPExecution(): Promise<void> {
    logger.info('🔄 Manually triggering TWAP execution...');
    await this.executeTWAPOrders();
  }

  /**
   * Manually trigger conditional order execution
   */
  public async triggerConditionalExecution(): Promise<void> {
    logger.info('🔄 Manually triggering conditional order execution...');
    await this.executeConditionalOrders();
  }
}

export const orderScheduler = OrderScheduler.getInstance();
