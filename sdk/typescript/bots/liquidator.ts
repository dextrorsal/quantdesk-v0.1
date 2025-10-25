// QuantDesk Liquidation Bot
// Professional liquidation implementation for QuantDesk perpetual DEX

import { QuantDeskClient, Position, MarketData } from '@quantdesk/sdk';

export interface LiquidationConfig {
  client: QuantDeskClient;
  markets: string[];           // Markets to monitor
  liquidationThreshold: number; // Liquidation threshold (e.g., 0.8 = 80%)
  maxGasPrice: number;         // Maximum gas price in SOL
  updateInterval: number;      // Update interval in milliseconds
  minLiquidationSize: number;  // Minimum position size to liquidate
  maxLiquidationSize: number;  // Maximum position size to liquidate
}

export class LiquidationBot {
  private client: QuantDeskClient;
  private config: LiquidationConfig;
  private isRunning: boolean = false;
  private positions: Position[] = [];
  private marketData: Map<string, MarketData> = new Map();

  constructor(config: LiquidationConfig) {
    this.client = config.client;
    this.config = config;
  }

  /**
   * Start the liquidation bot
   */
  async start(): Promise<void> {
    console.log(`Starting Liquidation Bot for markets: ${this.config.markets.join(', ')}`);
    this.isRunning = true;

    // Initialize bot
    await this.initialize();

    // Start main loop
    this.runMainLoop();
  }

  /**
   * Stop the liquidation bot
   */
  async stop(): Promise<void> {
    console.log('Stopping Liquidation Bot...');
    this.isRunning = false;
  }

  /**
   * Initialize the bot
   */
  private async initialize(): Promise<void> {
    try {
      // Get initial market data for all markets
      for (const market of this.config.markets) {
        const data = await this.client.getMarketData(market);
        this.marketData.set(market, data);
      }

      console.log(`Initialized Liquidation Bot`);
      console.log(`Markets: ${this.config.markets.join(', ')}`);
      console.log(`Liquidation Threshold: ${this.config.liquidationThreshold * 100}%`);
      console.log(`Max Gas Price: ${this.config.maxGasPrice} SOL`);
    } catch (error) {
      console.error('Failed to initialize Liquidation Bot:', error);
      throw error;
    }
  }

  /**
   * Main bot loop
   */
  private async runMainLoop(): Promise<void> {
    while (this.isRunning) {
      try {
        await this.updateMarketData();
        await this.scanForLiquidations();
        
        // Wait for next update
        await this.sleep(this.config.updateInterval);
      } catch (error) {
        console.error('Error in main loop:', error);
        await this.sleep(5000); // Wait 5 seconds before retry
      }
    }
  }

  /**
   * Update market data for all monitored markets
   */
  private async updateMarketData(): Promise<void> {
    try {
      for (const market of this.config.markets) {
        const data = await this.client.getMarketData(market);
        this.marketData.set(market, data);
      }
    } catch (error) {
      console.error('Failed to update market data:', error);
    }
  }

  /**
   * Scan for positions that need liquidation
   */
  private async scanForLiquidations(): Promise<void> {
    try {
      // Get all positions
      this.positions = await this.client.getPositions();
      
      // Filter positions in monitored markets
      const monitoredPositions = this.positions.filter(position => 
        this.config.markets.includes(position.market)
      );

      // Check each position for liquidation
      for (const position of monitoredPositions) {
        await this.checkPositionForLiquidation(position);
      }
    } catch (error) {
      console.error('Failed to scan for liquidations:', error);
    }
  }

  /**
   * Check if a position needs liquidation
   */
  private async checkPositionForLiquidation(position: Position): Promise<void> {
    try {
      const marketData = this.marketData.get(position.market);
      if (!marketData) {
        console.warn(`No market data for ${position.market}`);
        return;
      }

      // Calculate position health
      const health = await this.calculatePositionHealth(position, marketData);
      
      if (health <= this.config.liquidationThreshold) {
        console.log(`Position ${position.id} needs liquidation (health: ${health * 100}%)`);
        
        // Check if position meets size requirements
        if (this.isPositionLiquidatable(position)) {
          await this.liquidatePosition(position);
        }
      }
    } catch (error) {
      console.error(`Failed to check position ${position.id}:`, error);
    }
  }

  /**
   * Calculate position health (0-1, where 0 = fully liquidated)
   */
  private async calculatePositionHealth(position: Position, marketData: MarketData): Promise<number> {
    try {
      // Get current position value
      const currentValue = position.size * marketData.price;
      
      // Get collateral value
      const collateralValue = position.collateral;
      
      // Calculate health ratio
      const health = collateralValue / currentValue;
      
      return Math.max(0, Math.min(1, health));
    } catch (error) {
      console.error('Failed to calculate position health:', error);
      return 1; // Assume healthy if calculation fails
    }
  }

  /**
   * Check if position meets liquidation size requirements
   */
  private isPositionLiquidatable(position: Position): boolean {
    const size = Math.abs(position.size);
    return size >= this.config.minLiquidationSize && 
           size <= this.config.maxLiquidationSize;
  }

  /**
   * Liquidate a position
   */
  private async liquidatePosition(position: Position): Promise<void> {
    try {
      // Check gas price before liquidation
      const gasPrice = await this.getCurrentGasPrice();
      if (gasPrice > this.config.maxGasPrice) {
        console.log(`Gas price too high (${gasPrice} SOL), skipping liquidation`);
        return;
      }

      console.log(`Liquidating position: ${position.id}`);
      
      // Execute liquidation
      const liquidationResult = await this.client.liquidatePosition(position.id);
      
      if (liquidationResult.success) {
        console.log(`Successfully liquidated position: ${position.id}`);
        console.log(`Liquidation fee: ${liquidationResult.fee} SOL`);
        
        // Log liquidation details
        this.logLiquidationDetails(position, liquidationResult);
      } else {
        console.error(`Failed to liquidate position: ${position.id}`);
      }
    } catch (error) {
      console.error(`Failed to liquidate position ${position.id}:`, error);
    }
  }

  /**
   * Get current gas price
   */
  private async getCurrentGasPrice(): Promise<number> {
    try {
      // This would typically come from the Solana network
      // For now, return a mock value
      return 0.000005; // 0.000005 SOL per transaction
    } catch (error) {
      console.error('Failed to get gas price:', error);
      return 0.001; // Return high value to be safe
    }
  }

  /**
   * Log liquidation details
   */
  private logLiquidationDetails(position: Position, result: any): void {
    console.log('=== Liquidation Details ===');
    console.log(`Position ID: ${position.id}`);
    console.log(`Market: ${position.market}`);
    console.log(`Size: ${position.size}`);
    console.log(`Side: ${position.side}`);
    console.log(`Collateral: ${position.collateral}`);
    console.log(`Liquidation Fee: ${result.fee} SOL`);
    console.log(`Gas Used: ${result.gasUsed}`);
    console.log(`Transaction ID: ${result.txId}`);
    console.log('========================');
  }

  /**
   * Get bot status
   */
  getStatus(): {
    isRunning: boolean;
    markets: string[];
    positions: number;
    liquidations: number;
  } {
    return {
      isRunning: this.isRunning,
      markets: this.config.markets,
      positions: this.positions.length,
      liquidations: 0 // Would track total liquidations
    };
  }

  /**
   * Get liquidation statistics
   */
  getLiquidationStats(): {
    totalLiquidations: number;
    totalFeesEarned: number;
    averageLiquidationSize: number;
    successRate: number;
  } {
    // This would typically track real statistics
    return {
      totalLiquidations: 0,
      totalFeesEarned: 0,
      averageLiquidationSize: 0,
      successRate: 0
    };
  }

  /**
   * Sleep utility function
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Example usage
export async function runLiquidationBot() {
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com',
    programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    wallet: wallet // Your Solana wallet
  });

  const bot = new LiquidationBot({
    client: client,
    markets: ['SOL-PERP', 'ETH-PERP', 'BTC-PERP'],
    liquidationThreshold: 0.8,  // 80% liquidation threshold
    maxGasPrice: 0.001,          // Max 0.001 SOL gas price
    updateInterval: 2000,        // 2 seconds
    minLiquidationSize: 0.01,    // Min 0.01 SOL position size
    maxLiquidationSize: 100      // Max 100 SOL position size
  });

  try {
    await bot.start();
    
    // Run for 24 hours
    setTimeout(async () => {
      await bot.stop();
      console.log('Liquidation Bot stopped');
    }, 86400000);
    
  } catch (error) {
    console.error('Liquidation Bot error:', error);
  }
}
