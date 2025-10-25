// QuantDesk Market Maker Bot
// Professional market making implementation for QuantDesk perpetual DEX

import { QuantDeskClient, MarketData, Order, Position } from '@quantdesk/sdk';

export interface MarketMakerConfig {
  client: QuantDeskClient;
  market: string;
  spread: number;        // Spread percentage (e.g., 0.001 = 0.1%)
  size: number;          // Order size in base asset
  maxPositions: number;  // Maximum number of positions
  updateInterval: number; // Update interval in milliseconds
  minSpread: number;     // Minimum spread threshold
  maxSpread: number;     // Maximum spread threshold
}

export class MarketMakerBot {
  private client: QuantDeskClient;
  private config: MarketMakerConfig;
  private isRunning: boolean = false;
  private orders: Map<string, Order> = new Map();
  private positions: Position[] = [];
  private marketData: MarketData | null = null;

  constructor(config: MarketMakerConfig) {
    this.client = config.client;
    this.config = config;
  }

  /**
   * Start the market maker bot
   */
  async start(): Promise<void> {
    console.log(`Starting Market Maker Bot for ${this.config.market}`);
    this.isRunning = true;

    // Initialize bot
    await this.initialize();

    // Start main loop
    this.runMainLoop();
  }

  /**
   * Stop the market maker bot
   */
  async stop(): Promise<void> {
    console.log('Stopping Market Maker Bot...');
    this.isRunning = false;

    // Cancel all open orders
    await this.cancelAllOrders();
  }

  /**
   * Initialize the bot
   */
  private async initialize(): Promise<void> {
    try {
      // Get initial market data
      this.marketData = await this.client.getMarketData(this.config.market);
      
      // Get current positions
      this.positions = await this.client.getPositions();
      
      // Get existing orders
      const orders = await this.client.getOrders();
      orders.forEach(order => {
        if (order.market === this.config.market) {
          this.orders.set(order.id, order);
        }
      });

      console.log(`Initialized Market Maker Bot`);
      console.log(`Market: ${this.config.market}`);
      console.log(`Spread: ${this.config.spread * 100}%`);
      console.log(`Size: ${this.config.size}`);
      console.log(`Current Positions: ${this.positions.length}`);
      console.log(`Open Orders: ${this.orders.size}`);
    } catch (error) {
      console.error('Failed to initialize Market Maker Bot:', error);
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
        await this.updateOrders();
        await this.managePositions();
        
        // Wait for next update
        await this.sleep(this.config.updateInterval);
      } catch (error) {
        console.error('Error in main loop:', error);
        await this.sleep(5000); // Wait 5 seconds before retry
      }
    }
  }

  /**
   * Update market data
   */
  private async updateMarketData(): Promise<void> {
    try {
      this.marketData = await this.client.getMarketData(this.config.market);
    } catch (error) {
      console.error('Failed to update market data:', error);
    }
  }

  /**
   * Update orders based on market conditions
   */
  private async updateOrders(): Promise<void> {
    if (!this.marketData) return;

    const currentPrice = this.marketData.price;
    const spread = this.calculateSpread();
    
    if (spread < this.config.minSpread || spread > this.config.maxSpread) {
      console.log(`Spread ${spread * 100}% outside limits, skipping order update`);
      return;
    }

    // Calculate bid and ask prices
    const bidPrice = currentPrice * (1 - spread / 2);
    const askPrice = currentPrice * (1 + spread / 2);

    // Update bid order
    await this.updateBidOrder(bidPrice);
    
    // Update ask order
    await this.updateAskOrder(askPrice);
  }

  /**
   * Update bid order
   */
  private async updateBidOrder(price: number): Promise<void> {
    try {
      const existingBid = Array.from(this.orders.values())
        .find(order => order.side === 'buy' && order.market === this.config.market);

      if (existingBid) {
        // Cancel existing bid if price changed significantly
        if (Math.abs(existingBid.price - price) / price > 0.001) {
          await this.client.cancelOrder(existingBid.id);
          this.orders.delete(existingBid.id);
        } else {
          return; // Keep existing order
        }
      }

      // Place new bid order
      const order = await this.client.placeOrder({
        market: this.config.market,
        side: 'buy',
        size: this.config.size,
        price: price,
        orderType: 'limit'
      });

      this.orders.set(order.id, order);
      console.log(`Placed bid order: ${this.config.size} @ ${price}`);
    } catch (error) {
      console.error('Failed to update bid order:', error);
    }
  }

  /**
   * Update ask order
   */
  private async updateAskOrder(price: number): Promise<void> {
    try {
      const existingAsk = Array.from(this.orders.values())
        .find(order => order.side === 'sell' && order.market === this.config.market);

      if (existingAsk) {
        // Cancel existing ask if price changed significantly
        if (Math.abs(existingAsk.price - price) / price > 0.001) {
          await this.client.cancelOrder(existingAsk.id);
          this.orders.delete(existingAsk.id);
        } else {
          return; // Keep existing order
        }
      }

      // Place new ask order
      const order = await this.client.placeOrder({
        market: this.config.market,
        side: 'sell',
        size: this.config.size,
        price: price,
        orderType: 'limit'
      });

      this.orders.set(order.id, order);
      console.log(`Placed ask order: ${this.config.size} @ ${price}`);
    } catch (error) {
      console.error('Failed to update ask order:', error);
    }
  }

  /**
   * Manage positions
   */
  private async managePositions(): Promise<void> {
    try {
      // Update positions
      this.positions = await this.client.getPositions();
      
      // Check if we have too many positions
      if (this.positions.length >= this.config.maxPositions) {
        console.log(`Maximum positions reached (${this.config.maxPositions}), managing risk`);
        await this.manageRisk();
      }
    } catch (error) {
      console.error('Failed to manage positions:', error);
    }
  }

  /**
   * Manage risk by closing positions if needed
   */
  private async manageRisk(): Promise<void> {
    try {
      // Get portfolio PnL
      const pnl = await this.client.calculatePnL();
      
      if (pnl.totalPnL < -1000) { // Close positions if losing more than $1000
        console.log('Risk management: Closing positions due to losses');
        
        for (const position of this.positions) {
          if (position.market === this.config.market) {
            await this.client.closePosition(position.id);
            console.log(`Closed position: ${position.id}`);
          }
        }
      }
    } catch (error) {
      console.error('Failed to manage risk:', error);
    }
  }

  /**
   * Calculate dynamic spread based on market conditions
   */
  private calculateSpread(): number {
    if (!this.marketData) return this.config.spread;

    // Increase spread during high volatility
    const volatility = this.marketData.volatility || 0;
    const volatilityMultiplier = Math.min(1 + volatility * 2, 2); // Max 2x spread

    return this.config.spread * volatilityMultiplier;
  }

  /**
   * Cancel all open orders
   */
  private async cancelAllOrders(): Promise<void> {
    try {
      for (const order of this.orders.values()) {
        await this.client.cancelOrder(order.id);
        console.log(`Cancelled order: ${order.id}`);
      }
      this.orders.clear();
    } catch (error) {
      console.error('Failed to cancel orders:', error);
    }
  }

  /**
   * Get bot status
   */
  getStatus(): {
    isRunning: boolean;
    market: string;
    orders: number;
    positions: number;
    marketData: MarketData | null;
  } {
    return {
      isRunning: this.isRunning,
      market: this.config.market,
      orders: this.orders.size,
      positions: this.positions.length,
      marketData: this.marketData
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
export async function runMarketMakerBot() {
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com',
    programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    wallet: wallet // Your Solana wallet
  });

  const bot = new MarketMakerBot({
    client: client,
    market: 'SOL-PERP',
    spread: 0.001,        // 0.1% spread
    size: 0.1,           // 0.1 SOL per order
    maxPositions: 5,
    updateInterval: 5000, // 5 seconds
    minSpread: 0.0005,   // 0.05% minimum spread
    maxSpread: 0.002     // 0.2% maximum spread
  });

  try {
    await bot.start();
    
    // Run for 1 hour
    setTimeout(async () => {
      await bot.stop();
      console.log('Market Maker Bot stopped');
    }, 3600000);
    
  } catch (error) {
    console.error('Market Maker Bot error:', error);
  }
}
