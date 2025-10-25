// QuantDesk Market Data Monitoring Example
// Realistic market data monitoring and analysis implementation

import { QuantDeskClient, MarketData } from '@quantdesk/sdk';

/**
 * Market Data Monitoring Example - Realistic Implementation
 * This example demonstrates practical market data monitoring that users can actually implement
 */
export class MarketDataMonitoringExample {
  private client: QuantDeskClient;
  private updateInterval: number = 10000; // 10 seconds
  private isRunning: boolean = false;
  private marketData: Map<string, MarketData> = new Map();
  private priceHistory: Map<string, number[]> = new Map();

  constructor(client: QuantDeskClient, updateInterval: number = 10000) {
    this.client = client;
    this.updateInterval = updateInterval;
  }

  /**
   * Start market data monitoring
   */
  async startMonitoring(markets: string[]): Promise<void> {
    console.log(`🚀 Starting Market Data Monitoring for: ${markets.join(', ')}`);
    this.isRunning = true;

    try {
      // Initialize price history
      for (const market of markets) {
        this.priceHistory.set(market, []);
      }
      
      // Start monitoring loop
      this.startMonitoringLoop(markets);
      
    } catch (error) {
      console.error('❌ Failed to start market data monitoring:', error);
      throw error;
    }
  }

  /**
   * Stop market data monitoring
   */
  async stopMonitoring(): Promise<void> {
    console.log('🛑 Stopping Market Data Monitoring...');
    this.isRunning = false;
  }

  /**
   * Start monitoring loop
   */
  private startMonitoringLoop(markets: string[]): void {
    console.log(`\n🔄 Starting monitoring loop (${this.updateInterval / 1000}s intervals)...`);
    
    const monitor = async () => {
      if (!this.isRunning) return;
      
      try {
        await this.updateMarketData(markets);
        this.displayMarketData(markets);
        this.analyzePriceMovements(markets);
        
        // Schedule next update
        setTimeout(monitor, this.updateInterval);
      } catch (error) {
        console.error('❌ Error in monitoring loop:', error);
        
        // Continue monitoring even if there's an error
        setTimeout(monitor, this.updateInterval);
      }
    };
    
    // Start the monitoring loop
    setTimeout(monitor, this.updateInterval);
  }

  /**
   * Update market data for all monitored markets
   */
  private async updateMarketData(markets: string[]): Promise<void> {
    try {
      for (const market of markets) {
        const data = await this.client.getMarketData(market);
        this.marketData.set(market, data);
        
        // Update price history (keep last 20 prices)
        const history = this.priceHistory.get(market) || [];
        history.push(data.price);
        if (history.length > 20) {
          history.shift(); // Remove oldest price
        }
        this.priceHistory.set(market, history);
      }
    } catch (error) {
      console.error('❌ Failed to update market data:', error);
    }
  }

  /**
   * Display current market data
   */
  private displayMarketData(markets: string[]): void {
    console.log('\n📊 Current Market Data:');
    console.log(`Time: ${new Date().toLocaleTimeString()}`);
    
    for (const market of markets) {
      const data = this.marketData.get(market);
      if (data) {
        console.log(`\n${market}:`);
        console.log(`  Price: $${data.price.toFixed(2)}`);
        console.log(`  Volume: ${data.volume.toFixed(2)}`);
        console.log(`  Change 24h: ${(data.change24h * 100).toFixed(2)}%`);
        console.log(`  High 24h: $${data.high24h.toFixed(2)}`);
        console.log(`  Low 24h: $${data.low24h.toFixed(2)}`);
      }
    }
  }

  /**
   * Analyze price movements
   */
  private analyzePriceMovements(markets: string[]): void {
    console.log('\n📈 Price Movement Analysis:');
    
    for (const market of markets) {
      const history = this.priceHistory.get(market);
      if (history && history.length >= 2) {
        const currentPrice = history[history.length - 1];
        const previousPrice = history[history.length - 2];
        const change = ((currentPrice - previousPrice) / previousPrice) * 100;
        
        console.log(`${market}: ${change > 0 ? '📈' : '📉'} ${change.toFixed(3)}%`);
        
        // Simple trend analysis
        if (history.length >= 5) {
          const trend = this.calculateTrend(history.slice(-5));
          console.log(`  Trend: ${trend}`);
        }
      }
    }
  }

  /**
   * Calculate simple trend from price history
   */
  private calculateTrend(prices: number[]): string {
    if (prices.length < 2) return 'Unknown';
    
    const first = prices[0];
    const last = prices[prices.length - 1];
    const change = ((last - first) / first) * 100;
    
    if (change > 1) return 'Strong Up';
    if (change > 0.1) return 'Up';
    if (change < -1) return 'Strong Down';
    if (change < -0.1) return 'Down';
    return 'Sideways';
  }

  /**
   * Get market alerts
   */
  async getMarketAlerts(markets: string[]): Promise<string[]> {
    const alerts: string[] = [];
    
    try {
      for (const market of markets) {
        const data = this.marketData.get(market);
        if (data) {
          // Check for significant price movements
          if (Math.abs(data.change24h) > 0.1) { // 10% change
            alerts.push(`📊 ${market}: ${(data.change24h * 100).toFixed(2)}% change in 24h`);
          }
          
          // Check for high volume
          if (data.volume > 1000000) { // 1M+ volume
            alerts.push(`📈 ${market}: High volume detected (${data.volume.toFixed(0)})`);
          }
          
          // Check for price near 24h high/low
          const priceRange = data.high24h - data.low24h;
          const pricePosition = (data.price - data.low24h) / priceRange;
          
          if (pricePosition > 0.9) {
            alerts.push(`🔺 ${market}: Price near 24h high ($${data.price.toFixed(2)})`);
          } else if (pricePosition < 0.1) {
            alerts.push(`🔻 ${market}: Price near 24h low ($${data.price.toFixed(2)})`);
          }
        }
      }
    } catch (error) {
      console.error('❌ Failed to get market alerts:', error);
    }
    
    return alerts;
  }

  /**
   * Get price history for a market
   */
  getPriceHistory(market: string): number[] {
    return this.priceHistory.get(market) || [];
  }

  /**
   * Calculate simple moving average
   */
  calculateMovingAverage(market: string, period: number = 5): number | null {
    const history = this.priceHistory.get(market);
    if (!history || history.length < period) return null;
    
    const recentPrices = history.slice(-period);
    const sum = recentPrices.reduce((acc, price) => acc + price, 0);
    return sum / period;
  }

  /**
   * Get market summary
   */
  getMarketSummary(markets: string[]): {
    totalMarkets: number;
    averageChange: number;
    topGainer: string | null;
    topLoser: string | null;
  } {
    let totalChange = 0;
    let topGainer: string | null = null;
    let topLoser: string | null = null;
    let maxGain = -Infinity;
    let maxLoss = Infinity;
    
    for (const market of markets) {
      const data = this.marketData.get(market);
      if (data) {
        totalChange += data.change24h;
        
        if (data.change24h > maxGain) {
          maxGain = data.change24h;
          topGainer = market;
        }
        
        if (data.change24h < maxLoss) {
          maxLoss = data.change24h;
          topLoser = market;
        }
      }
    }
    
    return {
      totalMarkets: markets.length,
      averageChange: totalChange / markets.length,
      topGainer,
      topLoser
    };
  }

  /**
   * Export market data
   */
  async exportMarketData(markets: string[]): Promise<{
    timestamp: string;
    markets: { [key: string]: MarketData };
    priceHistory: { [key: string]: number[] };
  }> {
    try {
      const marketData: { [key: string]: MarketData } = {};
      const priceHistory: { [key: string]: number[] } = {};
      
      for (const market of markets) {
        const data = this.marketData.get(market);
        if (data) {
          marketData[market] = data;
        }
        
        const history = this.priceHistory.get(market);
        if (history) {
          priceHistory[market] = history;
        }
      }
      
      return {
        timestamp: new Date().toISOString(),
        markets: marketData,
        priceHistory
      };
    } catch (error) {
      console.error('❌ Failed to export market data:', error);
      throw error;
    }
  }
}

// Example usage - Realistic implementation
export async function runMarketDataMonitoringExample() {
  // Initialize QuantDesk client
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com',
    programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    wallet: wallet // Your Solana wallet instance
  });

  // Create market data monitor
  const monitor = new MarketDataMonitoringExample(client, 10000); // 10 second updates
  
  // Monitor common markets
  const markets = ['SOL-PERP', 'ETH-PERP', 'BTC-PERP'];
  
  try {
    // Start monitoring
    await monitor.startMonitoring(markets);
    
    // Run for 2 minutes (realistic demo duration)
    setTimeout(async () => {
      await monitor.stopMonitoring();
      console.log('✅ Market data monitoring demo completed');
    }, 120000); // 2 minutes
    
  } catch (error) {
    console.error('Market data monitoring failed:', error);
  }
}

// Export for use in other modules
export { MarketDataMonitoringExample };
