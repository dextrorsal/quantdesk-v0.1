// QuantDesk Portfolio Tracking Example
// Realistic portfolio monitoring and management implementation

import { QuantDeskClient, Portfolio, Position, MarketData } from '@quantdesk/sdk';

/**
 * Portfolio Tracking Example - Realistic Implementation
 * This example demonstrates practical portfolio monitoring that users can actually implement
 */
export class PortfolioTrackingExample {
  private client: QuantDeskClient;
  private updateInterval: number = 30000; // 30 seconds
  private isRunning: boolean = false;

  constructor(client: QuantDeskClient, updateInterval: number = 30000) {
    this.client = client;
    this.updateInterval = updateInterval;
  }

  /**
   * Start portfolio tracking
   */
  async startTracking(): Promise<void> {
    console.log('🚀 Starting Portfolio Tracking...');
    this.isRunning = true;

    try {
      // Initial portfolio snapshot
      await this.takePortfolioSnapshot();
      
      // Start monitoring loop
      this.startMonitoringLoop();
      
    } catch (error) {
      console.error('❌ Failed to start portfolio tracking:', error);
      throw error;
    }
  }

  /**
   * Stop portfolio tracking
   */
  async stopTracking(): Promise<void> {
    console.log('🛑 Stopping Portfolio Tracking...');
    this.isRunning = false;
  }

  /**
   * Take a portfolio snapshot
   */
  private async takePortfolioSnapshot(): Promise<void> {
    try {
      console.log('\n📸 Taking Portfolio Snapshot...');
      
      // Get portfolio overview
      const portfolio = await this.client.getPortfolio();
      
      // Get all positions
      const positions = await this.client.getPositions();
      
      // Display portfolio summary
      this.displayPortfolioSummary(portfolio);
      
      // Display position details
      if (positions.length > 0) {
        this.displayPositionDetails(positions);
      } else {
        console.log('📭 No open positions');
      }
      
      // Calculate portfolio metrics
      await this.calculatePortfolioMetrics(portfolio, positions);
      
    } catch (error) {
      console.error('❌ Failed to take portfolio snapshot:', error);
    }
  }

  /**
   * Display portfolio summary
   */
  private displayPortfolioSummary(portfolio: Portfolio): void {
    console.log('\n💰 Portfolio Summary:');
    console.log(`  Total Value: $${portfolio.totalValue.toFixed(2)}`);
    console.log(`  Total PnL: $${portfolio.totalPnL.toFixed(2)}`);
    console.log(`  PnL %: ${((portfolio.totalPnL / portfolio.totalValue) * 100).toFixed(2)}%`);
    console.log(`  Available Balance: $${portfolio.availableBalance.toFixed(2)}`);
    console.log(`  Margin Used: $${portfolio.marginUsed.toFixed(2)}`);
    console.log(`  Margin Available: $${portfolio.marginAvailable.toFixed(2)}`);
  }

  /**
   * Display position details
   */
  private displayPositionDetails(positions: Position[]): void {
    console.log('\n📋 Position Details:');
    
    positions.forEach((position, index) => {
      const pnlPercent = (position.pnl / (position.size * position.entryPrice)) * 100;
      
      console.log(`\n  ${index + 1}. ${position.market}`);
      console.log(`     Side: ${position.side}`);
      console.log(`     Size: ${position.size.toFixed(4)}`);
      console.log(`     Entry Price: $${position.entryPrice.toFixed(2)}`);
      console.log(`     Current Price: $${position.currentPrice.toFixed(2)}`);
      console.log(`     PnL: $${position.pnl.toFixed(2)} (${pnlPercent.toFixed(2)}%)`);
      console.log(`     Leverage: ${position.leverage}x`);
      console.log(`     Margin: $${position.margin.toFixed(2)}`);
      console.log(`     Liquidation Price: $${position.liquidationPrice.toFixed(2)}`);
    });
  }

  /**
   * Calculate portfolio metrics
   */
  private async calculatePortfolioMetrics(portfolio: Portfolio, positions: Position[]): Promise<void> {
    try {
      console.log('\n📊 Portfolio Metrics:');
      
      // Calculate total exposure
      const totalExposure = positions.reduce((sum, pos) => {
        return sum + (Math.abs(pos.size) * pos.currentPrice);
      }, 0);
      
      console.log(`  Total Exposure: $${totalExposure.toFixed(2)}`);
      
      // Calculate average leverage
      const avgLeverage = positions.length > 0 
        ? positions.reduce((sum, pos) => sum + pos.leverage, 0) / positions.length 
        : 0;
      
      console.log(`  Average Leverage: ${avgLeverage.toFixed(2)}x`);
      
      // Calculate risk metrics
      const riskMetrics = await this.calculateRiskMetrics(portfolio, positions);
      console.log(`  Portfolio Risk: ${riskMetrics.portfolioRisk.toFixed(2)}%`);
      console.log(`  Max Drawdown: ${riskMetrics.maxDrawdown.toFixed(2)}%`);
      
    } catch (error) {
      console.error('❌ Failed to calculate portfolio metrics:', error);
    }
  }

  /**
   * Calculate risk metrics
   */
  private async calculateRiskMetrics(portfolio: Portfolio, positions: Position[]): Promise<{
    portfolioRisk: number;
    maxDrawdown: number;
  }> {
    try {
      // Simple risk calculation based on leverage and position sizes
      let portfolioRisk = 0;
      let maxDrawdown = 0;
      
      for (const position of positions) {
        // Calculate position risk (simplified)
        const positionRisk = (position.leverage * Math.abs(position.size) * position.currentPrice) / portfolio.totalValue;
        portfolioRisk += positionRisk;
        
        // Calculate potential drawdown
        const potentialLoss = Math.abs(position.size) * position.currentPrice * 0.1; // 10% price move
        const drawdown = potentialLoss / portfolio.totalValue;
        maxDrawdown = Math.max(maxDrawdown, drawdown);
      }
      
      return {
        portfolioRisk: Math.min(portfolioRisk * 100, 100), // Cap at 100%
        maxDrawdown: maxDrawdown * 100
      };
    } catch (error) {
      console.error('❌ Failed to calculate risk metrics:', error);
      return { portfolioRisk: 0, maxDrawdown: 0 };
    }
  }

  /**
   * Start monitoring loop
   */
  private startMonitoringLoop(): void {
    console.log(`\n🔄 Starting monitoring loop (${this.updateInterval / 1000}s intervals)...`);
    
    const monitor = async () => {
      if (!this.isRunning) return;
      
      try {
        await this.takePortfolioSnapshot();
        
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
   * Get portfolio alerts
   */
  async getPortfolioAlerts(): Promise<string[]> {
    const alerts: string[] = [];
    
    try {
      const portfolio = await this.client.getPortfolio();
      const positions = await this.client.getPositions();
      
      // Check for high risk positions
      for (const position of positions) {
        const pnlPercent = (position.pnl / (position.size * position.entryPrice)) * 100;
        
        if (pnlPercent < -20) {
          alerts.push(`⚠️ High loss on ${position.market}: ${pnlPercent.toFixed(2)}%`);
        }
        
        if (position.leverage > 20) {
          alerts.push(`⚠️ High leverage on ${position.market}: ${position.leverage}x`);
        }
      }
      
      // Check portfolio health
      if (portfolio.totalPnL < -portfolio.totalValue * 0.1) {
        alerts.push(`🚨 Portfolio down 10%+: $${portfolio.totalPnL.toFixed(2)}`);
      }
      
    } catch (error) {
      console.error('❌ Failed to get portfolio alerts:', error);
    }
    
    return alerts;
  }

  /**
   * Export portfolio data
   */
  async exportPortfolioData(): Promise<{
    timestamp: string;
    portfolio: Portfolio;
    positions: Position[];
  }> {
    try {
      const portfolio = await this.client.getPortfolio();
      const positions = await this.client.getPositions();
      
      return {
        timestamp: new Date().toISOString(),
        portfolio,
        positions
      };
    } catch (error) {
      console.error('❌ Failed to export portfolio data:', error);
      throw error;
    }
  }
}

// Example usage - Realistic implementation
export async function runPortfolioTrackingExample() {
  // Initialize QuantDesk client
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com',
    programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    wallet: wallet // Your Solana wallet instance
  });

  // Create portfolio tracker
  const tracker = new PortfolioTrackingExample(client, 30000); // 30 second updates
  
  try {
    // Start tracking
    await tracker.startTracking();
    
    // Run for 5 minutes (realistic demo duration)
    setTimeout(async () => {
      await tracker.stopTracking();
      console.log('✅ Portfolio tracking demo completed');
    }, 300000); // 5 minutes
    
  } catch (error) {
    console.error('Portfolio tracking failed:', error);
  }
}

// Export for use in other modules
export { PortfolioTrackingExample };
