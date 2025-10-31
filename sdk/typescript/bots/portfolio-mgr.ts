// QuantDesk Portfolio Manager Bot
// Professional portfolio management implementation for QuantDesk perpetual DEX

import { QuantDeskClient, Position, MarketData, Portfolio } from '@quantdesk/sdk';

export interface PortfolioManagerConfig {
  client: QuantDeskClient;
  markets: string[];           // Markets to manage
  rebalanceThreshold: number;  // Rebalance threshold (e.g., 0.05 = 5%)
  maxLeverage: number;        // Maximum leverage per position
  riskLimit: number;          // Maximum risk per position (e.g., 0.1 = 10%)
  updateInterval: number;     // Update interval in milliseconds
  maxPositions: number;       // Maximum number of positions
  targetAllocation: Map<string, number>; // Target allocation per market
}

export class PortfolioManagerBot {
  private client: QuantDeskClient;
  private config: PortfolioManagerConfig;
  private isRunning: boolean = false;
  private portfolio: Portfolio | null = null;
  private positions: Position[] = [];
  private marketData: Map<string, MarketData> = new Map();

  constructor(config: PortfolioManagerConfig) {
    this.client = config.client;
    this.config = config;
  }

  /**
   * Start the portfolio manager bot
   */
  async start(): Promise<void> {
    console.log(`Starting Portfolio Manager Bot for markets: ${this.config.markets.join(', ')}`);
    this.isRunning = true;

    // Initialize bot
    await this.initialize();

    // Start main loop
    this.runMainLoop();
  }

  /**
   * Stop the portfolio manager bot
   */
  async stop(): Promise<void> {
    console.log('Stopping Portfolio Manager Bot...');
    this.isRunning = false;
  }

  /**
   * Initialize the bot
   */
  private async initialize(): Promise<void> {
    try {
      // Get initial portfolio data
      this.portfolio = await this.client.getPortfolio();
      
      // Get initial market data
      for (const market of this.config.markets) {
        const data = await this.client.getMarketData(market);
        this.marketData.set(market, data);
      }

      // Get current positions
      this.positions = await this.client.getPositions();

      console.log(`Initialized Portfolio Manager Bot`);
      console.log(`Markets: ${this.config.markets.join(', ')}`);
      console.log(`Rebalance Threshold: ${this.config.rebalanceThreshold * 100}%`);
      console.log(`Max Leverage: ${this.config.maxLeverage}x`);
      console.log(`Risk Limit: ${this.config.riskLimit * 100}%`);
      console.log(`Current Positions: ${this.positions.length}`);
      console.log(`Portfolio Value: $${this.portfolio.totalValue.toFixed(2)}`);
    } catch (error) {
      console.error('Failed to initialize Portfolio Manager Bot:', error);
      throw error;
    }
  }

  /**
   * Main bot loop
   */
  private async runMainLoop(): Promise<void> {
    while (this.isRunning) {
      try {
        await this.updatePortfolioData();
        await this.analyzePortfolio();
        await this.rebalancePortfolio();
        await this.manageRisk();
        
        // Wait for next update
        await this.sleep(this.config.updateInterval);
      } catch (error) {
        console.error('Error in main loop:', error);
        await this.sleep(5000); // Wait 5 seconds before retry
      }
    }
  }

  /**
   * Update portfolio and market data
   */
  private async updatePortfolioData(): Promise<void> {
    try {
      // Update portfolio
      this.portfolio = await this.client.getPortfolio();
      
      // Update market data
      for (const market of this.config.markets) {
        const data = await this.client.getMarketData(market);
        this.marketData.set(market, data);
      }

      // Update positions
      this.positions = await this.client.getPositions();
    } catch (error) {
      console.error('Failed to update portfolio data:', error);
    }
  }

  /**
   * Analyze current portfolio allocation
   */
  private async analyzePortfolio(): Promise<void> {
    try {
      if (!this.portfolio) return;

      console.log('=== Portfolio Analysis ===');
      console.log(`Total Value: $${this.portfolio.totalValue.toFixed(2)}`);
      console.log(`Total PnL: $${this.portfolio.totalPnL.toFixed(2)}`);
      console.log(`PnL %: ${((this.portfolio.totalPnL / this.portfolio.totalValue) * 100).toFixed(2)}%`);

      // Analyze allocation per market
      for (const market of this.config.markets) {
        const marketPositions = this.positions.filter(p => p.market === market);
        const marketValue = marketPositions.reduce((sum, pos) => sum + pos.value, 0);
        const marketAllocation = marketValue / this.portfolio.totalValue;
        const targetAllocation = this.config.targetAllocation.get(market) || 0;

        console.log(`${market}:`);
        console.log(`  Current: ${(marketAllocation * 100).toFixed(2)}%`);
        console.log(`  Target: ${(targetAllocation * 100).toFixed(2)}%`);
        console.log(`  Deviation: ${((marketAllocation - targetAllocation) * 100).toFixed(2)}%`);
      }
      console.log('========================');
    } catch (error) {
      console.error('Failed to analyze portfolio:', error);
    }
  }

  /**
   * Rebalance portfolio if needed
   */
  private async rebalancePortfolio(): Promise<void> {
    try {
      if (!this.portfolio) return;

      const rebalanceActions = this.calculateRebalanceActions();
      
      if (rebalanceActions.length > 0) {
        console.log(`Executing ${rebalanceActions.length} rebalance actions`);
        
        for (const action of rebalanceActions) {
          await this.executeRebalanceAction(action);
        }
      }
    } catch (error) {
      console.error('Failed to rebalance portfolio:', error);
    }
  }

  /**
   * Calculate rebalance actions needed
   */
  private calculateRebalanceActions(): RebalanceAction[] {
    const actions: RebalanceAction[] = [];

    if (!this.portfolio) return actions;

    for (const market of this.config.markets) {
      const marketPositions = this.positions.filter(p => p.market === market);
      const marketValue = marketPositions.reduce((sum, pos) => sum + pos.value, 0);
      const currentAllocation = marketValue / this.portfolio.totalValue;
      const targetAllocation = this.config.targetAllocation.get(market) || 0;
      
      const deviation = Math.abs(currentAllocation - targetAllocation);
      
      if (deviation > this.config.rebalanceThreshold) {
        const targetValue = this.portfolio.totalValue * targetAllocation;
        const valueDifference = targetValue - marketValue;
        
        actions.push({
          market: market,
          currentAllocation: currentAllocation,
          targetAllocation: targetAllocation,
          valueDifference: valueDifference,
          action: valueDifference > 0 ? 'buy' : 'sell',
          size: Math.abs(valueDifference)
        });
      }
    }

    return actions;
  }

  /**
   * Execute a rebalance action
   */
  private async executeRebalanceAction(action: RebalanceAction): Promise<void> {
    try {
      const marketData = this.marketData.get(action.market);
      if (!marketData) {
        console.warn(`No market data for ${action.market}`);
        return;
      }

      console.log(`Rebalancing ${action.market}: ${action.action} ${action.size.toFixed(4)}`);

      // Calculate position size
      const positionSize = action.size / marketData.price;
      
      // Check leverage limits
      const leverage = this.calculateLeverage(positionSize, marketData.price);
      if (leverage > this.config.maxLeverage) {
        console.log(`Leverage ${leverage.toFixed(2)}x exceeds limit ${this.config.maxLeverage}x, adjusting size`);
        const maxSize = (this.portfolio!.totalValue * this.config.maxLeverage) / marketData.price;
        const adjustedSize = Math.min(positionSize, maxSize);
        console.log(`Adjusted size: ${adjustedSize.toFixed(4)}`);
      }

      // Execute trade
      if (action.action === 'buy') {
        await this.client.openPosition({
          market: action.market,
          side: 'long',
          size: positionSize,
          leverage: Math.min(leverage, this.config.maxLeverage),
          entryPrice: marketData.price
        });
      } else {
        // Close existing positions first
        const marketPositions = this.positions.filter(p => p.market === action.market);
        for (const position of marketPositions) {
          await this.client.closePosition(position.id);
        }
      }

      console.log(`Rebalance action completed for ${action.market}`);
    } catch (error) {
      console.error(`Failed to execute rebalance action for ${action.market}:`, error);
    }
  }

  /**
   * Calculate leverage for a position
   */
  private calculateLeverage(size: number, price: number): number {
    if (!this.portfolio) return 1;
    
    const positionValue = size * price;
    return positionValue / this.portfolio.totalValue;
  }

  /**
   * Manage portfolio risk
   */
  private async manageRisk(): Promise<void> {
    try {
      if (!this.portfolio) return;

      // Check overall portfolio risk
      const portfolioRisk = this.calculatePortfolioRisk();
      
      if (portfolioRisk > this.config.riskLimit) {
        console.log(`Portfolio risk ${(portfolioRisk * 100).toFixed(2)}% exceeds limit ${(this.config.riskLimit * 100).toFixed(2)}%`);
        await this.reducePortfolioRisk();
      }

      // Check individual position risk
      for (const position of this.positions) {
        const positionRisk = this.calculatePositionRisk(position);
        
        if (positionRisk > this.config.riskLimit) {
          console.log(`Position ${position.id} risk ${(positionRisk * 100).toFixed(2)}% exceeds limit`);
          await this.client.closePosition(position.id);
        }
      }
    } catch (error) {
      console.error('Failed to manage risk:', error);
    }
  }

  /**
   * Calculate overall portfolio risk
   */
  private calculatePortfolioRisk(): number {
    if (!this.portfolio) return 0;

    // Simple risk calculation based on leverage and volatility
    let totalRisk = 0;
    
    for (const position of this.positions) {
      const marketData = this.marketData.get(position.market);
      if (marketData) {
        const volatility = marketData.volatility || 0.1;
        const leverage = Math.abs(position.size) * marketData.price / this.portfolio.totalValue;
        totalRisk += leverage * volatility;
      }
    }

    return Math.min(totalRisk, 1); // Cap at 100%
  }

  /**
   * Calculate individual position risk
   */
  private calculatePositionRisk(position: Position): number {
    if (!this.portfolio) return 0;

    const marketData = this.marketData.get(position.market);
    if (!marketData) return 0;

    const volatility = marketData.volatility || 0.1;
    const leverage = Math.abs(position.size) * marketData.price / this.portfolio.totalValue;
    
    return leverage * volatility;
  }

  /**
   * Reduce overall portfolio risk
   */
  private async reducePortfolioRisk(): Promise<void> {
    try {
      // Close positions with highest risk first
      const positionsByRisk = this.positions
        .map(pos => ({ position: pos, risk: this.calculatePositionRisk(pos) }))
        .sort((a, b) => b.risk - a.risk);

      // Close top 25% of positions by risk
      const positionsToClose = Math.ceil(positionsByRisk.length * 0.25);
      
      for (let i = 0; i < positionsToClose; i++) {
        const { position } = positionsByRisk[i];
        await this.client.closePosition(position.id);
        console.log(`Closed high-risk position: ${position.id}`);
      }
    } catch (error) {
      console.error('Failed to reduce portfolio risk:', error);
    }
  }

  /**
   * Get bot status
   */
  getStatus(): {
    isRunning: boolean;
    markets: string[];
    positions: number;
    portfolioValue: number;
    totalPnL: number;
    portfolioRisk: number;
  } {
    return {
      isRunning: this.isRunning,
      markets: this.config.markets,
      positions: this.positions.length,
      portfolioValue: this.portfolio?.totalValue || 0,
      totalPnL: this.portfolio?.totalPnL || 0,
      portfolioRisk: this.calculatePortfolioRisk()
    };
  }

  /**
   * Get portfolio statistics
   */
  getPortfolioStats(): {
    totalValue: number;
    totalPnL: number;
    pnlPercent: number;
    positions: number;
    averageLeverage: number;
    riskScore: number;
  } {
    if (!this.portfolio) {
      return {
        totalValue: 0,
        totalPnL: 0,
        pnlPercent: 0,
        positions: 0,
        averageLeverage: 0,
        riskScore: 0
      };
    }

    const averageLeverage = this.positions.reduce((sum, pos) => {
      const marketData = this.marketData.get(pos.market);
      if (marketData) {
        return sum + (Math.abs(pos.size) * marketData.price / this.portfolio!.totalValue);
      }
      return sum;
    }, 0) / this.positions.length;

    return {
      totalValue: this.portfolio.totalValue,
      totalPnL: this.portfolio.totalPnL,
      pnlPercent: (this.portfolio.totalPnL / this.portfolio.totalValue) * 100,
      positions: this.positions.length,
      averageLeverage: averageLeverage,
      riskScore: this.calculatePortfolioRisk() * 100
    };
  }

  /**
   * Sleep utility function
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Types
interface RebalanceAction {
  market: string;
  currentAllocation: number;
  targetAllocation: number;
  valueDifference: number;
  action: 'buy' | 'sell';
  size: number;
}

// Example usage
export async function runPortfolioManagerBot() {
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com',
    programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    wallet: wallet // Your Solana wallet
  });

  // Define target allocation
  const targetAllocation = new Map<string, number>();
  targetAllocation.set('SOL-PERP', 0.4);  // 40% SOL
  targetAllocation.set('ETH-PERP', 0.3);  // 30% ETH
  targetAllocation.set('BTC-PERP', 0.3);  // 30% BTC

  const bot = new PortfolioManagerBot({
    client: client,
    markets: ['SOL-PERP', 'ETH-PERP', 'BTC-PERP'],
    rebalanceThreshold: 0.05,    // 5% rebalance threshold
    maxLeverage: 10,             // Max 10x leverage
    riskLimit: 0.1,              // Max 10% risk per position
    updateInterval: 10000,        // 10 seconds
    maxPositions: 10,            // Max 10 positions
    targetAllocation: targetAllocation
  });

  try {
    await bot.start();
    
    // Run for 24 hours
    setTimeout(async () => {
      await bot.stop();
      console.log('Portfolio Manager Bot stopped');
    }, 86400000);
    
  } catch (error) {
    console.error('Portfolio Manager Bot error:', error);
  }
}
