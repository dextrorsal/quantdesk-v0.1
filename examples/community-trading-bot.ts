/**
 * QuantDesk Community Trading Bot Example
 * 
 * This is a simple trading bot example that demonstrates how to use
 * QuantDesk components to build automated trading strategies.
 * 
 * DISCLAIMER: This is for educational purposes only. Trading involves risk.
 */

import { QuantDeskSDK } from '@quantdesk/sdk';
import { MarketAnalysisAgent, RiskManagementAgent } from './examples/mikey-ai-agents';
import { EventEmitter } from 'events';

interface TradingBotConfig {
  symbols: string[];
  riskPercent: number;
  maxPositions: number;
  stopLossPercent: number;
  takeProfitPercent: number;
  apiKey: string;
  walletPrivateKey: string;
}

interface Position {
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;
  timestamp: Date;
}

export class SimpleTradingBot extends EventEmitter {
  private sdk: QuantDeskSDK;
  private analysisAgent: MarketAnalysisAgent;
  private riskAgent: RiskManagementAgent;
  private config: TradingBotConfig;
  private positions: Map<string, Position> = new Map();
  private isRunning = false;

  constructor(config: TradingBotConfig) {
    super();
    this.config = config;
    
    // Initialize SDK
    this.sdk = new QuantDeskSDK({
      apiUrl: 'https://api.quantdesk.com',
      network: 'mainnet-beta',
      apiKey: config.apiKey
    });

    // Initialize AI agents
    this.analysisAgent = new MarketAnalysisAgent(config.apiKey);
    this.riskAgent = new RiskManagementAgent(config.apiKey);
  }

  async start(): Promise<void> {
    console.log('🤖 Starting QuantDesk Trading Bot...');
    this.isRunning = true;

    // Connect wallet
    await this.sdk.wallet.connect();
    console.log('✅ Wallet connected');

    // Start monitoring loop
    this.startMonitoringLoop();
    
    this.emit('started');
  }

  async stop(): Promise<void> {
    console.log('🛑 Stopping trading bot...');
    this.isRunning = false;
    
    // Close all positions
    await this.closeAllPositions();
    
    this.emit('stopped');
  }

  private async startMonitoringLoop(): Promise<void> {
    while (this.isRunning) {
      try {
        await this.analyzeMarkets();
        await this.managePositions();
        await this.sleep(30000); // Wait 30 seconds
      } catch (error) {
        console.error('❌ Error in monitoring loop:', error);
        await this.sleep(5000); // Wait 5 seconds on error
      }
    }
  }

  private async analyzeMarkets(): Promise<void> {
    for (const symbol of this.config.symbols) {
      try {
        // Get market data
        const marketData = await this.sdk.market.getPrice(symbol);
        
        // Analyze market sentiment
        const analysis = await this.analysisAgent.analyzeMarket(symbol, {
          price: marketData.price,
          volume: marketData.volume,
          change24h: marketData.change24h
        });

        console.log(`📊 ${symbol} Analysis:`, {
          sentiment: analysis.sentiment,
          confidence: analysis.confidence,
          price: marketData.price
        });

        // Check if we should open a position
        if (this.shouldOpenPosition(symbol, analysis)) {
          await this.openPosition(symbol, analysis);
        }

      } catch (error) {
        console.error(`❌ Error analyzing ${symbol}:`, error);
      }
    }
  }

  private shouldOpenPosition(symbol: string, analysis: any): boolean {
    // Don't open if we already have a position
    if (this.positions.has(symbol)) {
      return false;
    }

    // Don't open if we're at max positions
    if (this.positions.size >= this.config.maxPositions) {
      return false;
    }

    // Only open if confidence is high and sentiment is clear
    return analysis.confidence > 0.7 && 
           (analysis.sentiment === 'bullish' || analysis.sentiment === 'bearish');
  }

  private async openPosition(symbol: string, analysis: any): Promise<void> {
    try {
      const marketData = await this.sdk.market.getPrice(symbol);
      const portfolio = await this.sdk.portfolio.get();
      
      // Calculate position size based on risk
      const riskAmount = portfolio.totalValue * (this.config.riskPercent / 100);
      const stopLossDistance = marketData.price * (this.config.stopLossPercent / 100);
      const positionSize = riskAmount / stopLossDistance;

      const side = analysis.sentiment === 'bullish' ? 'long' : 'short';
      const stopLoss = side === 'long' 
        ? marketData.price * (1 - this.config.stopLossPercent / 100)
        : marketData.price * (1 + this.config.stopLossPercent / 100);
      
      const takeProfit = side === 'long'
        ? marketData.price * (1 + this.config.takeProfitPercent / 100)
        : marketData.price * (1 - this.config.takeProfitPercent / 100);

      // Create order
      const order = await this.sdk.orders.create({
        symbol,
        side: side === 'long' ? 'buy' : 'sell',
        amount: positionSize,
        type: 'market'
      });

      // Store position
      const position: Position = {
        symbol,
        side,
        size: positionSize,
        entryPrice: marketData.price,
        stopLoss,
        takeProfit,
        timestamp: new Date()
      };

      this.positions.set(symbol, position);

      console.log(`🎯 Opened ${side} position for ${symbol}:`, {
        size: positionSize,
        entryPrice: marketData.price,
        stopLoss,
        takeProfit
      });

      this.emit('positionOpened', position);

    } catch (error) {
      console.error(`❌ Error opening position for ${symbol}:`, error);
    }
  }

  private async managePositions(): Promise<void> {
    for (const [symbol, position] of this.positions) {
      try {
        const currentPrice = await this.sdk.market.getPrice(symbol);
        
        // Check stop loss
        if (this.shouldStopLoss(position, currentPrice.price)) {
          await this.closePosition(symbol, 'stop-loss');
          continue;
        }

        // Check take profit
        if (this.shouldTakeProfit(position, currentPrice.price)) {
          await this.closePosition(symbol, 'take-profit');
          continue;
        }

        // Check if position is still valid based on analysis
        const analysis = await this.analysisAgent.analyzeMarket(symbol, {
          price: currentPrice.price,
          volume: currentPrice.volume,
          change24h: currentPrice.change24h
        });

        if (this.shouldClosePosition(position, analysis)) {
          await this.closePosition(symbol, 'analysis');
        }

      } catch (error) {
        console.error(`❌ Error managing position for ${symbol}:`, error);
      }
    }
  }

  private shouldStopLoss(position: Position, currentPrice: number): boolean {
    if (position.side === 'long') {
      return currentPrice <= position.stopLoss;
    } else {
      return currentPrice >= position.stopLoss;
    }
  }

  private shouldTakeProfit(position: Position, currentPrice: number): boolean {
    if (position.side === 'long') {
      return currentPrice >= position.takeProfit;
    } else {
      return currentPrice <= position.takeProfit;
    }
  }

  private shouldClosePosition(position: Position, analysis: any): boolean {
    // Close if sentiment changes against our position
    if (position.side === 'long' && analysis.sentiment === 'bearish') {
      return true;
    }
    if (position.side === 'short' && analysis.sentiment === 'bullish') {
      return true;
    }

    // Close if confidence drops significantly
    return analysis.confidence < 0.3;
  }

  private async closePosition(symbol: string, reason: string): Promise<void> {
    try {
      const position = this.positions.get(symbol);
      if (!position) return;

      // Create closing order
      const order = await this.sdk.orders.create({
        symbol,
        side: position.side === 'long' ? 'sell' : 'buy',
        amount: position.size,
        type: 'market'
      });

      // Calculate P&L
      const currentPrice = await this.sdk.market.getPrice(symbol);
      const pnl = this.calculatePnL(position, currentPrice.price);

      console.log(`🔒 Closed ${position.side} position for ${symbol} (${reason}):`, {
        pnl: pnl.toFixed(2),
        duration: Date.now() - position.timestamp.getTime()
      });

      this.positions.delete(symbol);
      this.emit('positionClosed', { position, reason, pnl });

    } catch (error) {
      console.error(`❌ Error closing position for ${symbol}:`, error);
    }
  }

  private calculatePnL(position: Position, currentPrice: number): number {
    if (position.side === 'long') {
      return (currentPrice - position.entryPrice) * position.size;
    } else {
      return (position.entryPrice - currentPrice) * position.size;
    }
  }

  private async closeAllPositions(): Promise<void> {
    console.log('🔒 Closing all positions...');
    
    for (const symbol of this.positions.keys()) {
      await this.closePosition(symbol, 'bot-stop');
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Public methods for monitoring
  getPositions(): Position[] {
    return Array.from(this.positions.values());
  }

  getStatus(): {
    isRunning: boolean;
    positionCount: number;
    symbols: string[];
  } {
    return {
      isRunning: this.isRunning,
      positionCount: this.positions.size,
      symbols: this.config.symbols
    };
  }
}

// Example usage
async function runTradingBot() {
  const config: TradingBotConfig = {
    symbols: ['SOL', 'BTC', 'ETH'],
    riskPercent: 2, // Risk 2% per trade
    maxPositions: 3,
    stopLossPercent: 5, // 5% stop loss
    takeProfitPercent: 10, // 10% take profit
    apiKey: 'your-openai-api-key',
    walletPrivateKey: 'your-solana-private-key'
  };

  const bot = new SimpleTradingBot(config);

  // Set up event listeners
  bot.on('started', () => {
    console.log('✅ Bot started successfully');
  });

  bot.on('positionOpened', (position) => {
    console.log('🎯 New position opened:', position.symbol);
  });

  bot.on('positionClosed', ({ position, reason, pnl }) => {
    console.log(`🔒 Position closed (${reason}): P&L = ${pnl.toFixed(2)}`);
  });

  bot.on('stopped', () => {
    console.log('🛑 Bot stopped');
  });

  // Start the bot
  await bot.start();

  // Monitor for 1 hour
  setTimeout(async () => {
    await bot.stop();
    process.exit(0);
  }, 60 * 60 * 1000);
}

// Export for use in other projects
export default SimpleTradingBot;

// Run if this file is executed directly
if (require.main === module) {
  runTradingBot().catch(console.error);
}
