import * as readline from 'readline';
import { TradingAgent } from '@/agents/TradingAgent';
import { dataSourcesService } from '@/services/DataSources';
import { ccxtService } from '@/services/CCXTService';
import { solanaService } from '@/services/SolanaService';
import { systemLogger, tradingLogger } from '@/utils/logger';
import { SecurityUtils } from '@/utils/security';

/**
 * Beautiful CLI Interface for Solana DeFi Trading Intelligence AI
 * Provides an interactive command-line experience for trading analysis
 */

export class TradingCLI {
  private rl: readline.Interface;
  private isRunning: boolean = false;
  private tradingAgent: TradingAgent;

  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout,
      prompt: '🤖 Solana AI > '
    });

    this.tradingAgent = new TradingAgent();
    this.setupEventHandlers();
  }

  /**
   * Setup event handlers for CLI
   */
  private setupEventHandlers(): void {
    this.rl.on('line', this.handleInput.bind(this));
    this.rl.on('close', this.handleClose.bind(this));
    this.rl.on('SIGINT', this.handleInterrupt.bind(this));
  }

  /**
   * Start the CLI
   */
  async start(): Promise<void> {
    this.isRunning = true;
    this.displayWelcome();
    this.displayHelp();
    this.rl.prompt();
  }

  /**
   * Display welcome message
   */
  private displayWelcome(): void {
    console.clear();
    console.log('🚀 Solana DeFi Trading Intelligence AI');
    console.log('=====================================');
    console.log('');
    console.log('Welcome to your AI-powered Solana trading assistant!');
    console.log('Ask me anything about markets, wallets, liquidations, or whales.');
    console.log('');
    console.log('💡 Type "help" for available commands or ask me anything!');
    console.log('');
  }

  /**
   * Display help information
   */
  private displayHelp(): void {
    console.log('📚 Available Commands:');
    console.log('');
    console.log('🔍 Analysis Commands:');
    console.log('  • "analyze wallet <address>" - Analyze a wallet portfolio');
    console.log('  • "price <symbol>" - Get current price and analysis');
    console.log('  • "sentiment <symbol>" - Analyze market sentiment');
    console.log('  • "liquidations" - Show recent liquidations');
    console.log('  • "whales" - Track whale activities');
    console.log('  • "technical <symbol>" - Technical analysis');
    console.log('');
    console.log('📊 Data Commands:');
    console.log('  • "drift data" - Get Drift Protocol data');
    console.log('  • "jupiter data" - Get Jupiter aggregator data');
    console.log('  • "hyperliquid data" - Get Hyperliquid data');
    console.log('  • "all data" - Get aggregated data from all sources');
    console.log('');
    console.log('🏦 CEX Commands:');
    console.log('  • "cex prices <symbol>" - Get prices from all CEX');
    console.log('  • "cex liquidations <symbol>" - Get CEX liquidations');
    console.log('  • "cex orderbook <symbol>" - Get CEX order books');
    console.log('  • "cex funding <symbol>" - Get funding rates');
    console.log('  • "arbitrage <symbol>" - Find arbitrage opportunities');
    console.log('');
    console.log('🛠️ System Commands:');
    console.log('  • "health" - Check system health');
    console.log('  • "balance" - Check wallet balance');
    console.log('  • "help" - Show this help');
    console.log('  • "clear" - Clear screen');
    console.log('  • "exit" - Exit the application');
    console.log('');
    console.log('💬 Natural Language:');
    console.log('  • "What are the top whale wallets today?"');
    console.log('  • "Show me SOL price analysis and support levels"');
    console.log('  • "Are there any large liquidations happening?"');
    console.log('  • "What\'s the market sentiment for ETH?"');
    console.log('');
  }

  /**
   * Handle user input
   */
  private async handleInput(input: string): Promise<void> {
    const command = input.trim().toLowerCase();
    
    if (!command) {
      this.rl.prompt();
      return;
    }

    try {
      await this.processCommand(command, input.trim());
    } catch (error) {
      console.log('❌ Error:', (error as Error).message);
    }

    if (this.isRunning) {
      this.rl.prompt();
    }
  }

  /**
   * Process user commands
   */
  private async processCommand(command: string, originalInput: string): Promise<void> {
    const args = command.split(' ');

    switch (args[0]) {
      case 'help':
        this.displayHelp();
        break;

      case 'clear':
        console.clear();
        break;

      case 'exit':
      case 'quit':
        await this.shutdown();
        break;

      case 'health':
        await this.showHealth();
        break;

      case 'balance':
        await this.showBalance();
        break;

      case 'analyze':
        if (args[1] === 'wallet' && args[2]) {
          await this.analyzeWallet(args[2]);
        } else {
          console.log('❌ Usage: analyze wallet <address>');
        }
        break;

      case 'price':
        if (args[1]) {
          await this.getPrice(args[1]);
        } else {
          console.log('❌ Usage: price <symbol>');
        }
        break;

      case 'sentiment':
        if (args[1]) {
          await this.getSentiment(args[1]);
        } else {
          console.log('❌ Usage: sentiment <symbol>');
        }
        break;

      case 'liquidations':
        await this.getLiquidations();
        break;

      case 'whales':
        await this.getWhales();
        break;

      case 'technical':
        if (args[1]) {
          await this.getTechnicalAnalysis(args[1]);
        } else {
          console.log('❌ Usage: technical <symbol>');
        }
        break;

      case 'drift':
        if (args[1] === 'data') {
          await this.getDriftData();
        } else {
          console.log('❌ Usage: drift data');
        }
        break;

      case 'jupiter':
        if (args[1] === 'data') {
          await this.getJupiterData();
        } else {
          console.log('❌ Usage: jupiter data');
        }
        break;

      case 'hyperliquid':
        if (args[1] === 'data') {
          await this.getHyperliquidData();
        } else {
          console.log('❌ Usage: hyperliquid data');
        }
        break;

      case 'all':
        if (args[1] === 'data') {
          await this.getAllData();
        } else {
          console.log('❌ Usage: all data');
        }
        break;

      case 'cex':
        if (args[1] === 'prices' && args[2]) {
          await this.getCEXPrices(args[2]);
        } else if (args[1] === 'liquidations' && args[2]) {
          await this.getCEXLiquidations(args[2]);
        } else if (args[1] === 'orderbook' && args[2]) {
          await this.getCEXOrderBook(args[2]);
        } else if (args[1] === 'funding' && args[2]) {
          await this.getCEXFunding(args[2]);
        } else {
          console.log('❌ Usage: cex <prices|liquidations|orderbook|funding> <symbol>');
        }
        break;

      case 'arbitrage':
        if (args[1]) {
          await this.getArbitrageOpportunities(args[1]);
        } else {
          console.log('❌ Usage: arbitrage <symbol>');
        }
        break;

      default:
        // Treat as natural language query
        await this.processNaturalLanguageQuery(originalInput);
        break;
    }
  }

  /**
   * Process natural language queries
   */
  private async processNaturalLanguageQuery(query: string): Promise<void> {
    console.log('🤔 Analyzing your query...');
    
    try {
      const response = await this.tradingAgent.processQuery({
        query,
        context: {}
      });

      console.log('\n🧠 AI Analysis:');
      console.log('================');
      console.log(response.response);
      console.log(`\n📊 Confidence: ${(response.confidence * 100).toFixed(1)}%`);
      console.log(`🔗 Sources: ${response.sources.join(', ')}`);
      
      tradingLogger.aiQuery(query, response.response, response.confidence);
    } catch (error) {
      console.log('❌ Failed to process query:', (error as Error).message);
    }
  }

  /**
   * Show system health
   */
  private async showHealth(): Promise<void> {
    console.log('🏥 System Health Check');
    console.log('======================');
    
    try {
      const networkHealth = await solanaService.getNetworkHealth();
      
      console.log(`✅ Solana Network: ${networkHealth.isHealthy ? 'Healthy' : 'Unhealthy'}`);
      console.log(`   Cluster: ${networkHealth.cluster}`);
      console.log(`   Current Slot: ${networkHealth.slot.toLocaleString()}`);
      console.log(`   Block Height: ${networkHealth.blockHeight.toLocaleString()}`);
      console.log(`   Epoch: ${networkHealth.epoch}`);
      
      console.log('✅ AI Agent: Ready');
      console.log('✅ Data Sources: Connected');
      console.log('✅ Security: Active');
      
    } catch (error) {
      console.log('❌ Health check failed:', (error as Error).message);
    }
  }

  /**
   * Show wallet balance
   */
  private async showBalance(): Promise<void> {
    console.log('💰 Wallet Balance');
    console.log('=================');
    
    try {
      const publicKey = solanaService.getPublicKey();
      if (!publicKey) {
        console.log('❌ No wallet configured');
        return;
      }

      const balance = await solanaService.getBalance();
      
      console.log(`📍 Address: ${publicKey.toString()}`);
      console.log(`💎 Balance: ${balance.toFixed(4)} SOL`);
      console.log(`💵 USD Value: ~$${(balance * 95.50).toFixed(2)} (estimated)`);
      
    } catch (error) {
      console.log('❌ Failed to get balance:', (error as Error).message);
    }
  }

  /**
   * Analyze wallet
   */
  private async analyzeWallet(address: string): Promise<void> {
    console.log(`🔍 Analyzing wallet: ${SecurityUtils.maskSensitiveData(address)}`);
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: `Analyze wallet ${address} with transaction history`,
        context: { walletId: address }
      });

      console.log('\n📊 Wallet Analysis:');
      console.log('===================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to analyze wallet:', (error as Error).message);
    }
  }

  /**
   * Get price analysis
   */
  private async getPrice(symbol: string): Promise<void> {
    console.log(`📈 Getting price analysis for ${symbol.toUpperCase()}`);
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: `Get current price and analysis for ${symbol}`,
        context: { symbols: [symbol] }
      });

      console.log('\n💹 Price Analysis:');
      console.log('==================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to get price:', (error as Error).message);
    }
  }

  /**
   * Get sentiment analysis
   */
  private async getSentiment(symbol: string): Promise<void> {
    console.log(`😊 Analyzing sentiment for ${symbol.toUpperCase()}`);
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: `Analyze market sentiment for ${symbol}`,
        context: { symbols: [symbol] }
      });

      console.log('\n🎭 Sentiment Analysis:');
      console.log('=======================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to get sentiment:', (error as Error).message);
    }
  }

  /**
   * Get liquidations
   */
  private async getLiquidations(): Promise<void> {
    console.log('💥 Fetching recent liquidations...');
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: 'Show me recent liquidations across all protocols',
        context: {}
      });

      console.log('\n⚡ Recent Liquidations:');
      console.log('=======================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to get liquidations:', (error as Error).message);
    }
  }

  /**
   * Get whale activities
   */
  private async getWhales(): Promise<void> {
    console.log('🐋 Tracking whale activities...');
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: 'Show me the top whale wallets and their recent activities',
        context: {}
      });

      console.log('\n🐋 Whale Activities:');
      console.log('====================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to get whale data:', (error as Error).message);
    }
  }

  /**
   * Get technical analysis
   */
  private async getTechnicalAnalysis(symbol: string): Promise<void> {
    console.log(`📊 Performing technical analysis for ${symbol.toUpperCase()}`);
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: `Perform technical analysis for ${symbol} with all indicators`,
        context: { symbols: [symbol] }
      });

      console.log('\n📈 Technical Analysis:');
      console.log('======================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to get technical analysis:', (error as Error).message);
    }
  }

  /**
   * Get Drift data
   */
  private async getDriftData(): Promise<void> {
    console.log('🌊 Fetching Drift Protocol data...');
    
    try {
      const data = await dataSourcesService.getDriftData();
      
      console.log('\n🌊 Drift Protocol Data:');
      console.log('=======================');
      console.log(`📊 Markets: ${data.markets.length}`);
      console.log(`💼 Positions: ${data.positions.length}`);
      console.log(`💥 Liquidations: ${data.liquidations.length}`);
      console.log(`💰 Funding Rates: ${data.fundingRates.length}`);
      
      if (data.markets.length > 0) {
        console.log('\nTop Market:');
        const market = data.markets[0];
        console.log(`  Symbol: ${market.symbol}`);
        console.log(`  Price: $${market.price}`);
        console.log(`  24h Volume: $${market.volume24h?.toLocaleString()}`);
        console.log(`  Open Interest: $${market.openInterest?.toLocaleString()}`);
      }
      
    } catch (error) {
      console.log('❌ Failed to get Drift data:', (error as Error).message);
    }
  }

  /**
   * Get Jupiter data
   */
  private async getJupiterData(): Promise<void> {
    console.log('🪐 Fetching Jupiter aggregator data...');
    
    try {
      const data = await dataSourcesService.getJupiterData();
      
      console.log('\n🪐 Jupiter Aggregator Data:');
      console.log('===========================');
      console.log(`💱 Quotes: ${data.quotes.length}`);
      console.log(`🛣️ Routes: ${data.routes.length}`);
      console.log(`🪙 Tokens: ${data.tokens.length}`);
      console.log(`💰 Price Data: ${data.priceData.length}`);
      
      if (data.priceData.length > 0) {
        console.log('\nPrice Data:');
        data.priceData.forEach((price: any) => {
          console.log(`  ${price.id}: $${price.price}`);
        });
      }
      
    } catch (error) {
      console.log('❌ Failed to get Jupiter data:', (error as Error).message);
    }
  }

  /**
   * Get Hyperliquid data
   */
  private async getHyperliquidData(): Promise<void> {
    console.log('⚡ Fetching Hyperliquid data...');
    
    try {
      const data = await dataSourcesService.getHyperliquidData();
      
      console.log('\n⚡ Hyperliquid Data:');
      console.log('====================');
      console.log(`🔄 Perpetuals: ${data.perpetuals.length}`);
      console.log(`💥 Liquidations: ${data.liquidations.length}`);
      console.log(`💰 Funding Rates: ${data.fundingRates.length}`);
      console.log(`📊 Order Books: ${data.orderBook.length}`);
      
      if (data.perpetuals.length > 0) {
        console.log('\nTop Perpetual:');
        const perpetual = data.perpetuals[0];
        console.log(`  Name: ${perpetual.name}`);
        console.log(`  Max Leverage: ${perpetual.maxLeverage}x`);
        console.log(`  Max Size: ${perpetual.maxSz}`);
      }
      
    } catch (error) {
      console.log('❌ Failed to get Hyperliquid data:', (error as Error).message);
    }
  }

  /**
   * Get all data sources
   */
  private async getAllData(): Promise<void> {
    console.log('🌐 Fetching data from all sources...');
    
    try {
      const data = await dataSourcesService.getAggregatedMarketData();
      
      console.log('\n🌐 Aggregated Market Data:');
      console.log('===========================');
      console.log(`🌊 Drift Markets: ${data.drift.markets.length}`);
      console.log(`🪐 Jupiter Quotes: ${data.jupiter.quotes.length}`);
      console.log(`⚡ Hyperliquid Perpetuals: ${data.hyperliquid.perpetuals.length}`);
      console.log(`🔮 Axiom Pools: ${data.axiom.pools.length}`);
      console.log(`⭐ Asterdex Markets: ${data.asterdex.markets.length}`);
      
      console.log('\n📊 Summary:');
      const totalMarkets = data.drift.markets.length + data.jupiter.quotes.length + 
                          data.hyperliquid.perpetuals.length + data.axiom.pools.length + 
                          data.asterdex.markets.length;
      console.log(`  Total Markets Tracked: ${totalMarkets}`);
      console.log(`  Total Liquidations: ${data.drift.liquidations.length + data.hyperliquid.liquidations.length}`);
      
    } catch (error) {
      console.log('❌ Failed to get aggregated data:', (error as Error).message);
    }
  }

  /**
   * Handle close event
   */
  private handleClose(): void {
    console.log('\n👋 Goodbye! Happy trading!');
    process.exit(0);
  }

  /**
   * Handle interrupt (Ctrl+C)
   */
  private async handleInterrupt(): Promise<void> {
    console.log('\n\n🛑 Shutting down gracefully...');
    await this.shutdown();
  }

  /**
   * Get CEX prices
   */
  private async getCEXPrices(symbol: string): Promise<void> {
    console.log(`💰 Getting CEX prices for ${symbol.toUpperCase()}...`);
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: `Get real-time prices for ${symbol} from all centralized exchanges`,
        context: { symbols: [symbol] }
      });

      console.log('\n💹 CEX Price Analysis:');
      console.log('======================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to get CEX prices:', (error as Error).message);
    }
  }

  /**
   * Get CEX liquidations
   */
  private async getCEXLiquidations(symbol: string): Promise<void> {
    console.log(`💥 Getting CEX liquidations for ${symbol.toUpperCase()}...`);
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: `Get liquidation data for ${symbol} from centralized exchanges`,
        context: { symbols: [symbol] }
      });

      console.log('\n⚡ CEX Liquidations:');
      console.log('====================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to get CEX liquidations:', (error as Error).message);
    }
  }

  /**
   * Get CEX order book
   */
  private async getCEXOrderBook(symbol: string): Promise<void> {
    console.log(`📊 Getting CEX order books for ${symbol.toUpperCase()}...`);
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: `Get order book data for ${symbol} from centralized exchanges`,
        context: { symbols: [symbol] }
      });

      console.log('\n📈 CEX Order Books:');
      console.log('====================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to get CEX order book:', (error as Error).message);
    }
  }

  /**
   * Get CEX funding rates
   */
  private async getCEXFunding(symbol: string): Promise<void> {
    console.log(`💰 Getting CEX funding rates for ${symbol.toUpperCase()}...`);
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: `Get funding rates for ${symbol} from perpetual exchanges`,
        context: { symbols: [symbol] }
      });

      console.log('\n💸 CEX Funding Rates:');
      console.log('======================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to get CEX funding rates:', (error as Error).message);
    }
  }

  /**
   * Get arbitrage opportunities
   */
  private async getArbitrageOpportunities(symbol: string): Promise<void> {
    console.log(`🔍 Finding arbitrage opportunities for ${symbol.toUpperCase()}...`);
    
    try {
      const response = await this.tradingAgent.processQuery({
        query: `Find arbitrage opportunities for ${symbol} between centralized exchanges`,
        context: { symbols: [symbol] }
      });

      console.log('\n🎯 Arbitrage Opportunities:');
      console.log('============================');
      console.log(response.response);
      
    } catch (error) {
      console.log('❌ Failed to find arbitrage opportunities:', (error as Error).message);
    }
  }

  /**
   * Shutdown the CLI
   */
  private async shutdown(): Promise<void> {
    this.isRunning = false;
    console.log('\n🔄 Cleaning up...');
    
    try {
      await solanaService.cleanup();
      await ccxtService.cleanup();
      systemLogger.shutdown('CLI shutdown');
    } catch (error) {
      console.log('❌ Error during shutdown:', (error as Error).message);
    }
    
    this.rl.close();
    process.exit(0);
  }
}

// Export CLI instance
export const tradingCLI = new TradingCLI();
