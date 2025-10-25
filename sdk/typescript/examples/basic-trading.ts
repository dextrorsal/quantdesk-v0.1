// QuantDesk Basic Trading Example
// Comprehensive example of QuantDesk SDK usage for basic trading operations

import { QuantDeskClient, MarketData, Position, Order } from '@quantdesk/sdk';

/**
 * Basic Trading Example
 * This example demonstrates how to use the QuantDesk SDK for basic trading operations
 */
export class BasicTradingExample {
  private client: QuantDeskClient;

  constructor(client: QuantDeskClient) {
    this.client = client;
  }

  /**
   * Run the complete trading example
   */
  async runExample(): Promise<void> {
    try {
      console.log('🚀 Starting QuantDesk Basic Trading Example');
      
      // Step 1: Initialize client
      await this.initializeClient();
      
      // Step 2: Get market information
      await this.getMarketInformation();
      
      // Step 3: Get portfolio overview
      await this.getPortfolioOverview();
      
      // Step 4: Place a sample order
      await this.placeSampleOrder();
      
      // Step 5: Monitor positions
      await this.monitorPositions();
      
      // Step 6: Get AI analysis
      await this.getAIAnalysis();
      
      console.log('✅ Basic Trading Example completed successfully');
      
    } catch (error) {
      console.error('❌ Basic Trading Example failed:', error);
      throw error;
    }
  }

  /**
   * Initialize the QuantDesk client
   */
  private async initializeClient(): Promise<void> {
    console.log('\n📡 Initializing QuantDesk Client...');
    
    try {
      await this.client.initialize();
      console.log('✅ Client initialized successfully');
    } catch (error) {
      console.error('❌ Failed to initialize client:', error);
      throw error;
    }
  }

  /**
   * Get market information
   */
  private async getMarketInformation(): Promise<void> {
    console.log('\n📊 Getting Market Information...');
    
    try {
      // Get available markets
      const markets = await this.client.getMarkets();
      console.log(`📈 Available markets: ${markets.length}`);
      markets.forEach(market => {
        console.log(`  - ${market.symbol}: ${market.name}`);
      });

      // Get detailed market data for SOL-PERP
      const solMarket = markets.find(m => m.symbol === 'SOL-PERP');
      if (solMarket) {
        const marketData = await this.client.getMarketData('SOL-PERP');
        console.log(`\n🔍 SOL-PERP Market Data:`);
        console.log(`  Price: $${marketData.price.toFixed(2)}`);
        console.log(`  Volume: ${marketData.volume.toFixed(2)}`);
        console.log(`  Change 24h: ${(marketData.change24h * 100).toFixed(2)}%`);
        console.log(`  High 24h: $${marketData.high24h.toFixed(2)}`);
        console.log(`  Low 24h: $${marketData.low24h.toFixed(2)}`);
      }
    } catch (error) {
      console.error('❌ Failed to get market information:', error);
      throw error;
    }
  }

  /**
   * Get portfolio overview
   */
  private async getPortfolioOverview(): Promise<void> {
    console.log('\n💼 Getting Portfolio Overview...');
    
    try {
      const portfolio = await this.client.getPortfolio();
      console.log(`💰 Portfolio Value: $${portfolio.totalValue.toFixed(2)}`);
      console.log(`📈 Total PnL: $${portfolio.totalPnL.toFixed(2)}`);
      console.log(`📊 PnL %: ${((portfolio.totalPnL / portfolio.totalValue) * 100).toFixed(2)}%`);
      console.log(`🔢 Total Positions: ${portfolio.totalPositions}`);
      console.log(`💵 Available Balance: $${portfolio.availableBalance.toFixed(2)}`);

      // Get individual positions
      const positions = await this.client.getPositions();
      if (positions.length > 0) {
        console.log(`\n📋 Current Positions:`);
        positions.forEach((position, index) => {
          console.log(`  ${index + 1}. ${position.market}`);
          console.log(`     Side: ${position.side}`);
          console.log(`     Size: ${position.size.toFixed(4)}`);
          console.log(`     Entry Price: $${position.entryPrice.toFixed(2)}`);
          console.log(`     Current Price: $${position.currentPrice.toFixed(2)}`);
          console.log(`     PnL: $${position.pnl.toFixed(2)}`);
          console.log(`     Leverage: ${position.leverage}x`);
        });
      } else {
        console.log('📭 No open positions');
      }
    } catch (error) {
      console.error('❌ Failed to get portfolio overview:', error);
      throw error;
    }
  }

  /**
   * Place a sample order
   */
  private async placeSampleOrder(): Promise<void> {
    console.log('\n📝 Placing Sample Order...');
    
    try {
      // Get current market data
      const marketData = await this.client.getMarketData('SOL-PERP');
      const currentPrice = marketData.price;
      
      // Place a small limit order
      const orderData = {
        market: 'SOL-PERP',
        side: 'buy' as const,
        size: 0.1, // 0.1 SOL
        price: currentPrice * 0.99, // 1% below current price
        orderType: 'limit' as const
      };

      console.log(`📋 Order Details:`);
      console.log(`  Market: ${orderData.market}`);
      console.log(`  Side: ${orderData.side}`);
      console.log(`  Size: ${orderData.size}`);
      console.log(`  Price: $${orderData.price.toFixed(2)}`);
      console.log(`  Type: ${orderData.orderType}`);

      const order = await this.client.placeOrder(orderData);
      console.log(`✅ Order placed successfully!`);
      console.log(`  Order ID: ${order.id}`);
      console.log(`  Status: ${order.status}`);
      console.log(`  Created: ${new Date(order.createdAt).toLocaleString()}`);

      // Monitor order status
      await this.monitorOrderStatus(order.id);

    } catch (error) {
      console.error('❌ Failed to place sample order:', error);
      throw error;
    }
  }

  /**
   * Monitor order status
   */
  private async monitorOrderStatus(orderId: string): Promise<void> {
    console.log(`\n👀 Monitoring Order Status: ${orderId}`);
    
    try {
      const maxChecks = 10;
      let checks = 0;

      while (checks < maxChecks) {
        const order = await this.client.getOrder(orderId);
        console.log(`  Status: ${order.status} (Check ${checks + 1}/${maxChecks})`);

        if (order.status === 'filled') {
          console.log(`✅ Order filled!`);
          console.log(`  Filled Size: ${order.filledSize}`);
          console.log(`  Filled Price: $${order.filledPrice.toFixed(2)}`);
          break;
        } else if (order.status === 'cancelled' || order.status === 'rejected') {
          console.log(`❌ Order ${order.status}`);
          break;
        }

        // Wait 2 seconds before next check
        await this.sleep(2000);
        checks++;
      }

      if (checks >= maxChecks) {
        console.log(`⏰ Order monitoring timeout`);
      }
    } catch (error) {
      console.error('❌ Failed to monitor order status:', error);
    }
  }

  /**
   * Monitor positions
   */
  private async monitorPositions(): Promise<void> {
    console.log('\n👀 Monitoring Positions...');
    
    try {
      const positions = await this.client.getPositions();
      
      if (positions.length === 0) {
        console.log('📭 No positions to monitor');
        return;
      }

      console.log(`📊 Monitoring ${positions.length} positions:`);
      
      for (const position of positions) {
        console.log(`\n📋 Position: ${position.market}`);
        console.log(`  Side: ${position.side}`);
        console.log(`  Size: ${position.size.toFixed(4)}`);
        console.log(`  Entry Price: $${position.entryPrice.toFixed(2)}`);
        console.log(`  Current Price: $${position.currentPrice.toFixed(2)}`);
        console.log(`  PnL: $${position.pnl.toFixed(2)}`);
        console.log(`  PnL %: ${((position.pnl / (position.size * position.entryPrice)) * 100).toFixed(2)}%`);
        console.log(`  Leverage: ${position.leverage}x`);
        console.log(`  Margin: $${position.margin.toFixed(2)}`);
        console.log(`  Liquidation Price: $${position.liquidationPrice.toFixed(2)}`);
      }
    } catch (error) {
      console.error('❌ Failed to monitor positions:', error);
    }
  }

  /**
   * Get AI analysis
   */
  private async getAIAnalysis(): Promise<void> {
    console.log('\n🤖 Getting AI Analysis...');
    
    try {
      // Get AI market analysis
      const analysis = await this.client.getAIAnalysis('SOL-PERP');
      console.log(`🔍 AI Market Analysis for SOL-PERP:`);
      console.log(`  Sentiment: ${analysis.sentiment}`);
      console.log(`  Confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
      console.log(`  Recommendation: ${analysis.recommendation}`);
      console.log(`  Risk Level: ${analysis.riskLevel}`);
      console.log(`  Key Insights:`);
      analysis.insights.forEach((insight, index) => {
        console.log(`    ${index + 1}. ${insight}`);
      });

      // Get trading signals
      const signals = await this.client.getTradingSignals();
      if (signals.length > 0) {
        console.log(`\n📡 Trading Signals:`);
        signals.forEach((signal, index) => {
          console.log(`  ${index + 1}. ${signal.market}: ${signal.action} (${signal.strength})`);
        });
      }

      // Get risk assessment
      const riskAssessment = await this.client.getRiskAssessment();
      console.log(`\n⚠️ Risk Assessment:`);
      console.log(`  Overall Risk: ${riskAssessment.overallRisk}`);
      console.log(`  Portfolio Risk: ${riskAssessment.portfolioRisk}`);
      console.log(`  Market Risk: ${riskAssessment.marketRisk}`);
      console.log(`  Recommendations:`);
      riskAssessment.recommendations.forEach((rec, index) => {
        console.log(`    ${index + 1}. ${rec}`);
      });

    } catch (error) {
      console.error('❌ Failed to get AI analysis:', error);
    }
  }

  /**
   * Sleep utility function
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Example usage
export async function runBasicTradingExample() {
  // Initialize QuantDesk client
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com',
    programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    wallet: wallet // Your Solana wallet instance
  });

  // Create and run the example
  const example = new BasicTradingExample(client);
  
  try {
    await example.runExample();
  } catch (error) {
    console.error('Example failed:', error);
  }
}

// Export for use in other modules
export { BasicTradingExample };
