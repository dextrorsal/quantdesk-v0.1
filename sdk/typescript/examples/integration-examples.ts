// QuantDesk Integration Examples
// Realistic integration examples that users can actually implement

import { QuantDeskClient, QuantDeskAPIClient, QuantDeskSecurity } from '@quantdesk/sdk';

/**
 * Integration Examples - Realistic Implementation
 * This example demonstrates practical integration patterns that users can actually implement
 */
export class QuantDeskIntegration {
  private client: QuantDeskClient;
  private apiClient: QuantDeskAPIClient;
  private security: QuantDeskSecurity;

  constructor(client: QuantDeskClient) {
    this.client = client;
    this.apiClient = new QuantDeskAPIClient(client);
    this.security = new QuantDeskSecurity(client);
  }

  /**
   * Initialize all components
   */
  async initialize(): Promise<void> {
    try {
      console.log('🚀 Initializing QuantDesk Integration...');
      
      await this.apiClient.initialize();
      console.log('✅ Integration initialized successfully');
      
    } catch (error) {
      console.error('❌ Failed to initialize integration:', error);
      throw error;
    }
  }

  /**
   * Example 1: Basic Market Data Integration
   */
  async basicMarketDataIntegration(): Promise<void> {
    try {
      console.log('\n📊 Basic Market Data Integration Example');
      
      // Get available markets
      const markets = await this.apiClient.getMarkets();
      console.log(`Available markets: ${markets.length}`);
      
      // Get market data for each market
      for (const market of markets.slice(0, 3)) { // Limit to first 3 for demo
        const marketData = await this.apiClient.getMarketData(market.symbol);
        console.log(`${market.symbol}: $${marketData.price.toFixed(2)}`);
      }
      
    } catch (error) {
      console.error('❌ Basic market data integration failed:', error);
    }
  }

  /**
   * Example 2: Portfolio Monitoring Integration
   */
  async portfolioMonitoringIntegration(): Promise<void> {
    try {
      console.log('\n💼 Portfolio Monitoring Integration Example');
      
      // Get portfolio overview
      const portfolio = await this.apiClient.getPortfolio();
      console.log(`Portfolio Value: $${portfolio.totalValue.toFixed(2)}`);
      console.log(`Total PnL: $${portfolio.totalPnL.toFixed(2)}`);
      
      // Get positions
      const positions = await this.apiClient.getPositions();
      console.log(`Open Positions: ${positions.length}`);
      
      // Display position details
      positions.forEach((position, index) => {
        console.log(`${index + 1}. ${position.market}: ${position.side} ${position.size} @ $${position.entryPrice}`);
      });
      
    } catch (error) {
      console.error('❌ Portfolio monitoring integration failed:', error);
    }
  }

  /**
   * Example 3: AI Integration
   */
  async aiIntegrationExample(): Promise<void> {
    try {
      console.log('\n🤖 AI Integration Example');
      
      // Get AI analysis for SOL-PERP
      const analysis = await this.apiClient.getAIAnalysis('SOL-PERP');
      console.log(`SOL-PERP Analysis:`);
      console.log(`  Sentiment: ${analysis.sentiment}`);
      console.log(`  Confidence: ${(analysis.confidence * 100).toFixed(1)}%`);
      console.log(`  Recommendation: ${analysis.recommendation}`);
      
      // Get trading signals
      const signals = await this.apiClient.getTradingSignals();
      console.log(`Trading Signals: ${signals.length}`);
      
      // Chat with MIKEY
      const chatResponse = await this.apiClient.chatWithMIKEY('What is your analysis of the current market?');
      console.log(`MIKEY Response: ${chatResponse.response.substring(0, 100)}...`);
      
    } catch (error) {
      console.error('❌ AI integration failed:', error);
    }
  }

  /**
   * Example 4: Order Management Integration
   */
  async orderManagementIntegration(): Promise<void> {
    try {
      console.log('\n📝 Order Management Integration Example');
      
      // Get current market data
      const marketData = await this.apiClient.getMarketData('SOL-PERP');
      const currentPrice = marketData.price;
      
      // Place a small test order
      const orderData = {
        market: 'SOL-PERP',
        side: 'buy' as const,
        size: 0.01, // Very small size for testing
        price: currentPrice * 0.99, // 1% below current price
        orderType: 'limit' as const
      };
      
      console.log(`Placing test order: ${orderData.size} ${orderData.market} @ $${orderData.price.toFixed(2)}`);
      
      const order = await this.apiClient.placeOrder(orderData);
      console.log(`Order placed: ${order.id}`);
      
      // Monitor order status
      await this.monitorOrderStatus(order.id);
      
    } catch (error) {
      console.error('❌ Order management integration failed:', error);
    }
  }

  /**
   * Monitor order status
   */
  private async monitorOrderStatus(orderId: string): Promise<void> {
    try {
      const maxChecks = 5;
      let checks = 0;
      
      while (checks < maxChecks) {
        const order = await this.apiClient.getOrder(orderId);
        console.log(`Order ${orderId} status: ${order.status}`);
        
        if (order.status === 'filled' || order.status === 'cancelled') {
          break;
        }
        
        await this.sleep(3000); // Wait 3 seconds
        checks++;
      }
      
    } catch (error) {
      console.error('❌ Order monitoring failed:', error);
    }
  }

  /**
   * Example 5: Security Integration
   */
  async securityIntegrationExample(): Promise<void> {
    try {
      console.log('\n🔒 Security Integration Example');
      
      // Validate market symbol
      this.security.validateMarketSymbol('SOL-PERP');
      console.log('✅ Market symbol validation passed');
      
      // Validate order data
      const orderData = {
        market: 'SOL-PERP',
        side: 'buy',
        size: 1.0,
        price: 100.0,
        orderType: 'limit'
      };
      
      this.security.validateOrderData(orderData);
      console.log('✅ Order data validation passed');
      
      // Check rate limit
      const rateLimitOk = this.security.checkRateLimit('placeOrder');
      console.log('✅ Rate limit check:', rateLimitOk);
      
      // Sanitize input
      const sanitizedInput = this.security.sanitizeInput('<script>alert("test")</script>');
      console.log('✅ Input sanitization:', sanitizedInput);
      
      // Get security status
      const securityStatus = this.security.getSecurityStatus();
      console.log('✅ Security status:', securityStatus);
      
    } catch (error) {
      console.error('❌ Security integration failed:', error);
    }
  }

  /**
   * Example 6: Error Handling Integration
   */
  async errorHandlingIntegration(): Promise<void> {
    try {
      console.log('\n⚠️ Error Handling Integration Example');
      
      // Test with invalid market symbol
      try {
        await this.apiClient.getMarketData('INVALID-MARKET');
      } catch (error) {
        console.log('✅ Caught expected error for invalid market:', error.message);
      }
      
      // Test with invalid order data
      try {
        await this.apiClient.placeOrder({
          market: 'SOL-PERP',
          side: 'invalid' as any,
          size: -1,
          price: 0,
          orderType: 'limit'
        });
      } catch (error) {
        console.log('✅ Caught expected error for invalid order:', error.message);
      }
      
      // Test with non-existent order ID
      try {
        await this.apiClient.getOrder('non-existent-order-id');
      } catch (error) {
        console.log('✅ Caught expected error for non-existent order:', error.message);
      }
      
    } catch (error) {
      console.error('❌ Error handling integration failed:', error);
    }
  }

  /**
   * Example 7: Real-time Data Integration
   */
  async realtimeDataIntegration(): Promise<void> {
    try {
      console.log('\n📡 Real-time Data Integration Example');
      
      // Monitor market data for 30 seconds
      const markets = ['SOL-PERP', 'ETH-PERP'];
      const startTime = Date.now();
      const duration = 30000; // 30 seconds
      
      console.log(`Monitoring ${markets.join(', ')} for ${duration / 1000} seconds...`);
      
      const monitor = async () => {
        if (Date.now() - startTime > duration) {
          console.log('✅ Real-time monitoring completed');
          return;
        }
        
        for (const market of markets) {
          try {
            const marketData = await this.apiClient.getMarketData(market);
            console.log(`${market}: $${marketData.price.toFixed(2)} (${(marketData.change24h * 100).toFixed(2)}%)`);
          } catch (error) {
            console.error(`❌ Failed to get data for ${market}:`, error.message);
          }
        }
        
        // Wait 5 seconds before next update
        setTimeout(monitor, 5000);
      };
      
      monitor();
      
    } catch (error) {
      console.error('❌ Real-time data integration failed:', error);
    }
  }

  /**
   * Example 8: Complete Trading Workflow
   */
  async completeTradingWorkflow(): Promise<void> {
    try {
      console.log('\n🔄 Complete Trading Workflow Example');
      
      // Step 1: Get market data
      const marketData = await this.apiClient.getMarketData('SOL-PERP');
      console.log(`Current SOL-PERP price: $${marketData.price.toFixed(2)}`);
      
      // Step 2: Get AI analysis
      const analysis = await this.apiClient.getAIAnalysis('SOL-PERP');
      console.log(`AI Analysis: ${analysis.sentiment} (${(analysis.confidence * 100).toFixed(1)}% confidence)`);
      
      // Step 3: Get portfolio status
      const portfolio = await this.apiClient.getPortfolio();
      console.log(`Portfolio Value: $${portfolio.totalValue.toFixed(2)}`);
      
      // Step 4: Place order based on AI recommendation
      if (analysis.recommendation === 'buy' && analysis.confidence > 0.7) {
        const orderData = {
          market: 'SOL-PERP',
          side: 'buy' as const,
          size: 0.01, // Small size for testing
          price: marketData.price * 0.99, // 1% below current price
          orderType: 'limit' as const
        };
        
        console.log('Placing order based on AI recommendation...');
        const order = await this.apiClient.placeOrder(orderData);
        console.log(`Order placed: ${order.id}`);
        
        // Step 5: Monitor order
        await this.monitorOrderStatus(order.id);
      } else {
        console.log('AI recommendation not strong enough for trading');
      }
      
    } catch (error) {
      console.error('❌ Complete trading workflow failed:', error);
    }
  }

  /**
   * Run all integration examples
   */
  async runAllExamples(): Promise<void> {
    try {
      console.log('🚀 Running All QuantDesk Integration Examples');
      
      await this.initialize();
      
      await this.basicMarketDataIntegration();
      await this.portfolioMonitoringIntegration();
      await this.aiIntegrationExample();
      await this.orderManagementIntegration();
      await this.securityIntegrationExample();
      await this.errorHandlingIntegration();
      await this.realtimeDataIntegration();
      await this.completeTradingWorkflow();
      
      console.log('\n✅ All integration examples completed successfully');
      
    } catch (error) {
      console.error('❌ Integration examples failed:', error);
    }
  }

  /**
   * Sleep utility function
   */
  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Example usage - Realistic implementation
export async function runIntegrationExamples() {
  // Initialize QuantDesk client
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com',
    programId: 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    wallet: wallet // Your Solana wallet instance
  });

  // Create integration example
  const integration = new QuantDeskIntegration(client);
  
  try {
    // Run all examples
    await integration.runAllExamples();
  } catch (error) {
    console.error('Integration examples failed:', error);
  }
}

// Export for use in other modules
export { QuantDeskIntegration };
