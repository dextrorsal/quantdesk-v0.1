// QuantDesk Basic Trading Example
// Realistic example of QuantDesk SDK usage for basic trading operations

import { QuantDeskClient, MarketData, Position, Order } from '@quantdesk/sdk';

/**
 * Basic Trading Example - Realistic Implementation
 * This example demonstrates practical QuantDesk SDK usage that users can actually implement
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
      
      // Step 4: Place a simple order
      await this.placeSimpleOrder();
      
      // Step 5: Monitor positions
      await this.monitorPositions();
      
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
   * Get market information - Realistic implementation
   */
  private async getMarketInformation(): Promise<void> {
    console.log('\n📊 Getting Market Information...');
    
    try {
      // Get available markets
      const markets = await this.client.getMarkets();
      console.log(`📈 Available markets: ${markets.length}`);
      
      // Show first 3 markets (realistic for demo)
      markets.slice(0, 3).forEach(market => {
        console.log(`  - ${market.symbol}: ${market.name}`);
      });

      // Get detailed market data for SOL-PERP (most common)
      const marketData = await this.client.getMarketData('SOL-PERP');
      console.log(`\n🔍 SOL-PERP Market Data:`);
      console.log(`  Price: $${marketData.price.toFixed(2)}`);
      console.log(`  Volume: ${marketData.volume.toFixed(2)}`);
      console.log(`  Change 24h: ${(marketData.change24h * 100).toFixed(2)}%`);
      
    } catch (error) {
      console.error('❌ Failed to get market information:', error);
      throw error;
    }
  }

  /**
   * Get portfolio overview - Realistic implementation
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

      // Get individual positions (realistic for demo)
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
   * Place a simple order - Realistic implementation
   */
  private async placeSimpleOrder(): Promise<void> {
    console.log('\n📝 Placing Simple Order...');
    
    try {
      // Get current market data
      const marketData = await this.client.getMarketData('SOL-PERP');
      const currentPrice = marketData.price;
      
      // Place a small limit order (realistic size)
      const orderData = {
        market: 'SOL-PERP',
        side: 'buy' as const,
        size: 0.01, // Small size for demo
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

      // Monitor order status (realistic timeout)
      await this.monitorOrderStatus(order.id);

    } catch (error) {
      console.error('❌ Failed to place simple order:', error);
      throw error;
    }
  }

  /**
   * Monitor order status - Realistic implementation
   */
  private async monitorOrderStatus(orderId: string): Promise<void> {
    console.log(`\n👀 Monitoring Order Status: ${orderId}`);
    
    try {
      const maxChecks = 5; // Realistic number of checks
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

        // Wait 3 seconds before next check (realistic interval)
        await this.sleep(3000);
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
   * Monitor positions - Realistic implementation
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
        console.log(`  Leverage: ${position.leverage}x`);
      }
    } catch (error) {
      console.error('❌ Failed to monitor positions:', error);
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
export async function runBasicTradingExample() {
  // Initialize QuantDesk client with realistic configuration
  const client = new QuantDeskClient({
    rpcUrl: 'https://api.devnet.solana.com', // Use devnet for testing
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
