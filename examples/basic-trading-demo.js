/**
 * QuantDesk Demo - Basic Trading Example
 * 
 * This example demonstrates how to connect to QuantDesk API
 * and perform basic trading operations.
 * 
 * Note: This is a simplified example for demonstration purposes.
 * Production applications should include proper error handling,
 * rate limiting, and security measures.
 */

const axios = require('axios');

class QuantDeskDemo {
  constructor(apiUrl = 'http://localhost:3002', apiKey = null) {
    this.apiUrl = apiUrl;
    this.apiKey = apiKey;
    this.authToken = null;
  }

  // Set authentication token
  setAuthToken(token) {
    this.authToken = token;
  }

  // Get headers for authenticated requests
  getHeaders() {
    const headers = {
      'Content-Type': 'application/json'
    };
    
    if (this.authToken) {
      headers['Authorization'] = `Bearer ${this.authToken}`;
    }
    
    return headers;
  }

  // 1. Get available markets
  async getMarkets() {
    try {
      console.log('📊 Fetching available markets...');
      const response = await axios.get(`${this.apiUrl}/api/supabase-oracle/markets`);
      
      if (response.data.success) {
        console.log(`✅ Found ${response.data.data.length} markets:`);
        response.data.data.forEach((market, index) => {
          console.log(`   ${index + 1}. ${market.symbol} (${market.base_asset}/${market.quote_asset})`);
          console.log(`      Max Leverage: ${market.max_leverage}x`);
          console.log(`      Min Order Size: ${market.min_order_size}`);
          console.log(`      Tick Size: ${market.tick_size}`);
        });
        return response.data.data;
      } else {
        console.log('❌ Failed to fetch markets:', response.data.error);
        return [];
      }
    } catch (error) {
      console.error('❌ Error fetching markets:', error.message);
      return [];
    }
  }

  // 2. Get portfolio metrics (requires authentication)
  async getPortfolioMetrics() {
    try {
      console.log('📈 Fetching portfolio metrics...');
      const response = await axios.get(`${this.apiUrl}/api/portfolio/metrics`, {
        headers: this.getHeaders()
      });
      
      if (response.data.success) {
        const metrics = response.data.data;
        console.log('✅ Portfolio Metrics:');
        console.log(`   Total Value: $${metrics.totalValue.toLocaleString()}`);
        console.log(`   Total PnL: $${metrics.totalPnL.toLocaleString()}`);
        console.log(`   Total PnL %: ${metrics.totalPnLPercentage.toFixed(2)}%`);
        console.log(`   Day PnL: $${metrics.dayPnL.toLocaleString()}`);
        console.log(`   Day PnL %: ${metrics.dayPnLPercentage.toFixed(2)}%`);
        console.log(`   Sharpe Ratio: ${metrics.sharpeRatio.toFixed(2)}`);
        console.log(`   Max Drawdown: ${(metrics.maxDrawdown * 100).toFixed(2)}%`);
        console.log(`   Win Rate: ${(metrics.winRate * 100).toFixed(2)}%`);
        console.log(`   Total Trades: ${metrics.totalTrades}`);
        console.log(`   Avg Trade Size: ${metrics.avgTradeSize.toFixed(3)}`);
        return metrics;
      } else {
        console.log('❌ Failed to fetch portfolio metrics:', response.data.error);
        return null;
      }
    } catch (error) {
      if (error.response?.status === 401) {
        console.log('⚠️ Portfolio metrics requires authentication');
      } else {
        console.error('❌ Error fetching portfolio metrics:', error.message);
      }
      return null;
    }
  }

  // 3. Get risk analysis (requires authentication)
  async getRiskAnalysis() {
    try {
      console.log('⚠️ Fetching risk analysis...');
      const response = await axios.get(`${this.apiUrl}/api/portfolio/risk`, {
        headers: this.getHeaders()
      });
      
      if (response.data.success) {
        const risk = response.data.data;
        console.log('✅ Risk Analysis:');
        console.log(`   Portfolio VaR: ${(risk.portfolioVaR * 100).toFixed(2)}%`);
        console.log(`   Portfolio CVaR: ${(risk.portfolioCVaR * 100).toFixed(2)}%`);
        console.log(`   Max Drawdown: ${(risk.maxDrawdown * 100).toFixed(2)}%`);
        console.log(`   Current Drawdown: ${(risk.currentDrawdown * 100).toFixed(2)}%`);
        console.log(`   Leverage Ratio: ${risk.leverageRatio.toFixed(2)}x`);
        console.log(`   Concentration Risk: ${(risk.concentrationRisk * 100).toFixed(2)}%`);
        console.log(`   Correlation Risk: ${(risk.correlationRisk * 100).toFixed(2)}%`);
        console.log(`   Overall Risk Score: ${risk.overallRiskScore.toFixed(1)}/100`);
        return risk;
      } else {
        console.log('❌ Failed to fetch risk analysis:', response.data.error);
        return null;
      }
    } catch (error) {
      if (error.response?.status === 401) {
        console.log('⚠️ Risk analysis requires authentication');
      } else {
        console.error('❌ Error fetching risk analysis:', error.message);
      }
      return null;
    }
  }

  // 4. Get JIT liquidity statistics
  async getJITLiquidityStats() {
    try {
      console.log('💧 Fetching JIT liquidity statistics...');
      const response = await axios.get(`${this.apiUrl}/api/jit-liquidity/stats`, {
        headers: this.getHeaders()
      });
      
      if (response.data.success) {
        const stats = response.data.data;
        console.log('✅ JIT Liquidity Statistics:');
        console.log(`   Active Auctions: ${stats.activeAuctions}`);
        console.log(`   Total Market Makers: ${stats.totalMarketMakers}`);
        console.log(`   Total Volume: $${stats.totalVolume.toLocaleString()}`);
        console.log(`   Total Fees: $${stats.totalFees.toLocaleString()}`);
        console.log(`   Price Improvements: ${stats.totalPriceImprovements}`);
        console.log(`   Avg Price Improvement: ${stats.averagePriceImprovement.toFixed(2)}%`);
        console.log(`   Liquidity Mining Programs: ${stats.liquidityMiningPrograms}`);
        console.log(`   Market Making Strategies: ${stats.marketMakingStrategies}`);
        return stats;
      } else {
        console.log('❌ Failed to fetch JIT liquidity stats:', response.data.error);
        return null;
      }
    } catch (error) {
      if (error.response?.status === 401) {
        console.log('⚠️ JIT liquidity stats requires authentication');
      } else {
        console.error('❌ Error fetching JIT liquidity stats:', error.message);
      }
      return null;
    }
  }

  // 5. Get liquidity mining programs
  async getLiquidityMiningPrograms() {
    try {
      console.log('⛏️ Fetching liquidity mining programs...');
      const response = await axios.get(`${this.apiUrl}/api/jit-liquidity/liquidity-mining`);
      
      if (response.data.success) {
        const data = response.data.data;
        console.log('✅ Liquidity Mining Programs:');
        console.log(`   Total Programs: ${data.totalPrograms}`);
        
        data.programs.forEach((program, index) => {
          console.log(`   ${index + 1}. ${program.name}:`);
          console.log(`      Description: ${program.description}`);
          console.log(`      Total Rewards: ${program.totalRewards.toLocaleString()} ${program.currency}`);
          console.log(`      Markets: ${program.marketIds.join(', ')}`);
          console.log(`      Status: ${program.status}`);
          console.log(`      Participants: ${program.participants.length}`);
          console.log(`      Rules: ${program.rules.length} tiers`);
        });
        return data.programs;
      } else {
        console.log('❌ Failed to fetch liquidity mining programs:', response.data.error);
        return [];
      }
    } catch (error) {
      console.error('❌ Error fetching liquidity mining programs:', error.message);
      return [];
    }
  }

  // 6. Demo WebSocket connection
  async demoWebSocket() {
    try {
      console.log('🔌 Testing WebSocket connection...');
      
      const WebSocket = require('ws');
      const ws = new WebSocket('ws://localhost:3002/ws');
      
      ws.on('open', () => {
        console.log('✅ WebSocket connected successfully');
        
        // Subscribe to market data
        ws.send(JSON.stringify({
          type: 'subscribe',
          channel: 'market_data',
          marketId: 'd87a99b4-148a-49c2-a2ad-ca1ee17a9372'
        }));
        
        console.log('📡 Subscribed to market data updates');
        
        // Close connection after 5 seconds
        setTimeout(() => {
          ws.close();
          console.log('🔌 WebSocket connection closed');
        }, 5000);
      });
      
      ws.on('message', (data) => {
        const message = JSON.parse(data);
        console.log('📨 Received WebSocket message:', message);
      });
      
      ws.on('error', (error) => {
        console.error('❌ WebSocket error:', error.message);
      });
      
    } catch (error) {
      console.error('❌ Error testing WebSocket:', error.message);
    }
  }

  // Run complete demo
  async runDemo() {
    console.log('🚀 QuantDesk API Demo Starting...\n');
    
    // Test public endpoints
    await this.getMarkets();
    console.log('\n');
    
    await this.getLiquidityMiningPrograms();
    console.log('\n');
    
    // Test authenticated endpoints (will show auth required messages)
    await this.getPortfolioMetrics();
    console.log('\n');
    
    await this.getRiskAnalysis();
    console.log('\n');
    
    await this.getJITLiquidityStats();
    console.log('\n');
    
    // Test WebSocket
    await this.demoWebSocket();
    
    console.log('\n🎉 Demo completed!');
    console.log('\n📚 Next Steps:');
    console.log('   1. Set up authentication to access protected endpoints');
    console.log('   2. Explore the full API documentation');
    console.log('   3. Build your own trading application');
    console.log('   4. Join our community for support and updates');
  }
}

// Run the demo
async function main() {
  const demo = new QuantDeskDemo();
  await demo.runDemo();
}

// Export for use in other projects
module.exports = QuantDeskDemo;

// Run if called directly
if (require.main === module) {
  main().catch(console.error);
}
