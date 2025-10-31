// QuantDesk Protocol Integration for MIKEY-AI
// Enhanced integration with wallet checking, portfolio analysis, and trading

import fetch from 'node-fetch';
import { DynamicTool } from 'langchain/tools';

export class QuantDeskProtocolTools {
  private readonly baseURL: string;

  constructor(baseURL?: string) {
    this.baseURL = baseURL || process.env.QUANTDESK_URL || 'http://localhost:3002';
  }

  static createCheckQuantDeskPortfolioTool(): DynamicTool {
    return new DynamicTool({
      name: 'check_quantdesk_portfolio',
      description: 'Check QuantDesk user portfolio balances and health',
      func: async (input: string) => {
        try {
          const baseURL = process.env.QUANTDESK_URL || 'http://localhost:3002';
          const body = JSON.parse(input || '{}');
          const wallet = body.wallet || body.wallet_address;
          if (!wallet) {
            return JSON.stringify({ success: false, error: 'wallet (public key) required' });
          }
          const res = await fetch(`${baseURL}/api/dev/user-portfolio/${wallet}`);
          const data = await res.json();
          return JSON.stringify(data);
        } catch (error: any) {
          return JSON.stringify({ success: false, error: error?.message || String(error) });
        }
      }
    });
  }

  static createGetQuantDeskMarketDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_quantdesk_market_data',
      description: 'Get QuantDesk market summary (live Pyth prices, volume, OI, leverage)',
      func: async (_input: string) => {
        try {
          const baseURL = process.env.QUANTDESK_URL || 'http://localhost:3002';

          // Aggregated market summary (uses pythOracleService.getAllPrices under the hood)
          const summaryResponse = await fetch(`${baseURL}/api/dev/market-summary`);
          const summaryData = await summaryResponse.json();

          return JSON.stringify({ success: true, data: summaryData });
        } catch (error: any) {
          return JSON.stringify({ success: false, error: `Error getting market summary: ${error?.message || String(error)}` });
        }
      }
    });
  }

  static createGetLivePriceTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_live_price',
      description: 'Get a single live asset price from QuantDesk oracle (Pyth with CoinGecko fallback). Input: { "asset": "SOL" }',
      func: async (input: string) => {
        try {
          const baseURL = process.env.QUANTDESK_URL || 'http://localhost:3002';
          const body = JSON.parse(input || '{}');
          const asset = (body.asset || 'SOL').toUpperCase();
          const res = await fetch(`${baseURL}/api/oracle/price/${asset}`);
          const data = await res.json();
          return JSON.stringify(data);
        } catch (error: any) {
          return JSON.stringify({ success: false, error: `Error getting live price: ${error?.message || String(error)}` });
        }
      }
    });
  }

  static createPlaceQuantDeskTradeTool(): DynamicTool {
    return new DynamicTool({
      name: 'place_quantdesk_trade',
      description: 'Place a trade via QuantDesk backend. Input: { "wallet":"...", "symbol":"BTC-PERP", "side":"buy|sell", "size":0.1, "orderType":"market|limit", "price"?:number, "leverage"?:number }',
      func: async (input: string) => {
        try {
          const baseURL = process.env.QUANTDESK_URL || 'http://localhost:3002';
          const payload = JSON.parse(input || '{}');
          const res = await fetch(`${baseURL}/api/orders`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await res.json();
          return JSON.stringify(data);
        } catch (error: any) {
          return JSON.stringify({ success: false, error: `Error placing trade: ${error?.message || String(error)}` });
        }
      }
    });
  }

  /**
   * Get QuantDesk protocol health and metrics
   */
  static createGetQuantDeskProtocolHealthTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_quantdesk_protocol_health',
      description: 'Get QuantDesk protocol health, metrics, and system status',
      func: async (input: string) => {
        try {
          const baseURL = process.env.QUANTDESK_URL || 'http://localhost:3002';

          // Get health status
          const healthResponse = await fetch(`${baseURL}/health`);
          const healthData = await healthResponse.json();

          // Get protocol metrics
          const metricsResponse = await fetch(`${baseURL}/api/metrics`);
          const metricsData = await metricsResponse.json();

          // Get system status
          const statusResponse = await fetch(`${baseURL}/api/status`);
          const statusData = await statusResponse.json();

          return JSON.stringify({
            success: true,
            health: healthData,
            metrics: metricsData,
            status: statusData,
            protocol: 'QuantDesk',
            network: 'devnet',
            timestamp: new Date().toISOString()
          });
        } catch (error) {
          return JSON.stringify({
            success: false,
            error: `Error getting QuantDesk protocol health: ${error.message}`
          });
        }
      }
    });
  }

  /**
   * Analyze wallet risk and portfolio health
   */
  static createAnalyzeWalletRiskTool(): DynamicTool {
    return new DynamicTool({
      name: 'analyze_wallet_risk',
      description: 'Analyze wallet risk, portfolio health, and provide recommendations',
      func: async (input: string) => {
        try {
          const walletAddress = input.trim();
          const baseURL = process.env.QUANTDESK_URL || 'http://localhost:3002';

          // Get comprehensive wallet data
          const connection = new Connection(
            process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com',
            'confirmed'
          );

          let solBalance = 0;
          try {
            const publicKey = new PublicKey(walletAddress);
            const balance = await connection.getBalance(publicKey);
            solBalance = balance / LAMPORTS_PER_SOL;
          } catch (error) {
            // Wallet might not exist
          }

          // Get QuantDesk data
          const portfolioResponse = await fetch(`${baseURL}/api/portfolio/${walletAddress}`);
          const portfolioData = await portfolioResponse.json();

          const positionsResponse = await fetch(`${baseURL}/api/positions/${walletAddress}`);
          const positionsData = await positionsResponse.json();

          // Calculate risk metrics
          const portfolioAny = portfolioData as any;
          const totalEquity = portfolioAny?.totalEquity || 0;
          const totalPnL = portfolioAny?.totalPnL || 0;
          const marginRatio = portfolioAny?.marginRatio || 0;
          const liquidationPrice = portfolioAny?.liquidationPrice || null;

          // Risk assessment
          let riskLevel = 'LOW';
          let recommendations = [];

          if (marginRatio > 0.8) {
            riskLevel = 'HIGH';
            recommendations.push('Consider reducing position sizes to lower margin ratio');
          } else if (marginRatio > 0.6) {
            riskLevel = 'MEDIUM';
            recommendations.push('Monitor margin ratio closely');
          }

          if (totalPnL < -totalEquity * 0.1) {
            riskLevel = 'HIGH';
            recommendations.push('Significant unrealized losses detected - consider risk management');
          }

          if (!recommendations.length) {
            recommendations.push('Portfolio appears healthy - continue monitoring');
          }

          return JSON.stringify({
            success: true,
            wallet: walletAddress,
            riskAnalysis: {
              riskLevel,
              totalEquity,
              totalPnL,
              marginRatio,
              liquidationPrice,
              solBalance: solBalance.toFixed(6)
            },
            recommendations,
            portfolio: portfolioData,
            positions: positionsData,
            timestamp: new Date().toISOString()
          });
        } catch (error) {
          return JSON.stringify({
            success: false,
            error: `Error analyzing wallet risk: ${error.message}`,
            wallet: input.trim()
          });
        }
      }
    });
  }

  /**
   * Get all QuantDesk protocol tools
   */
  static getAllTools(): DynamicTool[] {
    return [
      this.createCheckQuantDeskPortfolioTool(),
      this.createGetQuantDeskMarketDataTool(),
      this.createGetLivePriceTool(),
      this.createPlaceQuantDeskTradeTool(),
      this.createGetQuantDeskProtocolHealthTool(),
      this.createAnalyzeWalletRiskTool()
    ];
  }

  /**
   * Get all tools including demo mock tools (for demo mode)
   */
  static getAllToolsWithDemo(): DynamicTool[] {
    const { DemoMockTools } = require('./DemoMockTools');
    return [
      ...this.getAllTools(),
      ...DemoMockTools.getAllTools()
    ];
  }
}
