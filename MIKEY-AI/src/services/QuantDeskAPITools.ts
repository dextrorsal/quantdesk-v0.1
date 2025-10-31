import { DynamicTool } from '@langchain/core/tools';
import { config } from '@/config';

/**
 * Simplified QuantDesk API Tools for Mikey AI
 * These tools call your existing QuantDesk backend APIs instead of recreating data collection
 */

export class QuantDeskAPITools {
  
  /**
   * Get current prices from QuantDesk oracle
   */
  static createPriceTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_quantdesk_prices',
      description: 'Get current cryptocurrency prices from QuantDesk oracle service',
      func: async (input: string) => {
        try {
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/oracle/prices`);
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error fetching prices: ${error}`;
        }
      }
    });
  }

  /**
   * Get specific asset price
   */
  static createAssetPriceTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_asset_price',
      description: 'Get price for a specific asset (BTC, ETH, SOL, etc.)',
      func: async (input: string) => {
        try {
          const asset = input.trim().toUpperCase();
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/oracle/price/${asset}`);
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error fetching ${input} price: ${error}`;
        }
      }
    });
  }

  /**
   * Get market metrics from QuantDesk
   */
  static createMarketMetricsTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_market_metrics',
      description: 'Get QuantDesk market metrics and trading statistics',
      func: async (input: string) => {
        try {
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/grafana/metrics`);
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error fetching market metrics: ${error}`;
        }
      }
    });
  }

  /**
   * Get portfolio analytics
   */
  static createPortfolioAnalyticsTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_portfolio_analytics',
      description: 'Get portfolio analytics and performance metrics',
      func: async (input: string) => {
        try {
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/portfolio-analytics`);
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error fetching portfolio analytics: ${error}`;
        }
      }
    });
  }

  /**
   * Get all available tools
   */
  static getAllTools(): DynamicTool[] {
    return [
      this.createPriceTool(),
      this.createAssetPriceTool(),
      this.createMarketMetricsTool(),
      this.createPortfolioAnalyticsTool()
    ];
  }
}
