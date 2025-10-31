// QuantDesk Tools for Mikey AI - Hackathon Focus
import { DynamicTool } from '@langchain/core/tools';

export class QuantDeskTools {
  private readonly baseURL: string;
  private readonly apiKey?: string;

  constructor(baseURL?: string, apiKey?: string) {
    this.baseURL = baseURL || process.env.QUANTDESK_URL || 'http://localhost:3002';
    this.apiKey = apiKey;
  }

  /**
   * Get all available QuantDesk tools
   */
  static getAllTools(): DynamicTool[] {
    const tools = new QuantDeskTools();
    return [
      tools.createMarketDataTool(),
      tools.createPriceDataTool(),
      tools.createAccountTool(),
      tools.createTradingTool(),
      tools.createHealthTool()
    ];
  }

  /**
   * Market Data Tool - Get available markets
   */
  private createMarketDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_markets',
      description: 'Get available trading markets from QuantDesk',
      func: async (input: string) => {
        try {
          const response = await fetch(`${this.baseURL}/api/markets`);
          const data = await response.json() as any;
          
          if (data.success && data.markets) {
            return JSON.stringify({
              success: true,
              markets: data.markets.map((market: any) => ({
                symbol: market.symbol,
                baseAsset: market.baseAsset,
                quoteAsset: market.quoteAsset,
                type: market.type
              })),
              count: data.markets.length
            });
          }
          
          return JSON.stringify({ success: false, error: 'No markets found' });
        } catch (error) {
          return JSON.stringify({ success: false, error: error.message });
        }
      }
    });
  }

  /**
   * Price Data Tool - Get current prices
   */
  private createPriceDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_prices',
      description: 'Get current prices for trading pairs from QuantDesk',
      func: async (input: string) => {
        try {
          const response = await fetch(`${this.baseURL}/api/prices`);
          const data = await response.json() as any;
          
          if (data.success && data.prices) {
            return JSON.stringify({
              success: true,
              prices: data.prices.map((price: any) => ({
                symbol: price.symbol,
                price: price.price,
                change24h: price.change24h,
                volume: price.volume
              })),
              timestamp: new Date().toISOString()
            });
          }
          
          return JSON.stringify({ success: false, error: 'No price data found' });
        } catch (error) {
          return JSON.stringify({ success: false, error: error.message });
        }
      }
    });
  }

  /**
   * Account Tool - Get account information (requires auth)
   */
  private createAccountTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_account_info',
      description: 'Get account information from QuantDesk (requires authentication)',
      func: async (input: string) => {
        try {
          const headers: any = {
            'Content-Type': 'application/json'
          };
          
          if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
          }

          const response = await fetch(`${this.baseURL}/api/account`, {
            headers
          });
          
          if (response.status === 401) {
            return JSON.stringify({ 
              success: false, 
              error: 'Authentication required',
              message: 'Please provide API key for account access'
            });
          }
          
          const data = await response.json();
          return JSON.stringify({ success: true, account: data });
        } catch (error) {
          return JSON.stringify({ success: false, error: error.message });
        }
      }
    });
  }

  /**
   * Trading Tool - Get orders and positions (requires auth)
   */
  private createTradingTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_trading_data',
      description: 'Get trading data (orders, positions) from QuantDesk (requires authentication)',
      func: async (input: string) => {
        try {
          const headers: any = {
            'Content-Type': 'application/json'
          };
          
          if (this.apiKey) {
            headers['Authorization'] = `Bearer ${this.apiKey}`;
          }

          // Try to get orders
          const ordersResponse = await fetch(`${this.baseURL}/api/orders`, { headers });
          const ordersData = ordersResponse.status === 200 ? await ordersResponse.json() : null;

          // Try to get positions
          const positionsResponse = await fetch(`${this.baseURL}/api/positions`, { headers });
          const positionsData = positionsResponse.status === 200 ? await positionsResponse.json() : null;

          return JSON.stringify({
            success: true,
            orders: ordersData,
            positions: positionsData,
            authRequired: ordersResponse.status === 401 || positionsResponse.status === 401
          });
        } catch (error) {
          return JSON.stringify({ success: false, error: error.message });
        }
      }
    });
  }

  /**
   * Health Tool - Check QuantDesk API health
   */
  private createHealthTool(): DynamicTool {
    return new DynamicTool({
      name: 'check_quantdesk_health',
      description: 'Check QuantDesk API health status',
      func: async (input: string) => {
        try {
          const response = await fetch(`${this.baseURL}/health`);
          const data = await response.json() as any;
          
          return JSON.stringify({
            success: true,
            status: data.status,
            uptime: data.uptime,
            environment: data.environment,
            timestamp: data.timestamp
          });
        } catch (error) {
          return JSON.stringify({ success: false, error: error.message });
        }
      }
    });
  }
}
