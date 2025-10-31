import { DynamicTool } from '@langchain/core/tools';
import { config } from '../config';

/**
 * QuantDesk Trading Tools for Mikey AI
 * These tools allow Mikey AI to interact with your perpetual trading platform
 * - Place orders, manage positions, check account state
 * - All through your existing QuantDesk API
 */

export class QuantDeskTradingTools {
  
  /**
   * Get account state and balance
   */
  static createAccountStateTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_account_state',
      description: 'Get user account state, balances, and trading permissions',
      func: async (input: string) => {
        try {
          const userId = input.trim();
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/account-state/${userId}`, {
            headers: {
              'Authorization': `Bearer ${config.api.jwtSecret}`,
              'Content-Type': 'application/json'
            }
          });
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error getting account state: ${error}`;
        }
      }
    });
  }

  /**
   * Get user positions
   */
  static createGetPositionsTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_positions',
      description: 'Get all open positions for the user',
      func: async (input: string) => {
        try {
          const userId = input.trim();
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/positions`, {
            headers: {
              'Authorization': `Bearer ${config.api.jwtSecret}`,
              'Content-Type': 'application/json'
            }
          });
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error getting positions: ${error}`;
        }
      }
    });
  }

  /**
   * Place a trading order
   */
  static createPlaceOrderTool(): DynamicTool {
    return new DynamicTool({
      name: 'place_order',
      description: 'Place a trading order (buy/sell, market/limit)',
      func: async (input: string) => {
        try {
          const orderData = JSON.parse(input);
          const { symbol, side, size, orderType, price, leverage } = orderData;
          
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/orders`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${config.api.jwtSecret}`,
              'Content-Type': 'application/json'
            },
            body: JSON.stringify({
              symbol,
              side,
              size: parseFloat(size),
              orderType,
              price: price ? parseFloat(price) : undefined,
              leverage: leverage ? parseInt(leverage) : undefined
            })
          });
          
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error placing order: ${error}`;
        }
      }
    });
  }

  /**
   * Cancel an order
   */
  static createCancelOrderTool(): DynamicTool {
    return new DynamicTool({
      name: 'cancel_order',
      description: 'Cancel an existing order by ID',
      func: async (input: string) => {
        try {
          const orderId = input.trim();
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/orders/${orderId}/cancel`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${config.api.jwtSecret}`,
              'Content-Type': 'application/json'
            }
          });
          
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error canceling order: ${error}`;
        }
      }
    });
  }

  /**
   * Get order history
   */
  static createGetOrdersTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_orders',
      description: 'Get order history and status',
      func: async (input: string) => {
        try {
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/orders`, {
            headers: {
              'Authorization': `Bearer ${config.api.jwtSecret}`,
              'Content-Type': 'application/json'
            }
          });
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error getting orders: ${error}`;
        }
      }
    });
  }

  /**
   * Get market data
   */
  static createGetMarketsTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_markets',
      description: 'Get available trading markets and their data',
      func: async (input: string) => {
        try {
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/markets`);
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error getting markets: ${error}`;
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
      description: 'Get portfolio performance analytics and metrics',
      func: async (input: string) => {
        try {
          const userId = input.trim();
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/portfolio-analytics/${userId}`, {
            headers: {
              'Authorization': `Bearer ${config.api.jwtSecret}`,
              'Content-Type': 'application/json'
            }
          });
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error getting portfolio analytics: ${error}`;
        }
      }
    });
  }

  /**
   * Close a position
   */
  static createClosePositionTool(): DynamicTool {
    return new DynamicTool({
      name: 'close_position',
      description: 'Close an existing position by ID',
      func: async (input: string) => {
        try {
          const positionId = input.trim();
          const response = await fetch(`${config.api.quantdeskUrl || 'http://localhost:3002'}/api/positions/${positionId}/close`, {
            method: 'POST',
            headers: {
              'Authorization': `Bearer ${config.api.jwtSecret}`,
              'Content-Type': 'application/json'
            }
          });
          
          const data = await response.json();
          return JSON.stringify(data);
        } catch (error) {
          return `Error closing position: ${error}`;
        }
      }
    });
  }

  /**
   * Get all available trading tools
   */
  static getAllTools(): DynamicTool[] {
    return [
      this.createAccountStateTool(),
      this.createGetPositionsTool(),
      this.createPlaceOrderTool(),
      this.createCancelOrderTool(),
      this.createGetOrdersTool(),
      this.createGetMarketsTool(),
      this.createPortfolioAnalyticsTool(),
      this.createClosePositionTool()
    ];
  }
}