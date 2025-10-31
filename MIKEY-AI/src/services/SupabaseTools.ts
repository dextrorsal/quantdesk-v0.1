// Supabase Database Tools for Mikey AI - Hackathon Focus
import { DynamicTool } from '@langchain/core/tools';

export class SupabaseTools {
  private supabaseUrl: string;
  private supabaseKey: string;

  constructor(supabaseUrl?: string, supabaseKey?: string) {
    this.supabaseUrl = supabaseUrl || process.env.SUPABASE_URL || '';
    this.supabaseKey = supabaseKey || process.env.SUPABASE_ANON_KEY || '';
  }

  /**
   * Get all available Supabase tools
   */
  static getAllTools(): DynamicTool[] {
    const tools = new SupabaseTools();
    return [
      tools.createHistoricalDataTool(),
      tools.createNewsDataTool(),
      tools.createMarketDataTool(),
      tools.createUserDataTool()
    ];
  }

  /**
   * Historical Data Tool - Get price history from Supabase
   */
  private createHistoricalDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_historical_data',
      description: 'Get historical price data from Supabase database',
      func: async (input: string) => {
        try {
          // Parse input for symbol and timeframe
          const params = this.parseInput(input);
          const symbol = params.symbol || 'SOL';
          const timeframe = params.timeframe || '1h';
          const limit = params.limit || 100;

          const response = await fetch(`${this.supabaseUrl}/rest/v1/price_history`, {
            headers: {
              'apikey': this.supabaseKey,
              'Authorization': `Bearer ${this.supabaseKey}`,
              'Content-Type': 'application/json'
            },
            method: 'GET'
          });

          if (!response.ok) {
            return JSON.stringify({ 
              success: false, 
              error: `Database error: ${response.status}`,
              message: 'Check Supabase connection'
            });
          }

          const data = await response.json() as any[];
          return JSON.stringify({
            success: true,
            symbol,
            timeframe,
            data: data.slice(0, limit),
            count: data.length
          });
        } catch (error) {
          return JSON.stringify({ 
            success: false, 
            error: error.message,
            message: 'Supabase connection failed'
          });
        }
      }
    });
  }

  /**
   * News Data Tool - Get news from Supabase
   */
  private createNewsDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_news_data',
      description: 'Get news data from Supabase database',
      func: async (input: string) => {
        try {
          const params = this.parseInput(input);
          const limit = params.limit || 10;
          const symbol = params.symbol || '';

          let url = `${this.supabaseUrl}/rest/v1/news`;
          if (symbol) {
            url += `?symbol=eq.${symbol}`;
          }

          const response = await fetch(url, {
            headers: {
              'apikey': this.supabaseKey,
              'Authorization': `Bearer ${this.supabaseKey}`,
              'Content-Type': 'application/json'
            },
            method: 'GET'
          });

          if (!response.ok) {
            return JSON.stringify({ 
              success: false, 
              error: `Database error: ${response.status}`,
              message: 'Check Supabase connection'
            });
          }

          const data = await response.json() as any[];
          return JSON.stringify({
            success: true,
            news: data.slice(0, limit),
            count: data.length,
            symbol: symbol || 'all'
          });
        } catch (error) {
          return JSON.stringify({ 
            success: false, 
            error: error.message,
            message: 'Supabase connection failed'
          });
        }
      }
    });
  }

  /**
   * Market Data Tool - Get market data from Supabase
   */
  private createMarketDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_market_data',
      description: 'Get market data from Supabase database',
      func: async (input: string) => {
        try {
          const params = this.parseInput(input);
          const symbol = params.symbol || 'SOL';

          const response = await fetch(`${this.supabaseUrl}/rest/v1/market_data?symbol=eq.${symbol}`, {
            headers: {
              'apikey': this.supabaseKey,
              'Authorization': `Bearer ${this.supabaseKey}`,
              'Content-Type': 'application/json'
            },
            method: 'GET'
          });

          if (!response.ok) {
            return JSON.stringify({ 
              success: false, 
              error: `Database error: ${response.status}`,
              message: 'Check Supabase connection'
            });
          }

          const data = await response.json() as any[];
          return JSON.stringify({
            success: true,
            symbol,
            marketData: data,
            count: data.length
          });
        } catch (error) {
          return JSON.stringify({ 
            success: false, 
            error: error.message,
            message: 'Supabase connection failed'
          });
        }
      }
    });
  }

  /**
   * User Data Tool - Get user data from Supabase
   */
  private createUserDataTool(): DynamicTool {
    return new DynamicTool({
      name: 'get_user_data',
      description: 'Get user data from Supabase database',
      func: async (input: string) => {
        try {
          const params = this.parseInput(input);
          const userId = params.userId || 'current';

          const response = await fetch(`${this.supabaseUrl}/rest/v1/users?id=eq.${userId}`, {
            headers: {
              'apikey': this.supabaseKey,
              'Authorization': `Bearer ${this.supabaseKey}`,
              'Content-Type': 'application/json'
            },
            method: 'GET'
          });

          if (!response.ok) {
            return JSON.stringify({ 
              success: false, 
              error: `Database error: ${response.status}`,
              message: 'Check Supabase connection'
            });
          }

          const data = await response.json();
          return JSON.stringify({
            success: true,
            user: data[0] || null,
            userId
          });
        } catch (error) {
          return JSON.stringify({ 
            success: false, 
            error: error.message,
            message: 'Supabase connection failed'
          });
        }
      }
    });
  }

  /**
   * Parse input string for parameters
   */
  private parseInput(input: string): any {
    const params: any = {};
    
    // Simple parameter parsing
    if (input.includes('symbol=')) {
      const match = input.match(/symbol=([a-zA-Z0-9-]+)/);
      if (match) params.symbol = match[1];
    }
    
    if (input.includes('timeframe=')) {
      const match = input.match(/timeframe=([a-zA-Z0-9]+)/);
      if (match) params.timeframe = match[1];
    }
    
    if (input.includes('limit=')) {
      const match = input.match(/limit=(\d+)/);
      if (match) params.limit = parseInt(match[1]);
    }
    
    if (input.includes('userId=')) {
      const match = input.match(/userId=([a-zA-Z0-9-]+)/);
      if (match) params.userId = match[1];
    }
    
    return params;
  }
}
