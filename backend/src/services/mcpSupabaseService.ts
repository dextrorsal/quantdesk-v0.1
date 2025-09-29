import { Logger } from '../utils/logger';

const logger = new Logger();

export interface Market {
  id: string;
  symbol: string;
  base_asset: string;
  quote_asset: string;
  program_id: string;
  market_account: string;
  oracle_account: string;
  is_active: boolean;
  max_leverage: number;
  initial_margin_ratio: number;
  maintenance_margin_ratio: number;
  tick_size: number;
  step_size: number;
  min_order_size: number;
  max_order_size: number;
  funding_interval: number;
  last_funding_time?: Date;
  current_funding_rate: number;
  created_at: Date;
  updated_at: Date;
  metadata: any;
}

export interface OraclePrice {
  id: string;
  market_id: string;
  price: number;
  confidence: number;
  exponent: number;
  created_at: Date;
}

export class MCPSupabaseService {
  private static instance: MCPSupabaseService;
  private readonly PROJECT_ID: string;

  private constructor() {
    this.PROJECT_ID = process.env.SUPABASE_PROJECT_ID || 'vabqtnsrmvccgegzvztv';
  }

  public static getInstance(): MCPSupabaseService {
    if (!MCPSupabaseService.instance) {
      MCPSupabaseService.instance = new MCPSupabaseService();
    }
    return MCPSupabaseService.instance;
  }

  /**
   * Health check - verify database connection
   */
  public async healthCheck(): Promise<boolean> {
    try {
      // Simple query to test connection
      const result = await this.executeQuery('SELECT 1 as test');
      return result && result.length > 0;
    } catch (error) {
      logger.error('Supabase health check failed:', error);
      return false;
    }
  }

  /**
   * Execute a SQL query using MCP tools
   * Note: This is a placeholder - in a real implementation, you'd need to
   * integrate with the MCP Supabase tools directly
   */
  private async executeQuery(sql: string, params?: any[]): Promise<any> {
    // This would be replaced with actual MCP tool calls
    // For now, we'll use a mock implementation
    logger.info(`Executing SQL: ${sql}`);
    return [];
  }

  /**
   * Get all active markets
   */
  public async getMarkets(): Promise<Market[]> {
    try {
      const sql = 'SELECT * FROM markets WHERE is_active = true ORDER BY symbol';
      const result = await this.executeQuery(sql);
      
      return result.map((row: any) => ({
        ...row,
        created_at: new Date(row.created_at),
        updated_at: new Date(row.updated_at),
        last_funding_time: row.last_funding_time ? new Date(row.last_funding_time) : undefined
      }));
    } catch (error) {
      logger.error('Error getting markets:', error);
      throw error;
    }
  }

  /**
   * Get market by symbol
   */
  public async getMarketBySymbol(symbol: string): Promise<Market | null> {
    try {
      const sql = 'SELECT * FROM markets WHERE symbol = $1 AND is_active = true';
      const result = await this.executeQuery(sql, [symbol]);
      
      if (!result || result.length === 0) return null;
      
      const row = result[0];
      return {
        ...row,
        created_at: new Date(row.created_at),
        updated_at: new Date(row.updated_at),
        last_funding_time: row.last_funding_time ? new Date(row.last_funding_time) : undefined
      };
    } catch (error) {
      logger.error('Error getting market by symbol:', error);
      throw error;
    }
  }

  /**
   * Store oracle price data
   */
  public async storeOraclePrice(marketId: string, price: number, confidence: number, exponent: number): Promise<OraclePrice> {
    try {
      const sql = `
        INSERT INTO oracle_prices (market_id, price, confidence, exponent)
        VALUES ($1, $2, $3, $4)
        RETURNING *
      `;
      const result = await this.executeQuery(sql, [marketId, price, confidence, exponent]);
      
      const row = result[0];
      return {
        ...row,
        created_at: new Date(row.created_at)
      };
    } catch (error) {
      logger.error('Error storing oracle price:', error);
      throw error;
    }
  }

  /**
   * Get latest oracle prices for all markets
   */
  public async getLatestOraclePrices(): Promise<OraclePrice[]> {
    try {
      const sql = `
        SELECT DISTINCT ON (market_id) 
          op.*, m.symbol, m.base_asset, m.quote_asset
        FROM oracle_prices op
        JOIN markets m ON op.market_id = m.id
        WHERE m.is_active = true
        ORDER BY market_id, op.created_at DESC
      `;
      const result = await this.executeQuery(sql);
      
      return result.map((row: any) => ({
        id: row.id,
        market_id: row.market_id,
        price: row.price,
        confidence: row.confidence,
        exponent: row.exponent,
        created_at: new Date(row.created_at)
      }));
    } catch (error) {
      logger.error('Error getting latest oracle prices:', error);
      throw error;
    }
  }

  /**
   * Get user by wallet address
   */
  public async getUserByWallet(walletAddress: string): Promise<any> {
    try {
      const sql = 'SELECT * FROM users WHERE wallet_address = $1';
      const result = await this.executeQuery(sql, [walletAddress]);
      return result?.[0] || null;
    } catch (error) {
      logger.error('Error getting user by wallet:', error);
      throw error;
    }
  }

  /**
   * Create or update user
   */
  public async upsertUser(userData: any): Promise<any> {
    try {
      const sql = `
        INSERT INTO users (wallet_address, username, email, last_login, is_active)
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (wallet_address)
        DO UPDATE SET
          username = EXCLUDED.username,
          email = EXCLUDED.email,
          last_login = EXCLUDED.last_login,
          updated_at = NOW()
        RETURNING *
      `;
      const result = await this.executeQuery(sql, [
        userData.wallet_address,
        userData.username,
        userData.email,
        userData.last_login || new Date(),
        userData.is_active !== false
      ]);
      return result?.[0];
    } catch (error) {
      logger.error('Error upserting user:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const mcpSupabaseService = MCPSupabaseService.getInstance();
