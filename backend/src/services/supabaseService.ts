import { createClient, SupabaseClient } from '@supabase/supabase-js';

export interface SupabaseConfig {
  url: string;
  anonKey: string;
}

export class SupabaseService {
  private client: SupabaseClient;
  private config: SupabaseConfig;

  constructor(config: SupabaseConfig) {
    this.config = config;
    this.client = createClient(config.url, config.anonKey);
  }

  /**
   * Get the Supabase client instance
   */
  getClient(): SupabaseClient {
    return this.client;
  }

  /**
   * Execute a raw SQL query
   */
  async query(sql: string, params: any[] = []): Promise<any> {
    try {
      const { data, error } = await this.client.rpc('execute_sql', {
        sql,
        params
      });
      
      if (error) {
        throw new Error(`Supabase query error: ${error.message}`);
      }
      
      return data;
    } catch (error) {
      console.error('Supabase query error:', error);
      throw error;
    }
  }

  /**
   * Insert data into a table
   */
  async insert(table: string, data: any): Promise<any> {
    const { data: result, error } = await this.client
      .from(table)
      .insert(data)
      .select();

    if (error) {
      throw new Error(`Supabase insert error: ${error.message}`);
    }

    return result;
  }

  /**
   * Update data in a table
   */
  async update(table: string, data: any, filter: any): Promise<any> {
    const { data: result, error } = await this.client
      .from(table)
      .update(data)
      .match(filter)
      .select();

    if (error) {
      throw new Error(`Supabase update error: ${error.message}`);
    }

    return result;
  }

  /**
   * Select data from a table
   */
  async select(table: string, columns: string = '*', filter: any = {}): Promise<any> {
    const { data, error } = await this.client
      .from(table)
      .select(columns)
      .match(filter);

    if (error) {
      throw new Error(`Supabase select error: ${error.message}`);
    }

    return data;
  }

  /**
   * Delete data from a table
   */
  async delete(table: string, filter: any): Promise<any> {
    const { data, error } = await this.client
      .from(table)
      .delete()
      .match(filter)
      .select();

    if (error) {
      throw new Error(`Supabase delete error: ${error.message}`);
    }

    return data;
  }

  /**
   * Subscribe to real-time changes
   */
  subscribe(table: string, callback: (payload: any) => void, filter?: any) {
    return this.client
      .channel(`${table}_changes`)
      .on('postgres_changes', {
        event: '*',
        schema: 'public',
        table: table,
        filter: filter
      }, callback)
      .subscribe();
  }

  /**
   * Get markets data
   */
  async getMarkets(): Promise<any[]> {
    return this.select('markets', '*', { is_active: true });
  }

  /**
   * Get user by wallet address
   */
  async getUserByWallet(walletAddress: string): Promise<any> {
    const users = await this.select('users', '*', { wallet_address: walletAddress });
    return users?.[0] || null;
  }

  /**
   * Create or update user
   */
  async upsertUser(userData: any): Promise<any> {
    const { data, error } = await this.client
      .from('users')
      .upsert(userData, { onConflict: 'wallet_address' })
      .select();

    if (error) {
      throw new Error(`Supabase upsert user error: ${error.message}`);
    }

    return data?.[0];
  }

  /**
   * Store oracle price data
   */
  async storeOraclePrice(marketId: string, price: number, confidence: number, exponent: number): Promise<any> {
    return this.insert('oracle_prices', {
      market_id: marketId,
      price: price,
      confidence: confidence,
      exponent: exponent
    });
  }

  /**
   * Get latest oracle prices
   */
  async getLatestOraclePrices(): Promise<any[]> {
    const { data, error } = await this.client
      .from('oracle_prices')
      .select(`
        *,
        markets!inner(symbol, base_asset, quote_asset)
      `)
      .order('created_at', { ascending: false });

    if (error) {
      throw new Error(`Supabase get oracle prices error: ${error.message}`);
    }

    return data || [];
  }

  /**
   * Health check
   */
  async healthCheck(): Promise<boolean> {
    try {
      await this.client.from('markets').select('count').limit(1);
      return true;
    } catch (error) {
      console.error('Supabase health check failed:', error);
      return false;
    }
  }
}

// Create singleton instance
const supabaseConfig: SupabaseConfig = {
  url: process.env.SUPABASE_URL || '',
  anonKey: process.env.SUPABASE_ANON_KEY || ''
};

export const supabaseService = new SupabaseService(supabaseConfig);
