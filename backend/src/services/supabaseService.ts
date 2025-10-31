import { createClient, SupabaseClient } from '@supabase/supabase-js';

export interface SupabaseConfig {
  url: string;
  anonKey: string;
}

export class SupabaseService {
  private client: SupabaseClient;

  constructor(config: SupabaseConfig) {
    this.client = createClient(config.url, config.anonKey);
  }

  /**
   * Get the Supabase client instance
   */
  getClient(): SupabaseClient {
    return this.client;
  }

  /**
   * Execute a raw SQL query using Supabase RPC
   * @deprecated Use fluent API methods instead for better type safety and security
   */
  async query(sql: string, params: any[] = []): Promise<any> {
    console.warn('⚠️  Using deprecated query() method. Consider using fluent API methods instead.');
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
   * Execute a raw SQL query (alias for query method)
   * @deprecated Use fluent API methods instead for better type safety and security
   */
  async executeQuery(sql: string, params: any[] = []): Promise<any> {
    console.warn('⚠️  Using deprecated executeQuery() method. Consider using fluent API methods instead.');
    return this.query(sql, params);
  }

  /**
   * Execute a transaction
   */
  async transaction<T>(callback: (client: SupabaseClient) => Promise<T>): Promise<T> {
    // Supabase doesn't have traditional transactions, but we can simulate it
    // For now, just execute the callback with the client
    return callback(this.client);
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

    return data || [];
  }

  async getChannels(): Promise<any[]> {
    return this.select('chat_channels', '*', { is_active: true });
  }

  async createChannel(name: string, description: string, isPrivate: boolean = false, createdByUserId: string): Promise<any> {
    // User should already be authenticated, so we receive their ID directly.
    return this.insert('chat_channels', {
      name,
      description,
      is_private: isPrivate,
      created_by: createdByUserId
    });
  }

  async getMessages(channelId: string, limit: number = 50): Promise<any[]> {
    return this.select('chat_messages', '*', {
      channel_id: channelId,
      deleted_at: null
    }).then(messages => messages.slice(-limit));
  }

  async sendMessage(channelId: string, message: string, authorPubkey: string, mentions: string[] = []): Promise<any> {
    // User should already be authenticated, so we receive their pubkey directly.
    return this.insert('chat_messages', {
      channel_id: channelId,
      author_pubkey: authorPubkey,
      message,
      mentions
    });
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
    const users = await this.select('users', '*', { wallet_pubkey: walletAddress });
    return users?.[0] || null;
  }

  /**
   * Create or update user
   */
  async upsertUser(userData: any): Promise<any> {
    const { data, error } = await this.client
      .from('users')
      .upsert(userData, { onConflict: 'wallet_pubkey' })
      .select();

    if (error) {
      throw new Error(`Supabase upsert user error: ${error.message}`);
    }

    return data?.[0];
  }

  async upsert(table: string, data: any, conflict?: string): Promise<any> {
    const { data: result, error } = await this.client
      .from(table)
      .upsert(data, conflict ? { onConflict: conflict } : undefined)
      .select();
    if (error) throw new Error(`Supabase upsert error: ${error.message}`);
    return result?.[0] || result;
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
   * Get markets by symbol with proper filtering
   */
  async getMarketBySymbol(symbol: string): Promise<any> {
    const { data, error } = await this.client
      .from('markets')
      .select('id, symbol, base_asset, quote_asset, is_active')
      .eq('symbol', symbol)
      .eq('is_active', true)
      .single();

    if (error) {
      throw new Error(`Market not found: ${error.message}`);
    }

    return data;
  }

  /**
   * Insert order with proper validation
   */
  async insertOrder(orderData: {
    user_id: string;
    market_id: string;
    order_account: string;
    order_type: string;
    side: string;
    size: number;
    price?: number;
    leverage: number;
    status: string;
  }): Promise<any> {
    const { data, error } = await this.client
      .from('orders')
      .insert(orderData)
      .select('id')
      .single();

    if (error) {
      throw new Error(`Failed to insert order: ${error.message}`);
    }

    return data;
  }

  /**
   * Get pending orders for matching
   */
  async getPendingOrders(marketId: string, side: string, priceCondition?: { operator: string; value: number }): Promise<any[]> {
    let query = this.client
      .from('orders')
      .select('id, price, remaining_size, created_at')
      .eq('market_id', marketId)
      .eq('status', 'pending')
      .eq('side', side);

    if (priceCondition) {
      if (priceCondition.operator === '<=') {
        query = query.lte('price', priceCondition.value);
      } else if (priceCondition.operator === '>=') {
        query = query.gte('price', priceCondition.value);
      }
    }

    const { data, error } = await query
      .order('price', { ascending: side === 'buy' })
      .order('created_at', { ascending: true })
      .limit(100);

    if (error) {
      throw new Error(`Failed to get pending orders: ${error.message}`);
    }

    return data || [];
  }

  /**
   * Update order fill information
   */
  async updateOrderFill(orderId: string, fillSize: number): Promise<void> {
    // First get the current order to calculate new values
    const { data: order, error: getError } = await this.client
      .from('orders')
      .select('filled_size, remaining_size')
      .eq('id', orderId)
      .single();

    if (getError) {
      throw new Error(`Failed to get order: ${getError.message}`);
    }

    const newFilledSize = (order.filled_size || 0) + fillSize;
    const newRemainingSize = order.remaining_size - fillSize;
    const isFullyFilled = newRemainingSize <= 0;

    const { error } = await this.client
      .from('orders')
      .update({
        filled_size: newFilledSize,
        status: isFullyFilled ? 'filled' : 'pending',
        updated_at: new Date().toISOString(),
        filled_at: isFullyFilled ? new Date().toISOString() : undefined
      })
      .eq('id', orderId);

    if (error) {
      throw new Error(`Failed to update order fill: ${error.message}`);
    }
  }

  /**
   * Get or create position
   */
  async getOrCreatePosition(userId: string, marketId: string, side: string): Promise<any> {
    // First try to get existing position
    const { data: existingPosition, error: selectError } = await this.client
      .from('positions')
      .select('id, size, entry_price')
      .eq('user_id', userId)
      .eq('market_id', marketId)
      .eq('side', side)
      .single();

    if (selectError && selectError.code !== 'PGRST116') { // PGRST116 = no rows returned
      throw new Error(`Failed to get position: ${selectError.message}`);
    }

    if (existingPosition) {
      return existingPosition;
    }

    // Create new position
    const { data: newPosition, error: insertError } = await this.client
      .from('positions')
      .insert({
        user_id: userId,
        market_id: marketId,
        side: side,
        size: 0,
        entry_price: 0,
        status: 'open',
        created_at: new Date().toISOString()
      })
      .select('id, size, entry_price')
      .single();

    if (insertError) {
      throw new Error(`Failed to create position: ${insertError.message}`);
    }

    return newPosition;
  }

  /**
   * Update position size and average price
   */
  async updatePosition(positionId: string, sizeChange: number, fillPrice: number): Promise<void> {
    // First get the current position to calculate new values
    const { data: position, error: getError } = await this.client
      .from('positions')
      .select('size, entry_price')
      .eq('id', positionId)
      .single();

    if (getError) {
      throw new Error(`Failed to get position: ${getError.message}`);
    }

    const newSize = position.size + sizeChange;
    let newEntryPrice = position.entry_price;

    // Calculate new average price if size is not zero
    if (newSize !== 0) {
      newEntryPrice = (position.entry_price * position.size + fillPrice * sizeChange) / newSize;
    } else {
      newEntryPrice = 0;
    }

    const { error } = await this.client
      .from('positions')
      .update({
        size: newSize,
        entry_price: newEntryPrice,
        updated_at: new Date().toISOString()
      })
      .eq('id', positionId);

    if (error) {
      throw new Error(`Failed to update position: ${error.message}`);
    }
  }

  /**
   * Calculate position health factor
   */
  async calculatePositionHealth(positionId: string): Promise<number> {
    const { data, error } = await this.client
      .rpc('calculate_position_health', { position_id: positionId });

    if (error) {
      throw new Error(`Failed to calculate position health: ${error.message}`);
    }

    return data || 0;
  }

  /**
   * Update position health factor
   */
  async updatePositionHealth(positionId: string, healthFactor: number): Promise<void> {
    const { error } = await this.client
      .from('positions')
      .update({ health_factor: healthFactor })
      .eq('id', positionId);

    if (error) {
      throw new Error(`Failed to update position health: ${error.message}`);
    }
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

// Create singleton instance with lazy loading
let supabaseServiceInstance: SupabaseService | null = null;

export function getSupabaseService(): SupabaseService {
  if (!supabaseServiceInstance) {
    const supabaseConfig: SupabaseConfig = {
      url: process.env.SUPABASE_URL || '',
      anonKey: process.env.SUPABASE_SERVICE_ROLE_KEY || process.env.SUPABASE_ANON_KEY || ''
    };
    
    if (!supabaseConfig.url || !supabaseConfig.anonKey) {
      console.error('Supabase configuration missing:');
      console.error('SUPABASE_URL:', process.env.SUPABASE_URL ? '✅ Set' : '❌ Missing');
      console.error('SUPABASE_SERVICE_ROLE_KEY:', process.env.SUPABASE_SERVICE_ROLE_KEY ? '✅ Set' : '❌ Missing');
      console.error('SUPABASE_ANON_KEY:', process.env.SUPABASE_ANON_KEY ? '✅ Set' : '❌ Missing');
      throw new Error('Supabase configuration missing. Please check SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY environment variables.');
    }
    
    supabaseServiceInstance = new SupabaseService(supabaseConfig);
  }
  return supabaseServiceInstance;
}

// Export for backward compatibility (deprecated)
export const supabaseService = {
  getClient: () => getSupabaseService().getClient(),
  query: (sql: string, params: any[] = []) => getSupabaseService().query(sql, params),
  executeQuery: (sql: string, params: any[] = []) => getSupabaseService().executeQuery(sql, params),
  healthCheck: () => getSupabaseService().healthCheck(),
  insert: (table: string, data: any) => getSupabaseService().getClient().from(table).insert(data),
  select: (table: string, columns?: string) => getSupabaseService().getClient().from(table).select(columns),
  update: (table: string, data: any) => getSupabaseService().getClient().from(table).update(data),
  delete: (table: string) => getSupabaseService().getClient().from(table).delete()
};
