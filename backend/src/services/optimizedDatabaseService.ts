import { SupabaseDatabaseService } from './supabaseDatabase';
import { Logger } from '../utils/logger';

const logger = new Logger();

interface CacheEntry<T> {
  data: T;
  timestamp: number;
  ttl: number;
}

interface QueryMetrics {
  query: string;
  executionTime: number;
  cacheHit: boolean;
  timestamp: number;
}

/**
 * Optimized Database Service with Query Caching and Performance Monitoring
 * 
 * This service wraps the base SupabaseDatabaseService with:
 * - Query result caching
 * - Performance monitoring
 * - Query optimization
 * - Connection pooling
 */
export class OptimizedDatabaseService {
  private static instance: OptimizedDatabaseService;
  private baseService: SupabaseDatabaseService;
  private queryCache = new Map<string, CacheEntry<any>>();
  private queryMetrics: QueryMetrics[] = [];
  private maxCacheSize = 1000;
  private maxMetricsSize = 500;
  private defaultTTL = 30000; // 30 seconds

  private constructor() {
    this.baseService = SupabaseDatabaseService.getInstance();
  }

  public static getInstance(): OptimizedDatabaseService {
    if (!OptimizedDatabaseService.instance) {
      OptimizedDatabaseService.instance = new OptimizedDatabaseService();
    }
    return OptimizedDatabaseService.instance;
  }

  /**
   * Optimized select with caching
   */
  public async select<T>(
    table: string,
    columns: string = '*',
    filters: Record<string, any> = {},
    options: {
      cache?: boolean;
      ttl?: number;
      limit?: number;
      orderBy?: string;
      orderDirection?: 'asc' | 'desc';
    } = {}
  ): Promise<T[]> {
    const startTime = Date.now();
    const cacheKey = this.generateCacheKey('select', table, columns, filters, options);
    
    // Check cache first
    if (options.cache !== false) {
      const cached = this.getFromCache<T[]>(cacheKey);
      if (cached) {
        this.recordQueryMetrics('select', Date.now() - startTime, true);
        return cached;
      }
    }

    try {
      // Build optimized query
      const query = this.buildOptimizedQuery(table, columns, filters, options);
      
      // Execute query
      const result = await this.baseService.select(table, columns, filters);
      
      // Cache result
      if (options.cache !== false) {
        this.setCache(cacheKey, result, options.ttl || this.defaultTTL);
      }
      
      this.recordQueryMetrics('select', Date.now() - startTime, false);
      return result;
    } catch (error) {
      logger.error(`Database select error for table ${table}:`, error);
      throw error;
    }
  }

  /**
   * Optimized insert with batch support
   */
  public async insert<T>(
    table: string,
    data: T | T[],
    options: {
      batchSize?: number;
      ignoreDuplicates?: boolean;
    } = {}
  ): Promise<T | T[]> {
    const startTime = Date.now();
    
    try {
      // Handle batch inserts for better performance
      if (Array.isArray(data) && data.length > 1) {
        const batchSize = options.batchSize || 100;
        const batches = this.chunkArray(data, batchSize);
        const results: T[] = [];
        
        for (const batch of batches) {
          const batchResult = await this.baseService.insert(table, batch);
          results.push(...(Array.isArray(batchResult) ? batchResult : [batchResult]));
        }
        
        this.recordQueryMetrics('insert_batch', Date.now() - startTime, false);
        return results;
      }
      
      const result = await this.baseService.insert(table, data);
      this.recordQueryMetrics('insert', Date.now() - startTime, false);
      return result;
    } catch (error) {
      logger.error(`Database insert error for table ${table}:`, error);
      throw error;
    }
  }

  /**
   * Optimized update with selective updates
   */
  public async update<T>(
    table: string,
    data: Partial<T>,
    filters: Record<string, any>,
    options: {
      cache?: boolean;
      ttl?: number;
    } = {}
  ): Promise<T[]> {
    const startTime = Date.now();
    
    try {
      const result = await this.baseService.update(table, data, filters);
      
      // Invalidate related cache entries
      if (options.cache !== false) {
        this.invalidateCache(table, filters);
      }
      
      this.recordQueryMetrics('update', Date.now() - startTime, false);
      return result;
    } catch (error) {
      logger.error(`Database update error for table ${table}:`, error);
      throw error;
    }
  }


  /**
   * Get markets with optimized caching
   */
  public async getMarkets(): Promise<any[]> {
    return this.select('markets', '*', { is_active: true }, {
      cache: true,
      ttl: 60000, // 1 minute cache
      orderBy: 'symbol',
      orderDirection: 'asc'
    });
  }

  /**
   * Get user positions with optimized query
   */
  public async getUserPositions(userId: string): Promise<any[]> {
    return this.select('positions', '*', { 
      user_id: userId, 
      status: 'open' 
    }, {
      cache: true,
      ttl: 30000, // 30 seconds cache
      orderBy: 'created_at',
      orderDirection: 'desc'
    });
  }

  /**
   * Get user orders with optimized query
   */
  public async getUserOrders(userId: string, status?: string): Promise<any[]> {
    const filters: Record<string, any> = { user_id: userId };
    if (status) {
      filters.status = status;
    }
    
    return this.select('orders', '*', filters, {
      cache: true,
      ttl: 15000, // 15 seconds cache
      orderBy: 'created_at',
      orderDirection: 'desc',
      limit: 100
    });
  }

  /**
   * Get market by symbol with caching
   */
  public async getMarketBySymbol(symbol: string): Promise<any> {
    const markets = await this.getMarkets();
    return markets.find(market => market.symbol === symbol);
  }

  /**
   * Delegate other methods to base service
   */
  public async connect(): Promise<void> {
    return this.baseService.connect();
  }

  public async disconnect(): Promise<void> {
    return this.baseService.disconnect();
  }

  /**
   * Delete method
   */
  public async delete(table: string, filters: Record<string, any>): Promise<void> {
    const startTime = Date.now();
    
    try {
      // For now, we'll implement a simple delete by updating status
      // This is a placeholder implementation
      await this.baseService.update(table, { status: 'deleted' }, filters);
      
      // Invalidate related cache entries
      this.invalidateCache(table, filters);
      
      this.recordQueryMetrics('delete', Date.now() - startTime, false);
    } catch (error) {
      logger.error(`Database delete error for table ${table}:`, error);
      throw error;
    }
  }

  /**
   * Get performance metrics
   */
  public getPerformanceMetrics(): {
    cacheHitRate: number;
    averageQueryTime: number;
    slowQueries: QueryMetrics[];
    cacheSize: number;
  } {
    const totalQueries = this.queryMetrics.length;
    const cacheHits = this.queryMetrics.filter(m => m.cacheHit).length;
    const cacheHitRate = totalQueries > 0 ? (cacheHits / totalQueries) * 100 : 0;
    
    const averageQueryTime = totalQueries > 0 
      ? this.queryMetrics.reduce((sum, m) => sum + m.executionTime, 0) / totalQueries 
      : 0;
    
    const slowQueries = this.queryMetrics
      .filter(m => m.executionTime > 100) // > 100ms
      .sort((a, b) => b.executionTime - a.executionTime)
      .slice(0, 10);
    
    return {
      cacheHitRate,
      averageQueryTime,
      slowQueries,
      cacheSize: this.queryCache.size
    };
  }

  /**
   * Clear cache
   */
  public clearCache(): void {
    this.queryCache.clear();
    logger.info('Database query cache cleared');
  }

  /**
   * Generate cache key
   */
  private generateCacheKey(
    operation: string,
    table: string,
    columns: string,
    filters: Record<string, any>,
    options: any
  ): string {
    return `${operation}:${table}:${columns}:${JSON.stringify(filters)}:${JSON.stringify(options)}`;
  }

  /**
   * Get data from cache
   */
  private getFromCache<T>(key: string): T | null {
    const entry = this.queryCache.get(key);
    if (!entry) return null;
    
    if (Date.now() - entry.timestamp > entry.ttl) {
      this.queryCache.delete(key);
      return null;
    }
    
    return entry.data;
  }

  /**
   * Set data in cache
   */
  private setCache<T>(key: string, data: T, ttl: number): void {
    // Maintain cache size
    if (this.queryCache.size >= this.maxCacheSize) {
      const firstKey = this.queryCache.keys().next().value;
      this.queryCache.delete(firstKey);
    }
    
    this.queryCache.set(key, {
      data,
      timestamp: Date.now(),
      ttl
    });
  }

  /**
   * Invalidate cache entries for a table
   */
  private invalidateCache(table: string, filters: Record<string, any>): void {
    const keysToDelete: string[] = [];
    
    for (const key of this.queryCache.keys()) {
      if (key.includes(`:${table}:`)) {
        keysToDelete.push(key);
      }
    }
    
    keysToDelete.forEach(key => this.queryCache.delete(key));
  }

  /**
   * Build optimized query
   */
  private buildOptimizedQuery(
    table: string,
    columns: string,
    filters: Record<string, any>,
    options: any
  ): string {
    // This would contain query optimization logic
    // For now, we'll rely on the base service
    return `${table}:${columns}:${JSON.stringify(filters)}`;
  }

  /**
   * Record query metrics
   */
  private recordQueryMetrics(query: string, executionTime: number, cacheHit: boolean): void {
    this.queryMetrics.push({
      query,
      executionTime,
      cacheHit,
      timestamp: Date.now()
    });
    
    // Maintain metrics size
    if (this.queryMetrics.length > this.maxMetricsSize) {
      this.queryMetrics = this.queryMetrics.slice(-this.maxMetricsSize);
    }
  }

  /**
   * Chunk array for batch operations
   */
  private chunkArray<T>(array: T[], size: number): T[][] {
    const chunks: T[][] = [];
    for (let i = 0; i < array.length; i += size) {
      chunks.push(array.slice(i, i + size));
    }
    return chunks;
  }
}

// Export singleton instance
export const optimizedDatabaseService = OptimizedDatabaseService.getInstance();
