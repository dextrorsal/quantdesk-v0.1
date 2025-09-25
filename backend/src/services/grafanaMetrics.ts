import axios from 'axios';
import { Logger } from '../utils/logger';

const logger = new Logger();

export interface TradingMetrics {
  timestamp: Date;
  totalVolume24h: number;
  activeTraders: number;
  totalPositions: number;
  totalValueLocked: number;
  averagePositionSize: number;
  leverageDistribution: Record<string, number>;
  marketMetrics: {
    symbol: string;
    volume24h: number;
    priceChange24h: number;
    openInterest: number;
    longShortRatio: number;
  }[];
}

export interface SystemMetrics {
  timestamp: Date;
  apiResponseTime: number;
  databaseQueryTime: number;
  activeConnections: number;
  errorRate: number;
  memoryUsage: number;
  cpuUsage: number;
}

export class GrafanaMetricsService {
  private static instance: GrafanaMetricsService;
  private grafanaUrl: string;
  private grafanaApiKey?: string;
  private metricsBuffer: TradingMetrics[] = [];
  private systemMetricsBuffer: SystemMetrics[] = [];

  private constructor() {
    this.grafanaUrl = process.env.GRAFANA_URL || 'http://localhost:3000';
    this.grafanaApiKey = process.env.GRAFANA_API_KEY;
  }

  public static getInstance(): GrafanaMetricsService {
    if (!GrafanaMetricsService.instance) {
      GrafanaMetricsService.instance = new GrafanaMetricsService();
    }
    return GrafanaMetricsService.instance;
  }

  /**
   * Collect trading metrics from Supabase
   */
  public async collectTradingMetrics(): Promise<TradingMetrics> {
    try {
      const timestamp = new Date();
      
      // Get basic trading data
      const [markets, positions, trades] = await Promise.all([
        this.getMarketData(),
        this.getPositionData(),
        this.getTradeData()
      ]);

      const totalVolume24h = trades.reduce((sum, trade) => sum + (trade.volume || 0), 0);
      const activeTraders = new Set(trades.map(t => t.user_id)).size;
      const totalPositions = positions.length;
      const totalValueLocked = positions.reduce((sum, pos) => sum + (pos.notional_value || 0), 0);
      const averagePositionSize = totalPositions > 0 ? totalValueLocked / totalPositions : 0;

      // Calculate leverage distribution
      const leverageDistribution = this.calculateLeverageDistribution(positions);

      // Calculate market-specific metrics
      const marketMetrics = await this.calculateMarketMetrics(markets, trades, positions);

      const metrics: TradingMetrics = {
        timestamp,
        totalVolume24h,
        activeTraders,
        totalPositions,
        totalValueLocked,
        averagePositionSize,
        leverageDistribution,
        marketMetrics
      };

      this.metricsBuffer.push(metrics);
      
      // Keep only last 1000 metrics to prevent memory issues
      if (this.metricsBuffer.length > 1000) {
        this.metricsBuffer = this.metricsBuffer.slice(-1000);
      }

      logger.info(`📊 Collected trading metrics: ${activeTraders} traders, $${totalVolume24h.toFixed(2)} volume`);
      return metrics;

    } catch (error) {
      logger.error('❌ Error collecting trading metrics:', error);
      throw error;
    }
  }

  /**
   * Collect system performance metrics
   */
  public async collectSystemMetrics(): Promise<SystemMetrics> {
    try {
      const timestamp = new Date();
      
      // Measure API response time
      const apiStart = Date.now();
      // Mock database query for response time measurement
      await new Promise(resolve => setTimeout(resolve, Math.random() * 10));
      const apiResponseTime = Date.now() - apiStart;

      // Get system resource usage
      const memoryUsage = process.memoryUsage();
      const cpuUsage = process.cpuUsage();

      const metrics: SystemMetrics = {
        timestamp,
        apiResponseTime,
        databaseQueryTime: apiResponseTime, // Same for now
        activeConnections: 1, // Placeholder
        errorRate: 0, // Placeholder - would need error tracking
        memoryUsage: memoryUsage.heapUsed / 1024 / 1024, // MB
        cpuUsage: (cpuUsage.user + cpuUsage.system) / 1000000 // Convert to seconds
      };

      this.systemMetricsBuffer.push(metrics);
      
      // Keep only last 1000 metrics
      if (this.systemMetricsBuffer.length > 1000) {
        this.systemMetricsBuffer = this.systemMetricsBuffer.slice(-1000);
      }

      return metrics;

    } catch (error) {
      logger.error('❌ Error collecting system metrics:', error);
      throw error;
    }
  }

  /**
   * Send metrics to Grafana (if API key is configured)
   */
  public async sendMetricsToGrafana(metrics: TradingMetrics | SystemMetrics): Promise<void> {
    if (!this.grafanaApiKey) {
      logger.warn('⚠️ Grafana API key not configured, skipping metrics upload');
      return;
    }

    try {
      // This would send metrics to Grafana's metrics endpoint
      // Implementation depends on Grafana's metrics collection setup
      logger.info('📈 Metrics ready for Grafana (API key configured)');
    } catch (error) {
      logger.error('❌ Error sending metrics to Grafana:', error);
    }
  }

  /**
   * Get recent trading metrics for API endpoints
   */
  public getRecentTradingMetrics(limit: number = 100): TradingMetrics[] {
    return this.metricsBuffer.slice(-limit);
  }

  /**
   * Get recent system metrics for API endpoints
   */
  public getRecentSystemMetrics(limit: number = 100): SystemMetrics[] {
    return this.systemMetricsBuffer.slice(-limit);
  }

  private async getMarketData(): Promise<any[]> {
    try {
      // Use the existing Supabase Oracle API endpoint
      const response = await axios.get('http://localhost:3002/api/supabase-oracle/markets');
      if (response.data.success) {
        return response.data.data;
      }
      return [];
    } catch (error) {
      logger.error('Error fetching market data:', error);
      return [];
    }
  }

  private async getPositionData(): Promise<any[]> {
    try {
      // For now, return mock data since we don't have positions table yet
      // In production, this would query: SELECT * FROM positions WHERE status IN ('open', 'pending')
      return [
        { id: '1', market_id: 'd87a99b4-148a-49c2-a2ad-ca1ee17a9372', user_id: 'user1', side: 'long', leverage: 5, notional_value: 1000 },
        { id: '2', market_id: 'b28b505f-682a-473d-979c-01c05bf31b1c', user_id: 'user2', side: 'short', leverage: 10, notional_value: 2000 }
      ];
    } catch (error) {
      logger.error('Error fetching position data:', error);
      return [];
    }
  }

  private async getTradeData(): Promise<any[]> {
    try {
      // For now, return mock data since we don't have trades table yet
      // In production, this would query: SELECT * FROM trades WHERE created_at >= NOW() - INTERVAL '24 hours'
      return [
        { id: '1', market_id: 'd87a99b4-148a-49c2-a2ad-ca1ee17a9372', user_id: 'user1', volume: 100, price: 50000 },
        { id: '2', market_id: 'b28b505f-682a-473d-979c-01c05bf31b1c', user_id: 'user2', volume: 200, price: 3000 }
      ];
    } catch (error) {
      logger.error('Error fetching trade data:', error);
      return [];
    }
  }

  private calculateLeverageDistribution(positions: any[]): Record<string, number> {
    const distribution: Record<string, number> = {
      '1x-5x': 0,
      '5x-10x': 0,
      '10x-20x': 0,
      '20x+': 0
    };

    positions.forEach(pos => {
      const leverage = pos.leverage || 1;
      if (leverage <= 5) distribution['1x-5x']++;
      else if (leverage <= 10) distribution['5x-10x']++;
      else if (leverage <= 20) distribution['10x-20x']++;
      else distribution['20x+']++;
    });

    return distribution;
  }

  private async calculateMarketMetrics(markets: any[], trades: any[], positions: any[]): Promise<any[]> {
    return markets.map(market => {
      const marketTrades = trades.filter(t => t.market_id === market.id);
      const marketPositions = positions.filter(p => p.market_id === market.id);
      
      const volume24h = marketTrades.reduce((sum, trade) => sum + (trade.volume || 0), 0);
      const openInterest = marketPositions.reduce((sum, pos) => sum + (pos.notional_value || 0), 0);
      
      // Calculate long/short ratio
      const longPositions = marketPositions.filter(p => p.side === 'long').length;
      const shortPositions = marketPositions.filter(p => p.side === 'short').length;
      const longShortRatio = shortPositions > 0 ? longPositions / shortPositions : 0;

      return {
        symbol: market.symbol,
        volume24h,
        priceChange24h: 0, // Would need price history
        openInterest,
        longShortRatio
      };
    });
  }
}

export const grafanaMetricsService = GrafanaMetricsService.getInstance();
