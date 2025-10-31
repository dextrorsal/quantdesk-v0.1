/**
 * QuantDesk Client
 * 
 * Handles communication with QuantDesk backend API
 * Provides access to ML models, database, and trading strategies
 */

import axios, { AxiosInstance } from 'axios';
import { logger } from '../utils/logger';

export interface QuantDeskConfig {
  baseUrl: string;
  apiKey: string;
  timeout: number;
}

export interface MLPredictionRequest {
  symbol: string;
  timeframe: string;
  features?: any;
  modelType?: 'lorentzian' | 'logistic' | 'chandelier';
}

export interface MLPredictionResponse {
  prediction: number;
  confidence: number;
  modelType: string;
  features: any;
  timestamp: Date;
  backtestMetrics?: {
    winRate: number;
    totalReturn: number;
    maxDrawdown: number;
    sharpeRatio: number;
  };
}

export interface MarketDataRequest {
  symbol: string;
  timeframe: string;
  limit?: number;
}

export interface MarketDataResponse {
  symbol: string;
  timeframe: string;
  data: Array<{
    timestamp: Date;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }>;
}

export interface BacktestRequest {
  symbol: string;
  strategy: string;
  startDate: string;
  endDate: string;
  initialCapital: number;
}

export interface BacktestResponse {
  results: {
    totalReturn: number;
    winRate: number;
    maxDrawdown: number;
    sharpeRatio: number;
    totalTrades: number;
    profitableTrades: number;
  };
  trades: Array<{
    timestamp: Date;
    side: 'buy' | 'sell';
    price: number;
    quantity: number;
    pnl: number;
  }>;
}

export class QuantDeskClient {
  private client: AxiosInstance;
  private connected: boolean = false;

  constructor(config: QuantDeskConfig) {
    this.client = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout,
      headers: {
        'Authorization': `Bearer ${config.apiKey}`,
        'Content-Type': 'application/json'
      }
    });

    // Add request/response interceptors
    this.client.interceptors.request.use(
      (config) => {
        logger.debug(`QuantDesk API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        logger.error('QuantDesk API Request Error:', error);
        return Promise.reject(error);
      }
    );

    this.client.interceptors.response.use(
      (response) => {
        logger.debug(`QuantDesk API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        logger.error('QuantDesk API Response Error:', error);
        return Promise.reject(error);
      }
    );
  }

  public async connect(): Promise<void> {
    try {
      const response = await this.client.get('/health');
      this.connected = response.status === 200;
      logger.info('QuantDesk client connected successfully');
    } catch (error) {
      logger.error('Failed to connect to QuantDesk:', error);
      throw error;
    }
  }

  public isConnected(): boolean {
    return this.connected;
  }

  /**
   * Get ML model prediction
   */
  public async getMLPrediction(request: MLPredictionRequest): Promise<MLPredictionResponse> {
    try {
      const response = await this.client.post('/api/ml/predict', request);
      return response.data;
    } catch (error) {
      logger.error('ML prediction request failed:', error);
      throw error;
    }
  }

  /**
   * Get market data from QuantDesk database
   */
  public async getMarketData(request: MarketDataRequest): Promise<MarketDataResponse> {
    try {
      const response = await this.client.get('/api/market-data', {
        params: request
      });
      return response.data;
    } catch (error) {
      logger.error('Market data request failed:', error);
      throw error;
    }
  }

  /**
   * Run backtest with QuantDesk strategies
   */
  public async runBacktest(request: BacktestRequest): Promise<BacktestResponse> {
    try {
      const response = await this.client.post('/api/backtest', request);
      return response.data;
    } catch (error) {
      logger.error('Backtest request failed:', error);
      throw error;
    }
  }

  /**
   * Get available models and their performance metrics
   */
  public async getModelMetrics(): Promise<any> {
    try {
      const response = await this.client.get('/api/ml/models/metrics');
      return response.data;
    } catch (error) {
      logger.error('Model metrics request failed:', error);
      throw error;
    }
  }

  /**
   * Get trading signals from QuantDesk strategies
   */
  public async getTradingSignals(symbol: string, timeframe: string): Promise<any> {
    try {
      const response = await this.client.get('/api/signals', {
        params: { symbol, timeframe }
      });
      return response.data;
    } catch (error) {
      logger.error('Trading signals request failed:', error);
      throw error;
    }
  }

  /**
   * Get portfolio performance metrics
   */
  public async getPortfolioMetrics(): Promise<any> {
    try {
      const response = await this.client.get('/api/portfolio/metrics');
      return response.data;
    } catch (error) {
      logger.error('Portfolio metrics request failed:', error);
      throw error;
    }
  }

  /**
   * Get database schema information
   */
  public async getDatabaseInfo(): Promise<any> {
    try {
      const response = await this.client.get('/api/database/info');
      return response.data;
    } catch (error) {
      logger.error('Database info request failed:', error);
      throw error;
    }
  }

  /**
   * Get real-time performance metrics
   */
  public async getPerformanceMetrics(): Promise<any> {
    try {
      const response = await this.client.get('/api/performance/metrics');
      return response.data;
    } catch (error) {
      logger.error('Performance metrics request failed:', error);
      throw error;
    }
  }
}
