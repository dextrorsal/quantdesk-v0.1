/**
 * MIKEY-AI Client
 * 
 * Handles communication with MIKEY-AI system
 * Provides access to AI agent, CCXT data, and Solana intelligence
 */

import axios, { AxiosInstance } from 'axios';
import { logger } from '../utils/logger';

export interface MikeyAIConfig {
  baseUrl: string;
  apiKey: string;
  timeout: number;
}

export interface AIQueryRequest {
  query: string;
  context?: any;
  includeMarketData?: boolean;
  includeWhaleData?: boolean;
  includeArbitrageData?: boolean;
}

export interface AIQueryResponse {
  response: string;
  sources: string[];
  confidence: number;
  timestamp: Date;
  data: any;
}

export interface MarketDataRequest {
  symbol: string;
  exchanges?: string[];
  includeOrderBook?: boolean;
  includeFundingRates?: boolean;
}

export interface MarketDataResponse {
  symbol: string;
  exchanges: Array<{
    exchange: string;
    price: number;
    volume24h: number;
    spread: number;
    orderBook?: any;
    fundingRate?: number;
  }>;
  summary: {
    averagePrice: number;
    totalVolume24h: number;
    priceRange: { highest: number; lowest: number };
    spreads: Array<{ exchange: string; spread: number }>;
  };
}

export interface ArbitrageRequest {
  symbol: string;
  minSpreadPercent?: number;
  exchanges?: string[];
}

export interface ArbitrageResponse {
  symbol: string;
  opportunities: Array<{
    buyExchange: string;
    sellExchange: string;
    buyPrice: number;
    sellPrice: number;
    spreadPercent: number;
    potentialProfit: number;
    volume: number;
  }>;
  summary: {
    totalOpportunities: number;
    bestOpportunity: any;
    averageSpread: number;
    exchangesAnalyzed: number;
  };
}

export interface WhaleTrackingRequest {
  symbol: string;
  threshold?: number;
  timeframe?: string;
}

export interface WhaleTrackingResponse {
  symbol: string;
  whales: Array<{
    walletAddress: string;
    totalValue: number;
    recentActivity: {
      transactions24h: number;
      volume24h: number;
      largestTransaction: number;
    };
    portfolio: any;
  }>;
  summary: {
    totalWhales: number;
    totalVolume: number;
    averageWhaleSize: number;
    marketImpact: string;
  };
}

export interface LiquidationRequest {
  symbol?: string;
  protocol?: string;
  timeframe?: string;
}

export interface LiquidationResponse {
  liquidations: Array<{
    liquidationId: string;
    timestamp: Date;
    protocol: string;
    walletAddress: string;
    positionType: string;
    size: number;
    liquidationPrice: number;
    marketPrice: number;
    pnl: number;
  }>;
  summary: {
    totalLiquidations: number;
    totalVolume: number;
    largestLiquidation: number;
    protocolBreakdown: any;
    cascadeRisk: string;
  };
}

export class MikeyAIClient {
  private client: AxiosInstance;
  private connected: boolean = false;

  constructor(config: MikeyAIConfig) {
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
        logger.debug(`MIKEY-AI API Request: ${config.method?.toUpperCase()} ${config.url}`);
        return config;
      },
      (error) => {
        logger.error('MIKEY-AI API Request Error:', error);
        return Promise.reject(error);
      }
    );

    this.client.interceptors.response.use(
      (response) => {
        logger.debug(`MIKEY-AI API Response: ${response.status} ${response.config.url}`);
        return response;
      },
      (error) => {
        logger.error('MIKEY-AI API Response Error:', error);
        return Promise.reject(error);
      }
    );
  }

  public async connect(): Promise<void> {
    try {
      const response = await this.client.get('/health');
      this.connected = response.status === 200;
      logger.info('MIKEY-AI client connected successfully');
    } catch (error) {
      logger.error('Failed to connect to MIKEY-AI:', error);
      throw error;
    }
  }

  public isConnected(): boolean {
    return this.connected;
  }

  /**
   * Process natural language query with AI agent
   */
  public async processQuery(request: AIQueryRequest): Promise<AIQueryResponse> {
    try {
      const response = await this.client.post('/api/query', request);
      return response.data;
    } catch (error) {
      logger.error('AI query request failed:', error);
      throw error;
    }
  }

  /**
   * Get market data from 100+ exchanges via CCXT
   */
  public async getMarketData(request: MarketDataRequest): Promise<MarketDataResponse> {
    try {
      const response = await this.client.post('/api/market-data', request);
      return response.data;
    } catch (error) {
      logger.error('Market data request failed:', error);
      throw error;
    }
  }

  /**
   * Detect arbitrage opportunities across exchanges
   */
  public async detectArbitrage(request: ArbitrageRequest): Promise<ArbitrageResponse> {
    try {
      const response = await this.client.post('/api/arbitrage', request);
      return response.data;
    } catch (error) {
      logger.error('Arbitrage detection request failed:', error);
      throw error;
    }
  }

  /**
   * Track whale wallet activities
   */
  public async trackWhales(request: WhaleTrackingRequest): Promise<WhaleTrackingResponse> {
    try {
      const response = await this.client.post('/api/whales', request);
      return response.data;
    } catch (error) {
      logger.error('Whale tracking request failed:', error);
      throw error;
    }
  }

  /**
   * Get liquidation data across protocols
   */
  public async getLiquidations(request: LiquidationRequest): Promise<LiquidationResponse> {
    try {
      const response = await this.client.post('/api/liquidations', request);
      return response.data;
    } catch (error) {
      logger.error('Liquidation data request failed:', error);
      throw error;
    }
  }

  /**
   * Get technical analysis from MIKEY-AI
   */
  public async getTechnicalAnalysis(symbol: string, indicators?: string[]): Promise<any> {
    try {
      const response = await this.client.post('/api/technical-analysis', {
        symbol,
        indicators: indicators || ['RSI', 'MACD', 'Bollinger', 'Support', 'Resistance']
      });
      return response.data;
    } catch (error) {
      logger.error('Technical analysis request failed:', error);
      throw error;
    }
  }

  /**
   * Get sentiment analysis
   */
  public async getSentimentAnalysis(symbol: string, sources?: string[]): Promise<any> {
    try {
      const response = await this.client.post('/api/sentiment', {
        symbol,
        sources: sources || ['twitter', 'reddit', 'news']
      });
      return response.data;
    } catch (error) {
      logger.error('Sentiment analysis request failed:', error);
      throw error;
    }
  }

  /**
   * Get Solana-specific data
   */
  public async getSolanaData(address?: string): Promise<any> {
    try {
      const response = await this.client.get('/api/solana', {
        params: address ? { address } : {}
      });
      return response.data;
    } catch (error) {
      logger.error('Solana data request failed:', error);
      throw error;
    }
  }

  /**
   * Get funding rates across perpetual exchanges
   */
  public async getFundingRates(symbol: string): Promise<any> {
    try {
      const response = await this.client.get('/api/funding-rates', {
        params: { symbol }
      });
      return response.data;
    } catch (error) {
      logger.error('Funding rates request failed:', error);
      throw error;
    }
  }

  /**
   * Get order book data
   */
  public async getOrderBookData(symbol: string, exchanges?: string[]): Promise<any> {
    try {
      const response = await this.client.post('/api/order-book', {
        symbol,
        exchanges
      });
      return response.data;
    } catch (error) {
      logger.error('Order book data request failed:', error);
      throw error;
    }
  }
}
