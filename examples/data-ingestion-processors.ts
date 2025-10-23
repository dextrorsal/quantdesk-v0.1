/**
 * QuantDesk Data Ingestion Processor Examples
 * 
 * This file demonstrates reusable data processing patterns for real-time market data.
 * These processors are open source and can be used by the community.
 */

import { EventEmitter } from 'events';
import WebSocket from 'ws';
import axios from 'axios';

// Example: Market Data Processor
export class MarketDataProcessor extends EventEmitter {
  private ws: WebSocket | null = null;
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;

  constructor(
    private config: {
      wsUrl: string;
      symbols: string[];
      apiKey?: string;
    }
  ) {
    super();
  }

  async start(): Promise<void> {
    try {
      await this.connect();
      this.setupEventHandlers();
    } catch (error) {
      console.error('Failed to start market data processor:', error);
      throw error;
    }
  }

  private async connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      const wsUrl = this.config.apiKey ? 
        `${this.config.wsUrl}?api_key=${this.config.apiKey}` : 
        this.config.wsUrl;

      this.ws = new WebSocket(wsUrl);

      this.ws.on('open', () => {
        console.log('Connected to market data feed');
        this.isConnected = true;
        this.reconnectAttempts = 0;
        this.subscribeToSymbols();
        resolve();
      });

      this.ws.on('error', (error) => {
        console.error('WebSocket error:', error);
        reject(error);
      });

      this.ws.on('close', () => {
        console.log('WebSocket connection closed');
        this.isConnected = false;
        this.handleReconnect();
      });
    });
  }

  private setupEventHandlers(): void {
    if (!this.ws) return;

    this.ws.on('message', (data: Buffer) => {
      try {
        const message = JSON.parse(data.toString());
        this.processMarketData(message);
      } catch (error) {
        console.error('Failed to parse market data:', error);
      }
    });
  }

  private subscribeToSymbols(): void {
    if (!this.ws || !this.isConnected) return;

    const subscriptionMessage = {
      type: 'subscribe',
      symbols: this.config.symbols
    };

    this.ws.send(JSON.stringify(subscriptionMessage));
  }

  private processMarketData(data: any): void {
    // Normalize different data formats
    const normalizedData = this.normalizeMarketData(data);
    
    // Emit processed data
    this.emit('marketData', normalizedData);
    
    // Store in database (if configured)
    this.storeMarketData(normalizedData);
  }

  private normalizeMarketData(data: any): {
    symbol: string;
    price: number;
    volume: number;
    timestamp: Date;
    source: string;
  } {
    // Handle different data formats from various exchanges
    if (data.type === 'ticker') {
      return {
        symbol: data.symbol,
        price: parseFloat(data.price),
        volume: parseFloat(data.volume || 0),
        timestamp: new Date(data.timestamp || Date.now()),
        source: 'websocket'
      };
    }

    if (data.type === 'trade') {
      return {
        symbol: data.symbol,
        price: parseFloat(data.price),
        volume: parseFloat(data.size || 0),
        timestamp: new Date(data.time || Date.now()),
        source: 'websocket'
      };
    }

    // Default normalization
    return {
      symbol: data.symbol || 'UNKNOWN',
      price: parseFloat(data.price || 0),
      volume: parseFloat(data.volume || 0),
      timestamp: new Date(),
      source: 'websocket'
    };
  }

  private async storeMarketData(data: {
    symbol: string;
    price: number;
    volume: number;
    timestamp: Date;
    source: string;
  }): Promise<void> {
    try {
      // Example: Store in database
      // await this.database.storeMarketData(data);
      console.log('Stored market data:', data.symbol, data.price);
    } catch (error) {
      console.error('Failed to store market data:', error);
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      console.error('Max reconnection attempts reached');
      this.emit('error', new Error('Connection lost'));
      return;
    }

    this.reconnectAttempts++;
    const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1);

    console.log(`Reconnecting in ${delay}ms (attempt ${this.reconnectAttempts})`);
    
    setTimeout(() => {
      this.connect().catch(error => {
        console.error('Reconnection failed:', error);
      });
    }, delay);
  }

  stop(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
    this.isConnected = false;
  }
}

// Example: Price Aggregator
export class PriceAggregator {
  private priceCache = new Map<string, {
    price: number;
    volume: number;
    timestamp: Date;
    sources: string[];
  }>();

  constructor(private config: {
    aggregationWindow: number; // milliseconds
    minSources: number;
  }) {}

  addPriceData(data: {
    symbol: string;
    price: number;
    volume: number;
    timestamp: Date;
    source: string;
  }): void {
    const existing = this.priceCache.get(data.symbol);
    
    if (!existing) {
      this.priceCache.set(data.symbol, {
        price: data.price,
        volume: data.volume,
        timestamp: data.timestamp,
        sources: [data.source]
      });
      return;
    }

    // Weighted average based on volume
    const totalVolume = existing.volume + data.volume;
    const weightedPrice = (existing.price * existing.volume + data.price * data.volume) / totalVolume;

    existing.price = weightedPrice;
    existing.volume = totalVolume;
    existing.timestamp = data.timestamp;
    
    if (!existing.sources.includes(data.source)) {
      existing.sources.push(data.source);
    }
  }

  getAggregatedPrice(symbol: string): {
    price: number;
    volume: number;
    timestamp: Date;
    sources: string[];
    confidence: number;
  } | null {
    const data = this.priceCache.get(symbol);
    
    if (!data) return null;

    const confidence = Math.min(data.sources.length / this.config.minSources, 1);
    
    return {
      ...data,
      confidence
    };
  }

  getAllPrices(): Map<string, {
    price: number;
    volume: number;
    timestamp: Date;
    sources: string[];
    confidence: number;
  }> {
    const result = new Map();
    
    for (const [symbol, data] of this.priceCache) {
      const confidence = Math.min(data.sources.length / this.config.minSources, 1);
      result.set(symbol, {
        ...data,
        confidence
      });
    }
    
    return result;
  }

  clearOldData(maxAge: number): void {
    const cutoff = new Date(Date.now() - maxAge);
    
    for (const [symbol, data] of this.priceCache) {
      if (data.timestamp < cutoff) {
        this.priceCache.delete(symbol);
      }
    }
  }
}

// Example: Data Validator
export class DataValidator {
  private priceHistory = new Map<string, number[]>();
  private maxHistoryLength = 100;

  validatePriceData(data: {
    symbol: string;
    price: number;
    volume: number;
    timestamp: Date;
  }): {
    isValid: boolean;
    warnings: string[];
    confidence: number;
  } {
    const warnings: string[] = [];
    let confidence = 1.0;

    // Check for reasonable price range
    if (data.price <= 0) {
      warnings.push('Price is zero or negative');
      confidence *= 0.1;
    }

    // Check for reasonable volume
    if (data.volume < 0) {
      warnings.push('Volume is negative');
      confidence *= 0.5;
    }

    // Check for price spikes
    const history = this.priceHistory.get(data.symbol) || [];
    if (history.length > 0) {
      const lastPrice = history[history.length - 1];
      const priceChange = Math.abs(data.price - lastPrice) / lastPrice;
      
      if (priceChange > 0.1) { // 10% change
        warnings.push('Large price change detected');
        confidence *= 0.7;
      }
    }

    // Update price history
    history.push(data.price);
    if (history.length > this.maxHistoryLength) {
      history.shift();
    }
    this.priceHistory.set(data.symbol, history);

    return {
      isValid: warnings.length === 0,
      warnings,
      confidence
    };
  }

  validateVolumeData(data: {
    symbol: string;
    volume: number;
    timestamp: Date;
  }): {
    isValid: boolean;
    warnings: string[];
  } {
    const warnings: string[] = [];

    if (data.volume < 0) {
      warnings.push('Volume cannot be negative');
    }

    if (data.volume === 0) {
      warnings.push('Zero volume may indicate stale data');
    }

    return {
      isValid: warnings.length === 0,
      warnings
    };
  }
}

// Example: Data Pipeline
export class DataPipeline {
  private processors: MarketDataProcessor[] = [];
  private aggregator: PriceAggregator;
  private validator: DataValidator;

  constructor(config: {
    sources: Array<{
      wsUrl: string;
      symbols: string[];
      apiKey?: string;
    }>;
    aggregationWindow: number;
    minSources: number;
  }) {
    this.aggregator = new PriceAggregator({
      aggregationWindow: config.aggregationWindow,
      minSources: config.minSources
    });

    this.validator = new DataValidator();

    // Create processors for each source
    for (const sourceConfig of config.sources) {
      const processor = new MarketDataProcessor(sourceConfig);
      processor.on('marketData', (data) => this.handleMarketData(data));
      this.processors.push(processor);
    }
  }

  async start(): Promise<void> {
    console.log('Starting data pipeline...');
    
    for (const processor of this.processors) {
      try {
        await processor.start();
      } catch (error) {
        console.error('Failed to start processor:', error);
      }
    }

    // Start aggregation timer
    setInterval(() => {
      this.publishAggregatedData();
    }, this.aggregator['config'].aggregationWindow);
  }

  private handleMarketData(data: {
    symbol: string;
    price: number;
    volume: number;
    timestamp: Date;
    source: string;
  }): void {
    // Validate data
    const validation = this.validator.validatePriceData(data);
    
    if (!validation.isValid) {
      console.warn('Invalid price data:', validation.warnings);
      return;
    }

    // Add to aggregator
    this.aggregator.addPriceData(data);
  }

  private publishAggregatedData(): void {
    const aggregatedPrices = this.aggregator.getAllPrices();
    
    for (const [symbol, data] of aggregatedPrices) {
      if (data.confidence >= 0.5) { // Only publish if confidence is high enough
        this.emit('aggregatedPrice', {
          symbol,
          ...data
        });
      }
    }

    // Clean up old data
    this.aggregator.clearOldData(5 * 60 * 1000); // 5 minutes
  }

  stop(): void {
    for (const processor of this.processors) {
      processor.stop();
    }
  }
}

// Example: REST API Data Source
export class RestApiDataSource {
  constructor(private config: {
    baseUrl: string;
    apiKey?: string;
    symbols: string[];
    interval: number; // milliseconds
  }) {}

  async start(): Promise<void> {
    console.log('Starting REST API data source...');
    
    // Fetch data at regular intervals
    setInterval(async () => {
      try {
        await this.fetchData();
      } catch (error) {
        console.error('Failed to fetch data:', error);
      }
    }, this.config.interval);
  }

  private async fetchData(): Promise<void> {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    };

    if (this.config.apiKey) {
      headers['Authorization'] = `Bearer ${this.config.apiKey}`;
    }

    for (const symbol of this.config.symbols) {
      try {
        const response = await axios.get(
          `${this.config.baseUrl}/ticker/${symbol}`,
          { headers }
        );

        const data = response.data;
        
        // Process the data
        this.processApiData({
          symbol,
          price: parseFloat(data.price),
          volume: parseFloat(data.volume || 0),
          timestamp: new Date(data.timestamp || Date.now()),
          source: 'rest-api'
        });
      } catch (error) {
        console.error(`Failed to fetch data for ${symbol}:`, error);
      }
    }
  }

  private processApiData(data: {
    symbol: string;
    price: number;
    volume: number;
    timestamp: Date;
    source: string;
  }): void {
    // Emit or process the data
    console.log('Processed API data:', data.symbol, data.price);
  }
}

export default {
  MarketDataProcessor,
  PriceAggregator,
  DataValidator,
  DataPipeline,
  RestApiDataSource
};
