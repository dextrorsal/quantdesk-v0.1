/**
 * QuantDesk API Integration Example
 * 
 * This example demonstrates how to integrate QuantDesk APIs into your application.
 * It shows various API patterns and best practices for building trading applications.
 */

import axios, { AxiosInstance, AxiosResponse } from 'axios';
import { EventEmitter } from 'events';

interface QuantDeskAPIConfig {
  baseUrl: string;
  apiKey?: string;
  timeout?: number;
  retries?: number;
}

interface MarketData {
  symbol: string;
  price: number;
  volume: number;
  change24h: number;
  timestamp: Date;
}

interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  amount: number;
  price?: number;
  type: 'market' | 'limit';
  status: 'pending' | 'filled' | 'cancelled';
  createdAt: Date;
}

interface Portfolio {
  totalValue: number;
  positions: Array<{
    symbol: string;
    amount: number;
    value: number;
    pnl: number;
  }>;
}

export class QuantDeskAPIClient extends EventEmitter {
  private client: AxiosInstance;
  private config: QuantDeskAPIConfig;
  private authToken: string | null = null;

  constructor(config: QuantDeskAPIConfig) {
    super();
    this.config = config;
    
    this.client = axios.create({
      baseURL: config.baseUrl,
      timeout: config.timeout || 10000,
      headers: {
        'Content-Type': 'application/json',
        'User-Agent': 'QuantDesk-API-Client/1.0'
      }
    });

    this.setupInterceptors();
  }

  private setupInterceptors(): void {
    // Request interceptor
    this.client.interceptors.request.use(
      (config) => {
        // Add API key if available
        if (this.config.apiKey) {
          config.headers['X-API-Key'] = this.config.apiKey;
        }

        // Add auth token if available
        if (this.authToken) {
          config.headers['Authorization'] = `Bearer ${this.authToken}`;
        }

        return config;
      },
      (error) => Promise.reject(error)
    );

    // Response interceptor
    this.client.interceptors.response.use(
      (response) => response,
      async (error) => {
        const originalRequest = error.config;

        // Retry logic for network errors
        if (error.code === 'ECONNABORTED' || error.code === 'ENOTFOUND') {
          if (!originalRequest._retry && this.config.retries) {
            originalRequest._retry = true;
            await this.sleep(1000);
            return this.client(originalRequest);
          }
        }

        // Handle 401 errors (unauthorized)
        if (error.response?.status === 401) {
          this.authToken = null;
          this.emit('unauthorized');
        }

        return Promise.reject(error);
      }
    );
  }

  // Authentication
  async authenticate(email: string, password: string): Promise<string> {
    try {
      const response = await this.client.post('/auth/login', {
        email,
        password
      });

      this.authToken = response.data.token;
      this.emit('authenticated', this.authToken);
      
      return this.authToken;
    } catch (error) {
      throw new Error(`Authentication failed: ${error.message}`);
    }
  }

  async logout(): Promise<void> {
    try {
      await this.client.post('/auth/logout');
      this.authToken = null;
      this.emit('loggedOut');
    } catch (error) {
      console.warn('Logout failed:', error.message);
    }
  }

  // Market Data API
  async getMarketData(symbol: string): Promise<MarketData> {
    try {
      const response = await this.client.get(`/market-data/${symbol}`);
      return this.transformMarketData(response.data);
    } catch (error) {
      throw new Error(`Failed to get market data for ${symbol}: ${error.message}`);
    }
  }

  async getMarketHistory(symbol: string, timeframe: string = '1h'): Promise<MarketData[]> {
    try {
      const response = await this.client.get(`/market-data/${symbol}/history`, {
        params: { timeframe }
      });
      return response.data.map((item: any) => this.transformMarketData(item));
    } catch (error) {
      throw new Error(`Failed to get market history for ${symbol}: ${error.message}`);
    }
  }

  async getMarketSummary(): Promise<{
    totalVolume: number;
    activeMarkets: number;
    priceChanges: Array<{ symbol: string; change24h: number }>;
  }> {
    try {
      const response = await this.client.get('/market-summary');
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get market summary: ${error.message}`);
    }
  }

  // Order Management API
  async createOrder(orderData: {
    symbol: string;
    side: 'buy' | 'sell';
    amount: number;
    price?: number;
    type: 'market' | 'limit';
  }): Promise<Order> {
    try {
      const response = await this.client.post('/orders', orderData);
      return this.transformOrder(response.data);
    } catch (error) {
      throw new Error(`Failed to create order: ${error.message}`);
    }
  }

  async getOrders(status?: string): Promise<Order[]> {
    try {
      const params = status ? { status } : {};
      const response = await this.client.get('/orders', { params });
      return response.data.map((item: any) => this.transformOrder(item));
    } catch (error) {
      throw new Error(`Failed to get orders: ${error.message}`);
    }
  }

  async cancelOrder(orderId: string): Promise<void> {
    try {
      await this.client.delete(`/orders/${orderId}`);
      this.emit('orderCancelled', orderId);
    } catch (error) {
      throw new Error(`Failed to cancel order ${orderId}: ${error.message}`);
    }
  }

  // Portfolio API
  async getPortfolio(): Promise<Portfolio> {
    try {
      const response = await this.client.get('/portfolio');
      return response.data;
    } catch (error) {
      throw new Error(`Failed to get portfolio: ${error.message}`);
    }
  }

  async getBalance(symbol: string): Promise<number> {
    try {
      const response = await this.client.get(`/balance/${symbol}`);
      return response.data.balance;
    } catch (error) {
      throw new Error(`Failed to get balance for ${symbol}: ${error.message}`);
    }
  }

  // WebSocket API (simulated with polling)
  async startMarketDataStream(symbols: string[], callback: (data: MarketData) => void): Promise<void> {
    const streamInterval = setInterval(async () => {
      try {
        for (const symbol of symbols) {
          const marketData = await this.getMarketData(symbol);
          callback(marketData);
        }
      } catch (error) {
        console.error('Error in market data stream:', error);
      }
    }, 1000); // Update every second

    // Store interval for cleanup
    (this as any)._streamInterval = streamInterval;
  }

  async stopMarketDataStream(): Promise<void> {
    const streamInterval = (this as any)._streamInterval;
    if (streamInterval) {
      clearInterval(streamInterval);
      (this as any)._streamInterval = null;
    }
  }

  // Utility methods
  private transformMarketData(data: any): MarketData {
    return {
      symbol: data.symbol,
      price: parseFloat(data.price),
      volume: parseFloat(data.volume),
      change24h: parseFloat(data.change24h),
      timestamp: new Date(data.timestamp)
    };
  }

  private transformOrder(data: any): Order {
    return {
      id: data.id,
      symbol: data.symbol,
      side: data.side,
      amount: parseFloat(data.amount),
      price: data.price ? parseFloat(data.price) : undefined,
      type: data.type,
      status: data.status,
      createdAt: new Date(data.createdAt)
    };
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Health check
  async healthCheck(): Promise<boolean> {
    try {
      const response = await this.client.get('/health');
      return response.status === 200;
    } catch (error) {
      return false;
    }
  }
}

// Example usage and integration patterns
export class TradingApp {
  private apiClient: QuantDeskAPIClient;
  private isRunning = false;

  constructor(config: QuantDeskAPIConfig) {
    this.apiClient = new QuantDeskAPIClient(config);
    this.setupEventHandlers();
  }

  private setupEventHandlers(): void {
    this.apiClient.on('authenticated', (token) => {
      console.log('✅ Authenticated successfully');
    });

    this.apiClient.on('unauthorized', () => {
      console.log('❌ Authentication expired');
    });

    this.apiClient.on('orderCancelled', (orderId) => {
      console.log(`🔒 Order ${orderId} cancelled`);
    });
  }

  async start(): Promise<void> {
    console.log('🚀 Starting trading application...');
    this.isRunning = true;

    // Check API health
    const isHealthy = await this.apiClient.healthCheck();
    if (!isHealthy) {
      throw new Error('API is not healthy');
    }

    // Authenticate
    await this.apiClient.authenticate('your-email@example.com', 'your-password');

    // Start market data stream
    await this.apiClient.startMarketDataStream(['SOL', 'BTC'], (data) => {
      console.log(`📊 ${data.symbol}: $${data.price} (${data.change24h}%)`);
    });

    // Monitor portfolio
    this.startPortfolioMonitoring();
  }

  async stop(): Promise<void> {
    console.log('🛑 Stopping trading application...');
    this.isRunning = false;

    await this.apiClient.stopMarketDataStream();
    await this.apiClient.logout();
  }

  private async startPortfolioMonitoring(): Promise<void> {
    while (this.isRunning) {
      try {
        const portfolio = await this.apiClient.getPortfolio();
        console.log(`💰 Portfolio Value: $${portfolio.totalValue.toFixed(2)}`);
        
        await this.sleep(30000); // Check every 30 seconds
      } catch (error) {
        console.error('Error monitoring portfolio:', error);
        await this.sleep(5000);
      }
    }
  }

  // Trading methods
  async placeMarketOrder(symbol: string, side: 'buy' | 'sell', amount: number): Promise<Order> {
    return await this.apiClient.createOrder({
      symbol,
      side,
      amount,
      type: 'market'
    });
  }

  async placeLimitOrder(symbol: string, side: 'buy' | 'sell', amount: number, price: number): Promise<Order> {
    return await this.apiClient.createOrder({
      symbol,
      side,
      amount,
      price,
      type: 'limit'
    });
  }

  async getOpenOrders(): Promise<Order[]> {
    return await this.apiClient.getOrders('pending');
  }

  async cancelAllOrders(): Promise<void> {
    const orders = await this.getOpenOrders();
    for (const order of orders) {
      await this.apiClient.cancelOrder(order.id);
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Example integration with React
export const useQuantDeskAPI = (config: QuantDeskAPIConfig) => {
  const [client, setClient] = React.useState<QuantDeskAPIClient | null>(null);
  const [isConnected, setIsConnected] = React.useState(false);
  const [marketData, setMarketData] = React.useState<MarketData[]>([]);

  React.useEffect(() => {
    const apiClient = new QuantDeskAPIClient(config);
    setClient(apiClient);

    apiClient.on('authenticated', () => setIsConnected(true));
    apiClient.on('unauthorized', () => setIsConnected(false));

    return () => {
      apiClient.stopMarketDataStream();
    };
  }, [config]);

  const authenticate = async (email: string, password: string) => {
    if (client) {
      await client.authenticate(email, password);
    }
  };

  const getMarketData = async (symbol: string) => {
    if (client) {
      const data = await client.getMarketData(symbol);
      setMarketData(prev => [...prev.filter(d => d.symbol !== symbol), data]);
      return data;
    }
  };

  return {
    client,
    isConnected,
    marketData,
    authenticate,
    getMarketData
  };
};

// Example integration with Node.js
export const createTradingBot = (config: QuantDeskAPIConfig) => {
  const app = new TradingApp(config);

  return {
    start: () => app.start(),
    stop: () => app.stop(),
    placeOrder: (symbol: string, side: 'buy' | 'sell', amount: number) => 
      app.placeMarketOrder(symbol, side, amount),
    getPortfolio: () => app.apiClient.getPortfolio(),
    getOrders: () => app.apiClient.getOrders()
  };
};

export default QuantDeskAPIClient;
