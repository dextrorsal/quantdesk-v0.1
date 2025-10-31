// Enhanced WebSocket Service for Portfolio Updates
// Provides robust real-time portfolio data with exponential backoff reconnection

import { Logger } from '../utils/logger';

const logger = new Logger();

export interface PortfolioWebSocketMessage {
  type: 'portfolio_update' | 'error' | 'ping' | 'pong' | 'authenticated' | 'auth_error';
  data: any;
  timestamp: number;
}

export interface PortfolioUpdateData {
  userId: string;
  totalValue: number;
  totalUnrealizedPnl: number;
  totalRealizedPnl: number;
  marginRatio: number;
  healthFactor: number;
  totalCollateral: number;
  usedMargin: number;
  availableMargin: number;
  positions: Array<{
    id: string;
    symbol: string;
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnl: number;
    unrealizedPnlPercent: number;
    margin: number;
    leverage: number;
    side: 'long' | 'short';
  }>;
  timestamp: number;
}

export interface WebSocketConfig {
  url: string;
  maxReconnectAttempts?: number;
  baseReconnectDelay?: number;
  maxReconnectDelay?: number;
  heartbeatInterval?: number;
  connectionTimeout?: number;
  enableJitter?: boolean;
}

export interface ConnectionStats {
  isConnected: boolean;
  reconnectAttempts: number;
  lastConnected: number | null;
  lastDisconnected: number | null;
  totalReconnects: number;
  connectionUptime: number;
}

export class PortfolioWebSocketService {
  private ws: WebSocket | null = null;
  private config: Required<WebSocketConfig>;
  private reconnectAttempts = 0;
  private isConnected = false;
  private isConnecting = false;
  private reconnectTimeout: NodeJS.Timeout | null = null;
  private heartbeatInterval: NodeJS.Timeout | null = null;
  private connectionTimeout: NodeJS.Timeout | null = null;
  private subscribers: Set<(data: PortfolioUpdateData) => void> = new Set();
  private authToken: string | null = null;
  private userId: string | null = null;
  private stats: ConnectionStats = {
    isConnected: false,
    reconnectAttempts: 0,
    lastConnected: null,
    lastDisconnected: null,
    totalReconnects: 0,
    connectionUptime: 0
  };

  constructor(config: WebSocketConfig) {
    this.config = {
      url: config.url,
      maxReconnectAttempts: config.maxReconnectAttempts || 10,
      baseReconnectDelay: config.baseReconnectDelay || 1000,
      maxReconnectDelay: config.maxReconnectDelay || 30000,
      heartbeatInterval: config.heartbeatInterval || 30000,
      connectionTimeout: config.connectionTimeout || 10000,
      enableJitter: config.enableJitter !== false
    };
  }

  /**
   * Connect to WebSocket with authentication
   */
  public async connect(authToken: string, userId: string): Promise<void> {
    this.authToken = authToken;
    this.userId = userId;

    return new Promise((resolve, reject) => {
      if (this.isConnecting || this.isConnected) {
        resolve();
        return;
      }

      this.isConnecting = true;
      logger.debug(`🔌 Connecting to portfolio WebSocket: ${this.config.url}`);

      try {
        this.ws = new WebSocket(this.config.url);

        // Set connection timeout
        this.connectionTimeout = setTimeout(() => {
          if (!this.isConnected) {
            logger.warn('⏰ WebSocket connection timeout');
            this.ws?.close();
            this.isConnecting = false;
            reject(new Error('Connection timeout'));
          }
        }, this.config.connectionTimeout);

        this.ws.onopen = () => {
          logger.info('✅ Portfolio WebSocket connected');
          this.isConnected = true;
          this.isConnecting = false;
          this.reconnectAttempts = 0;
          this.stats.isConnected = true;
          this.stats.lastConnected = Date.now();
          this.stats.totalReconnects++;

          // Clear connection timeout
          if (this.connectionTimeout) {
            clearTimeout(this.connectionTimeout);
            this.connectionTimeout = null;
          }

          // Start heartbeat
          this.startHeartbeat();

          // Authenticate
          this.authenticate();

          resolve();
        };

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data);
        };

        this.ws.onclose = (event) => {
          logger.info(`🔌 Portfolio WebSocket disconnected: ${event.code} ${event.reason}`);
          this.handleDisconnect();
        };

        this.ws.onerror = (error) => {
          logger.error('❌ Portfolio WebSocket error:', error);
          this.isConnecting = false;
          reject(error);
        };

      } catch (error) {
        this.isConnecting = false;
        reject(error);
      }
    });
  }

  /**
   * Disconnect from WebSocket
   */
  public disconnect(): void {
    logger.info('🔌 Disconnecting from portfolio WebSocket');
    
    this.isConnected = false;
    this.isConnecting = false;
    
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }

    this.stopHeartbeat();
    this.clearReconnectTimeout();
    this.clearConnectionTimeout();
  }

  /**
   * Subscribe to portfolio updates
   */
  public subscribe(callback: (data: PortfolioUpdateData) => void): () => void {
    this.subscribers.add(callback);

    // If connected, subscribe to portfolio updates
    if (this.isConnected && this.userId) {
      this.send({
        type: 'subscribe_portfolio',
        data: {},
        timestamp: Date.now()
      });
    }

    // Return unsubscribe function
    return () => {
      this.subscribers.delete(callback);
    };
  }

  /**
   * Get connection statistics
   */
  public getStats(): ConnectionStats {
    return { ...this.stats };
  }

  /**
   * Check if connected
   */
  public isServiceConnected(): boolean {
    return this.isConnected;
  }

  /**
   * Force reconnection
   */
  public async forceReconnect(): Promise<void> {
    logger.info('🔄 Forcing portfolio WebSocket reconnection');
    this.disconnect();
    this.reconnectAttempts = 0;
    
    if (this.authToken && this.userId) {
      await this.connect(this.authToken, this.userId);
    }
  }

  /**
   * Handle incoming messages
   */
  private handleMessage(data: string): void {
    try {
      const message: PortfolioWebSocketMessage = JSON.parse(data);
      
      switch (message.type) {
        case 'portfolio_update':
          this.handlePortfolioUpdate(message.data);
          break;
        case 'authenticated':
          logger.debug('✅ Portfolio WebSocket authenticated');
          this.subscribeToPortfolio();
          break;
        case 'auth_error':
          logger.error('❌ Portfolio WebSocket authentication failed:', message.data);
          break;
        case 'pong':
          // Heartbeat response
          break;
        default:
          logger.debug('📨 Unknown portfolio WebSocket message type:', message.type);
      }
    } catch (error) {
      logger.error('Error parsing portfolio WebSocket message:', error);
    }
  }

  /**
   * Handle portfolio update
   */
  private handlePortfolioUpdate(data: PortfolioUpdateData): void {
    logger.debug(`📊 Portfolio update received for user ${data.userId}`);
    
    // Notify all subscribers
    this.subscribers.forEach(callback => {
      try {
        callback(data);
      } catch (error) {
        logger.error('Error in portfolio update callback:', error);
      }
    });
  }

  /**
   * Authenticate with WebSocket
   */
  private authenticate(): void {
    if (this.authToken) {
      this.send({
        type: 'authenticate',
        data: { token: this.authToken },
        timestamp: Date.now()
      });
    }
  }

  /**
   * Subscribe to portfolio updates
   */
  private subscribeToPortfolio(): void {
    this.send({
      type: 'subscribe_portfolio',
      data: {},
      timestamp: Date.now()
    });
  }

  /**
   * Handle disconnection
   */
  private handleDisconnect(): void {
    this.isConnected = false;
    this.isConnecting = false;
    this.stats.isConnected = false;
    this.stats.lastDisconnected = Date.now();
    
    this.stopHeartbeat();
    this.clearConnectionTimeout();

    // Attempt reconnection with exponential backoff
    this.scheduleReconnect();
  }

  /**
   * Schedule reconnection with exponential backoff and jitter
   */
  private scheduleReconnect(): void {
    if (this.reconnectAttempts >= this.config.maxReconnectAttempts) {
      logger.error('❌ Max reconnection attempts reached for portfolio WebSocket');
      return;
    }

    this.reconnectAttempts++;
    this.stats.reconnectAttempts = this.reconnectAttempts;

    // Calculate delay with exponential backoff
    let delay = Math.min(
      this.config.baseReconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      this.config.maxReconnectDelay
    );

    // Add jitter to prevent thundering herd
    if (this.config.enableJitter) {
      delay += Math.random() * 1000;
    }

    logger.info(`🔄 Scheduling portfolio WebSocket reconnection in ${Math.round(delay)}ms (attempt ${this.reconnectAttempts})`);

    this.reconnectTimeout = setTimeout(async () => {
      if (this.authToken && this.userId) {
        try {
          await this.connect(this.authToken, this.userId);
        } catch (error) {
          logger.error('Portfolio WebSocket reconnection failed:', error);
          // Will automatically schedule another attempt
        }
      }
    }, delay);
  }

  /**
   * Start heartbeat
   */
  private startHeartbeat(): void {
    this.stopHeartbeat();
    
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected && this.ws) {
        this.send({
          type: 'ping',
          data: { timestamp: Date.now() },
          timestamp: Date.now()
        });
      }
    }, this.config.heartbeatInterval);
  }

  /**
   * Stop heartbeat
   */
  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval);
      this.heartbeatInterval = null;
    }
  }

  /**
   * Clear reconnect timeout
   */
  private clearReconnectTimeout(): void {
    if (this.reconnectTimeout) {
      clearTimeout(this.reconnectTimeout);
      this.reconnectTimeout = null;
    }
  }

  /**
   * Clear connection timeout
   */
  private clearConnectionTimeout(): void {
    if (this.connectionTimeout) {
      clearTimeout(this.connectionTimeout);
      this.connectionTimeout = null;
    }
  }

  /**
   * Send message to WebSocket
   */
  private send(message: PortfolioWebSocketMessage): void {
    if (this.isConnected && this.ws) {
      try {
        this.ws.send(JSON.stringify(message));
      } catch (error) {
        logger.error('Error sending portfolio WebSocket message:', error);
      }
    }
  }
}

// Export singleton instance
export const portfolioWebSocketService = new PortfolioWebSocketService({
  url: import.meta.env.VITE_WS_URL || 'ws://localhost:3002',
  maxReconnectAttempts: 10,
  baseReconnectDelay: 1000,
  maxReconnectDelay: 30000,
  heartbeatInterval: 30000,
  connectionTimeout: 10000,
  enableJitter: true
});
