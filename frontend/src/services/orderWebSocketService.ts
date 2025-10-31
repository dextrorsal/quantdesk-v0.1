import { io, Socket } from 'socket.io-client';
import { logger } from '../utils/logger';

export interface OrderUpdate {
  symbol: string;
  orderId: string;
  status: 'pending' | 'filled' | 'partially_filled' | 'cancelled' | 'failed';
  filledSize: number;
  averageFillPrice?: number;
  userId: string;
  timestamp: number;
  smartContractTx?: string;
  smartContractPositionId?: string;
  atomicPositionCreation?: boolean;
}

export interface OrderStatusCallback {
  (update: OrderUpdate): void;
}

export class OrderWebSocketService {
  private static instance: OrderWebSocketService;
  private socket: Socket | null = null;
  private isConnected = false;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;
  private reconnectDelay = 1000;
  private maxReconnectDelay = 16000;
  private orderStatusCallbacks: Map<string, OrderStatusCallback[]> = new Map();

  private constructor() {}

  public static getInstance(): OrderWebSocketService {
    if (!OrderWebSocketService.instance) {
      OrderWebSocketService.instance = new OrderWebSocketService();
    }
    return OrderWebSocketService.instance;
  }

  /**
   * Initialize WebSocket connection
   */
  public initialize(serverUrl: string, token: string): void {
    if (this.socket?.connected) {
      logger.warn('Order WebSocket already connected');
      return;
    }

    try {
      this.socket = io(serverUrl, {
        auth: {
          token: token
        },
        transports: ['websocket'],
        timeout: 10000,
        reconnection: true,
        reconnectionAttempts: this.maxReconnectAttempts,
        reconnectionDelay: this.reconnectDelay,
        reconnectionDelayMax: this.maxReconnectDelay
      });

      this.setupEventHandlers();
      logger.info('Order WebSocket service initialized');
    } catch (error) {
      logger.error('Failed to initialize Order WebSocket service:', error);
    }
  }

  /**
   * Setup WebSocket event handlers
   */
  private setupEventHandlers(): void {
    if (!this.socket) return;

    this.socket.on('connect', () => {
      this.isConnected = true;
      this.reconnectAttempts = 0;
      logger.info('Order WebSocket connected');
      
      // Subscribe to order updates
      this.subscribeToOrderUpdates();
    });

    this.socket.on('disconnect', (reason) => {
      this.isConnected = false;
      logger.warn('Order WebSocket disconnected:', reason);
    });

    this.socket.on('connect_error', (error) => {
      logger.error('Order WebSocket connection error:', error);
      this.handleReconnect();
    });

    this.socket.on('order_update', (update: OrderUpdate) => {
      this.handleOrderUpdate(update);
    });

    this.socket.on('error', (error) => {
      logger.error('Order WebSocket error:', error);
    });
  }

  /**
   * Subscribe to order updates
   */
  private subscribeToOrderUpdates(): void {
    if (!this.socket?.connected) return;

    this.socket.emit('subscribe_orders');
    logger.info('Subscribed to order updates');
  }

  /**
   * Unsubscribe from order updates
   */
  public unsubscribeFromOrderUpdates(): void {
    if (!this.socket?.connected) return;

    this.socket.emit('unsubscribe_orders');
    logger.info('Unsubscribed from order updates');
  }

  /**
   * Handle order update from server
   */
  private handleOrderUpdate(update: OrderUpdate): void {
    logger.info('Received order update:', update);
    
    // Notify all registered callbacks
    const callbacks = this.orderStatusCallbacks.get(update.orderId) || [];
    callbacks.forEach(callback => {
      try {
        callback(update);
      } catch (error) {
        logger.error('Error in order status callback:', error);
      }
    });

    // Also notify general order update listeners
    const generalCallbacks = this.orderStatusCallbacks.get('*') || [];
    generalCallbacks.forEach(callback => {
      try {
        callback(update);
      } catch (error) {
        logger.error('Error in general order status callback:', error);
      }
    });
  }

  /**
   * Register callback for order status updates
   */
  public onOrderStatusUpdate(orderId: string, callback: OrderStatusCallback): void {
    if (!this.orderStatusCallbacks.has(orderId)) {
      this.orderStatusCallbacks.set(orderId, []);
    }
    this.orderStatusCallbacks.get(orderId)!.push(callback);
  }

  /**
   * Register callback for all order status updates
   */
  public onAllOrderStatusUpdates(callback: OrderStatusCallback): void {
    this.onOrderStatusUpdate('*', callback);
  }

  /**
   * Remove callback for order status updates
   */
  public offOrderStatusUpdate(orderId: string, callback: OrderStatusCallback): void {
    const callbacks = this.orderStatusCallbacks.get(orderId);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }

  /**
   * Handle reconnection with exponential backoff
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts >= this.maxReconnectAttempts) {
      logger.error('Max reconnection attempts reached');
      return;
    }

    this.reconnectAttempts++;
    const delay = Math.min(
      this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1),
      this.maxReconnectDelay
    );

    logger.info(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`);

    setTimeout(() => {
      if (this.socket && !this.socket.connected) {
        this.socket.connect();
      }
    }, delay);
  }

  /**
   * Get connection status
   */
  public getConnectionStatus(): boolean {
    return this.isConnected && this.socket?.connected === true;
  }

  /**
   * Disconnect WebSocket
   */
  public disconnect(): void {
    if (this.socket) {
      this.unsubscribeFromOrderUpdates();
      this.socket.disconnect();
      this.socket = null;
      this.isConnected = false;
      this.orderStatusCallbacks.clear();
      logger.info('Order WebSocket disconnected');
    }
  }

  /**
   * Get socket instance for advanced usage
   */
  public getSocket(): Socket | null {
    return this.socket;
  }
}

// Export singleton instance
export const orderWebSocketService = OrderWebSocketService.getInstance();
