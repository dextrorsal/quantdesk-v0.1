import { create } from 'zustand';
import { orderWebSocketService, OrderUpdate } from '../services/orderWebSocketService';
import { logger } from '../utils/logger';

export interface Order {
  id: string;
  symbol: string;
  side: 'buy' | 'sell';
  size: number;
  price?: number;
  orderType: 'market' | 'limit';
  status: 'pending' | 'filled' | 'partially_filled' | 'cancelled' | 'failed';
  filledSize: number;
  averageFillPrice?: number;
  createdAt: string;
  updatedAt: string;
  smartContractTx?: string;
  smartContractPositionId?: string;
  atomicPositionCreation?: boolean;
  errorMessage?: string;
}

export interface OrderStore {
  // State
  orders: Map<string, Order>;
  isLoading: boolean;
  error: string | null;
  isConnected: boolean;

  // Actions
  addOrder: (order: Order) => void;
  updateOrder: (orderId: string, updates: Partial<Order>) => void;
  removeOrder: (orderId: string) => void;
  clearOrders: () => void;
  setLoading: (loading: boolean) => void;
  setError: (error: string | null) => void;
  setConnected: (connected: boolean) => void;

  // WebSocket actions
  initializeWebSocket: () => void;
  disconnectWebSocket: () => void;
  handleOrderUpdate: (update: OrderUpdate) => void;

  // Getters
  getOrder: (orderId: string) => Order | undefined;
  getOrdersBySymbol: (symbol: string) => Order[];
  getOrdersByStatus: (status: Order['status']) => Order[];
  getAllOrders: () => Order[];
}

export const useOrderStore = create<OrderStore>((set, get) => ({
  // Initial state
  orders: new Map(),
  isLoading: false,
  error: null,
  isConnected: false,

  // Actions
  addOrder: (order: Order) => {
    set((state) => {
      const newOrders = new Map(state.orders);
      newOrders.set(order.id, order);
      return { orders: newOrders };
    });
    logger.info(`Added order ${order.id} to store`);
  },

  updateOrder: (orderId: string, updates: Partial<Order>) => {
    set((state) => {
      const newOrders = new Map(state.orders);
      const existingOrder = newOrders.get(orderId);
      
      if (existingOrder) {
        const updatedOrder = { ...existingOrder, ...updates, updatedAt: new Date().toISOString() };
        newOrders.set(orderId, updatedOrder);
        logger.info(`Updated order ${orderId} in store`);
      }
      
      return { orders: newOrders };
    });
  },

  removeOrder: (orderId: string) => {
    set((state) => {
      const newOrders = new Map(state.orders);
      newOrders.delete(orderId);
      return { orders: newOrders };
    });
    logger.info(`Removed order ${orderId} from store`);
  },

  clearOrders: () => {
    set({ orders: new Map() });
    logger.info('Cleared all orders from store');
  },

  setLoading: (loading: boolean) => {
    set({ isLoading: loading });
  },

  setError: (error: string | null) => {
    set({ error });
  },

  setConnected: (connected: boolean) => {
    set({ isConnected: connected });
  },

  // WebSocket actions
  initializeWebSocket: () => {
    const token = localStorage.getItem('token');
    if (!token) {
      logger.warn('No authentication token found for order WebSocket');
      return;
    }

    const serverUrl = process.env.REACT_APP_WS_URL || 'http://localhost:3002';
    orderWebSocketService.initialize(serverUrl, token);

    // Subscribe to all order updates
    orderWebSocketService.onAllOrderStatusUpdates((update: OrderUpdate) => {
      get().handleOrderUpdate(update);
    });

    // Monitor connection status
    const checkConnection = () => {
      const isConnected = orderWebSocketService.getConnectionStatus();
      get().setConnected(isConnected);
    };

    // Check connection status periodically
    const connectionInterval = setInterval(checkConnection, 1000);

    // Cleanup interval on store destruction
    return () => {
      clearInterval(connectionInterval);
    };
  },

  disconnectWebSocket: () => {
    orderWebSocketService.disconnect();
    get().setConnected(false);
  },

  handleOrderUpdate: (update: OrderUpdate) => {
    const { orderId, status, filledSize, averageFillPrice, smartContractTx, smartContractPositionId, atomicPositionCreation } = update;
    
    // Update existing order or create new one
    const existingOrder = get().orders.get(orderId);
    
    if (existingOrder) {
      // Update existing order
      get().updateOrder(orderId, {
        status,
        filledSize,
        averageFillPrice,
        smartContractTx,
        smartContractPositionId,
        atomicPositionCreation
      });
    } else {
      // Create new order from update
      const newOrder: Order = {
        id: orderId,
        symbol: update.symbol,
        side: update.symbol.includes('BUY') ? 'buy' : 'sell', // This should be determined from the actual order data
        size: filledSize, // This should be the original order size
        price: averageFillPrice,
        orderType: 'market', // This should be determined from the actual order data
        status,
        filledSize,
        averageFillPrice,
        createdAt: new Date(update.timestamp).toISOString(),
        updatedAt: new Date(update.timestamp).toISOString(),
        smartContractTx,
        smartContractPositionId,
        atomicPositionCreation
      };
      
      get().addOrder(newOrder);
    }

    logger.info(`Handled order update for ${orderId}:`, update);
  },

  // Getters
  getOrder: (orderId: string) => {
    return get().orders.get(orderId);
  },

  getOrdersBySymbol: (symbol: string) => {
    const orders = Array.from(get().orders.values());
    return orders.filter(order => order.symbol === symbol);
  },

  getOrdersByStatus: (status: Order['status']) => {
    const orders = Array.from(get().orders.values());
    return orders.filter(order => order.status === status);
  },

  getAllOrders: () => {
    return Array.from(get().orders.values()).sort((a, b) => 
      new Date(b.createdAt).getTime() - new Date(a.createdAt).getTime()
    );
  }
}));

// Export convenience hooks
export const useOrders = () => useOrderStore(state => state.getAllOrders());
export const useOrder = (orderId: string) => useOrderStore(state => state.getOrder(orderId));
export const useOrdersBySymbol = (symbol: string) => useOrderStore(state => state.getOrdersBySymbol(symbol));
export const useOrdersByStatus = (status: Order['status']) => useOrderStore(state => state.getOrdersByStatus(status));
export const useOrderConnection = () => useOrderStore(state => ({
  isConnected: state.isConnected,
  isLoading: state.isLoading,
  error: state.error
}));
