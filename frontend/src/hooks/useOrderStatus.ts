import React, { useEffect, useCallback, useRef, useState } from 'react';
import { orderWebSocketService, OrderUpdate, OrderStatusCallback } from '../services/orderWebSocketService';
import { logger } from '../utils/logger';

export interface UseOrderStatusOptions {
  orderId?: string;
  onOrderUpdate?: (update: OrderUpdate) => void;
  autoConnect?: boolean;
}

export interface UseOrderStatusReturn {
  isConnected: boolean;
  subscribeToOrder: (orderId: string, callback: OrderStatusCallback) => void;
  unsubscribeFromOrder: (orderId: string, callback: OrderStatusCallback) => void;
  subscribeToAllOrders: (callback: OrderStatusCallback) => void;
  unsubscribeFromAllOrders: (callback: OrderStatusCallback) => void;
  connect: () => void;
  disconnect: () => void;
}

/**
 * Hook for managing order status updates via WebSocket
 */
export const useOrderStatus = (options: UseOrderStatusOptions = {}): UseOrderStatusReturn => {
  const { orderId, onOrderUpdate, autoConnect = true } = options;
  const callbacksRef = useRef<Map<string, OrderStatusCallback[]>>(new Map());

  // Initialize WebSocket connection
  useEffect(() => {
    if (autoConnect) {
      const token = localStorage.getItem('token');
      if (token) {
        const serverUrl = process.env.REACT_APP_WS_URL || 'http://localhost:3002';
        orderWebSocketService.initialize(serverUrl, token);
      } else {
        logger.warn('No authentication token found for order WebSocket');
      }
    }

    return () => {
      // Cleanup on unmount
      orderWebSocketService.disconnect();
    };
  }, [autoConnect]);

  // Handle order updates
  useEffect(() => {
    if (onOrderUpdate) {
      if (orderId) {
        orderWebSocketService.onOrderStatusUpdate(orderId, onOrderUpdate);
      } else {
        orderWebSocketService.onAllOrderStatusUpdates(onOrderUpdate);
      }
    }

    return () => {
      if (onOrderUpdate) {
        if (orderId) {
          orderWebSocketService.offOrderStatusUpdate(orderId, onOrderUpdate);
        } else {
          orderWebSocketService.offOrderStatusUpdate('*', onOrderUpdate);
        }
      }
    };
  }, [orderId, onOrderUpdate]);

  // Subscribe to specific order updates
  const subscribeToOrder = useCallback((orderId: string, callback: OrderStatusCallback) => {
    orderWebSocketService.onOrderStatusUpdate(orderId, callback);
    
    // Track callback for cleanup
    if (!callbacksRef.current.has(orderId)) {
      callbacksRef.current.set(orderId, []);
    }
    callbacksRef.current.get(orderId)!.push(callback);
  }, []);

  // Unsubscribe from specific order updates
  const unsubscribeFromOrder = useCallback((orderId: string, callback: OrderStatusCallback) => {
    orderWebSocketService.offOrderStatusUpdate(orderId, callback);
    
    // Remove from tracked callbacks
    const callbacks = callbacksRef.current.get(orderId);
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }, []);

  // Subscribe to all order updates
  const subscribeToAllOrders = useCallback((callback: OrderStatusCallback) => {
    orderWebSocketService.onAllOrderStatusUpdates(callback);
    
    // Track callback for cleanup
    if (!callbacksRef.current.has('*')) {
      callbacksRef.current.set('*', []);
    }
    callbacksRef.current.get('*')!.push(callback);
  }, []);

  // Unsubscribe from all order updates
  const unsubscribeFromAllOrders = useCallback((callback: OrderStatusCallback) => {
    orderWebSocketService.offOrderStatusUpdate('*', callback);
    
    // Remove from tracked callbacks
    const callbacks = callbacksRef.current.get('*');
    if (callbacks) {
      const index = callbacks.indexOf(callback);
      if (index > -1) {
        callbacks.splice(index, 1);
      }
    }
  }, []);

  // Manual connect
  const connect = useCallback(() => {
    const token = localStorage.getItem('token');
    if (token) {
      const serverUrl = process.env.REACT_APP_WS_URL || 'http://localhost:3002';
      orderWebSocketService.initialize(serverUrl, token);
    }
  }, []);

  // Manual disconnect
  const disconnect = useCallback(() => {
    orderWebSocketService.disconnect();
  }, []);

  // Cleanup all tracked callbacks on unmount
  useEffect(() => {
    return () => {
      callbacksRef.current.forEach((callbacks, orderId) => {
        callbacks.forEach(callback => {
          if (orderId === '*') {
            orderWebSocketService.offOrderStatusUpdate('*', callback);
          } else {
            orderWebSocketService.offOrderStatusUpdate(orderId, callback);
          }
        });
      });
      callbacksRef.current.clear();
    };
  }, []);

  return {
    isConnected: orderWebSocketService.getConnectionStatus(),
    subscribeToOrder,
    unsubscribeFromOrder,
    subscribeToAllOrders,
    unsubscribeFromAllOrders,
    connect,
    disconnect
  };
};

/**
 * Hook for tracking a specific order's status
 */
export const useOrderStatusTracking = (orderId: string) => {
  const [orderStatus, setOrderStatus] = useState<OrderUpdate | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const handleOrderUpdate = useCallback((update: OrderUpdate) => {
    if (update.orderId === orderId) {
      setOrderStatus(update);
      setIsLoading(false);
    }
  }, [orderId]);

  const { isConnected } = useOrderStatus({
    orderId,
    onOrderUpdate: handleOrderUpdate
  });

  return {
    orderStatus,
    isLoading,
    isConnected
  };
};
