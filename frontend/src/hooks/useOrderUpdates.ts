import { useEffect, useState, useCallback } from 'react';

export interface OrderStatusUpdate {
  orderId: string;
  status: 'pending' | 'filled' | 'cancelled' | 'expired' | 'partially_filled';
  symbol: string;
  filledSize?: number;
  averageFillPrice?: number;
  timestamp: number;
}

export interface UseOrderUpdatesReturn {
  orderUpdates: OrderStatusUpdate[];
  subscribeToOrderUpdates: (callback: (update: OrderStatusUpdate) => void) => () => void;
  clearOrderUpdates: () => void;
}

/**
 * Hook for handling real-time order status updates
 */
export const useOrderUpdates = (): UseOrderUpdatesReturn => {
  const [orderUpdates, setOrderUpdates] = useState<OrderStatusUpdate[]>([]);

  const subscribeToOrderUpdates = useCallback((callback: (update: OrderStatusUpdate) => void) => {
    const handleOrderUpdate = (event: CustomEvent) => {
      const update: OrderStatusUpdate = event.detail;
      console.log('📋 Received order update:', update);
      
      // Update local state
      setOrderUpdates(prev => {
        const existingIndex = prev.findIndex(o => o.orderId === update.orderId);
        if (existingIndex >= 0) {
          // Update existing order
          const newUpdates = [...prev];
          newUpdates[existingIndex] = update;
          return newUpdates;
        } else {
          // Add new order update
          return [update, ...prev].slice(0, 50); // Keep last 50 updates
        }
      });
      
      // Call the callback
      callback(update);
    };

    // Listen for order status updates
    window.addEventListener('orderStatusUpdate', handleOrderUpdate as EventListener);

    // Return cleanup function
    return () => {
      window.removeEventListener('orderStatusUpdate', handleOrderUpdate as EventListener);
    };
  }, []);

  const clearOrderUpdates = useCallback(() => {
    setOrderUpdates([]);
  }, []);

  return {
    orderUpdates,
    subscribeToOrderUpdates,
    clearOrderUpdates
  };
};

/**
 * Hook for getting order status for a specific order
 */
export const useOrderStatus = (orderId: string): OrderStatusUpdate | null => {
  const { orderUpdates } = useOrderUpdates();
  
  return orderUpdates.find(update => update.orderId === orderId) || null;
};

/**
 * Hook for getting all orders for a specific symbol
 */
export const useOrdersForSymbol = (symbol: string): OrderStatusUpdate[] => {
  const { orderUpdates } = useOrderUpdates();
  
  return orderUpdates.filter(update => update.symbol === symbol);
};
