import { describe, it, expect, vi, beforeEach } from 'vitest';
import { useOrderStore } from '../stores/orderStore';
import { OrderUpdate } from '../services/orderWebSocketService';

// Mock the orderWebSocketService
vi.mock('../services/orderWebSocketService', () => ({
  orderWebSocketService: {
    initialize: vi.fn(),
    connect: vi.fn(),
    disconnect: vi.fn(),
    onOrderStatusUpdate: vi.fn(),
    offOrderStatusUpdate: vi.fn(),
  }
}));

describe('Frontend Order Integration', () => {
  beforeEach(() => {
    // Reset store state
    useOrderStore.getState().clearOrders();
  });

  describe('Order Status Updates with Smart Contract Data', () => {
    it('should handle order updates with atomic position creation', () => {
      const store = useOrderStore.getState();
      
      // Simulate order update from backend with atomic execution
      const orderUpdate: OrderUpdate = {
        symbol: 'BTC-PERP',
        orderId: 'order-123',
        status: 'filled',
        filledSize: 1,
        averageFillPrice: 50000,
        userId: 'user-123',
        timestamp: Date.now(),
        smartContractTx: 'tx_abc123def456',
        smartContractPositionId: 'pos_xyz789',
        atomicPositionCreation: true
      };

      // Handle the order update
      store.handleOrderUpdate(orderUpdate);

      // Verify order was created with smart contract data
      const order = store.getOrder('order-123');
      expect(order).toBeDefined();
      expect(order?.status).toBe('filled');
      expect(order?.smartContractTx).toBe('tx_abc123def456');
      expect(order?.smartContractPositionId).toBe('pos_xyz789');
      expect(order?.atomicPositionCreation).toBe(true);
    });

    it('should handle order updates without atomic position creation', () => {
      const store = useOrderStore.getState();
      
      // Simulate order update without atomic execution
      const orderUpdate: OrderUpdate = {
        symbol: 'ETH-PERP',
        orderId: 'order-456',
        status: 'partially_filled',
        filledSize: 0.5,
        averageFillPrice: 3000,
        userId: 'user-456',
        timestamp: Date.now(),
        smartContractTx: undefined,
        smartContractPositionId: undefined,
        atomicPositionCreation: false
      };

      // Handle the order update
      store.handleOrderUpdate(orderUpdate);

      // Verify order was created without smart contract data
      const order = store.getOrder('order-456');
      expect(order).toBeDefined();
      expect(order?.status).toBe('partially_filled');
      expect(order?.smartContractTx).toBeUndefined();
      expect(order?.smartContractPositionId).toBeUndefined();
      expect(order?.atomicPositionCreation).toBe(false);
    });

    it('should update existing orders with new smart contract data', () => {
      const store = useOrderStore.getState();
      
      // Create initial order
      const initialOrder = {
        id: 'order-789',
        symbol: 'SOL-PERP',
        side: 'buy' as const,
        size: 10,
        price: 100,
        orderType: 'market' as const,
        status: 'pending' as const,
        filledSize: 0,
        createdAt: new Date().toISOString(),
        updatedAt: new Date().toISOString(),
      };
      
      store.addOrder(initialOrder);

      // Simulate order update with smart contract execution
      const orderUpdate: OrderUpdate = {
        symbol: 'SOL-PERP',
        orderId: 'order-789',
        status: 'filled',
        filledSize: 10,
        averageFillPrice: 100,
        userId: 'user-789',
        timestamp: Date.now(),
        smartContractTx: 'tx_sol123',
        smartContractPositionId: 'pos_sol456',
        atomicPositionCreation: true
      };

      // Handle the order update
      store.handleOrderUpdate(orderUpdate);

      // Verify order was updated with smart contract data
      const updatedOrder = store.getOrder('order-789');
      expect(updatedOrder?.status).toBe('filled');
      expect(updatedOrder?.smartContractTx).toBe('tx_sol123');
      expect(updatedOrder?.smartContractPositionId).toBe('pos_sol456');
      expect(updatedOrder?.atomicPositionCreation).toBe(true);
    });
  });

  describe('Order Status Consistency', () => {
    it('should maintain consistent order status across updates', () => {
      const store = useOrderStore.getState();
      
      // Create order
      const orderUpdate: OrderUpdate = {
        symbol: 'BTC-PERP',
        orderId: 'order-consistency-test',
        status: 'pending',
        filledSize: 0,
        userId: 'user-test',
        timestamp: Date.now(),
      };

      store.handleOrderUpdate(orderUpdate);
      let order = store.getOrder('order-consistency-test');
      expect(order?.status).toBe('pending');

      // Update to filled
      const filledUpdate: OrderUpdate = {
        ...orderUpdate,
        status: 'filled',
        filledSize: 1,
        averageFillPrice: 50000,
        smartContractTx: 'tx_filled',
        atomicPositionCreation: true
      };

      store.handleOrderUpdate(filledUpdate);
      order = store.getOrder('order-consistency-test');
      expect(order?.status).toBe('filled');
      expect(order?.filledSize).toBe(1);
      expect(order?.smartContractTx).toBe('tx_filled');
    });
  });

  describe('Error Handling', () => {
    it('should handle order updates with error messages', () => {
      const store = useOrderStore.getState();
      
      // Simulate failed order
      const failedOrderUpdate: OrderUpdate = {
        symbol: 'BTC-PERP',
        orderId: 'order-failed',
        status: 'failed',
        filledSize: 0,
        userId: 'user-test',
        timestamp: Date.now(),
      };

      store.handleOrderUpdate(failedOrderUpdate);

      const order = store.getOrder('order-failed');
      expect(order?.status).toBe('failed');
      expect(order?.filledSize).toBe(0);
    });
  });
});
