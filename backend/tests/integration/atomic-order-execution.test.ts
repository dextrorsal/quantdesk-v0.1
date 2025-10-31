import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import { matchingService } from '../../src/services/matching';
import { smartContractService } from '../../src/services/smartContractService';
import { pythOracleService } from '../../src/services/pythOracleService';
import { WebSocketService } from '../../src/services/websocket';
import { getSupabaseService } from '../../src/services/supabaseService';

// Mock dependencies
vi.mock('../../src/services/smartContractService');
vi.mock('../../src/services/pythOracleService');
vi.mock('../../src/services/websocket');
vi.mock('../../src/services/supabaseService');

describe('Atomic Order Execution Integration', () => {
  let mockSupabase: any;
  let mockWebSocket: any;

  beforeEach(() => {
    // Reset all mocks
    vi.clearAllMocks();

    // Mock Supabase service
    mockSupabase = {
      getMarketBySymbol: vi.fn().mockResolvedValue({ id: 'market-1', symbol: 'BTC-PERP' }),
      insertOrder: vi.fn().mockResolvedValue({ id: 'order-123' }),
      getPendingOrders: vi.fn().mockResolvedValue([]),
      getClient: vi.fn().mockReturnValue({
        from: vi.fn().mockReturnValue({
          update: vi.fn().mockReturnValue({
            eq: vi.fn().mockResolvedValue({ data: null, error: null })
          }),
          select: vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              single: vi.fn().mockResolvedValue({ 
                data: { smart_contract_tx: 'tx-123', smart_contract_position_id: 'pos-123' }, 
                error: null 
              })
            })
          })
        })
      })
    };

    (getSupabaseService as any).mockReturnValue(mockSupabase);

    // Mock WebSocket service
    mockWebSocket = {
      broadcastToUser: vi.fn(),
      broadcast: vi.fn()
    };
    (WebSocketService as any).current = mockWebSocket;

    // Mock Oracle service
    (pythOracleService.getLatestPrice as any).mockResolvedValue(50000);

    // Mock Smart Contract service
    (smartContractService.executeOrder as any).mockResolvedValue({
      success: true,
      transactionSignature: 'tx-123',
      positionId: 'pos-123'
    });
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('Atomic Order Execution Flow', () => {
    it('should execute order with atomic position creation', async () => {
      // Arrange
      const orderInput = {
        userId: 'user-123',
        symbol: 'BTC-PERP',
        side: 'buy' as const,
        size: 1,
        orderType: 'market' as const,
        leverage: 2
      };

      // Act
      const result = await matchingService.placeOrder(orderInput);

      // Assert
      expect(result).toEqual({
        orderId: 'order-123',
        filled: true,
        fills: [{ price: 50000, size: 1 }],
        averageFillPrice: 50000
      });

      // Verify smart contract execution was called
      expect(smartContractService.executeOrder).toHaveBeenCalledWith({
        orderId: 'order-123',
        userId: 'user-123',
        marketSymbol: 'BTC-PERP',
        side: 'long',
        size: 1,
        price: 50000,
        leverage: 2,
        orderType: 'market'
      });

      // Verify order was updated with smart contract details
      expect(mockSupabase.getClient().from().update().eq).toHaveBeenCalledWith(
        expect.objectContaining({
          smart_contract_tx: 'tx-123',
          smart_contract_position_id: 'pos-123'
        })
      );

      // Verify WebSocket broadcast includes smart contract status
      expect(mockWebSocket.broadcastToUser).toHaveBeenCalledWith(
        'user-123',
        'order_update',
        expect.objectContaining({
          symbol: 'BTC-PERP',
          orderId: 'order-123',
          status: 'filled',
          smartContractTx: 'tx-123',
          smartContractPositionId: 'pos-123',
          atomicPositionCreation: true
        })
      );
    });

    it('should handle smart contract execution failure with rollback', async () => {
      // Arrange
      const orderInput = {
        userId: 'user-123',
        symbol: 'BTC-PERP',
        side: 'buy' as const,
        size: 1,
        orderType: 'market' as const,
        leverage: 2
      };

      // Mock smart contract failure
      (smartContractService.executeOrder as any).mockResolvedValue({
        success: false,
        error: 'Insufficient collateral'
      });

      // Act & Assert
      await expect(matchingService.placeOrder(orderInput)).rejects.toThrow('Smart contract execution failed: Insufficient collateral');

      // Verify order was rolled back to failed status
      expect(mockSupabase.getClient().from().update().eq).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'failed',
          error_message: 'Insufficient collateral'
        })
      );

      // Verify WebSocket broadcast was not sent for failed orders
      expect(mockWebSocket.broadcastToUser).not.toHaveBeenCalled();
    });

    it('should use cached Oracle prices to reduce redundant calls', async () => {
      // Arrange
      const orderInput = {
        userId: 'user-123',
        symbol: 'BTC-PERP',
        side: 'buy' as const,
        size: 1,
        orderType: 'market' as const,
        leverage: 2
      };

      // Act - Place multiple orders quickly
      await matchingService.placeOrder(orderInput);
      await matchingService.placeOrder({ ...orderInput, userId: 'user-456' });

      // Assert - Oracle should only be called once due to caching
      expect(pythOracleService.getLatestPrice).toHaveBeenCalledTimes(1);
    });

    it('should handle partial fills correctly', async () => {
      // Arrange
      const orderInput = {
        userId: 'user-123',
        symbol: 'BTC-PERP',
        side: 'buy' as const,
        size: 2,
        orderType: 'limit' as const,
        price: 49000,
        leverage: 2
      };

      // Mock partial fill scenario
      mockSupabase.getPendingOrders.mockResolvedValue([
        { id: 'maker-order-1', remaining_size: '1', price: '49000' }
      ]);

      // Act
      const result = await matchingService.placeOrder(orderInput);

      // Assert
      expect(result.filled).toBe(false);
      expect(result.fills).toEqual([{ price: 49000, size: 1, makerOrderId: 'maker-order-1' }]);

      // Verify order status is partially_filled
      expect(mockSupabase.getClient().from().update().eq).toHaveBeenCalledWith(
        expect.objectContaining({
          status: 'partially_filled',
          filled_size: 1
        })
      );

      // Verify smart contract execution was not called for partial fills
      expect(smartContractService.executeOrder).not.toHaveBeenCalled();
    });

    it('should validate order authorization before execution', async () => {
      // Arrange
      const orderInput = {
        userId: 'user-123',
        symbol: 'BTC-PERP',
        side: 'buy' as const,
        size: 1000, // Large size that should trigger authorization failure
        orderType: 'market' as const,
        leverage: 10
      };

      // Mock authorization failure
      const mockOrderAuthorizationService = await import('../../src/services/orderAuthorizationService');
      (mockOrderAuthorizationService.orderAuthorizationService.authorizeOrder as any).mockResolvedValue({
        authorized: false,
        reason: 'Position size exceeds maximum allowed',
        code: 'POSITION_SIZE_EXCEEDED',
        riskLevel: 'HIGH'
      });

      // Act & Assert
      await expect(matchingService.placeOrder(orderInput)).rejects.toThrow('Order authorization failed: Position size exceeds maximum allowed');

      // Verify smart contract execution was not called
      expect(smartContractService.executeOrder).not.toHaveBeenCalled();
    });
  });

  describe('Error Handling and Recovery', () => {
    it('should handle database errors gracefully', async () => {
      // Arrange
      const orderInput = {
        userId: 'user-123',
        symbol: 'BTC-PERP',
        side: 'buy' as const,
        size: 1,
        orderType: 'market' as const,
        leverage: 2
      };

      // Mock database error
      mockSupabase.insertOrder.mockRejectedValue(new Error('Database connection failed'));

      // Act & Assert
      await expect(matchingService.placeOrder(orderInput)).rejects.toThrow('Database connection failed');

      // Verify error handling service was called
      const mockErrorHandlingService = await import('../../src/services/errorHandlingService');
      expect(mockErrorHandlingService.errorHandlingService.handleError).toHaveBeenCalled();
    });

    it('should handle Oracle service failures', async () => {
      // Arrange
      const orderInput = {
        userId: 'user-123',
        symbol: 'BTC-PERP',
        side: 'buy' as const,
        size: 1,
        orderType: 'market' as const,
        leverage: 2
      };

      // Mock Oracle failure
      (pythOracleService.getLatestPrice as any).mockResolvedValue(null);

      // Act & Assert
      await expect(matchingService.placeOrder(orderInput)).rejects.toThrow('Price unavailable');
    });
  });

  describe('Performance and Optimization', () => {
    it('should meet performance targets for order execution', async () => {
      // Arrange
      const orderInput = {
        userId: 'user-123',
        symbol: 'BTC-PERP',
        side: 'buy' as const,
        size: 1,
        orderType: 'market' as const,
        leverage: 2
      };

      const startTime = Date.now();

      // Act
      await matchingService.placeOrder(orderInput);

      const executionTime = Date.now() - startTime;

      // Assert - Should complete within 200ms
      expect(executionTime).toBeLessThan(200);
    });

    it('should handle concurrent order processing', async () => {
      // Arrange
      const orders = Array.from({ length: 10 }, (_, i) => ({
        userId: `user-${i}`,
        symbol: 'BTC-PERP',
        side: 'buy' as const,
        size: 1,
        orderType: 'market' as const,
        leverage: 2
      }));

      // Act - Process orders concurrently
      const results = await Promise.all(
        orders.map(order => matchingService.placeOrder(order))
      );

      // Assert - All orders should be processed successfully
      expect(results).toHaveLength(10);
      results.forEach(result => {
        expect(result.filled).toBe(true);
        expect(result.orderId).toBeDefined();
      });

      // Verify Oracle was called efficiently (cached)
      expect(pythOracleService.getLatestPrice).toHaveBeenCalledTimes(1);
    });
  });
});
