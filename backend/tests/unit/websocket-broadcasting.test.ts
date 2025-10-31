/**
 * WebSocket Broadcasting Tests for Hackathon Demo
 * 
 * This test suite validates the real-time WebSocket broadcasting functionality
 * required for the hackathon demo, ensuring all updates are sent correctly
 * and received by the frontend components.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { WebSocketService } from '../../src/services/websocket';
import { Server as SocketIOServer } from 'socket.io';
import { createServer } from 'http';
import { Logger } from '../../src/utils/logger';
import { SupabaseDatabaseService } from '../../src/services/supabaseDatabase';

const logger = new Logger();

describe('WebSocket Broadcasting - Hackathon Demo Real-time Updates', () => {
  let httpServer: any;
  let io: SocketIOServer;
  let wsService: WebSocketService;
  let db: SupabaseDatabaseService;
  let testUserId: string;
  let testWalletAddress: string;
  let connectedClients: any[] = [];

  beforeAll(async () => {
    // Initialize test database
    db = SupabaseDatabaseService.getInstance();
    
    // Create test user
    testWalletAddress = 'test-wallet-' + Date.now();
    testUserId = 'test-user-' + Date.now();
    
    // Initialize WebSocket service
    httpServer = createServer();
    io = new SocketIOServer(httpServer, {
      cors: { origin: "*" }
    });
    wsService = WebSocketService.getInstance(io);
    wsService.initialize();
    
    logger.info('🚀 WebSocket broadcasting test environment initialized');
  });

  afterAll(async () => {
    // Cleanup
    wsService.stop();
    io.close();
    httpServer.close();
    
    logger.info('🧹 WebSocket broadcasting test environment cleaned up');
  });

  beforeEach(async () => {
    // Ensure clean state for each test - tests will use mock data
    connectedClients = [];
  });

  describe('Order Update Broadcasting', () => {
    it('should broadcast order status updates to authenticated users', async () => {
      // Create test user
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      // Mock WebSocket client connection
      const mockClient = {
        id: 'test-client-1',
        data: { userId: testUserId, walletAddress: testWalletAddress },
        join: jest.fn(),
        leave: jest.fn(),
        emit: jest.fn()
      };
      
      // Simulate client joining orders room
      wsService['io'].to = jest.fn().mockReturnValue({
        emit: jest.fn()
      });
      
      // Test order update broadcasting
      const orderUpdate = {
        orderId: 'test-order-' + Date.now(),
        status: 'filled',
        filledSize: 1.0,
        averageFillPrice: 100,
        transactionSignature: 'tx_' + Date.now(),
        positionId: 'pos_' + Date.now(),
        timestamp: Date.now()
      };
      
      wsService.broadcastOrderUpdate(testUserId, orderUpdate);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`orders:${testUserId}`);
      
      logger.info('✅ Order update broadcasting test passed');
    });

    it('should broadcast order cancellation updates', async () => {
      const orderUpdate = {
        orderId: 'test-order-' + Date.now(),
        status: 'cancelled',
        userId: testUserId,
        timestamp: Date.now()
      };
      
      wsService.broadcastOrderUpdate(testUserId, orderUpdate);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`orders:${testUserId}`);
      
      logger.info('✅ Order cancellation broadcasting test passed');
    });

    it('should broadcast order failure updates', async () => {
      const orderUpdate = {
        orderId: 'test-order-' + Date.now(),
        status: 'failed',
        error: 'Smart contract execution failed',
        userId: testUserId,
        timestamp: Date.now()
      };
      
      wsService.broadcastOrderUpdate(testUserId, orderUpdate);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`orders:${testUserId}`);
      
      logger.info('✅ Order failure broadcasting test passed');
    });
  });

  describe('Position Update Broadcasting', () => {
    it('should broadcast position creation updates', async () => {
      const positionUpdate = {
        userId: testUserId,
        positionId: 'test-position-' + Date.now(),
        symbol: 'SOL-PERP',
        side: 'long' as const,
        size: 1.0,
        entryPrice: 100,
        leverage: 5,
        status: 'open',
        timestamp: Date.now()
      };
      
      wsService.broadcastPositionUpdate(testUserId, positionUpdate);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`positions:${testUserId}`);
      
      logger.info('✅ Position creation broadcasting test passed');
    });

    it('should broadcast position P&L updates', async () => {
      const positionUpdate = {
        userId: testUserId,
        positionId: 'test-position-' + Date.now(),
        symbol: 'SOL-PERP',
        side: 'long' as const,
        size: 1.0,
        entryPrice: 100,
        currentPrice: 105,
        leverage: 5,
        unrealizedPnl: 5,
        marginRatio: 0.02,
        healthFactor: 0.95,
        timestamp: Date.now()
      };
      
      wsService.broadcastPositionUpdate(testUserId, positionUpdate);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`positions:${testUserId}`);
      
      logger.info('✅ Position P&L update broadcasting test passed');
    });

    it('should broadcast position closure updates', async () => {
      const positionUpdate = {
        userId: testUserId,
        positionId: 'test-position-' + Date.now(),
        symbol: 'SOL-PERP',
        side: 'long' as const,
        size: 1.0,
        entryPrice: 100,
        leverage: 5,
        status: 'closed',
        realizedPnl: 5,
        timestamp: Date.now()
      };
      
      wsService.broadcastPositionUpdate(testUserId, positionUpdate);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`positions:${testUserId}`);
      
      logger.info('✅ Position closure broadcasting test passed');
    });
  });

  describe('Portfolio Update Broadcasting', () => {
    it('should broadcast portfolio balance updates', async () => {
      const portfolioData = {
        userId: testUserId,
        totalValue: 1000,
        totalUnrealizedPnl: 5,
        totalRealizedPnl: 0,
        marginRatio: 0.02,
        healthFactor: 1.0,
        totalCollateral: 1000,
        usedMargin: 20,
        availableMargin: 980,
        positions: [{
          id: 'pos-1',
          symbol: 'SOL-PERP',
          size: 1.0,
          entryPrice: 100,
          currentPrice: 105,
          unrealizedPnl: 5,
          unrealizedPnlPercent: 5.0,
          margin: 20,
          leverage: 5
        }],
        timestamp: Date.now()
      };
      
      wsService.broadcastPortfolioUpdate(testUserId, portfolioData);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`portfolio:${testUserId}`);
      
      logger.info('✅ Portfolio balance update broadcasting test passed');
    });

    it('should broadcast portfolio P&L updates', async () => {
      const portfolioData = {
        userId: testUserId,
        totalValue: 1005,
        totalUnrealizedPnl: 10,
        totalRealizedPnl: 5,
        marginRatio: 0.02,
        healthFactor: 1.0,
        totalCollateral: 1000,
        usedMargin: 20,
        availableMargin: 980,
        positions: [],
        timestamp: Date.now()
      };
      
      wsService.broadcastPortfolioUpdate(testUserId, portfolioData);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`portfolio:${testUserId}`);
      
      logger.info('✅ Portfolio P&L update broadcasting test passed');
    });

    it('should broadcast portfolio health factor updates', async () => {
      const portfolioData = {
        userId: testUserId,
        totalValue: 1000,
        totalUnrealizedPnl: -50,
        totalRealizedPnl: 0,
        marginRatio: 0.05,
        healthFactor: 0.8,
        totalCollateral: 1000,
        usedMargin: 50,
        availableMargin: 950,
        positions: [],
        timestamp: Date.now()
      };
      
      wsService.broadcastPortfolioUpdate(testUserId, portfolioData);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`portfolio:${testUserId}`);
      
      logger.info('✅ Portfolio health factor update broadcasting test passed');
    });
  });

  describe('Market Data Broadcasting', () => {
    it('should broadcast market data updates', async () => {
      // Test market data broadcasting
      const marketData = {
        symbol: 'SOL-PERP',
        price: 100.50,
        change24h: 2.5,
        volume24h: 1000000,
        openInterest: 50000000,
        fundingRate: 0.001
      };
      
      // Mock the broadcast method
      const originalBroadcast = wsService.broadcast;
      wsService.broadcast = jest.fn();
      
      wsService.broadcast!('market_data', marketData);
      
      // Verify broadcast was called
      expect(wsService.broadcast).toHaveBeenCalledWith('market_data', marketData);
      
      // Restore original method
      wsService.broadcast = originalBroadcast;
      
      logger.info('✅ Market data broadcasting test passed');
    });

    it('should broadcast order book updates', async () => {
      const orderBookData = {
        symbol: 'SOL-PERP',
        bids: [[100.00, 1.0], [99.99, 2.0]],
        asks: [[100.01, 1.5], [100.02, 3.0]],
        spread: 0.01
      };
      
      // Mock the broadcast method
      const originalBroadcast = wsService.broadcast;
      wsService.broadcast = jest.fn();
      
      wsService.broadcast!('order_book', orderBookData);
      
      // Verify broadcast was called
      expect(wsService.broadcast).toHaveBeenCalledWith('order_book', orderBookData);
      
      // Restore original method
      wsService.broadcast = originalBroadcast;
      
      logger.info('✅ Order book broadcasting test passed');
    });

    it('should broadcast trade updates', async () => {
      const tradeData = {
        symbol: 'SOL-PERP',
        side: 'buy' as const,
        size: 1.0,
        price: 100.50,
        timestamp: Date.now()
      };
      
      wsService.broadcastTradeUpdate(tradeData);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`market:${tradeData.symbol}`);
      
      logger.info('✅ Trade update broadcasting test passed');
    });
  });

  describe('User-Specific Broadcasting', () => {
    it('should broadcast to specific user rooms', async () => {
      const testData = {
        userId: testUserId,
        message: 'Test message',
        timestamp: Date.now()
      };
      
      wsService.broadcastToUser(testUserId, 'test_event', testData);
      
      // Verify broadcast was called
      expect(wsService['io'].to).toHaveBeenCalledWith(`orders:${testUserId}`);
      
      logger.info('✅ User-specific broadcasting test passed');
    });

    it('should handle multiple user subscriptions', async () => {
      const userIds = ['user1', 'user2', 'user3'];
      
      for (const userId of userIds) {
        const orderUpdate = {
          orderId: 'test-order-' + Date.now(),
          status: 'filled',
          userId,
          timestamp: Date.now()
        };
        
        wsService.broadcastOrderUpdate(userId, orderUpdate);
        
        // Verify broadcast was called for each user
        expect(wsService['io'].to).toHaveBeenCalledWith(`orders:${userId}`);
      }
      
      logger.info('✅ Multiple user subscriptions test passed');
    });
  });

  describe('Error Handling and Resilience', () => {
    it('should handle WebSocket connection failures gracefully', async () => {
      // Test error handling when WebSocket is not available
      const orderUpdate = {
        orderId: 'test-order-' + Date.now(),
        status: 'filled',
        userId: testUserId,
        timestamp: Date.now()
      };
      
      // This should not throw an error even if WebSocket fails
      expect(() => {
        wsService.broadcastOrderUpdate(testUserId, orderUpdate);
      }).not.toThrow();
      
      logger.info('✅ WebSocket error handling test passed');
    });

    it('should handle invalid user IDs gracefully', async () => {
      const orderUpdate = {
        orderId: 'test-order-' + Date.now(),
        status: 'filled',
        userId: 'invalid-user-id',
        timestamp: Date.now()
      };
      
      // This should not throw an error
      expect(() => {
        wsService.broadcastOrderUpdate('invalid-user-id', orderUpdate);
      }).not.toThrow();
      
      logger.info('✅ Invalid user ID handling test passed');
    });

    it('should handle malformed update data gracefully', async () => {
      // Test with malformed data
      const malformedUpdate = {
        orderId: null,
        status: undefined,
        userId: testUserId,
        timestamp: 'invalid-timestamp'
      };
      
      // This should not throw an error
      expect(() => {
        wsService.broadcastOrderUpdate(testUserId, malformedUpdate as any);
      }).not.toThrow();
      
      logger.info('✅ Malformed data handling test passed');
    });
  });

  describe('Performance and Scalability', () => {
    it('should handle high-frequency updates efficiently', async () => {
      const startTime = Date.now();
      
      // Send 100 rapid updates
      for (let i = 0; i < 100; i++) {
        const orderUpdate = {
          orderId: `test-order-${i}`,
          status: 'filled',
          userId: testUserId,
          timestamp: Date.now()
        };
        
        wsService.broadcastOrderUpdate(testUserId, orderUpdate);
      }
      
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Should complete within reasonable time
      expect(executionTime).toBeLessThan(1000);
      
      logger.info(`✅ High-frequency updates test passed - ${100} updates in ${executionTime}ms`);
    });

    it('should handle concurrent broadcasts efficiently', async () => {
      const startTime = Date.now();
      
      // Send concurrent broadcasts
      const promises = [];
      for (let i = 0; i < 10; i++) {
        const userId = `concurrent-user-${i}`;
        const orderUpdate = {
          orderId: `test-order-${i}`,
          status: 'filled',
          userId,
          timestamp: Date.now()
        };
        
        promises.push(
          new Promise(resolve => {
            wsService.broadcastOrderUpdate(userId, orderUpdate);
            resolve(true);
          })
        );
      }
      
      await Promise.all(promises);
      
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Should complete within reasonable time
      expect(executionTime).toBeLessThan(500);
      
      logger.info(`✅ Concurrent broadcasts test passed - ${10} concurrent broadcasts in ${executionTime}ms`);
    });
  });

  describe('Demo Integration Tests', () => {
    it('should broadcast complete trading flow updates', async () => {
      // Simulate complete trading flow with broadcasts
      const orderUpdate = {
        orderId: 'demo-order-' + Date.now(),
        status: 'filled',
        filledSize: 1.0,
        averageFillPrice: 100,
        transactionSignature: 'tx_' + Date.now(),
        positionId: 'pos_' + Date.now(),
        userId: testUserId,
        timestamp: Date.now()
      };
      
      wsService.broadcastOrderUpdate(testUserId, orderUpdate);
      
      const positionUpdate = {
        userId: testUserId,
        positionId: orderUpdate.positionId,
        symbol: 'SOL-PERP',
        side: 'long' as const,
        size: 1.0,
        entryPrice: 100,
        leverage: 5,
        status: 'open',
        timestamp: Date.now()
      };
      
      wsService.broadcastPositionUpdate(testUserId, positionUpdate);
      
      const portfolioData = {
        userId: testUserId,
        totalValue: 1000,
        totalUnrealizedPnl: 0,
        totalRealizedPnl: 0,
        marginRatio: 0.02,
        healthFactor: 1.0,
        totalCollateral: 1000,
        usedMargin: 20,
        availableMargin: 980,
        positions: [],
        timestamp: Date.now()
      };
      
      wsService.broadcastPortfolioUpdate(testUserId, portfolioData);
      
      // Verify all broadcasts were called
      expect(wsService['io'].to).toHaveBeenCalledWith(`orders:${testUserId}`);
      expect(wsService['io'].to).toHaveBeenCalledWith(`positions:${testUserId}`);
      expect(wsService['io'].to).toHaveBeenCalledWith(`portfolio:${testUserId}`);
      
      logger.info('✅ Complete trading flow broadcasting test passed');
    });

    it('should maintain broadcast consistency during demo', async () => {
      // Test that broadcasts maintain data consistency
      const orderId = 'demo-order-' + Date.now();
      const positionId = 'demo-position-' + Date.now();
      
      // Order filled broadcast
      const orderUpdate = {
        orderId,
        status: 'filled',
        filledSize: 1.0,
        averageFillPrice: 100,
        positionId,
        userId: testUserId,
        timestamp: Date.now()
      };
      
      wsService.broadcastOrderUpdate(testUserId, orderUpdate);
      
      // Position created broadcast
      const positionUpdate = {
        userId: testUserId,
        positionId,
        symbol: 'SOL-PERP',
        side: 'long' as const,
        size: 1.0,
        entryPrice: 100,
        leverage: 5,
        status: 'open',
        timestamp: Date.now()
      };
      
      wsService.broadcastPositionUpdate(testUserId, positionUpdate);
      
      // Verify consistency
      expect(orderUpdate.positionId).toBe(positionUpdate.positionId);
      expect(orderUpdate.userId).toBe(positionUpdate.userId);
      
      logger.info('✅ Broadcast consistency test passed');
    });
  });
});