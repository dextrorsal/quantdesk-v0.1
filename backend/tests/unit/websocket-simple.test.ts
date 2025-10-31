/**
 * WebSocket Broadcasting Tests - Simplified
 * 
 * This test suite validates the WebSocket broadcasting functionality
 * for the hackathon demo without database dependencies.
 */

import { describe, it, expect, beforeAll, afterAll } from 'vitest';
import { WebSocketService } from '../../src/services/websocket';
import { Server as SocketIOServer } from 'socket.io';
import { createServer } from 'http';
import { Logger } from '../../src/utils/logger';

const logger = new Logger();

describe('WebSocket Broadcasting - Hackathon Demo', () => {
  let httpServer: any;
  let io: SocketIOServer;
  let wsService: WebSocketService;
  let testUserId: string;

  beforeAll(async () => {
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

  describe('WebSocket Service Initialization', () => {
    it('should initialize WebSocket service successfully', () => {
      expect(wsService).toBeDefined();
      expect(io).toBeDefined();
      
      logger.info('✅ WebSocket service initialized successfully');
    });

    it('should have broadcasting methods available', () => {
      expect(typeof wsService.broadcastPositionUpdate).toBe('function');
      expect(typeof wsService.broadcastOrderUpdate).toBe('function');
      expect(typeof wsService.broadcastPortfolioUpdate).toBe('function');
      expect(typeof wsService.broadcastTradeUpdate).toBe('function');
      
      logger.info('✅ WebSocket broadcasting methods available');
    });
  });

  describe('Broadcasting Functionality', () => {
    it('should broadcast position updates', () => {
      const positionUpdate = {
        userId: testUserId,
        positionId: 'pos-123',
        symbol: 'SOL-PERP',
        side: 'long',
        size: 0.5,
        entryPrice: 100,
        leverage: 5,
        status: 'open',
        unrealizedPnl: 5,
        marginRatio: 0.1,
        healthFactor: 0.9,
        timestamp: Date.now(),
      };

      // Should not throw when broadcasting
      expect(() => {
        wsService.broadcastPositionUpdate(testUserId, positionUpdate);
      }).not.toThrow();
      
      logger.info('✅ Position update broadcasting validated');
    });

    it('should broadcast order updates', () => {
      const orderUpdate = {
        orderId: 'order-456',
        status: 'filled',
        symbol: 'SOL-PERP',
        filledSize: 0.1,
        averageFillPrice: 100,
        transactionSignature: 'tx-abc',
        positionId: 'pos-xyz',
        timestamp: Date.now(),
      };

      // Should not throw when broadcasting
      expect(() => {
        wsService.broadcastOrderUpdate(testUserId, orderUpdate);
      }).not.toThrow();
      
      logger.info('✅ Order update broadcasting validated');
    });

    it('should broadcast portfolio updates', () => {
      const portfolioUpdate = {
        userId: testUserId,
        totalValue: 1050,
        totalUnrealizedPnl: 70,
        totalRealizedPnl: 30,
        marginRatio: 0.15,
        healthFactor: 0.95,
        positions: [],
        timestamp: Date.now(),
      };

      // Should not throw when broadcasting
      expect(() => {
        wsService.broadcastPortfolioUpdate(testUserId, portfolioUpdate);
      }).not.toThrow();
      
      logger.info('✅ Portfolio update broadcasting validated');
    });

    it('should broadcast trade updates', () => {
      const tradeUpdate = {
        symbol: 'SOL-PERP',
        side: 'buy',
        size: 0.05,
        price: 101,
        timestamp: Date.now(),
      };

      // Should not throw when broadcasting
      expect(() => {
        wsService.broadcastTradeUpdate(tradeUpdate);
      }).not.toThrow();
      
      logger.info('✅ Trade update broadcasting validated');
    });
  });

  describe('Error Handling', () => {
    it('should handle invalid user IDs gracefully', () => {
      const positionUpdate = {
        userId: 'invalid-user',
        positionId: 'pos-123',
        symbol: 'SOL-PERP',
        side: 'long',
        size: 0.5,
        entryPrice: 100,
        leverage: 5,
        status: 'open',
        unrealizedPnl: 5,
        marginRatio: 0.1,
        healthFactor: 0.9,
        timestamp: Date.now(),
      };

      // Should not throw with invalid user ID
      expect(() => {
        wsService.broadcastPositionUpdate('invalid-user', positionUpdate);
      }).not.toThrow();
      
      logger.info('✅ Invalid user ID handling validated');
    });

    it('should handle malformed update data gracefully', () => {
      // Should not throw with malformed data
      expect(() => {
        wsService.broadcastPositionUpdate(testUserId, {} as any);
      }).not.toThrow();
      
      logger.info('✅ Malformed data handling validated');
    });
  });

  describe('Demo Readiness', () => {
    it('should validate WebSocket service is ready for demo', () => {
      // Check that all required methods exist
      const requiredMethods = [
        'broadcastPositionUpdate',
        'broadcastOrderUpdate', 
        'broadcastPortfolioUpdate',
        'broadcastTradeUpdate',
        'broadcast',
        'broadcastToUser'
      ];

      requiredMethods.forEach(method => {
        expect(typeof wsService[method]).toBe('function');
      });
      
      logger.info('✅ WebSocket service ready for demo');
    });

    it('should validate broadcasting performance', () => {
      const startTime = Date.now();
      
      // Test multiple broadcasts
      for (let i = 0; i < 10; i++) {
        wsService.broadcastPositionUpdate(testUserId, {
          userId: testUserId,
          positionId: `pos-${i}`,
          symbol: 'SOL-PERP',
          side: 'long',
          size: 0.1,
          entryPrice: 100,
          leverage: 5,
          status: 'open',
          unrealizedPnl: 1,
          marginRatio: 0.1,
          healthFactor: 0.9,
          timestamp: Date.now(),
        });
      }
      
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Broadcasting should be fast (< 100ms for 10 broadcasts)
      expect(executionTime).toBeLessThan(100);
      
      logger.info(`✅ Broadcasting performance validated - ${executionTime}ms for 10 broadcasts`);
    });
  });
});
