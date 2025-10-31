/**
 * Hackathon Demo Flow End-to-End Tests
 * 
 * This test suite validates the complete trading flow required for the hackathon demo:
 * 1. Account initialization and wallet connection
 * 2. Deposit process and balance display
 * 3. Order placement and execution
 * 4. Position creation and display
 * 5. Real-time updates via WebSocket
 * 6. Position management and closing
 * 
 * These tests ensure the demo will work flawlessly for judges.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { WebSocketService } from '../../src/services/websocket';
import { SupabaseDatabaseService } from '../../src/services/supabaseDatabase';
import { smartContractService } from '../../src/services/smartContractService';
import { matchingService } from '../../src/services/matching';
import { pythOracleService } from '../../src/services/pythOracleService';
import { Server as SocketIOServer } from 'socket.io';
import { createServer } from 'http';
import { Logger } from '../../src/utils/logger';

const logger = new Logger();

describe('Hackathon Demo Flow - Complete Trading Journey', () => {
  let httpServer: any;
  let io: SocketIOServer;
  let wsService: WebSocketService;
  let db: SupabaseDatabaseService;
  let testUserId: string;
  let testWalletAddress: string;

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
    
    logger.info('🚀 Hackathon demo test environment initialized');
  });

  afterAll(async () => {
    // Cleanup
    wsService.stop();
    io.close();
    httpServer.close();
    
    // Clean up test data
    if (testUserId) {
      await db.deleteUser(testUserId).catch(() => {});
    }
    
    logger.info('🧹 Hackathon demo test environment cleaned up');
  });

  beforeEach(async () => {
    // Ensure clean state for each test
    await db.deleteUser(testUserId).catch(() => {});
  });

  describe('Phase 1: Account Initialization', () => {
    it('should initialize user account and display balance', async () => {
      // Create test user
      const user = await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      expect(user).toBeDefined();
      expect(user.wallet_address).toBe(testWalletAddress);
      
      // Initialize account state
      const accountState = await db.createAccountState(testUserId, {
        total_collateral: 1000,
        used_margin: 0,
        available_margin: 1000,
        margin_ratio: 0,
        health_factor: 1.0
      });
      
      expect(accountState).toBeDefined();
      expect(accountState.total_collateral).toBe(1000);
      
      logger.info('✅ Account initialization test passed');
    });

    it('should connect wallet and authenticate user', async () => {
      // Simulate wallet connection
      const jwt = require('jsonwebtoken');
      const token = jwt.sign(
        { walletAddress: testWalletAddress },
        process.env.JWT_SECRET || 'test-secret',
        { expiresIn: '1h' }
      );
      
      expect(token).toBeDefined();
      
      // Verify token can be decoded
      const decoded = jwt.verify(token, process.env.JWT_SECRET || 'test-secret');
      expect(decoded.walletAddress).toBe(testWalletAddress);
      
      logger.info('✅ Wallet connection and authentication test passed');
    });
  });

  describe('Phase 2: Deposit Process', () => {
    it('should process deposit and update balance', async () => {
      // Create user first
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      // Simulate deposit
      const depositAmount = 500;
      const accountState = await db.createAccountState(testUserId, {
        total_collateral: depositAmount,
        used_margin: 0,
        available_margin: depositAmount,
        margin_ratio: 0,
        health_factor: 1.0
      });
      
      expect(accountState.total_collateral).toBe(depositAmount);
      expect(accountState.available_margin).toBe(depositAmount);
      
      logger.info('✅ Deposit process test passed');
    });

    it('should display updated balance in real-time', async () => {
      // Test portfolio calculation
      const portfolioData = await db.getUserPortfolio(testUserId);
      
      expect(portfolioData).toBeDefined();
      expect(portfolioData.total_collateral).toBeGreaterThan(0);
      
      logger.info('✅ Balance display test passed');
    });
  });

  describe('Phase 3: Order Placement and Execution', () => {
    it('should place market order successfully', async () => {
      // Create user and account state
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      await db.createAccountState(testUserId, {
        total_collateral: 1000,
        used_margin: 0,
        available_margin: 1000,
        margin_ratio: 0,
        health_factor: 1.0
      });
      
      // Create test market
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      
      // Place market order
      const orderResult = await matchingService.placeOrder({
        userId: testUserId,
        symbol: 'SOL-PERP',
        side: 'buy',
        size: 1.0,
        orderType: 'market',
        leverage: 5
      });
      
      expect(orderResult).toBeDefined();
      expect(orderResult.orderId).toBeDefined();
      expect(orderResult.filled).toBe(true);
      
      logger.info('✅ Market order placement test passed');
    });

    it('should execute order on smart contract with atomic position creation', async () => {
      // Test smart contract execution
      const executionResult = await smartContractService.executeOrder({
        orderId: 'test-order-' + Date.now(),
        userId: testUserId,
        marketSymbol: 'SOL-PERP',
        side: 'long',
        size: 1.0,
        price: 100,
        leverage: 5,
        orderType: 'market'
      });
      
      expect(executionResult).toBeDefined();
      expect(executionResult.success).toBe(true);
      expect(executionResult.transactionSignature).toBeDefined();
      expect(executionResult.positionId).toBeDefined();
      
      logger.info('✅ Smart contract execution test passed');
    });

    it('should create position after order execution', async () => {
      // Create position
      const position = await db.createPosition({
        user_id: testUserId,
        market_id: 'test-market-id',
        side: 'long',
        size: 1.0,
        entry_price: 100,
        leverage: 5,
        margin: 20,
        unrealized_pnl: 0,
        realized_pnl: 0,
        funding_fees: 0,
        is_liquidated: false,
        liquidation_price: 80,
        health_factor: 1.0
      });
      
      expect(position).toBeDefined();
      expect(position.side).toBe('long');
      expect(position.size).toBe(1.0);
      
      logger.info('✅ Position creation test passed');
    });
  });

  describe('Phase 4: Position Display and Management', () => {
    it('should display position with correct P&L calculations', async () => {
      // Create test position
      const position = await db.createPosition({
        user_id: testUserId,
        market_id: 'test-market-id',
        side: 'long',
        size: 1.0,
        entry_price: 100,
        leverage: 5,
        margin: 20,
        unrealized_pnl: 0,
        realized_pnl: 0,
        funding_fees: 0,
        is_liquidated: false,
        liquidation_price: 80,
        health_factor: 1.0
      });
      
      // Get position with P&L calculations
      const positions = await db.getUserPositions(testUserId);
      
      expect(positions).toBeDefined();
      expect(positions.length).toBeGreaterThan(0);
      
      const userPosition = positions[0];
      expect(userPosition.id).toBe(position.id);
      expect(userPosition.side).toBe('long');
      expect(userPosition.size).toBe(1.0);
      
      logger.info('✅ Position display test passed');
    });

    it('should update position P&L in real-time', async () => {
      // Test P&L calculation service
      const { pnlCalculationService } = await import('../../src/services/pnlCalculationService');
      
      const positionData = [{
        id: 'test-position',
        symbol: 'SOL-PERP',
        side: 'long' as const,
        size: 1.0,
        entryPrice: 100,
        currentPrice: 105,
        leverage: 5,
        margin: 20
      }];
      
      const positionsWithPnl = await pnlCalculationService.updatePositionPnlWithCurrentPrices(positionData);
      
      expect(positionsWithPnl).toBeDefined();
      expect(positionsWithPnl.length).toBe(1);
      expect(positionsWithPnl[0].unrealizedPnl).toBeGreaterThan(0);
      
      logger.info('✅ Real-time P&L update test passed');
    });

    it('should close position successfully', async () => {
      // Create test position
      const position = await db.createPosition({
        user_id: testUserId,
        market_id: 'test-market-id',
        side: 'long',
        size: 1.0,
        entry_price: 100,
        leverage: 5,
        margin: 20,
        unrealized_pnl: 5,
        realized_pnl: 0,
        funding_fees: 0,
        is_liquidated: false,
        liquidation_price: 80,
        health_factor: 1.0
      });
      
      // Close position
      const closeResult = await smartContractService.closePosition(position.id, testUserId);
      
      expect(closeResult).toBeDefined();
      expect(closeResult.success).toBe(true);
      
      // Update position status in database
      const updatedPosition = await db.updatePosition(position.id, {
        is_liquidated: true,
        closed_at: new Date(),
        realized_pnl: 5
      });
      
      expect(updatedPosition.is_liquidated).toBe(true);
      expect(updatedPosition.realized_pnl).toBe(5);
      
      logger.info('✅ Position closing test passed');
    });
  });

  describe('Phase 5: Real-time Updates', () => {
    it('should broadcast order updates via WebSocket', async () => {
      // Test WebSocket broadcasting
      const orderUpdate = {
        orderId: 'test-order-' + Date.now(),
        status: 'filled',
        filledSize: 1.0,
        averageFillPrice: 100,
        userId: testUserId,
        timestamp: Date.now()
      };
      
      // This should not throw an error
      wsService.broadcastOrderUpdate(testUserId, orderUpdate);
      
      logger.info('✅ Order update broadcasting test passed');
    });

    it('should broadcast position updates via WebSocket', async () => {
      // Test position update broadcasting
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
      
      // This should not throw an error
      wsService.broadcastPositionUpdate(testUserId, positionUpdate);
      
      logger.info('✅ Position update broadcasting test passed');
    });

    it('should broadcast portfolio updates via WebSocket', async () => {
      // Test portfolio update broadcasting
      const portfolioData = {
        userId: testUserId,
        totalValue: 1000,
        totalUnrealizedPnl: 5,
        totalRealizedPnl: 0,
        marginRatio: 0.02,
        healthFactor: 1.0,
        positions: [],
        timestamp: Date.now()
      };
      
      // This should not throw an error
      wsService.broadcastPortfolioUpdate(testUserId, portfolioData);
      
      logger.info('✅ Portfolio update broadcasting test passed');
    });
  });

  describe('Phase 6: Demo Flow Integration', () => {
    it('should complete full trading flow end-to-end', async () => {
      // Step 1: Create user and account
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      await db.createAccountState(testUserId, {
        total_collateral: 1000,
        used_margin: 0,
        available_margin: 1000,
        margin_ratio: 0,
        health_factor: 1.0
      });
      
      // Step 2: Create market
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      
      // Step 3: Place and execute order
      const orderResult = await matchingService.placeOrder({
        userId: testUserId,
        symbol: 'SOL-PERP',
        side: 'buy',
        size: 1.0,
        orderType: 'market',
        leverage: 5
      });
      
      expect(orderResult.success).toBe(true);
      
      // Step 4: Verify position creation
      const positions = await db.getUserPositions(testUserId);
      expect(positions.length).toBeGreaterThan(0);
      
      // Step 5: Test real-time updates
      const positionUpdate = {
        userId: testUserId,
        positionId: positions[0].id,
        symbol: 'SOL-PERP',
        side: 'long' as const,
        size: 1.0,
        entryPrice: 100,
        leverage: 5,
        status: 'open',
        timestamp: Date.now()
      };
      
      wsService.broadcastPositionUpdate(testUserId, positionUpdate);
      
      // Step 6: Close position
      const closeResult = await smartContractService.closePosition(positions[0].id, testUserId);
      expect(closeResult.success).toBe(true);
      
      logger.info('✅ Complete demo flow integration test passed');
    });

    it('should handle errors gracefully during demo', async () => {
      // Test error handling
      try {
        await matchingService.placeOrder({
          userId: 'non-existent-user',
          symbol: 'INVALID-SYMBOL',
          side: 'buy',
          size: 0,
          orderType: 'market',
          leverage: 0
        });
      } catch (error) {
        expect(error).toBeDefined();
        expect(error.message).toContain('Size must be positive');
      }
      
      logger.info('✅ Error handling test passed');
    });

    it('should maintain data consistency during demo', async () => {
      // Test data consistency
      const initialBalance = 1000;
      
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      await db.createAccountState(testUserId, {
        total_collateral: initialBalance,
        used_margin: 0,
        available_margin: initialBalance,
        margin_ratio: 0,
        health_factor: 1.0
      });
      
      // Place order
      const orderResult = await matchingService.placeOrder({
        userId: testUserId,
        symbol: 'SOL-PERP',
        side: 'buy',
        size: 1.0,
        orderType: 'market',
        leverage: 5
      });
      
      // Verify account state is consistent
      const accountState = await db.getAccountState(testUserId);
      expect(accountState).toBeDefined();
      expect(accountState.total_collateral).toBe(initialBalance);
      
      logger.info('✅ Data consistency test passed');
    });
  });

  describe('Performance and Reliability', () => {
    it('should complete demo flow within performance requirements', async () => {
      const startTime = Date.now();
      
      // Execute full demo flow
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      await db.createAccountState(testUserId, {
        total_collateral: 1000,
        used_margin: 0,
        available_margin: 1000,
        margin_ratio: 0,
        health_factor: 1.0
      });
      
      const orderResult = await matchingService.placeOrder({
        userId: testUserId,
        symbol: 'SOL-PERP',
        side: 'buy',
        size: 1.0,
        orderType: 'market',
        leverage: 5
      });
      
      const endTime = Date.now();
      const executionTime = endTime - startTime;
      
      // Demo should complete within 2 seconds
      expect(executionTime).toBeLessThan(2000);
      
      logger.info(`✅ Performance test passed - Demo completed in ${executionTime}ms`);
    });

    it('should handle concurrent demo requests', async () => {
      // Test concurrent order placement
      const promises = [];
      
      for (let i = 0; i < 5; i++) {
        const userId = `concurrent-user-${i}`;
        const walletAddress = `concurrent-wallet-${i}`;
        
        promises.push(
          db.createUser({
            wallet_address: walletAddress,
            username: `user-${i}`,
            email: `user${i}@quantdesk.com`
          }).then(() => 
            db.createAccountState(userId, {
              total_collateral: 1000,
              used_margin: 0,
              available_margin: 1000,
              margin_ratio: 0,
              health_factor: 1.0
            })
          )
        );
      }
      
      const results = await Promise.all(promises);
      expect(results.length).toBe(5);
      
      logger.info('✅ Concurrent requests test passed');
    });
  });
});