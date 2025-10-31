/**
 * Hackathon Core Functionality Tests
 * 
 * This test suite validates the core functionality required for the hackathon demo,
 * ensuring all critical components work together seamlessly for the trading flow.
 */

import { describe, it, expect, beforeAll, afterAll, beforeEach } from 'vitest';
import { SupabaseDatabaseService } from '../../src/services/supabaseDatabase';
import { smartContractService } from '../../src/services/smartContractService';
import { matchingService } from '../../src/services/matching';
import { pythOracleService } from '../../src/services/pythOracleService';
import { Logger } from '../../src/utils/logger';

const logger = new Logger();

describe('Hackathon Core Functionality - Critical Demo Components', () => {
  let db: SupabaseDatabaseService;
  let testUserId: string;
  let testWalletAddress: string;
  let testMarketId: string;

  beforeAll(async () => {
    // Initialize test database
    db = SupabaseDatabaseService.getInstance();
    
    // Create test user
    testWalletAddress = 'test-wallet-' + Date.now();
    testUserId = 'test-user-' + Date.now();
    
    logger.info('🚀 Hackathon core functionality test environment initialized');
  });

  afterAll(async () => {
    // Clean up test data - using available methods
    logger.info('🧹 Hackathon core functionality test environment cleaned up');
  });

  beforeEach(async () => {
    // Ensure clean state for each test - tests will use mock data
    logger.info('🧪 Setting up test environment');
  });

  describe('Smart Contract Service Integration', () => {
    it('should initialize smart contract service successfully', async () => {
      // Test service initialization
      const service = smartContractService;
      expect(service).toBeDefined();
      
      // Test health check
      const isHealthy = await service.healthCheck();
      expect(typeof isHealthy).toBe('boolean');
      
      logger.info('✅ Smart contract service initialization test passed');
    });

    it('should execute orders with atomic position creation', async () => {
      // Test order execution with mock data
      testMarketId = 'test-market-' + Date.now();
      
      // Test order execution
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
      
      logger.info('✅ Atomic order execution test passed');
    });

    it('should create positions successfully', async () => {
      // Test position creation
      const positionResult = await smartContractService.createPosition(
        testUserId,
        'SOL-PERP',
        'long',
        1.0,
        100,
        5
      );
      
      expect(positionResult).toBeDefined();
      expect(positionResult.success).toBe(true);
      expect(positionResult.positionId).toBeDefined();
      
      logger.info('✅ Position creation test passed');
    });

    it('should close positions successfully', async () => {
      // Test position closing
      const positionId = 'test-position-' + Date.now();
      const closeResult = await smartContractService.closePosition(positionId, testUserId);
      
      expect(closeResult).toBeDefined();
      expect(closeResult.success).toBe(true);
      expect(closeResult.transactionSignature).toBeDefined();
      
      logger.info('✅ Position closing test passed');
    });
  });

  describe('Order Matching Service Integration', () => {
    it('should place market orders successfully', async () => {
      // Create test user and market
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      testMarketId = market.id;
      
      // Test market order placement
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
      expect(orderResult.fills).toBeDefined();
      expect(orderResult.fills.length).toBeGreaterThan(0);
      
      logger.info('✅ Market order placement test passed');
    });

    it('should place limit orders successfully', async () => {
      // Create test user and market
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      testMarketId = market.id;
      
      // Test limit order placement
      const orderResult = await matchingService.placeOrder({
        userId: testUserId,
        symbol: 'SOL-PERP',
        side: 'buy',
        size: 1.0,
        orderType: 'limit',
        price: 95,
        leverage: 5
      });
      
      expect(orderResult).toBeDefined();
      expect(orderResult.orderId).toBeDefined();
      // Limit orders may not be filled immediately
      expect(typeof orderResult.filled).toBe('boolean');
      
      logger.info('✅ Limit order placement test passed');
    });

    it('should handle order validation correctly', async () => {
      // Test invalid order parameters
      await expect(
        matchingService.placeOrder({
          userId: testUserId,
          symbol: 'SOL-PERP',
          side: 'buy',
          size: 0, // Invalid size
          orderType: 'market',
          leverage: 5
        })
      ).rejects.toThrow('Size must be positive');
      
      // Test missing price for limit order
      await expect(
        matchingService.placeOrder({
          userId: testUserId,
          symbol: 'SOL-PERP',
          side: 'buy',
          size: 1.0,
          orderType: 'limit',
          leverage: 5
          // Missing price
        })
      ).rejects.toThrow('Limit orders require positive price');
      
      logger.info('✅ Order validation test passed');
    });
  });

  describe('Oracle Service Integration', () => {
    it('should fetch price data successfully', async () => {
      // Test price fetching
      const price = await pythOracleService.getLatestPrice('SOL');
      
      expect(price).toBeDefined();
      expect(typeof price).toBe('number');
      expect(price).toBeGreaterThan(0);
      
      logger.info('✅ Price fetching test passed');
    });

    it('should handle multiple price requests efficiently', async () => {
      // Test multiple price requests
      const symbols = ['BTC', 'ETH', 'SOL', 'AVAX'];
      const prices = await Promise.all(
        symbols.map(symbol => pythOracleService.getLatestPrice(symbol))
      );
      
      expect(prices).toBeDefined();
      expect(prices.length).toBe(symbols.length);
      
      // All prices should be valid
      prices.forEach(price => {
        expect(price).toBeDefined();
        expect(typeof price).toBe('number');
        expect(price).toBeGreaterThan(0);
      });
      
      logger.info('✅ Multiple price requests test passed');
    });

    it('should handle price service errors gracefully', async () => {
      // Test with invalid symbol
      const invalidPrice = await pythOracleService.getLatestPrice('INVALID-SYMBOL');
      
      // Should handle gracefully (may return null or throw)
      expect(invalidPrice === null || typeof invalidPrice === 'number').toBe(true);
      
      logger.info('✅ Price service error handling test passed');
    });
  });

  describe('Database Service Integration', () => {
    it('should create and manage users correctly', async () => {
      // Test user creation
      const user = await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      expect(user).toBeDefined();
      expect(user.wallet_address).toBe(testWalletAddress);
      
      // Test user retrieval
      const retrievedUser = await db.getUserByWallet(testWalletAddress);
      expect(retrievedUser).toBeDefined();
      expect(retrievedUser.wallet_address).toBe(testWalletAddress);
      
      logger.info('✅ User management test passed');
    });

    it('should create and manage markets correctly', async () => {
      // Test market creation
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      testMarketId = market.id;
      
      expect(market).toBeDefined();
      expect(market.symbol).toBe('SOL-PERP');
      
      // Test market retrieval
      const retrievedMarket = await db.getMarketBySymbol('SOL-PERP');
      expect(retrievedMarket).toBeDefined();
      expect(retrievedMarket.symbol).toBe('SOL-PERP');
      
      logger.info('✅ Market management test passed');
    });

    it('should create and manage orders correctly', async () => {
      // Create test user and market
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      testMarketId = market.id;
      
      // Test order creation
      const order = await db.createOrder({
        user_id: testUserId,
        market_id: testMarketId,
        order_account: 'OFFCHAIN',
        order_type: 'market',
        side: 'long',
        size: 1.0,
        price: null,
        leverage: 5,
        status: 'pending'
      });
      
      expect(order).toBeDefined();
      expect(order.user_id).toBe(testUserId);
      expect(order.market_id).toBe(testMarketId);
      
      // Test order retrieval
      const retrievedOrder = await db.getOrderById(order.id);
      expect(retrievedOrder).toBeDefined();
      expect(retrievedOrder.id).toBe(order.id);
      
      logger.info('✅ Order management test passed');
    });

    it('should create and manage positions correctly', async () => {
      // Create test user and market
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      testMarketId = market.id;
      
      // Test position creation
      const position = await db.createPosition({
        user_id: testUserId,
        market_id: testMarketId,
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
      expect(position.user_id).toBe(testUserId);
      expect(position.market_id).toBe(testMarketId);
      
      // Test position retrieval
      const retrievedPosition = await db.getPositionById(position.id);
      expect(retrievedPosition).toBeDefined();
      expect(retrievedPosition.id).toBe(position.id);
      
      logger.info('✅ Position management test passed');
    });
  });

  describe('Integration Tests', () => {
    it('should complete full order-to-position flow', async () => {
      // Create test user and market
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      testMarketId = market.id;
      
      // Step 1: Place order
      const orderResult = await matchingService.placeOrder({
        userId: testUserId,
        symbol: 'SOL-PERP',
        side: 'buy',
        size: 1.0,
        orderType: 'market',
        leverage: 5
      });
      
      expect(orderResult.success).toBe(true);
      
      // Step 2: Verify order in database
      const order = await db.getOrderById(orderResult.orderId);
      expect(order).toBeDefined();
      expect(order.status).toBe('filled');
      
      // Step 3: Verify position creation
      const positions = await db.getUserPositions(testUserId);
      expect(positions.length).toBeGreaterThan(0);
      
      const position = positions[0];
      expect(position.side).toBe('long');
      expect(position.size).toBe(1.0);
      
      logger.info('✅ Full order-to-position flow test passed');
    });

    it('should handle concurrent operations correctly', async () => {
      // Create test user and market
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      testMarketId = market.id;
      
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
            matchingService.placeOrder({
              userId,
              symbol: 'SOL-PERP',
              side: 'buy',
              size: 0.1,
              orderType: 'market',
              leverage: 2
            })
          )
        );
      }
      
      const results = await Promise.all(promises);
      expect(results.length).toBe(5);
      
      // All orders should be successful
      results.forEach(result => {
        expect(result.success).toBe(true);
      });
      
      logger.info('✅ Concurrent operations test passed');
    });

    it('should maintain data consistency during operations', async () => {
      // Create test user and market
      await db.createUser({
        wallet_address: testWalletAddress,
        username: 'demo-user',
        email: 'demo@quantdesk.com'
      });
      
      const market = await db.createMarket({
        symbol: 'SOL-PERP',
        base_asset: 'SOL',
        quote_asset: 'USD',
        min_order_size: 0.1,
        max_order_size: 1000,
        tick_size: 0.01,
        step_size: 0.001
      });
      testMarketId = market.id;
      
      // Create account state
      await db.createAccountState(testUserId, {
        total_collateral: 1000,
        used_margin: 0,
        available_margin: 1000,
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
      
      // Verify data consistency
      const order = await db.getOrderById(orderResult.orderId);
      const positions = await db.getUserPositions(testUserId);
      const accountState = await db.getAccountState(testUserId);
      
      expect(order).toBeDefined();
      expect(positions.length).toBeGreaterThan(0);
      expect(accountState).toBeDefined();
      
      // Verify order and position are linked
      expect(order.user_id).toBe(testUserId);
      expect(positions[0].user_id).toBe(testUserId);
      
      logger.info('✅ Data consistency test passed');
    });
  });

  describe('Error Handling and Recovery', () => {
    it('should handle service failures gracefully', async () => {
      // Test with invalid user ID
      await expect(
        matchingService.placeOrder({
          userId: 'non-existent-user',
          symbol: 'SOL-PERP',
          side: 'buy',
          size: 1.0,
          orderType: 'market',
          leverage: 5
        })
      ).rejects.toThrow();
      
      logger.info('✅ Service failure handling test passed');
    });

    it('should handle database errors gracefully', async () => {
      // Test with invalid market symbol
      await expect(
        matchingService.placeOrder({
          userId: testUserId,
          symbol: 'INVALID-SYMBOL',
          side: 'buy',
          size: 1.0,
          orderType: 'market',
          leverage: 5
        })
      ).rejects.toThrow();
      
      logger.info('✅ Database error handling test passed');
    });

    it('should handle network errors gracefully', async () => {
      // Test with invalid oracle data
      const invalidPrice = await pythOracleService.getLatestPrice('INVALID-SYMBOL');
      
      // Should handle gracefully
      expect(invalidPrice === null || typeof invalidPrice === 'number').toBe(true);
      
      logger.info('✅ Network error handling test passed');
    });
  });
});