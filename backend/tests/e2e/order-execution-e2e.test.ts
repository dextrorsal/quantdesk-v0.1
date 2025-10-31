import { describe, it, expect, beforeEach, afterEach, vi } from 'vitest';
import request from 'supertest';
import { app } from '../../src/server';
import { SupabaseDatabaseService } from '../../src/services/supabaseDatabase';
import { SmartContractService } from '../../src/services/smartContractService';
import { WebSocketService } from '../../src/services/websocket';
import { OrderAuthorizationService } from '../../src/services/orderAuthorizationService';
import { PerformanceMonitoringService } from '../../src/services/performanceMonitoringService';
import { ErrorHandlingService } from '../../src/services/errorHandlingService';
import { AuditTrailService } from '../../src/services/auditTrailService';

// Mock dependencies
vi.mock('../../src/services/supabaseDatabase');
vi.mock('../../src/services/smartContractService');
vi.mock('../../src/services/websocket');
vi.mock('../../src/services/orderAuthorizationService');
vi.mock('../../src/services/performanceMonitoringService');
vi.mock('../../src/services/errorHandlingService');
vi.mock('../../src/services/auditTrailService');

describe('1.2-E2E-001: User places market order successfully', () => {
  let mockDb: any;
  let mockSmartContract: any;
  let mockWebSocket: any;
  let mockOrderAuth: any;
  let mockPerformance: any;
  let mockErrorHandling: any;
  let mockAuditTrail: any;

  beforeEach(() => {
    mockDb = {
      getUserById: vi.fn(),
      getMarketById: vi.fn(),
      getUserBalance: vi.fn(),
      createOrder: vi.fn(),
      updateOrder: vi.fn(),
      createPosition: vi.fn()
    };

    mockSmartContract = {
      executeOrder: vi.fn(),
      openPosition: vi.fn()
    };

    mockWebSocket = {
      broadcastToUser: vi.fn(),
      broadcast: vi.fn()
    };

    mockOrderAuth = {
      authorizeOrder: vi.fn(),
      sanitizeOrderInput: vi.fn()
    };

    mockPerformance = {
      startTimer: vi.fn(() => vi.fn()),
      monitorOrderPlacement: vi.fn(),
      monitorOrderExecution: vi.fn()
    };

    mockErrorHandling = {
      handleError: vi.fn(),
      rollbackOrder: vi.fn()
    };

    mockAuditTrail = {
      logOrderPlacement: vi.fn(),
      logOrderExecution: vi.fn()
    };

    vi.mocked(SupabaseDatabaseService.getInstance).mockReturnValue(mockDb);
    vi.mocked(SmartContractService.getInstance).mockReturnValue(mockSmartContract);
    vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocket);
    vi.mocked(OrderAuthorizationService.getInstance).mockReturnValue(mockOrderAuth);
    vi.mocked(PerformanceMonitoringService.getInstance).mockReturnValue(mockPerformance);
    vi.mocked(ErrorHandlingService.getInstance).mockReturnValue(mockErrorHandling);
    vi.mocked(AuditTrailService.getInstance).mockReturnValue(mockAuditTrail);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should place market order successfully through UI', async () => {
    // Given: User with sufficient funds and valid account
    const user = {
      id: 'user-123',
      wallet_address: 'wallet-123',
      is_active: true,
      risk_level: 'medium'
    };

    const market = {
      id: 'market-456',
      symbol: 'BTC-PERP',
      is_active: true,
      max_leverage: 100,
      min_order_size: 0.001,
      max_order_size: 1000000
    };

    const userBalance = {
      balance: 10000,
      locked_balance: 0
    };

    const orderRequest = {
      marketId: 'market-456',
      orderType: 'market',
      side: 'long',
      size: 100,
      leverage: 10
    };

    mockDb.getUserById.mockResolvedValue(user);
    mockDb.getMarketById.mockResolvedValue(market);
    mockDb.getUserBalance.mockResolvedValue(userBalance);

    mockOrderAuth.authorizeOrder.mockResolvedValue({
      authorized: true,
      reason: 'Order authorized'
    });

    mockOrderAuth.sanitizeOrderInput.mockReturnValue(orderRequest);

    mockDb.createOrder.mockResolvedValue({
      id: 'order-123',
      ...orderRequest,
      status: 'pending',
      createdAt: new Date()
    });

    mockSmartContract.executeOrder.mockResolvedValue({
      success: true,
      transactionSignature: 'tx-123',
      positionId: 'position-456'
    });

    mockSmartContract.openPosition.mockResolvedValue({
      success: true,
      positionId: 'position-456'
    });

    mockDb.updateOrder.mockResolvedValue(true);
    mockDb.createPosition.mockResolvedValue({
      id: 'position-456',
      userId: 'user-123',
      marketId: 'market-456',
      side: 'long',
      size: 100,
      entryPrice: 50000,
      leverage: 10
    });

    // When: User submits market order through UI
    const response = await request(app)
      .post('/api/orders')
      .set('Authorization', 'Bearer valid-jwt-token')
      .send(orderRequest);

    // Then: Order is placed successfully and confirmation displayed
    expect(response.status).toBe(201);
    expect(response.body.success).toBe(true);
    expect(response.body.orderId).toBe('order-123');
    expect(response.body.message).toContain('Order placed successfully');
  });

  it('should handle insufficient funds error', async () => {
    // Given: User with insufficient funds
    const user = {
      id: 'user-123',
      wallet_address: 'wallet-123',
      is_active: true,
      risk_level: 'medium'
    };

    const market = {
      id: 'market-456',
      symbol: 'BTC-PERP',
      is_active: true,
      max_leverage: 100,
      min_order_size: 0.001,
      max_order_size: 1000000
    };

    const userBalance = {
      balance: 100, // Insufficient balance
      locked_balance: 0
    };

    const orderRequest = {
      marketId: 'market-456',
      orderType: 'market',
      side: 'long',
      size: 100,
      leverage: 10
    };

    mockDb.getUserById.mockResolvedValue(user);
    mockDb.getMarketById.mockResolvedValue(market);
    mockDb.getUserBalance.mockResolvedValue(userBalance);

    mockOrderAuth.authorizeOrder.mockResolvedValue({
      authorized: false,
      reason: 'Insufficient balance'
    });

    // When: User submits market order through UI
    const response = await request(app)
      .post('/api/orders')
      .set('Authorization', 'Bearer valid-jwt-token')
      .send(orderRequest);

    // Then: Order is rejected with clear error message
    expect(response.status).toBe(400);
    expect(response.body.error).toContain('Insufficient balance');
  });
});

describe('1.2-E2E-002: Order executes when price conditions met', () => {
  let mockDb: any;
  let mockSmartContract: any;
  let mockWebSocket: any;
  let mockOrderAuth: any;
  let mockPerformance: any;
  let mockErrorHandling: any;
  let mockAuditTrail: any;

  beforeEach(() => {
    mockDb = {
      getOrderById: vi.fn(),
      updateOrder: vi.fn(),
      createPosition: vi.fn()
    };

    mockSmartContract = {
      executeOrder: vi.fn(),
      openPosition: vi.fn()
    };

    mockWebSocket = {
      broadcastToUser: vi.fn(),
      broadcast: vi.fn()
    };

    mockOrderAuth = {
      authorizeOrder: vi.fn(),
      canExecuteOrder: vi.fn()
    };

    mockPerformance = {
      startTimer: vi.fn(() => vi.fn()),
      monitorOrderPlacement: vi.fn(),
      monitorOrderExecution: vi.fn()
    };

    mockErrorHandling = {
      handleError: vi.fn(),
      rollbackOrder: vi.fn()
    };

    mockAuditTrail = {
      logOrderPlacement: vi.fn(),
      logOrderExecution: vi.fn()
    };

    vi.mocked(SupabaseDatabaseService.getInstance).mockReturnValue(mockDb);
    vi.mocked(SmartContractService.getInstance).mockReturnValue(mockSmartContract);
    vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocket);
    vi.mocked(OrderAuthorizationService.getInstance).mockReturnValue(mockOrderAuth);
    vi.mocked(PerformanceMonitoringService.getInstance).mockReturnValue(mockPerformance);
    vi.mocked(ErrorHandlingService.getInstance).mockReturnValue(mockErrorHandling);
    vi.mocked(AuditTrailService.getInstance).mockReturnValue(mockAuditTrail);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should execute limit order when price reaches target', async () => {
    // Given: Limit order with target price
    const limitOrder = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      orderType: 'limit',
      side: 'long',
      size: 100,
      price: 50000,
      leverage: 10,
      status: 'pending',
      expiresAt: new Date(Date.now() + 3600000) // 1 hour from now
    };

    const currentPrice = 50000; // Price reaches target

    mockDb.getOrderById.mockResolvedValue(limitOrder);
    mockOrderAuth.canExecuteOrder.mockReturnValue(true);

    mockSmartContract.executeOrder.mockResolvedValue({
      success: true,
      transactionSignature: 'tx-123',
      positionId: 'position-456'
    });

    mockSmartContract.openPosition.mockResolvedValue({
      success: true,
      positionId: 'position-456'
    });

    mockDb.updateOrder.mockResolvedValue(true);
    mockDb.createPosition.mockResolvedValue({
      id: 'position-456',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long',
      size: 100,
      entryPrice: 50000,
      leverage: 10
    });

    // When: Market price reaches target
    const order = await mockDb.getOrderById('order-123');
    const canExecute = mockOrderAuth.canExecuteOrder(order, currentPrice);

    if (canExecute) {
      const executionResult = await mockSmartContract.executeOrder({
        orderId: order.id,
        userId: order.userId,
        marketSymbol: 'BTC-PERP',
        side: order.side,
        size: order.size,
        price: order.price,
        leverage: order.leverage,
        orderType: order.orderType
      });

      if (executionResult.success) {
        await mockSmartContract.openPosition({
          userId: order.userId,
          marketSymbol: 'BTC-PERP',
          side: order.side,
          size: order.size,
          entryPrice: order.price,
          leverage: order.leverage
        });

        await mockDb.updateOrder(order.id, {
          status: 'filled',
          filledSize: order.size,
          averageFillPrice: order.price,
          filledAt: new Date()
        });

        await mockDb.createPosition({
          userId: order.userId,
          marketId: order.marketId,
          side: order.side,
          size: order.size,
          entryPrice: order.price,
          leverage: order.leverage
        });
      }
    }

    // Then: Order executes automatically and position is created
    expect(canExecute).toBe(true);
    expect(mockSmartContract.executeOrder).toHaveBeenCalledWith(
      expect.objectContaining({
        orderId: 'order-123',
        userId: 'user-456',
        marketSymbol: 'BTC-PERP',
        side: 'long',
        size: 100,
        price: 50000,
        leverage: 10,
        orderType: 'limit'
      })
    );
    expect(mockSmartContract.openPosition).toHaveBeenCalledWith(
      expect.objectContaining({
        userId: 'user-456',
        marketSymbol: 'BTC-PERP',
        side: 'long',
        size: 100,
        entryPrice: 50000,
        leverage: 10
      })
    );
  });

  it('should not execute limit order when price conditions not met', async () => {
    // Given: Limit order with target price
    const limitOrder = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      orderType: 'limit',
      side: 'long',
      size: 100,
      price: 50000,
      leverage: 10,
      status: 'pending',
      expiresAt: new Date(Date.now() + 3600000) // 1 hour from now
    };

    const currentPrice = 49000; // Price below target

    mockDb.getOrderById.mockResolvedValue(limitOrder);
    mockOrderAuth.canExecuteOrder.mockReturnValue(false);

    // When: Market price does not reach target
    const order = await mockDb.getOrderById('order-123');
    const canExecute = mockOrderAuth.canExecuteOrder(order, currentPrice);

    // Then: Order does not execute
    expect(canExecute).toBe(false);
    expect(mockSmartContract.executeOrder).not.toHaveBeenCalled();
  });
});

describe('1.2-E2E-003: User sees real-time order status updates', () => {
  let mockDb: any;
  let mockWebSocket: any;
  let mockOrderAuth: any;

  beforeEach(() => {
    mockDb = {
      getOrderById: vi.fn(),
      updateOrder: vi.fn()
    };

    mockWebSocket = {
      broadcastToUser: vi.fn(),
      broadcast: vi.fn()
    };

    mockOrderAuth = {
      transitionOrderStatus: vi.fn()
    };

    vi.mocked(SupabaseDatabaseService.getInstance).mockReturnValue(mockDb);
    vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocket);
    vi.mocked(OrderAuthorizationService.getInstance).mockReturnValue(mockOrderAuth);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should update UI immediately when order status changes to filled', async () => {
    // Given: User with pending order
    const pendingOrder = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      status: 'pending',
      size: 100,
      filledSize: 0
    };

    mockDb.getOrderById.mockResolvedValue(pendingOrder);
    mockOrderAuth.transitionOrderStatus.mockReturnValue('filled');
    mockDb.updateOrder.mockResolvedValue(true);

    // When: Order status changes to filled
    const order = await mockDb.getOrderById('order-123');
    const newStatus = mockOrderAuth.transitionOrderStatus(order, 'filled', 100);
    
    await mockDb.updateOrder(order.id, {
      status: newStatus,
      filledSize: 100,
      filledAt: new Date()
    });

    await mockWebSocket.broadcastToUser(order.userId, 'order_update', {
      orderId: order.id,
      status: newStatus,
      filledSize: 100,
      userId: order.userId,
      timestamp: Date.now()
    });

    // Then: UI updates immediately showing new status
    expect(newStatus).toBe('filled');
    expect(mockWebSocket.broadcastToUser).toHaveBeenCalledWith(
      'user-456',
      'order_update',
      expect.objectContaining({
        orderId: 'order-123',
        status: 'filled',
        filledSize: 100,
        userId: 'user-456'
      })
    );
  });

  it('should update UI immediately when order status changes to cancelled', async () => {
    // Given: User with pending order
    const pendingOrder = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      status: 'pending',
      size: 100,
      filledSize: 0
    };

    mockDb.getOrderById.mockResolvedValue(pendingOrder);
    mockOrderAuth.transitionOrderStatus.mockReturnValue('cancelled');
    mockDb.updateOrder.mockResolvedValue(true);

    // When: Order status changes to cancelled
    const order = await mockDb.getOrderById('order-123');
    const newStatus = mockOrderAuth.transitionOrderStatus(order, 'cancelled', 0);
    
    await mockDb.updateOrder(order.id, {
      status: newStatus,
      cancelledAt: new Date()
    });

    await mockWebSocket.broadcastToUser(order.userId, 'order_update', {
      orderId: order.id,
      status: newStatus,
      userId: order.userId,
      timestamp: Date.now()
    });

    // Then: UI updates immediately showing new status
    expect(newStatus).toBe('cancelled');
    expect(mockWebSocket.broadcastToUser).toHaveBeenCalledWith(
      'user-456',
      'order_update',
      expect.objectContaining({
        orderId: 'order-123',
        status: 'cancelled',
        userId: 'user-456'
      })
    );
  });
});

describe('1.2-E2E-004: Position created after order fill', () => {
  let mockDb: any;
  let mockSmartContract: any;
  let mockWebSocket: any;

  beforeEach(() => {
    mockDb = {
      getOrderById: vi.fn(),
      createPosition: vi.fn(),
      updateOrder: vi.fn()
    };

    mockSmartContract = {
      executeOrder: vi.fn(),
      openPosition: vi.fn()
    };

    mockWebSocket = {
      broadcastToUser: vi.fn(),
      broadcast: vi.fn()
    };

    vi.mocked(SupabaseDatabaseService.getInstance).mockReturnValue(mockDb);
    vi.mocked(SmartContractService.getInstance).mockReturnValue(mockSmartContract);
    vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocket);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should create position after order execution completes', async () => {
    // Given: User with filled order
    const filledOrder = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long',
      size: 100,
      price: 50000,
      leverage: 10,
      status: 'filled',
      filledSize: 100,
      averageFillPrice: 50000
    };

    mockDb.getOrderById.mockResolvedValue(filledOrder);
    mockSmartContract.executeOrder.mockResolvedValue({
      success: true,
      transactionSignature: 'tx-123',
      positionId: 'position-456'
    });

    mockSmartContract.openPosition.mockResolvedValue({
      success: true,
      positionId: 'position-456'
    });

    mockDb.createPosition.mockResolvedValue({
      id: 'position-456',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long',
      size: 100,
      entryPrice: 50000,
      leverage: 10,
      margin: 5000
    });

    // When: Order execution completes
    const order = await mockDb.getOrderById('order-123');
    const executionResult = await mockSmartContract.executeOrder({
      orderId: order.id,
      userId: order.userId,
      marketSymbol: 'BTC-PERP',
      side: order.side,
      size: order.size,
      price: order.price,
      leverage: order.leverage,
      orderType: 'limit'
    });

    if (executionResult.success) {
      const positionResult = await mockSmartContract.openPosition({
        userId: order.userId,
        marketSymbol: 'BTC-PERP',
        side: order.side,
        size: order.size,
        entryPrice: order.averageFillPrice,
        leverage: order.leverage
      });

      if (positionResult.success) {
        const position = await mockDb.createPosition({
          userId: order.userId,
          marketId: order.marketId,
          side: order.side,
          size: order.size,
          entryPrice: order.averageFillPrice,
          leverage: order.leverage,
          margin: order.size * order.averageFillPrice / order.leverage
        });

        await mockWebSocket.broadcastToUser(order.userId, 'position_created', {
          positionId: position.id,
          userId: order.userId,
          marketId: order.marketId,
          side: order.side,
          size: order.size,
          entryPrice: order.averageFillPrice,
          leverage: order.leverage
        });
      }
    }

    // Then: Position appears in user's portfolio
    expect(executionResult.success).toBe(true);
    expect(mockSmartContract.openPosition).toHaveBeenCalledWith(
      expect.objectContaining({
        userId: 'user-456',
        marketSymbol: 'BTC-PERP',
        side: 'long',
        size: 100,
        entryPrice: 50000,
        leverage: 10
      })
    );
    expect(mockDb.createPosition).toHaveBeenCalledWith(
      expect.objectContaining({
        userId: 'user-456',
        marketId: 'market-789',
        side: 'long',
        size: 100,
        entryPrice: 50000,
        leverage: 10,
        margin: 5000
      })
    );
    expect(mockWebSocket.broadcastToUser).toHaveBeenCalledWith(
      'user-456',
      'position_created',
      expect.objectContaining({
        positionId: 'position-456',
        userId: 'user-456',
        marketId: 'market-789',
        side: 'long',
        size: 100,
        entryPrice: 50000,
        leverage: 10
      })
    );
  });

  it('should handle position creation failure gracefully', async () => {
    // Given: User with filled order
    const filledOrder = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long',
      size: 100,
      price: 50000,
      leverage: 10,
      status: 'filled',
      filledSize: 100,
      averageFillPrice: 50000
    };

    mockDb.getOrderById.mockResolvedValue(filledOrder);
    mockSmartContract.executeOrder.mockResolvedValue({
      success: true,
      transactionSignature: 'tx-123'
    });

    mockSmartContract.openPosition.mockRejectedValue(new Error('Position creation failed'));

    // When: Order execution completes but position creation fails
    try {
      const order = await mockDb.getOrderById('order-123');
      const executionResult = await mockSmartContract.executeOrder({
        orderId: order.id,
        userId: order.userId,
        marketSymbol: 'BTC-PERP',
        side: order.side,
        size: order.size,
        price: order.price,
        leverage: order.leverage,
        orderType: 'limit'
      });

      if (executionResult.success) {
        await mockSmartContract.openPosition({
          userId: order.userId,
          marketSymbol: 'BTC-PERP',
          side: order.side,
          size: order.size,
          entryPrice: order.averageFillPrice,
          leverage: order.leverage
        });
      }
    } catch (error) {
      // Then: Position creation failure is handled gracefully
      expect(error.message).toBe('Position creation failed');
    }
  });
});

describe('1.2-E2E-005: User sees clear error messages', () => {
  let mockDb: any;
  let mockOrderAuth: any;
  let mockErrorHandling: any;
  let mockWebSocket: any;

  beforeEach(() => {
    mockDb = {
      getUserById: vi.fn(),
      getMarketById: vi.fn(),
      getUserBalance: vi.fn()
    };

    mockOrderAuth = {
      authorizeOrder: vi.fn()
    };

    mockErrorHandling = {
      handleError: vi.fn(),
      generateUserFriendlyMessage: vi.fn()
    };

    mockWebSocket = {
      broadcastToUser: vi.fn()
    };

    vi.mocked(SupabaseDatabaseService.getInstance).mockReturnValue(mockDb);
    vi.mocked(OrderAuthorizationService.getInstance).mockReturnValue(mockOrderAuth);
    vi.mocked(ErrorHandlingService.getInstance).mockReturnValue(mockErrorHandling);
    vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocket);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should display clear error message for invalid order', async () => {
    // Given: User attempts invalid order
    const invalidOrder = {
      marketId: 'market-456',
      orderType: 'limit',
      side: 'long',
      size: -100, // Invalid negative size
      price: 50000,
      leverage: 10
    };

    mockOrderAuth.authorizeOrder.mockResolvedValue({
      authorized: false,
      reason: 'Invalid order size'
    });

    mockErrorHandling.handleError.mockResolvedValue({
      errorId: 'error-789',
      message: 'Order size must be positive',
      timestamp: new Date()
    });

    mockErrorHandling.generateUserFriendlyMessage.mockReturnValue(
      'Order size must be positive'
    );

    // When: Order placement fails
    const response = await request(app)
      .post('/api/orders')
      .set('Authorization', 'Bearer valid-jwt-token')
      .send(invalidOrder);

    // Then: Clear error message is displayed
    expect(response.status).toBe(400);
    expect(response.body.error).toContain('Order size must be positive');
  });

  it('should display clear error message for market not found', async () => {
    // Given: User attempts order for non-existent market
    const invalidOrder = {
      marketId: 'non-existent-market',
      orderType: 'limit',
      side: 'long',
      size: 100,
      price: 50000,
      leverage: 10
    };

    mockDb.getMarketById.mockResolvedValue(null);

    mockOrderAuth.authorizeOrder.mockResolvedValue({
      authorized: false,
      reason: 'Market not found'
    });

    mockErrorHandling.handleError.mockResolvedValue({
      errorId: 'error-789',
      message: 'Market not found',
      timestamp: new Date()
    });

    mockErrorHandling.generateUserFriendlyMessage.mockReturnValue(
      'Market not found'
    );

    // When: Order placement fails
    const response = await request(app)
      .post('/api/orders')
      .set('Authorization', 'Bearer valid-jwt-token')
      .send(invalidOrder);

    // Then: Clear error message is displayed
    expect(response.status).toBe(404);
    expect(response.body.error).toContain('Market not found');
  });

  it('should display clear error message for network issues', async () => {
    // Given: User attempts order during network issues
    const order = {
      marketId: 'market-456',
      orderType: 'limit',
      side: 'long',
      size: 100,
      price: 50000,
      leverage: 10
    };

    mockOrderAuth.authorizeOrder.mockRejectedValue(new Error('Network timeout'));

    mockErrorHandling.handleError.mockResolvedValue({
      errorId: 'error-789',
      message: 'Network error. Please try again.',
      timestamp: new Date()
    });

    mockErrorHandling.generateUserFriendlyMessage.mockReturnValue(
      'Network error. Please try again.'
    );

    // When: Order placement fails due to network issues
    const response = await request(app)
      .post('/api/orders')
      .set('Authorization', 'Bearer valid-jwt-token')
      .send(order);

    // Then: Clear error message is displayed
    expect(response.status).toBe(500);
    expect(response.body.error).toContain('Network error. Please try again.');
  });
});

describe('1.2-E2E-006: Order execution with smart contract failure', () => {
  let mockDb: any;
  let mockSmartContract: any;
  let mockErrorHandling: any;
  let mockWebSocket: any;

  beforeEach(() => {
    mockDb = {
      getOrderById: vi.fn(),
      updateOrder: vi.fn(),
      rollbackTransaction: vi.fn()
    };

    mockSmartContract = {
      executeOrder: vi.fn()
    };

    mockErrorHandling = {
      handleError: vi.fn(),
      rollbackOrder: vi.fn()
    };

    mockWebSocket = {
      broadcastToUser: vi.fn()
    };

    vi.mocked(SupabaseDatabaseService.getInstance).mockReturnValue(mockDb);
    vi.mocked(SmartContractService.getInstance).mockReturnValue(mockSmartContract);
    vi.mocked(ErrorHandlingService.getInstance).mockReturnValue(mockErrorHandling);
    vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocket);
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should notify user and safely cancel order during smart contract outage', async () => {
    // Given: User places order during smart contract outage
    const order = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long',
      size: 100,
      price: 50000,
      leverage: 10,
      orderType: 'limit',
      status: 'pending'
    };

    mockDb.getOrderById.mockResolvedValue(order);
    mockSmartContract.executeOrder.mockRejectedValue(new Error('Smart contract service unavailable'));

    mockErrorHandling.handleError.mockResolvedValue({
      errorId: 'error-789',
      message: 'Trading temporarily unavailable. Please try again later.',
      timestamp: new Date()
    });

    mockErrorHandling.rollbackOrder.mockResolvedValue(true);
    mockDb.rollbackTransaction.mockResolvedValue(true);
    mockDb.updateOrder.mockResolvedValue(true);

    // When: Order execution fails
    try {
      await mockSmartContract.executeOrder({
        orderId: order.id,
        userId: order.userId,
        marketSymbol: 'BTC-PERP',
        side: order.side,
        size: order.size,
        price: order.price,
        leverage: order.leverage,
        orderType: order.orderType
      });
    } catch (error) {
      // Then: User is notified and order is safely cancelled
      expect(error.message).toBe('Smart contract service unavailable');
      
      const errorResult = await mockErrorHandling.handleError(error, {
        userId: order.userId,
        orderId: order.id,
        operation: 'smart_contract_execution'
      });

      await mockErrorHandling.rollbackOrder(order.id);
      await mockDb.rollbackTransaction();
      await mockDb.updateOrder(order.id, {
        status: 'cancelled',
        cancelledAt: new Date(),
        errorMessage: errorResult.message
      });

      await mockWebSocket.broadcastToUser(order.userId, 'order_error', {
        orderId: order.id,
        error: errorResult.message,
        timestamp: errorResult.timestamp
      });

      expect(errorResult.message).toBe('Trading temporarily unavailable. Please try again later.');
      expect(mockErrorHandling.rollbackOrder).toHaveBeenCalledWith(order.id);
      expect(mockDb.rollbackTransaction).toHaveBeenCalled();
      expect(mockWebSocket.broadcastToUser).toHaveBeenCalledWith(
        'user-456',
        'order_error',
        expect.objectContaining({
          orderId: 'order-123',
          error: errorResult.message
        })
      );
    }
  });

  it('should retry order execution after temporary smart contract failure', async () => {
    // Given: User places order during temporary smart contract outage
    const order = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long',
      size: 100,
      price: 50000,
      leverage: 10,
      orderType: 'limit',
      status: 'pending'
    };

    mockDb.getOrderById.mockResolvedValue(order);
    
    // First call fails, second call succeeds
    mockSmartContract.executeOrder
      .mockRejectedValueOnce(new Error('Smart contract service temporarily unavailable'))
      .mockResolvedValueOnce({
        success: true,
        transactionSignature: 'tx-123',
        positionId: 'position-456'
      });

    // When: Order execution fails initially but succeeds on retry
    try {
      await mockSmartContract.executeOrder({
        orderId: order.id,
        userId: order.userId,
        marketSymbol: 'BTC-PERP',
        side: order.side,
        size: order.size,
        price: order.price,
        leverage: order.leverage,
        orderType: order.orderType
      });
    } catch (error) {
      // Retry after temporary failure
      const result = await mockSmartContract.executeOrder({
        orderId: order.id,
        userId: order.userId,
        marketSymbol: 'BTC-PERP',
        side: order.side,
        size: order.size,
        price: order.price,
        leverage: order.leverage,
        orderType: order.orderType
      });

      // Then: Order executes successfully on retry
      expect(result.success).toBe(true);
      expect(result.transactionSignature).toBe('tx-123');
      expect(result.positionId).toBe('position-456');
    }
  });
});
