import { describe, it, expect, beforeEach, vi, afterEach } from 'vitest';
import { OrderAuthorizationService } from '../../src/services/orderAuthorizationService';
import { SupabaseDatabaseService } from '../../src/services/supabaseDatabase';

// Mock dependencies
vi.mock('../../src/services/supabaseDatabase');
vi.mock('../../src/services/auditTrailService');

describe('1.2-UNIT-001: Order validation logic', () => {
  let orderAuthService: OrderAuthorizationService;
  let mockDb: any;

  beforeEach(() => {
    mockDb = {
      getUserById: vi.fn(),
      getMarketById: vi.fn(),
      getUserBalance: vi.fn()
    };
    
    vi.mocked(SupabaseDatabaseService.getInstance).mockReturnValue(mockDb);
    orderAuthService = OrderAuthorizationService.getInstance();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should validate order parameters with valid data', async () => {
    // Given: Order parameters with valid data
    const validOrder = {
      userId: 'user-123',
      marketId: 'market-456',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10
    };

    mockDb.getUserById.mockResolvedValue({
      id: 'user-123',
      is_active: true,
      risk_level: 'medium'
    });

    mockDb.getMarketById.mockResolvedValue({
      id: 'market-456',
      is_active: true,
      max_leverage: 100,
      min_order_size: 0.001,
      max_order_size: 1000000
    });

    mockDb.getUserBalance.mockResolvedValue({
      balance: 10000,
      locked_balance: 0
    });

    // When: Order validation method is called
    const result = await orderAuthService.authorizeOrder(validOrder);

    // Then: Validation passes and order is accepted
    expect(result.authorized).toBe(true);
    expect(result.reason).toBe('Order authorized');
  });

  it('should reject order with invalid user', async () => {
    // Given: Order parameters with invalid user
    const invalidOrder = {
      userId: 'invalid-user',
      marketId: 'market-456',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10
    };

    mockDb.getUserById.mockResolvedValue(null);

    // When: Order validation method is called
    const result = await orderAuthService.authorizeOrder(invalidOrder);

    // Then: Validation fails and order is rejected
    expect(result.authorized).toBe(false);
    expect(result.reason).toContain('User not found');
  });

  it('should reject order with insufficient balance', async () => {
    // Given: Order parameters with insufficient balance
    const orderWithInsufficientBalance = {
      userId: 'user-123',
      marketId: 'market-456',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10
    };

    mockDb.getUserById.mockResolvedValue({
      id: 'user-123',
      is_active: true,
      risk_level: 'medium'
    });

    mockDb.getMarketById.mockResolvedValue({
      id: 'market-456',
      is_active: true,
      max_leverage: 100,
      min_order_size: 0.001,
      max_order_size: 1000000
    });

    mockDb.getUserBalance.mockResolvedValue({
      balance: 100, // Insufficient balance
      locked_balance: 0
    });

    // When: Order validation method is called
    const result = await orderAuthService.authorizeOrder(orderWithInsufficientBalance);

    // Then: Validation fails and order is rejected
    expect(result.authorized).toBe(false);
    expect(result.reason).toContain('Insufficient balance');
  });
});

describe('1.2-UNIT-002: Order parameter sanitization', () => {
  let orderAuthService: OrderAuthorizationService;

  beforeEach(() => {
    orderAuthService = OrderAuthorizationService.getInstance();
  });

  it('should sanitize order parameters with potentially malicious input', () => {
    // Given: Order parameters with potentially malicious input
    const maliciousInput = {
      userId: 'user-123<script>alert("xss")</script>',
      marketId: 'market-456; DROP TABLE orders;',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10
    };

    // When: Sanitization method is called
    const sanitized = orderAuthService.sanitizeOrderInput(maliciousInput);

    // Then: Input is cleaned and safe for processing
    expect(sanitized.userId).toBe('user-123');
    expect(sanitized.marketId).toBe('market-456');
    expect(sanitized.userId).not.toContain('<script>');
    expect(sanitized.marketId).not.toContain('DROP TABLE');
  });

  it('should handle null and undefined values', () => {
    // Given: Order parameters with null/undefined values
    const inputWithNulls = {
      userId: 'user-123',
      marketId: 'market-456',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: null as any,
      leverage: undefined as any
    };

    // When: Sanitization method is called
    const sanitized = orderAuthService.sanitizeOrderInput(inputWithNulls);

    // Then: Null/undefined values are handled safely
    expect(sanitized.price).toBeNull();
    expect(sanitized.leverage).toBeUndefined();
  });
});

describe('1.2-UNIT-003: Order execution logic validation', () => {
  let orderAuthService: OrderAuthorizationService;

  beforeEach(() => {
    orderAuthService = OrderAuthorizationService.getInstance();
  });

  it('should validate order execution conditions', () => {
    // Given: Order with execution conditions met
    const order = {
      id: 'order-123',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: 50000,
      status: 'pending' as const,
      expiresAt: new Date(Date.now() + 3600000) // 1 hour from now
    };

    const currentPrice = 50000;

    // When: Execution logic is triggered
    const canExecute = orderAuthService.canExecuteOrder(order, currentPrice);

    // Then: Order execution proceeds with proper validation
    expect(canExecute).toBe(true);
  });

  it('should reject execution for expired orders', () => {
    // Given: Order with execution conditions not met (expired)
    const expiredOrder = {
      id: 'order-123',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: 50000,
      status: 'pending' as const,
      expiresAt: new Date(Date.now() - 3600000) // 1 hour ago
    };

    const currentPrice = 50000;

    // When: Execution logic is triggered
    const canExecute = orderAuthService.canExecuteOrder(expiredOrder, currentPrice);

    // Then: Order execution is rejected
    expect(canExecute).toBe(false);
  });

  it('should validate limit order price conditions', () => {
    // Given: Limit order with price conditions not met
    const limitOrder = {
      id: 'order-123',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: 50000,
      status: 'pending' as const,
      expiresAt: new Date(Date.now() + 3600000)
    };

    const currentPrice = 49000; // Below limit price

    // When: Execution logic is triggered
    const canExecute = orderAuthService.canExecuteOrder(limitOrder, currentPrice);

    // Then: Order execution is rejected due to price conditions
    expect(canExecute).toBe(false);
  });
});

describe('1.2-UNIT-004: Smart contract instruction validation', () => {
  let orderAuthService: OrderAuthorizationService;

  beforeEach(() => {
    orderAuthService = OrderAuthorizationService.getInstance();
  });

  it('should validate smart contract instruction parameters', () => {
    // Given: Order execution parameters
    const executionParams = {
      orderId: 'order-123',
      userId: 'user-456',
      marketSymbol: 'BTC-PERP',
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10,
      orderType: 'limit' as const
    };

    // When: Smart contract instruction is prepared
    const instruction = orderAuthService.prepareSmartContractInstruction(executionParams);

    // Then: Instruction is valid and executable
    expect(instruction).toBeDefined();
    expect(instruction.orderId).toBe('order-123');
    expect(instruction.userId).toBe('user-456');
    expect(instruction.marketSymbol).toBe('BTC-PERP');
    expect(instruction.side).toBe('long');
    expect(instruction.size).toBe(100);
    expect(instruction.price).toBe(50000);
    expect(instruction.leverage).toBe(10);
  });

  it('should reject invalid smart contract instruction parameters', () => {
    // Given: Invalid order execution parameters
    const invalidParams = {
      orderId: '',
      userId: 'user-456',
      marketSymbol: 'BTC-PERP',
      side: 'long' as const,
      size: -100, // Invalid negative size
      price: 50000,
      leverage: 10,
      orderType: 'limit' as const
    };

    // When: Smart contract instruction is prepared
    expect(() => {
      orderAuthService.prepareSmartContractInstruction(invalidParams);
    }).toThrow('Invalid order parameters');
  });
});

describe('1.2-UNIT-005: Order status state machine', () => {
  let orderAuthService: OrderAuthorizationService;

  beforeEach(() => {
    orderAuthService = OrderAuthorizationService.getInstance();
  });

  it('should transition order from pending to filled', () => {
    // Given: Order in pending status
    const order = {
      id: 'order-123',
      status: 'pending' as const,
      size: 100,
      filledSize: 0
    };

    // When: Status transition is triggered
    const newStatus = orderAuthService.transitionOrderStatus(order, 'filled', 100);

    // Then: Order moves to correct new status
    expect(newStatus).toBe('filled');
  });

  it('should transition order from pending to partially_filled', () => {
    // Given: Order in pending status
    const order = {
      id: 'order-123',
      status: 'pending' as const,
      size: 100,
      filledSize: 0
    };

    // When: Status transition is triggered
    const newStatus = orderAuthService.transitionOrderStatus(order, 'partially_filled', 50);

    // Then: Order moves to correct new status
    expect(newStatus).toBe('partially_filled');
  });

  it('should reject invalid status transitions', () => {
    // Given: Order in filled status
    const order = {
      id: 'order-123',
      status: 'filled' as const,
      size: 100,
      filledSize: 100
    };

    // When: Invalid status transition is attempted
    expect(() => {
      orderAuthService.transitionOrderStatus(order, 'pending', 0);
    }).toThrow('Invalid status transition from filled to pending');
  });
});

describe('1.2-UNIT-006: Position creation logic validation', () => {
  let orderAuthService: OrderAuthorizationService;

  beforeEach(() => {
    orderAuthService = OrderAuthorizationService.getInstance();
  });

  it('should validate position creation with filled order', () => {
    // Given: Filled order with position data
    const filledOrder = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10,
      status: 'filled' as const,
      filledSize: 100,
      averageFillPrice: 50000
    };

    // When: Position creation logic is triggered
    const position = orderAuthService.createPositionFromOrder(filledOrder);

    // Then: Position is created with correct parameters
    expect(position).toBeDefined();
    expect(position.userId).toBe('user-456');
    expect(position.marketId).toBe('market-789');
    expect(position.side).toBe('long');
    expect(position.size).toBe(100);
    expect(position.entryPrice).toBe(50000);
    expect(position.leverage).toBe(10);
  });

  it('should reject position creation for unfilled orders', () => {
    // Given: Unfilled order
    const unfilledOrder = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10,
      status: 'pending' as const,
      filledSize: 0
    };

    // When: Position creation logic is triggered
    expect(() => {
      orderAuthService.createPositionFromOrder(unfilledOrder);
    }).toThrow('Cannot create position from unfilled order');
  });
});

describe('1.2-UNIT-007: Error message generation logic', () => {
  let orderAuthService: OrderAuthorizationService;

  beforeEach(() => {
    orderAuthService = OrderAuthorizationService.getInstance();
  });

  it('should generate appropriate error message for order execution failure', () => {
    // Given: Order execution failure
    const error = new Error('Smart contract execution failed');
    const orderId = 'order-123';

    // When: Error handling is triggered
    const errorMessage = orderAuthService.generateOrderErrorMessage(error, orderId);

    // Then: Appropriate error message is generated
    expect(errorMessage).toContain('Order execution failed');
    expect(errorMessage).toContain('order-123');
    expect(errorMessage).toContain('Smart contract execution failed');
  });

  it('should generate user-friendly error messages', () => {
    // Given: Technical error
    const technicalError = new Error('RPC_CONNECTION_TIMEOUT');
    const orderId = 'order-123';

    // When: Error handling is triggered
    const errorMessage = orderAuthService.generateOrderErrorMessage(technicalError, orderId);

    // Then: User-friendly error message is generated
    expect(errorMessage).not.toContain('RPC_CONNECTION_TIMEOUT');
    expect(errorMessage).toContain('network');
    expect(errorMessage).toContain('try again');
  });
});

describe('1.2-UNIT-008: Order authorization validation', () => {
  let orderAuthService: OrderAuthorizationService;
  let mockDb: any;

  beforeEach(() => {
    mockDb = {
      getUserById: vi.fn(),
      getMarketById: vi.fn(),
      getUserBalance: vi.fn()
    };
    
    vi.mocked(SupabaseDatabaseService.getInstance).mockReturnValue(mockDb);
    orderAuthService = OrderAuthorizationService.getInstance();
  });

  afterEach(() => {
    vi.clearAllMocks();
  });

  it('should reject unauthorized order request', async () => {
    // Given: Unauthorized order request
    const unauthorizedOrder = {
      userId: 'user-123',
      marketId: 'market-456',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10
    };

    mockDb.getUserById.mockResolvedValue({
      id: 'user-123',
      is_active: false, // Inactive user
      risk_level: 'medium'
    });

    // When: Authorization check is performed
    const result = await orderAuthService.authorizeOrder(unauthorizedOrder);

    // Then: Order is rejected with security error
    expect(result.authorized).toBe(false);
    expect(result.reason).toContain('User account is inactive');
  });

  it('should reject orders from high-risk users', async () => {
    // Given: Order from high-risk user
    const highRiskOrder = {
      userId: 'user-123',
      marketId: 'market-456',
      orderType: 'limit' as const,
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10
    };

    mockDb.getUserById.mockResolvedValue({
      id: 'user-123',
      is_active: true,
      risk_level: 'high' // High risk user
    });

    // When: Authorization check is performed
    const result = await orderAuthService.authorizeOrder(highRiskOrder);

    // Then: Order is rejected with security error
    expect(result.authorized).toBe(false);
    expect(result.reason).toContain('High risk user');
  });
});

describe('1.2-UNIT-009: Position creation atomicity validation', () => {
  let orderAuthService: OrderAuthorizationService;

  beforeEach(() => {
    orderAuthService = OrderAuthorizationService.getInstance();
  });

  it('should validate atomic transaction for position creation', async () => {
    // Given: Order execution with position creation failure
    const order = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10,
      status: 'filled' as const,
      filledSize: 100,
      averageFillPrice: 50000
    };

    // Mock position creation failure
    const mockCreatePosition = vi.fn().mockRejectedValue(new Error('Position creation failed'));

    // When: Atomic transaction is processed
    try {
      await orderAuthService.executeAtomicOrderTransaction(order, mockCreatePosition);
    } catch (error) {
      // Then: Order execution is rolled back to maintain consistency
      expect(error.message).toContain('Position creation failed');
      expect(mockCreatePosition).toHaveBeenCalled();
    }
  });

  it('should complete atomic transaction successfully', async () => {
    // Given: Order execution with successful position creation
    const order = {
      id: 'order-123',
      userId: 'user-456',
      marketId: 'market-789',
      side: 'long' as const,
      size: 100,
      price: 50000,
      leverage: 10,
      status: 'filled' as const,
      filledSize: 100,
      averageFillPrice: 50000
    };

    // Mock successful position creation
    const mockCreatePosition = vi.fn().mockResolvedValue({ id: 'position-123' });

    // When: Atomic transaction is processed
    const result = await orderAuthService.executeAtomicOrderTransaction(order, mockCreatePosition);

    // Then: Transaction completes successfully
    expect(result.success).toBe(true);
    expect(result.positionId).toBe('position-123');
  });
});
