import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import { MatchingService } from '../../src/services/matching';
import { smartContractService } from '../../src/services/smartContractService';
import { WebSocketService } from '../../src/services/websocket';
import { getSupabaseService } from '../../src/services/supabaseService';

// Mock dependencies
vi.mock('../../src/services/smartContractService');
vi.mock('../../src/services/websocket');
vi.mock('../../src/services/supabaseService');

describe('Order → Position Flow E2E Test', () => {
  let matchingService: MatchingService;
  let mockSupabase: any;
  let mockWebSocket: any;

  const MOCK_USER_ID = 'test-user-123';
  const MOCK_MARKET_ID = 'market-123';
  const MOCK_ORDER_ID = 'order-123';

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Mock Supabase service
    mockSupabase = {
      getMarketBySymbol: vi.fn().mockResolvedValue({ id: MOCK_MARKET_ID }),
      insertOrder: vi.fn().mockResolvedValue({ id: MOCK_ORDER_ID }),
      getPendingOrders: vi.fn().mockResolvedValue([]),
      getOrCreatePosition: vi.fn().mockResolvedValue({ 
        id: 'position-123', 
        size: 0, 
        entry_price: 0 
      }),
      updatePosition: vi.fn().mockResolvedValue({}),
      calculatePositionHealth: vi.fn().mockResolvedValue(100),
      updatePositionHealth: vi.fn().mockResolvedValue({}),
      getClient: vi.fn().mockReturnValue({
        from: vi.fn().mockReturnValue({
          update: vi.fn().mockReturnValue({
            eq: vi.fn().mockResolvedValue({})
          }),
          select: vi.fn().mockReturnValue({
            eq: vi.fn().mockReturnValue({
              single: vi.fn().mockResolvedValue({ smart_contract_tx: 'tx-123' })
            })
          })
        })
      })
    };

    // Mock WebSocket service
    mockWebSocket = {
      broadcast: vi.fn(),
      broadcastToUser: vi.fn()
    };

    // Mock smart contract service
    (smartContractService.executeOrder as any) = vi.fn().mockResolvedValue({
      success: true,
      transactionSignature: 'tx-123',
      positionId: 'pos-123'
    });

    (smartContractService.createPosition as any) = vi.fn().mockResolvedValue({
      success: true,
      positionId: 'pos-123'
    });

    // Setup mocks
    vi.mocked(getSupabaseService).mockReturnValue(mockSupabase);
    vi.mocked(WebSocketService.getInstance).mockReturnValue(mockWebSocket as any);

    matchingService = MatchingService.getInstance();
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  it('should complete order → position flow successfully', async () => {
    // Mock oracle price
    const mockPythOracleService = await import('../../src/services/pythOracleService');
    vi.mocked(mockPythOracleService.pythOracleService.getLatestPrice).mockResolvedValue(50000);

    // Place order
    const result = await matchingService.placeOrder({
      userId: MOCK_USER_ID,
      symbol: 'BTC/USD',
      side: 'buy',
      size: 0.001,
      orderType: 'market',
      leverage: 1
    });

    // Verify order was created
    expect(result).toBeDefined();
    expect(result.orderId).toBe(MOCK_ORDER_ID);
    expect(result.filled).toBe(true);
    expect(result.fills).toHaveLength(1);
    expect(result.fills[0].price).toBe(50000);
    expect(result.fills[0].size).toBe(0.001);

    // Verify market lookup
    expect(mockSupabase.getMarketBySymbol).toHaveBeenCalledWith('BTC/USD');

    // Verify order insertion
    expect(mockSupabase.insertOrder).toHaveBeenCalledWith({
      user_id: MOCK_USER_ID,
      market_id: MOCK_MARKET_ID,
      order_account: 'OFFCHAIN',
      order_type: 'market',
      side: 'long',
      size: 0.001,
      price: null,
      leverage: 1,
      status: 'pending'
    });

    // Verify smart contract execution
    expect(smartContractService.executeOrder).toHaveBeenCalledWith({
      orderId: MOCK_ORDER_ID,
      userId: MOCK_USER_ID,
      marketSymbol: 'BTC/USD',
      side: 'long',
      size: 0.001,
      price: 50000,
      leverage: 1,
      orderType: 'market'
    });

    // Verify position creation
    expect(mockSupabase.getOrCreatePosition).toHaveBeenCalledWith(
      MOCK_USER_ID,
      MOCK_MARKET_ID,
      'long'
    );

    // Verify position update
    expect(mockSupabase.updatePosition).toHaveBeenCalledWith(
      'position-123',
      0.001,
      50000
    );

    // Verify WebSocket broadcasts
    expect(mockWebSocket.broadcastToUser).toHaveBeenCalledWith(
      MOCK_USER_ID,
      'order_update',
      expect.objectContaining({
        symbol: 'BTC/USD',
        orderId: MOCK_ORDER_ID,
        status: 'filled',
        filledSize: 0.001,
        averageFillPrice: 50000,
        userId: MOCK_USER_ID
      })
    );

    expect(mockWebSocket.broadcastToUser).toHaveBeenCalledWith(
      MOCK_USER_ID,
      'position_update',
      expect.objectContaining({
        positionId: 'position-123',
        healthFactor: 100,
        symbol: 'BTC/USD',
        side: 'long'
      })
    );
  });

  it('should handle order placement errors gracefully', async () => {
    // Mock oracle service to throw error
    const mockPythOracleService = await import('../../src/services/pythOracleService');
    vi.mocked(mockPythOracleService.pythOracleService.getLatestPrice).mockResolvedValue(null);

    // Place order should fail
    await expect(matchingService.placeOrder({
      userId: MOCK_USER_ID,
      symbol: 'BTC/USD',
      side: 'buy',
      size: 0.001,
      orderType: 'market',
      leverage: 1
    })).rejects.toThrow('Price unavailable');
  });

  it('should handle smart contract execution failure', async () => {
    // Mock oracle price
    const mockPythOracleService = await import('../../src/services/pythOracleService');
    vi.mocked(mockPythOracleService.pythOracleService.getLatestPrice).mockResolvedValue(50000);

    // Mock smart contract failure
    (smartContractService.executeOrder as any).mockResolvedValue({
      success: false,
      error: 'Smart contract execution failed'
    });

    // Place order should fail
    await expect(matchingService.placeOrder({
      userId: MOCK_USER_ID,
      symbol: 'BTC/USD',
      side: 'buy',
      size: 0.001,
      orderType: 'market',
      leverage: 1
    })).rejects.toThrow('Smart contract execution failed');

    // Verify order status was updated to failed
    expect(mockSupabase.getClient().from().update().eq).toHaveBeenCalledWith(
      expect.objectContaining({
        status: 'failed',
        error_message: 'Smart contract execution failed'
      })
    );
  });

  it('should handle position creation failure gracefully', async () => {
    // Mock oracle price
    const mockPythOracleService = await import('../../src/services/pythOracleService');
    vi.mocked(mockPythOracleService.pythOracleService.getLatestPrice).mockResolvedValue(50000);

    // Mock position creation failure
    (smartContractService.createPosition as any).mockResolvedValue({
      success: false,
      error: 'Position creation failed'
    });

    // Place order should still succeed (position creation failure is non-critical)
    const result = await matchingService.placeOrder({
      userId: MOCK_USER_ID,
      symbol: 'BTC/USD',
      side: 'buy',
      size: 0.001,
      orderType: 'market',
      leverage: 1
    });

    expect(result).toBeDefined();
    expect(result.filled).toBe(true);

    // Verify position creation was attempted
    expect(smartContractService.createPosition).toHaveBeenCalledWith(
      MOCK_USER_ID,
      'BTC/USD',
      'long',
      0.001,
      50000,
      1
    );
  });
});
