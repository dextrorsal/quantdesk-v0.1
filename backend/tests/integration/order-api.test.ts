import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import request from 'supertest';
import express from 'express';
import { orderRoutes } from '../../src/routes/orders';
import { matchingService } from '../../src/services/matching';
import { authMiddleware } from '../../src/middleware/auth';

// Mock dependencies
vi.mock('../../src/services/matching');
vi.mock('../../src/middleware/auth');

describe('Order API Endpoint Tests', () => {
  let app: express.Application;
  let mockMatchingService: any;

  beforeEach(() => {
    vi.clearAllMocks();
    
    // Mock auth middleware
    vi.mocked(authMiddleware).mockImplementation((req: any, res: any, next: any) => {
      req.user = { id: 'test-user-123' };
      next();
    });

    // Mock matching service
    mockMatchingService = {
      placeOrder: vi.fn()
    };
    vi.mocked(matchingService).mockReturnValue(mockMatchingService);

    // Create Express app
    app = express();
    app.use(express.json());
    app.use('/api/orders', authMiddleware as any, orderRoutes);
  });

  afterEach(() => {
    vi.restoreAllMocks();
  });

  describe('POST /api/orders', () => {
    it('should place order successfully with valid data', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy',
        size: 0.001,
        orderType: 'market',
        leverage: 1
      };

      const mockResult = {
        orderId: 'order-123',
        filled: true,
        fills: [{ price: 50000, size: 0.001 }],
        averageFillPrice: 50000
      };

      mockMatchingService.placeOrder.mockResolvedValue(mockResult);

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(200);

      expect(response.body).toEqual({
        success: true,
        message: 'Order placed successfully',
        data: mockResult
      });

      expect(mockMatchingService.placeOrder).toHaveBeenCalledWith({
        userId: 'test-user-123',
        symbol: 'BTC/USD',
        side: 'buy',
        size: 0.001,
        orderType: 'market',
        price: undefined,
        leverage: 1
      });
    });

    it('should reject order with missing required fields', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy'
        // Missing size and orderType
      };

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(400);

      expect(response.body).toEqual({
        success: false,
        error: 'Missing required fields',
        details: 'symbol, side, size, orderType are required',
        code: 'MISSING_FIELDS'
      });
    });

    it('should reject order with invalid size', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy',
        size: -0.001, // Invalid negative size
        orderType: 'market',
        leverage: 1
      };

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(400);

      expect(response.body).toEqual({
        success: false,
        error: 'Invalid size',
        details: 'Size must be a positive number',
        code: 'INVALID_SIZE'
      });
    });

    it('should reject limit order without price', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy',
        size: 0.001,
        orderType: 'limit',
        leverage: 1
        // Missing price for limit order
      };

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(400);

      expect(response.body).toEqual({
        success: false,
        error: 'Invalid price',
        details: 'Limit orders require a positive price',
        code: 'INVALID_PRICE'
      });
    });

    it('should reject order with invalid leverage', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy',
        size: 0.001,
        orderType: 'market',
        leverage: 150 // Invalid leverage > 100
      };

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(400);

      expect(response.body).toEqual({
        success: false,
        error: 'Invalid leverage',
        details: 'Leverage must be between 1 and 100',
        code: 'INVALID_LEVERAGE'
      });
    });

    it('should reject order with invalid side', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'invalid', // Invalid side
        size: 0.001,
        orderType: 'market',
        leverage: 1
      };

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(400);

      expect(response.body).toEqual({
        success: false,
        error: 'Invalid side',
        details: 'Side must be either "buy" or "sell"',
        code: 'INVALID_SIDE'
      });
    });

    it('should reject order with invalid order type', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy',
        size: 0.001,
        orderType: 'invalid', // Invalid order type
        leverage: 1
      };

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(400);

      expect(response.body).toEqual({
        success: false,
        error: 'Invalid order type',
        details: 'Order type must be either "market" or "limit"',
        code: 'INVALID_ORDER_TYPE'
      });
    });

    it('should handle matching service errors', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy',
        size: 0.001,
        orderType: 'market',
        leverage: 1
      };

      mockMatchingService.placeOrder.mockRejectedValue(new Error('Size must be positive'));

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(400);

      expect(response.body).toEqual({
        success: false,
        error: 'Invalid order parameters',
        details: 'Size must be positive',
        code: 'INVALID_SIZE'
      });
    });

    it('should handle price unavailable errors', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy',
        size: 0.001,
        orderType: 'market',
        leverage: 1
      };

      mockMatchingService.placeOrder.mockRejectedValue(new Error('Price unavailable'));

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(503);

      expect(response.body).toEqual({
        success: false,
        error: 'Market data unavailable',
        details: 'Unable to get current market price',
        code: 'PRICE_UNAVAILABLE'
      });
    });

    it('should handle smart contract execution errors', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy',
        size: 0.001,
        orderType: 'market',
        leverage: 1
      };

      mockMatchingService.placeOrder.mockRejectedValue(new Error('Smart contract execution failed: Transaction failed'));

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(500);

      expect(response.body).toEqual({
        success: false,
        error: 'Order execution failed',
        details: 'Order was created but failed to execute on blockchain',
        code: 'SMART_CONTRACT_ERROR'
      });
    });

    it('should handle generic errors', async () => {
      const orderData = {
        symbol: 'BTC/USD',
        side: 'buy',
        size: 0.001,
        orderType: 'market',
        leverage: 1
      };

      mockMatchingService.placeOrder.mockRejectedValue(new Error('Unknown error'));

      const response = await request(app)
        .post('/api/orders')
        .send(orderData)
        .expect(500);

      expect(response.body).toEqual({
        success: false,
        error: 'Order placement failed',
        details: 'Unknown error',
        code: 'ORDER_ERROR'
      });
    });
  });
});
