import express from 'express';
import { SupabaseDatabaseService } from '../services/supabaseDatabase';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';
import { AuthenticatedRequest } from '../middleware/auth';
import { matchingService } from '../services/matching';
import { orderRateLimit } from '../middleware/rateLimiting';

const router = express.Router();
const logger = new Logger();
const db = SupabaseDatabaseService.getInstance();

// Debug endpoint to test JWT_SECRET
router.get('/debug', asyncHandler(async (req, res) => {
  const jwt = require('jsonwebtoken');
  const token = jwt.sign({ wallet_pubkey: 'test-wallet-address', iat: Math.floor(Date.now() / 1000), exp: Math.floor(Date.now() / 1000) + (60 * 60) }, 'test-jwt-secret');
  
  res.json({
    jwtSecret: process.env.JWT_SECRET,
    testToken: token,
    configJwtSecret: require('../config/environment').config.JWT_SECRET
  });
}));

// Get user orders
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { status } = req.query;

  try {
    const orders = await db.getUserOrders(req.userId, status as string);
    
    res.json({
      success: true,
      orders: orders.map(order => ({
        id: order.id,
        marketId: order.market_id,
        orderType: order.order_type,
        side: order.side,
        size: order.size,
        price: order.price,
        stopPrice: order.stop_price,
        trailingDistance: order.trailing_distance,
        leverage: order.leverage,
        status: order.status,
        filledSize: order.filled_size,
        remainingSize: order.size - (order.filled_size || 0),
        averageFillPrice: order.average_fill_price,
        createdAt: order.created_at,
        updatedAt: order.updated_at,
        expiresAt: order.expires_at,
        filledAt: order.filled_at,
        cancelledAt: order.cancelled_at
      }))
    });

  } catch (error) {
    logger.error('Error fetching orders:', error);
    res.status(500).json({
      error: 'Failed to fetch orders',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get order by ID
router.get('/:id', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { id } = req.params;

  try {
    const order = await db.getOrderById(id);
    
    if (!order) {
      return res.status(404).json({
        error: 'Order not found',
        code: 'ORDER_NOT_FOUND'
      });
    }

    // Check if user owns this order
    if (order.user_id !== req.userId) {
      return res.status(403).json({
        error: 'Access denied',
        code: 'ACCESS_DENIED'
      });
    }

    res.json({
      success: true,
      order: {
        id: order.id,
        marketId: order.market_id,
        orderType: order.order_type,
        side: order.side,
        size: order.size,
        price: order.price,
        stopPrice: order.stop_price,
        trailingDistance: order.trailing_distance,
        leverage: order.leverage,
        status: order.status,
        filledSize: order.filled_size,
        remainingSize: order.size - (order.filled_size || 0),
        averageFillPrice: order.average_fill_price,
        createdAt: order.created_at,
        updatedAt: order.updated_at,
        expiresAt: order.expires_at,
        filledAt: order.filled_at,
        cancelledAt: order.cancelled_at
      }
    });

  } catch (error) {
    logger.error(`Error fetching order ${id}:`, error);
    res.status(500).json({
      error: 'Failed to fetch order',
      code: 'FETCH_ERROR'
    });
  }
}));

// Place order
router.post('/', orderRateLimit, asyncHandler(async (req: AuthenticatedRequest, res) => {
  console.log('🔍 Order placement request received:', req.body);
  console.log('🔍 User from request:', req.user);
  
  const { symbol, side, size, orderType, price, leverage } = req.body || {};

  // Enhanced validation
  if (!symbol || !side || !size || !orderType) {
    return res.status(400).json({ 
      success: false,
      error: 'Missing required fields',
      details: 'symbol, side, size, orderType are required',
      code: 'MISSING_FIELDS'
    });
  }

  // Validate data types and ranges
  const parsedSize = parseFloat(size);
  const parsedPrice = price != null ? parseFloat(price) : undefined;
  const parsedLeverage = leverage != null ? parseInt(leverage) : undefined;

  if (isNaN(parsedSize) || parsedSize <= 0) {
    return res.status(400).json({
      success: false,
      error: 'Invalid size',
      details: 'Size must be a positive number',
      code: 'INVALID_SIZE'
    });
  }

  if (orderType === 'limit' && (parsedPrice == null || isNaN(parsedPrice) || parsedPrice <= 0)) {
    return res.status(400).json({
      success: false,
      error: 'Invalid price',
      details: 'Limit orders require a positive price',
      code: 'INVALID_PRICE'
    });
  }

  if (parsedLeverage != null && (isNaN(parsedLeverage) || parsedLeverage < 1 || parsedLeverage > 100)) {
    return res.status(400).json({
      success: false,
      error: 'Invalid leverage',
      details: 'Leverage must be between 1 and 100',
      code: 'INVALID_LEVERAGE'
    });
  }

  if (!['buy', 'sell'].includes(side)) {
    return res.status(400).json({
      success: false,
      error: 'Invalid side',
      details: 'Side must be either "buy" or "sell"',
      code: 'INVALID_SIDE'
    });
  }

  if (!['market', 'limit'].includes(orderType)) {
    return res.status(400).json({
      success: false,
      error: 'Invalid order type',
      details: 'Order type must be either "market" or "limit"',
      code: 'INVALID_ORDER_TYPE'
    });
  }

  try {
    console.log('🔍 Calling matchingService.placeOrder with:', {
      userId: req.userId,
      symbol,
      side,
      size: parsedSize,
      orderType,
      price: parsedPrice,
      leverage: parsedLeverage
    });
    
    const result = await matchingService.placeOrder({
      userId: req.userId,
      symbol,
      side,
      size: parsedSize,
      orderType,
      price: parsedPrice,
      leverage: parsedLeverage
    });

    res.json({ 
      success: true, 
      message: 'Order placed successfully',
      data: result 
    });
  } catch (error) {
    logger.error('Error placing order:', error);
    
    // Determine error type and provide appropriate response
    const errorMessage = error instanceof Error ? error.message : 'Unknown error occurred';
    
    if (errorMessage.includes('Size must be positive')) {
      return res.status(400).json({
        success: false,
        error: 'Invalid order parameters',
        details: errorMessage,
        code: 'INVALID_SIZE'
      });
    }
    
    if (errorMessage.includes('Limit orders require')) {
      return res.status(400).json({
        success: false,
        error: 'Invalid order parameters',
        details: errorMessage,
        code: 'INVALID_PRICE'
      });
    }
    
    if (errorMessage.includes('Price unavailable')) {
      return res.status(503).json({
        success: false,
        error: 'Market data unavailable',
        details: 'Unable to get current market price',
        code: 'PRICE_UNAVAILABLE'
      });
    }
    
    if (errorMessage.includes('Smart contract execution failed')) {
      return res.status(500).json({
        success: false,
        error: 'Order execution failed',
        details: 'Order was created but failed to execute on blockchain',
        code: 'SMART_CONTRACT_ERROR'
      });
    }
    
    // Generic error response
    res.status(500).json({
      success: false,
      error: 'Order placement failed',
      details: errorMessage,
      code: 'ORDER_ERROR'
    });
  }
}));

// Cancel order
router.post('/:id/cancel', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { id } = req.params;

  try {
    const order = await db.getOrderById(id);
    
    if (!order) {
      return res.status(404).json({
        error: 'Order not found',
        code: 'ORDER_NOT_FOUND'
      });
    }

    // Check if user owns this order
    if (order.user_id !== req.userId) {
      return res.status(403).json({
        error: 'Access denied',
        code: 'ACCESS_DENIED'
      });
    }

    // Check if order can be cancelled
    if (order.status !== 'pending') {
      return res.status(400).json({
        error: 'Order cannot be cancelled',
        code: 'ORDER_NOT_CANCELLABLE'
      });
    }

    // Update order status
    const updatedOrder = await db.updateOrder(id, {
      status: 'cancelled',
      cancelled_at: new Date()
    });

    // 🚀 NEW: Broadcast order cancellation
    const { WebSocketService } = await import('../services/websocket');
    WebSocketService.current?.broadcast?.('order_update', {
      symbol: order.market_id || 'Unknown',
      orderId: id,
      status: 'cancelled',
      userId: req.userId,
      timestamp: Date.now()
    });

    res.json({
      success: true,
      order: {
        id: updatedOrder.id,
        status: updatedOrder.status,
        cancelledAt: updatedOrder.cancelled_at
      }
    });

  } catch (error) {
    logger.error(`Error cancelling order ${id}:`, error);
    res.status(500).json({
      error: 'Failed to cancel order',
      code: 'CANCEL_ERROR'
    });
  }
}));

export default router;
