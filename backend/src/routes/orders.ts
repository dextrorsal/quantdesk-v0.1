import express from 'express';
import { DatabaseService } from '../services/database';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandler';
import { AuthenticatedRequest } from '../middleware/auth';
import { matchingService } from '../services/matching';

const router = express.Router();
const logger = new Logger();
const db = DatabaseService.getInstance();

// Get user orders
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { status } = req.query;

  try {
    const orders = await db.getUserOrders(req.user!.id, status as string);
    
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
    if (order.user_id !== req.user!.id) {
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
router.post('/', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { symbol, side, size, orderType, price, leverage } = req.body || {};

  if (!symbol || !side || !size || !orderType) {
    return res.status(400).json({ error: 'symbol, side, size, orderType required' });
  }

  try {
    const result = await matchingService.placeOrder({
      userId: req.user!.id,
      symbol,
      side,
      size: parseFloat(size),
      orderType,
      price: price != null ? parseFloat(price) : undefined,
      leverage: leverage != null ? parseInt(leverage) : undefined
    });

    res.json({ success: true, ...result });
  } catch (error) {
    logger.error('Error placing order:', error);
    res.status(400).json({ error: (error as Error).message || 'ORDER_ERROR' });
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
    if (order.user_id !== req.user!.id) {
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
