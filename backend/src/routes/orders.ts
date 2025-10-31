import { Router } from 'express';
import { Logger } from '../utils/logger';
import { SupabaseDatabaseService } from '../services/supabaseDatabase';
import { asyncHandler } from '../middleware/errorHandling';
import { AuthenticatedRequest } from '../middleware/auth';

const router = Router();
const logger = new Logger();
const db = SupabaseDatabaseService.getInstance();

/**
 * Orders Routes
 * 
 * Real order management using Supabase database
 */

// Get user orders
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    logger.info(`Getting orders for user: ${req.userId}`);
    const orders = await db.getUserOrders(req.userId);
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
        leverage: order.leverage,
        status: order.status,
        filledSize: order.filled_size,
        averageFillPrice: order.average_fill_price,
        createdAt: order.created_at,
        updatedAt: order.updated_at
      }))
    });
  } catch (error) {
    logger.error('Error getting orders:', error);
    res.status(500).json({ error: 'Failed to fetch orders' });
  }
}));

// Create new order
router.post('/', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    logger.info('Creating new order');
    const { marketId, orderType, side, size, price, stopPrice, leverage, expiresAt } = req.body;
    
    if (!marketId || !orderType || !side || !size || leverage === undefined) {
      return res.status(400).json({ 
        error: 'Missing required fields',
        required: ['marketId', 'orderType', 'side', 'size', 'leverage']
      });
    }
    
    const order = await db.createOrder({
      user_id: req.userId,
      market_id: marketId,
      order_account: '', // Will be set by Solana program
      order_type: orderType,
      side,
      size,
      price,
      stop_price: stopPrice,
      leverage,
      status: 'pending',
      expires_at: expiresAt ? new Date(expiresAt) : undefined
    });
    
    res.json({ 
      success: true,
      order: {
        id: order.id,
        marketId: order.market_id,
        orderType: order.order_type,
        side: order.side,
        size: order.size,
        price: order.price,
        leverage: order.leverage,
        status: order.status,
        createdAt: order.created_at
      }
    });
  } catch (error) {
    logger.error('Error creating order:', error);
    res.status(500).json({ error: 'Failed to create order' });
  }
}));

// Cancel order
router.delete('/:id', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { id } = req.params;
    logger.info(`Cancelling order: ${id}`);
    
    // Verify order belongs to user
    const order = await db.getOrderById(id);
    if (!order || order.user_id !== req.userId) {
      return res.status(404).json({ error: 'Order not found' });
    }
    
    await db.updateOrder(id, { 
      status: 'cancelled',
      cancelled_at: new Date()
    });
    
    res.json({ success: true, message: 'Order cancelled' });
  } catch (error) {
    logger.error('Error cancelling order:', error);
    res.status(500).json({ error: 'Failed to cancel order' });
  }
}));

export default router;
