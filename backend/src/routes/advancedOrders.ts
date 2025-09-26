import { Router, Request, Response } from 'express';
import { advancedOrderService, AdvancedOrder, OrderType, PositionSide, TimeInForce } from '../services/advancedOrderService';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

/**
 * POST /api/advanced-orders
 * Place a new advanced order
 */
router.post('/', async (req: Request, res: Response) => {
  try {
    const orderData: Partial<AdvancedOrder> = req.body;
    
    // Validate required fields
    if (!orderData.user_id || !orderData.market_id || !orderData.order_type || !orderData.side || !orderData.size || !orderData.leverage) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: user_id, market_id, order_type, side, size, leverage'
      });
    }

    const order = await advancedOrderService.placeOrder(orderData);
    
    res.status(201).json({
      success: true,
      data: order,
      message: 'Advanced order placed successfully'
    });

  } catch (error) {
    logger.error('Error placing advanced order:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to place advanced order',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/advanced-orders/user/:userId
 * Get all orders for a specific user
 */
router.get('/user/:userId', async (req: Request, res: Response) => {
  try {
    const { userId } = req.params;
    const { status } = req.query;

    const orders = await advancedOrderService.getUserOrders(userId, status as any);
    
    res.json({
      success: true,
      data: orders,
      count: orders.length
    });

  } catch (error) {
    logger.error('Error fetching user orders:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch user orders',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/advanced-orders/:orderId
 * Get a specific order by ID
 */
router.get('/:orderId', async (req: Request, res: Response) => {
  try {
    const { orderId } = req.params;

    const order = await advancedOrderService.getOrderById(orderId);
    
    if (!order) {
      return res.status(404).json({
        success: false,
        error: 'Order not found'
      });
    }

    res.json({
      success: true,
      data: order
    });

  } catch (error) {
    logger.error('Error fetching order:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch order',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * DELETE /api/advanced-orders/:orderId
 * Cancel an order
 */
router.delete('/:orderId', async (req: Request, res: Response) => {
  try {
    const { orderId } = req.params;
    const { userId } = req.body;

    if (!userId) {
      return res.status(400).json({
        success: false,
        error: 'User ID is required'
      });
    }

    const success = await advancedOrderService.cancelOrder(orderId, userId);
    
    if (success) {
      res.json({
        success: true,
        message: 'Order cancelled successfully'
      });
    } else {
      res.status(400).json({
        success: false,
        error: 'Failed to cancel order'
      });
    }

  } catch (error) {
    logger.error('Error cancelling order:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to cancel order',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/advanced-orders/execute-conditional
 * Execute conditional orders for a market (called by keeper bots)
 */
router.post('/execute-conditional', async (req: Request, res: Response) => {
  try {
    const { marketId, currentPrice } = req.body;

    if (!marketId || !currentPrice) {
      return res.status(400).json({
        success: false,
        error: 'Market ID and current price are required'
      });
    }

    const results = await advancedOrderService.executeConditionalOrders(marketId, currentPrice);
    
    res.json({
      success: true,
      data: results,
      count: results.length,
      message: `${results.length} conditional orders processed`
    });

  } catch (error) {
    logger.error('Error executing conditional orders:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to execute conditional orders',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/advanced-orders/execute-twap
 * Execute TWAP orders (called by scheduler)
 */
router.post('/execute-twap', async (req: Request, res: Response) => {
  try {
    const results = await advancedOrderService.executeTWAPOrders();
    
    res.json({
      success: true,
      data: results,
      count: results.length,
      message: `${results.length} TWAP orders processed`
    });

  } catch (error) {
    logger.error('Error executing TWAP orders:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to execute TWAP orders',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/advanced-orders/types
 * Get available order types
 */
router.get('/types', (req: Request, res: Response) => {
  res.json({
    success: true,
    data: {
      order_types: Object.values(OrderType),
      position_sides: Object.values(PositionSide),
      time_in_force: Object.values(TimeInForce)
    }
  });
});

/**
 * GET /api/advanced-orders/stats
 * Get order statistics
 */
router.get('/stats', async (req: Request, res: Response) => {
  try {
    // This would typically query the database for order statistics
    const stats = {
      total_orders: 0,
      pending_orders: 0,
      filled_orders: 0,
      cancelled_orders: 0,
      orders_by_type: {},
      orders_by_status: {},
      total_volume: 0,
      average_order_size: 0
    };

    res.json({
      success: true,
      data: stats
    });

  } catch (error) {
    logger.error('Error fetching order stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch order statistics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
