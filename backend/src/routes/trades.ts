import express from 'express';
import { SupabaseDatabaseService } from '../services/supabaseDatabase';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';
import { AuthenticatedRequest } from '../middleware/auth';

const router = express.Router();
const logger = new Logger();
const db = SupabaseDatabaseService.getInstance();

// Get user trades
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const limit = parseInt(req.query.limit as string) || 100;

  try {
    const trades = await db.getUserTrades(req.userId, limit);
    
    res.json({
      success: true,
      trades: trades.map(trade => ({
        id: trade.id,
        marketId: trade.market_id,
        positionId: trade.position_id,
        orderId: trade.order_id,
        side: trade.side,
        size: trade.size,
        price: trade.price,
        value: trade.size * trade.price,
        fees: trade.fees,
        pnl: trade.pnl,
        timestamp: trade.created_at
      }))
    });

  } catch (error) {
    logger.error('Error fetching trades:', error);
    res.status(500).json({
      error: 'Failed to fetch trades',
      code: 'FETCH_ERROR'
    });
  }
}));

export default router;
