import express from 'express';
import { DatabaseService } from '../services/database';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandler';
import { AuthenticatedRequest } from '../middleware/auth';

const router = express.Router();
const logger = new Logger();
const db = DatabaseService.getInstance();

// Get user positions
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const positions = await db.getUserPositions(req.user!.id);
    
    res.json({
      success: true,
      positions: positions.map(position => ({
        id: position.id,
        marketId: position.market_id,
        side: position.side,
        size: position.size,
        entryPrice: position.entry_price,
        currentPrice: position.current_price,
        margin: position.margin,
        leverage: position.leverage,
        unrealizedPnl: position.unrealized_pnl,
        realizedPnl: position.realized_pnl,
        fundingFees: position.funding_fees,
        isLiquidated: position.is_liquidated,
        liquidationPrice: position.liquidation_price,
        healthFactor: position.health_factor,
        createdAt: position.created_at,
        updatedAt: position.updated_at
      }))
    });

  } catch (error) {
    logger.error('Error fetching positions:', error);
    res.status(500).json({
      error: 'Failed to fetch positions',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get position by ID
router.get('/:id', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { id } = req.params;

  try {
    const position = await db.getPositionById(id);
    
    if (!position) {
      return res.status(404).json({
        error: 'Position not found',
        code: 'POSITION_NOT_FOUND'
      });
    }

    // Check if user owns this position
    if (position.user_id !== req.user!.id) {
      return res.status(403).json({
        error: 'Access denied',
        code: 'ACCESS_DENIED'
      });
    }

    res.json({
      success: true,
      position: {
        id: position.id,
        marketId: position.market_id,
        side: position.side,
        size: position.size,
        entryPrice: position.entry_price,
        currentPrice: position.current_price,
        margin: position.margin,
        leverage: position.leverage,
        unrealizedPnl: position.unrealized_pnl,
        realizedPnl: position.realized_pnl,
        fundingFees: position.funding_fees,
        isLiquidated: position.is_liquidated,
        liquidationPrice: position.liquidation_price,
        healthFactor: position.health_factor,
        createdAt: position.created_at,
        updatedAt: position.updated_at,
        closedAt: position.closed_at
      }
    });

  } catch (error) {
    logger.error(`Error fetching position ${id}:`, error);
    res.status(500).json({
      error: 'Failed to fetch position',
      code: 'FETCH_ERROR'
    });
  }
}));

export default router;
