import express from 'express';
import { SupabaseDatabaseService } from '../services/supabaseDatabase';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';
import { AuthenticatedRequest } from '../middleware/auth';
import { pnlCalculationService } from '../services/pnlCalculationService';

const router = express.Router();
const logger = new Logger();
const db = SupabaseDatabaseService.getInstance();

// Get user positions
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const positions = await db.getUserPositions(req.userId);
    
    // 🚀 NEW: Calculate P&L using centralized service
    const positionsWithPnl = await pnlCalculationService.updatePositionPnlWithCurrentPrices(
      positions.map(position => ({
        id: position.id,
        symbol: position.market_id, // This should be symbol, but using market_id for now
        side: position.side,
        size: position.size,
        entryPrice: position.entry_price,
        currentPrice: position.current_price,
        leverage: position.leverage,
        margin: position.margin
      }))
    );
    
    res.json({
      success: true,
      positions: positionsWithPnl.map((positionWithPnl, index) => {
        const originalPosition = positions[index];
        return {
          id: positionWithPnl.id,
          marketId: originalPosition.market_id,
          symbol: positionWithPnl.symbol,
          side: positionWithPnl.side,
          size: positionWithPnl.size,
          entryPrice: positionWithPnl.entryPrice,
          currentPrice: positionWithPnl.currentPrice,
          margin: positionWithPnl.margin,
          leverage: positionWithPnl.leverage,
          unrealizedPnl: positionWithPnl.unrealizedPnl,
          unrealizedPnlPercent: positionWithPnl.unrealizedPnlPercent,
          liquidationPrice: positionWithPnl.liquidationPrice,
          healthFactor: positionWithPnl.healthFactor,
          marginRatio: positionWithPnl.marginRatio,
          isLiquidated: originalPosition.is_liquidated,
          createdAt: originalPosition.created_at,
          updatedAt: originalPosition.updated_at
        };
      })
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
    if (position.user_id !== req.userId) {
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

// Close position
router.post('/:id/close', asyncHandler(async (req: AuthenticatedRequest, res) => {
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
    if (position.user_id !== req.userId) {
      return res.status(403).json({
        error: 'Access denied',
        code: 'ACCESS_DENIED'
      });
    }

    // Check if position can be closed
    if (position.is_liquidated) {
      return res.status(400).json({
        error: 'Position is already liquidated',
        code: 'POSITION_LIQUIDATED'
      });
    }

    // Close position in database
    const closedPosition = await db.updatePosition(id, {
      is_liquidated: true,
      closed_at: new Date(),
      realized_pnl: position.unrealized_pnl || 0
    });

    // 🚀 NEW: Close position on smart contract
    try {
      const { smartContractService } = await import('../services/smartContractService');
      const smartContractResult = await smartContractService.closePosition(id, req.userId);

      if (smartContractResult.success) {
        console.log(`✅ Position ${id} closed on smart contract:`, smartContractResult.transactionSignature);
      } else {
        console.error(`❌ Smart contract position closing failed:`, smartContractResult.error);
      }
    } catch (error) {
      console.error(`❌ Error closing position ${id} on smart contract:`, error);
    }

    // 🚀 NEW: Broadcast position closure
    const { WebSocketService } = await import('../services/websocket');
    WebSocketService.current?.broadcast?.('position_update', {
      positionId: id,
      status: 'closed',
      userId: req.userId,
      realizedPnl: position.unrealized_pnl || 0,
      timestamp: Date.now()
    });

    res.json({
      success: true,
      position: {
        id: closedPosition.id,
        status: 'closed',
        realizedPnl: closedPosition.realized_pnl,
        closedAt: closedPosition.closed_at
      }
    });

  } catch (error) {
    logger.error(`Error closing position ${id}:`, error);
    res.status(500).json({
      error: 'Failed to close position',
      code: 'CLOSE_ERROR'
    });
  }
}));

export default router;
