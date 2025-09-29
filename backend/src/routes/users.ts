import express from 'express';
import { DatabaseService } from '../services/database';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandler';
import { AuthenticatedRequest } from '../middleware/auth';

const router = express.Router();
const logger = new Logger();
const db = DatabaseService.getInstance();

// Get user profile
router.get('/profile', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const user = await db.getUserById(req.user!.id);
    
    if (!user) {
      return res.status(404).json({
        error: 'User not found',
        code: 'USER_NOT_FOUND'
      });
    }

    res.json({
      success: true,
      user: {
        id: user.id,
        walletAddress: user.wallet_address,
        username: user.username,
        email: user.email,
        kycStatus: user.kyc_status,
        riskLevel: user.risk_level,
        totalVolume: user.total_volume,
        totalTrades: user.total_trades,
        createdAt: user.created_at,
        lastLogin: user.last_login
      }
    });

  } catch (error) {
    logger.error('Error fetching user profile:', error);
    res.status(500).json({
      error: 'Failed to fetch user profile',
      code: 'FETCH_ERROR'
    });
  }
}));

// Update user profile
router.put('/profile', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const { username, email } = req.body;

  try {
    const updates: any = {};
    
    if (username !== undefined) {
      updates.username = username;
    }
    
    if (email !== undefined) {
      updates.email = email;
    }

    const updatedUser = await db.updateUser(req.user!.id, updates);

    res.json({
      success: true,
      user: {
        id: updatedUser.id,
        walletAddress: updatedUser.wallet_address,
        username: updatedUser.username,
        email: updatedUser.email,
        kycStatus: updatedUser.kyc_status,
        riskLevel: updatedUser.risk_level,
        totalVolume: updatedUser.total_volume,
        totalTrades: updatedUser.total_trades,
        createdAt: updatedUser.created_at,
        lastLogin: updatedUser.last_login
      }
    });

  } catch (error) {
    logger.error('Error updating user profile:', error);
    res.status(500).json({
      error: 'Failed to update user profile',
      code: 'UPDATE_ERROR'
    });
  }
}));

// Get user statistics
router.get('/stats', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const userId = req.user!.id;

    // Get position statistics
    const positions = await db.query(
      `SELECT 
         COUNT(*) as total_positions,
         SUM(CASE WHEN size > 0 THEN 1 ELSE 0 END) as active_positions,
         SUM(CASE WHEN is_liquidated THEN 1 ELSE 0 END) as liquidated_positions,
         SUM(realized_pnl) as total_realized_pnl,
         SUM(unrealized_pnl) as total_unrealized_pnl,
         SUM(funding_fees) as total_funding_fees
       FROM positions 
       WHERE user_id = $1`,
      [userId]
    );

    // Get trade statistics
    const trades = await db.query(
      `SELECT 
         COUNT(*) as total_trades,
         SUM(size * price) as total_volume,
         SUM(fees) as total_fees,
         SUM(pnl) as total_pnl
       FROM trades 
       WHERE user_id = $1`,
      [userId]
    );

    // Get order statistics
    const orders = await db.query(
      `SELECT 
         COUNT(*) as total_orders,
         COUNT(CASE WHEN status = 'filled' THEN 1 END) as filled_orders,
         COUNT(CASE WHEN status = 'cancelled' THEN 1 END) as cancelled_orders,
         COUNT(CASE WHEN status = 'pending' THEN 1 END) as pending_orders
       FROM orders 
       WHERE user_id = $1`,
      [userId]
    );

    const positionStats = positions.rows[0];
    const tradeStats = trades.rows[0];
    const orderStats = orders.rows[0];

    res.json({
      success: true,
      stats: {
        positions: {
          total: parseInt(positionStats.total_positions),
          active: parseInt(positionStats.active_positions),
          liquidated: parseInt(positionStats.liquidated_positions),
          totalRealizedPnl: parseFloat(positionStats.total_realized_pnl) || 0,
          totalUnrealizedPnl: parseFloat(positionStats.total_unrealized_pnl) || 0,
          totalFundingFees: parseFloat(positionStats.total_funding_fees) || 0
        },
        trades: {
          total: parseInt(tradeStats.total_trades),
          totalVolume: parseFloat(tradeStats.total_volume) || 0,
          totalFees: parseFloat(tradeStats.total_fees) || 0,
          totalPnl: parseFloat(tradeStats.total_pnl) || 0
        },
        orders: {
          total: parseInt(orderStats.total_orders),
          filled: parseInt(orderStats.filled_orders),
          cancelled: parseInt(orderStats.cancelled_orders),
          pending: parseInt(orderStats.pending_orders)
        }
      }
    });

  } catch (error) {
    logger.error('Error fetching user statistics:', error);
    res.status(500).json({
      error: 'Failed to fetch user statistics',
      code: 'FETCH_ERROR'
    });
  }
}));

export default router;
