import express from 'express';
import { DatabaseService } from '../services/database';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandler';
import { AuthenticatedRequest } from '../middleware/auth';

const router = express.Router();
const logger = new Logger();
const db = DatabaseService.getInstance();

// Get system statistics
router.get('/stats', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    // Get user statistics
    const users = await db.query(
      `SELECT 
         COUNT(*) as total_users,
         COUNT(CASE WHEN is_active THEN 1 END) as active_users,
         COUNT(CASE WHEN created_at >= NOW() - INTERVAL '24 hours' THEN 1 END) as new_users_24h
       FROM users`
    );

    // Get market statistics
    const markets = await db.query(
      `SELECT 
         COUNT(*) as total_markets,
         COUNT(CASE WHEN is_active THEN 1 END) as active_markets
       FROM markets`
    );

    // Get trading statistics
    const trading = await db.query(
      `SELECT 
         COUNT(*) as total_trades,
         SUM(size * price) as total_volume,
         SUM(fees) as total_fees,
         COUNT(DISTINCT user_id) as active_traders
       FROM trades 
       WHERE created_at >= NOW() - INTERVAL '24 hours'`
    );

    // Get position statistics
    const positions = await db.query(
      `SELECT 
         COUNT(*) as total_positions,
         COUNT(CASE WHEN size > 0 THEN 1 END) as active_positions,
         COUNT(CASE WHEN is_liquidated THEN 1 END) as liquidated_positions,
         SUM(size * entry_price) as total_open_interest
       FROM positions`
    );

    res.json({
      success: true,
      stats: {
        users: users.rows[0],
        markets: markets.rows[0],
        trading: trading.rows[0],
        positions: positions.rows[0]
      }
    });

  } catch (error) {
    logger.error('Error fetching admin statistics:', error);
    res.status(500).json({
      error: 'Failed to fetch admin statistics',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get all users (admin only)
router.get('/users', asyncHandler(async (req: AuthenticatedRequest, res) => {
  const limit = parseInt(req.query.limit as string) || 100;
  const offset = parseInt(req.query.offset as string) || 0;

  try {
    const users = await db.query(
      `SELECT id, wallet_address, username, email, kyc_status, risk_level, 
              total_volume, total_trades, created_at, last_login, is_active
       FROM users 
       ORDER BY created_at DESC 
       LIMIT $1 OFFSET $2`,
      [limit, offset]
    );

    res.json({
      success: true,
      users: users.rows
    });

  } catch (error) {
    logger.error('Error fetching users:', error);
    res.status(500).json({
      error: 'Failed to fetch users',
      code: 'FETCH_ERROR'
    });
  }
}));

// Get system health
router.get('/health', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const dbHealth = await db.healthCheck();
    
    res.json({
      success: true,
      health: {
        database: dbHealth,
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    logger.error('Error checking system health:', error);
    res.status(500).json({
      error: 'Failed to check system health',
      code: 'HEALTH_CHECK_ERROR'
    });
  }
}));

export default router;
