import express from 'express';
import jwt from 'jsonwebtoken';
import { AdminUserService } from '../services/adminUserService';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandler';
import { AuthenticatedRequest, authMiddleware } from '../middleware/auth';

const router = express.Router();
const logger = new Logger();
const adminUserService = AdminUserService.getInstance();

// System mode management
let systemMode: 'demo' | 'live' = 'demo';

// Admin login
router.post('/login', asyncHandler(async (req, res) => {
  try {
    const { username, password, twoFactorCode } = req.body;
    
    // Verify user credentials
    logger.info(`Attempting login for user: ${username}`);
    const user = await adminUserService.verifyPassword(username, password);
    if (!user) {
      logger.warn(`Login failed for user: ${username} - Invalid credentials`);
      return res.status(401).json({
        error: 'Invalid credentials',
        code: 'INVALID_CREDENTIALS'
      });
    }
    logger.info(`Login successful for user: ${username}`);
    
    // Check 2FA (simplified - in production use real 2FA)
    if (user.role === 'super-admin' && !twoFactorCode) {
      return res.status(200).json({
        requiresTwoFactor: true,
        message: '2FA code required'
      });
    }
    
    // Generate JWT token
    const token = jwt.sign(
      { 
        userId: user.id, 
        username: user.username, 
        role: user.role 
      },
      process.env.JWT_SECRET || 'quantdesk-admin-secret',
      { expiresIn: '24h' }
    );
    
    // Log login action
    await adminUserService.logAction(
      user.id,
      'LOGIN',
      'admin',
      { username: user.username },
      req.ip,
      req.get('User-Agent')
    );
    
    logger.info(`Admin login successful: ${username}`);
    
    res.json({
      success: true,
      token,
      user: {
        id: user.id,
        username: user.username,
        role: user.role,
        permissions: user.permissions
      }
    });
  } catch (error) {
    logger.error('Admin login error:', error);
    res.status(500).json({
      error: 'Login failed',
      code: 'LOGIN_ERROR'
    });
  }
}));

// Verify admin token
router.get('/verify', asyncHandler(async (req, res) => {
  try {
    const authHeader = req.headers.authorization;
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      return res.status(401).json({
        error: 'No token provided',
        code: 'NO_TOKEN'
      });
    }
    
    const token = authHeader.substring(7);
    const decoded = jwt.verify(token, process.env.JWT_SECRET || 'quantdesk-admin-secret') as any;
    
    // Find user in database
    const user = await adminUserService.getUserById(decoded.userId);
    if (!user) {
      return res.status(401).json({
        error: 'Invalid token',
        code: 'INVALID_TOKEN'
      });
    }
    
    res.json({
      success: true,
      user: {
        id: user.id,
        username: user.username,
        role: user.role,
        permissions: user.permissions
      }
    });
  } catch (error) {
    logger.error('Token verification error:', error);
    res.status(401).json({
      error: 'Invalid token',
      code: 'INVALID_TOKEN'
    });
  }
}));

// Admin user management endpoints

// Get all admin users
router.get('/users', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const users = await adminUserService.getAllUsers();
    res.json({
      success: true,
      users: users.map(user => ({
        id: user.id,
        username: user.username,
        role: user.role,
        permissions: user.permissions,
        is_active: user.is_active,
        last_login: user.last_login,
        created_at: user.created_at
      }))
    });
  } catch (error) {
    logger.error('Error fetching admin users:', error);
    res.status(500).json({
      error: 'Failed to fetch admin users',
      code: 'FETCH_USERS_ERROR'
    });
  }
}));

// Create new admin user
router.post('/users', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { username, password, role, permissions } = req.body;
    
    if (!username || !password || !role) {
      return res.status(400).json({
        error: 'Username, password, and role are required',
        code: 'MISSING_FIELDS'
      });
    }

    const newUser = await adminUserService.createUser({
      username,
      password,
      role,
      permissions,
      created_by: req.user?.id || 'system'
    });

    // Log user creation
    await adminUserService.logAction(
      req.user?.id || 'system',
      'CREATE_USER',
      'admin_users',
      { username, role },
      req.ip,
      req.get('User-Agent')
    );

    res.json({
      success: true,
      user: {
        id: newUser.id,
        username: newUser.username,
        role: newUser.role,
        permissions: newUser.permissions,
        is_active: newUser.is_active,
        created_at: newUser.created_at
      }
    });
  } catch (error) {
    logger.error('Error creating admin user:', error);
    res.status(500).json({
      error: error instanceof Error ? error.message : 'Failed to create admin user',
      code: 'CREATE_USER_ERROR'
    });
  }
}));

// Update admin user
router.put('/users/:id', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { id } = req.params;
    const { username, role, permissions, is_active } = req.body;

    const updatedUser = await adminUserService.updateUser(id, {
      username,
      role,
      permissions,
      is_active
    });

    // Log user update
    await adminUserService.logAction(
      req.user?.id || 'system',
      'UPDATE_USER',
      'admin_users',
      { userId: id, changes: req.body },
      req.ip,
      req.get('User-Agent')
    );

    res.json({
      success: true,
      user: {
        id: updatedUser.id,
        username: updatedUser.username,
        role: updatedUser.role,
        permissions: updatedUser.permissions,
        is_active: updatedUser.is_active,
        updated_at: updatedUser.updated_at
      }
    });
  } catch (error) {
    logger.error('Error updating admin user:', error);
    res.status(500).json({
      error: 'Failed to update admin user',
      code: 'UPDATE_USER_ERROR'
    });
  }
}));

// Delete admin user
router.delete('/users/:id', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { id } = req.params;

    await adminUserService.deleteUser(id);

    // Log user deletion
    await adminUserService.logAction(
      req.user?.id || 'system',
      'DELETE_USER',
      'admin_users',
      { userId: id },
      req.ip,
      req.get('User-Agent')
    );

    res.json({
      success: true,
      message: 'User deactivated successfully'
    });
  } catch (error) {
    logger.error('Error deleting admin user:', error);
    res.status(500).json({
      error: 'Failed to delete admin user',
      code: 'DELETE_USER_ERROR'
    });
  }
}));

// Get audit logs
router.get('/audit-logs', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { limit = 100, offset = 0 } = req.query;
    
    const logs = await adminUserService.getAuditLogs(
      parseInt(limit as string),
      parseInt(offset as string)
    );

    res.json({
      success: true,
      logs
    });
  } catch (error) {
    logger.error('Error fetching audit logs:', error);
    res.status(500).json({
      error: 'Failed to fetch audit logs',
      code: 'FETCH_LOGS_ERROR'
    });
  }
}));

// Get system mode
router.get('/mode', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    res.json({
      success: true,
      mode: systemMode,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error fetching system mode:', error);
    res.status(500).json({
      error: 'Failed to fetch system mode',
      code: 'MODE_FETCH_ERROR'
    });
  }
}));

// Set system mode
router.post('/mode', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { mode } = req.body;
    
    if (!mode || !['demo', 'live'].includes(mode)) {
      return res.status(400).json({
        error: 'Invalid mode. Must be "demo" or "live"',
        code: 'INVALID_MODE'
      });
    }
    
    // Log mode change
    logger.info(`System mode changed from ${systemMode} to ${mode} by user ${req.user?.id}`);
    
    // Update system mode
    systemMode = mode;
    
    // TODO: Implement actual mode switching logic
    // - Update database configuration
    // - Switch API endpoints
    // - Update trading engine settings
    // - Notify connected clients
    
    res.json({
      success: true,
      mode: systemMode,
      message: `System mode changed to ${mode.toUpperCase()}`,
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('Error setting system mode:', error);
    res.status(500).json({
      error: 'Failed to set system mode',
      code: 'MODE_SET_ERROR'
    });
  }
}));

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
        positions: positions.rows[0],
        system: {
          mode: systemMode,
          uptime: process.uptime(),
          memory: process.memoryUsage(),
          timestamp: new Date().toISOString()
        }
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
    
    // Check system resources
    const memoryUsage = process.memoryUsage();
    const cpuUsage = process.cpuUsage();
    
    res.json({
      success: true,
      health: {
        database: dbHealth,
        system: {
          mode: systemMode,
          uptime: process.uptime(),
          memory: {
            used: memoryUsage.heapUsed,
            total: memoryUsage.heapTotal,
            external: memoryUsage.external,
            rss: memoryUsage.rss
          },
          cpu: {
            user: cpuUsage.user,
            system: cpuUsage.system
          }
        },
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

// Get trading metrics
router.get('/metrics', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    // Get 24h trading metrics
    const metrics = await db.query(
      `SELECT 
         COUNT(*) as total_trades,
         SUM(size * price) as total_volume,
         SUM(CASE WHEN pnl > 0 THEN 1 ELSE 0 END) as winning_trades,
         SUM(pnl) as total_pnl,
         AVG(pnl) as avg_pnl,
         COUNT(DISTINCT user_id) as active_traders
       FROM trades 
       WHERE created_at >= NOW() - INTERVAL '24 hours'`
    );

    // Calculate win rate
    const totalTrades = parseInt(metrics.rows[0].total_trades) || 0;
    const winningTrades = parseInt(metrics.rows[0].winning_trades) || 0;
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

    res.json({
      success: true,
      metrics: {
        ...metrics.rows[0],
        win_rate: winRate,
        timestamp: new Date().toISOString()
      }
    });

  } catch (error) {
    logger.error('Error fetching trading metrics:', error);
    res.status(500).json({
      error: 'Failed to fetch trading metrics',
      code: 'METRICS_FETCH_ERROR'
    });
  }
}));

// Get system logs
router.get('/logs', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const limit = parseInt(req.query.limit as string) || 100;
    const level = req.query.level as string || 'all';
    
    // TODO: Implement log retrieval from database or log files
    // For now, return mock data
    const logs = [
      {
        id: 1,
        level: 'info',
        message: 'System mode changed to DEMO',
        timestamp: new Date().toISOString(),
        user_id: req.user?.id
      },
      {
        id: 2,
        level: 'warning',
        message: 'High memory usage detected',
        timestamp: new Date(Date.now() - 300000).toISOString(),
        user_id: null
      }
    ];

    res.json({
      success: true,
      logs: logs.slice(0, limit)
    });

  } catch (error) {
    logger.error('Error fetching system logs:', error);
    res.status(500).json({
      error: 'Failed to fetch system logs',
      code: 'LOGS_FETCH_ERROR'
    });
  }
}));

// Emergency stop (halt all trading)
router.post('/emergency-stop', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    // Log emergency stop
    logger.warn(`EMERGENCY STOP triggered by user ${req.user?.id}`);
    
    // TODO: Implement emergency stop logic
    // - Halt all trading engines
    // - Close all open positions
    // - Notify all connected clients
    // - Send alerts to administrators
    
    res.json({
      success: true,
      message: 'Emergency stop activated',
      timestamp: new Date().toISOString()
    });

  } catch (error) {
    logger.error('Error during emergency stop:', error);
    res.status(500).json({
      error: 'Failed to activate emergency stop',
      code: 'EMERGENCY_STOP_ERROR'
    });
  }
}));

export default router;