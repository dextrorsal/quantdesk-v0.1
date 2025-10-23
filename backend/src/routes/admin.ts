import express from 'express';
import jwt from 'jsonwebtoken';
import passport from 'passport';
import { AdminUserService } from '../services/adminUserService';
import { SupabaseDatabaseService } from '../services/supabaseDatabase';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';
import { AuthenticatedRequest, authMiddleware } from '../middleware/auth';
import { generateAdminJWT } from '../middleware/adminAuth';

const router = express.Router();
const logger = new Logger();
const adminUserService = AdminUserService.getInstance();
const db = SupabaseDatabaseService.getInstance();

// System mode management
let systemMode: 'demo' | 'live' = 'demo';

// OAuth routes
// Root route for admin API
router.get('/', (req, res) => {
  res.json({
    success: true,
    message: 'QuantDesk Admin API',
    version: '1.0.0',
    endpoints: {
      login: '/api/admin/login',
      google_auth: '/api/admin/auth/google',
      github_auth: '/api/admin/auth/github',
      verify: '/api/admin/verify'
    }
  });
});

// GitHub OAuth routes
router.get('/auth/github', 
  passport.authenticate('github', { scope: ['user:email'] })
);

router.get('/auth/github/callback',
  passport.authenticate('github', { failureRedirect: `${process.env.FRONTEND_URL}/admin/login?error=auth_failed` }),
  (req, res) => {
    // Generate JWT token for admin session
    const admin = req.user as any;
    
    if (!admin) {
      return res.redirect(`${process.env.FRONTEND_URL}/admin/login?error=not_authorized`);
    }
    
    const token = generateAdminJWT(admin);
    // In development, redirect to standalone admin dashboard
    // In production, redirect to admin dashboard on same domain
    const adminUrl = process.env.NODE_ENV === 'production' 
      ? `${process.env.FRONTEND_URL}/admin-dashboard/`
      : 'http://localhost:5173';
    res.redirect(`${adminUrl}?token=${token}`);
  }
);

// Google OAuth routes
router.get('/auth/google',
  passport.authenticate('google', { scope: ['profile', 'email'] })
);

router.get('/auth/google/callback',
  passport.authenticate('google', { failureRedirect: `${process.env.FRONTEND_URL}/admin/login?error=auth_failed` }),
  async (req, res) => {
    try {
      console.log('Google OAuth callback reached');
      console.log('req.user:', req.user);
      
      // Check if Google user is admin
      const admin = req.user as any;
      
      if (!admin) {
        console.log('No admin user found in req.user');
        return res.redirect(`${process.env.FRONTEND_URL}/admin/login?error=not_authorized`);
      }
      
      console.log('Admin user found:', admin.email);
      const token = generateAdminJWT(admin);
        console.log('Generated JWT token:', token.substring(0, 50) + '...');
        console.log('Redirecting to admin dashboard with token');
        // In development, redirect to standalone admin dashboard
        // In production, redirect to admin dashboard on same domain
        const adminUrl = process.env.NODE_ENV === 'production' 
          ? `${process.env.FRONTEND_URL}/admin-dashboard/`
          : 'http://localhost:5173';
        res.redirect(`${adminUrl}?token=${token}`);
    } catch (error) {
      console.error('Error in Google OAuth callback:', error);
      res.status(500).json({
        success: false,
        error: 'Internal server error',
        code: 'INTERNAL_ERROR',
        message: error.message
      });
    }
  }
);

// Create authorized admin user endpoint (for production use)
router.post('/create-admin', asyncHandler(async (req, res) => {
  try {
    const { email, google_id, github_id, role = 'super_admin' } = req.body;
    
    if (!email) {
      return res.status(400).json({
        error: 'Email is required',
        code: 'MISSING_EMAIL'
      });
    }
    
    if (!google_id && !github_id) {
      return res.status(400).json({
        error: 'Either Google ID or GitHub ID is required',
        code: 'MISSING_OAUTH_ID'
      });
    }
    
    // Check if admin user already exists using fluent API
    const existingUsers = await db.select('admin_users', '*', {
      email: email
    });
    
    // Also check by OAuth IDs
    const googleUsers = google_id ? await db.select('admin_users', '*', { google_id }) : [];
    const githubUsers = github_id ? await db.select('admin_users', '*', { github_id }) : [];
    
    const allExistingUsers = [...existingUsers, ...googleUsers, ...githubUsers];
    
    if (allExistingUsers.length > 0) {
      return res.status(409).json({
        error: 'Admin user already exists',
        code: 'USER_EXISTS',
        user: allExistingUsers[0]
      });
    }
    
    
    // Create admin user using fluent API
    const newUser = await db.insert('admin_users', {
      username: email.split('@')[0], // username from email
      email: email,
      password_hash: 'oauth_only', // password hash
      role: role,
      permissions: JSON.stringify(['*']), // all permissions
      is_active: true,
      google_id: google_id || null,
      github_id: github_id || null,
      oauth_provider: google_id ? 'google' : 'github'
    });

    res.json({
      success: true,
      message: 'Admin user created successfully',
      user: newUser[0]
    });
    
  } catch (error) {
    logger.error('Error in create-admin:', error);
    res.status(500).json({
      error: 'Internal server error',
      code: 'INTERNAL_ERROR'
    });
  }
}));

// List all admin users
router.get('/list-admins', asyncHandler(async (req, res) => {
  try {
    const admins = await db.select('admin_users', 'id, username, email, role, is_active, google_id, github_id, oauth_provider, created_at, last_login');
    
    // Sort by created_at descending
    const sortedAdmins = admins.sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime());
    
    res.json({
      success: true,
      admins: sortedAdmins
    });
    
  } catch (error) {
    logger.error('Error in list-admins:', error);
    res.status(500).json({
      error: 'Internal server error',
      code: 'INTERNAL_ERROR'
    });
  }
}));

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
      process.env.JWT_SECRET as string,
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
    const decoded = jwt.verify(token, process.env.JWT_SECRET as string) as any;
    
    // Find user in database
    const user = await adminUserService.getUserById(decoded.adminId);
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
    // Get user statistics using fluent API
    const allUsers = await db.select('users', '*');
    const totalUsers = allUsers.length;
    const activeUsers = allUsers.filter(user => user.is_active).length;
    const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const newUsers24h = allUsers.filter(user => new Date(user.created_at) >= yesterday).length;

    // Get market statistics using fluent API
    const allMarkets = await db.select('markets', '*');
    const totalMarkets = allMarkets.length;
    const activeMarkets = allMarkets.filter(market => market.is_active).length;

    // Get trading statistics using fluent API
    const allTrades = await db.select('trades', '*');
    const yesterdayTrades = allTrades.filter(trade => new Date(trade.created_at) >= yesterday);
    const totalTrades = yesterdayTrades.length;
    const totalVolume = yesterdayTrades.reduce((sum, trade) => sum + (trade.size * trade.price), 0);
    const totalFees = yesterdayTrades.reduce((sum, trade) => sum + (trade.fees || 0), 0);
    const activeTraders = new Set(yesterdayTrades.map(trade => trade.user_id)).size;

    // Get position statistics using fluent API
    const allPositions = await db.select('positions', '*');
    const totalPositions = allPositions.length;
    const activePositions = allPositions.filter(pos => pos.size > 0).length;
    const liquidatedPositions = allPositions.filter(pos => pos.is_liquidated).length;
    const totalOpenInterest = allPositions.reduce((sum, pos) => sum + (pos.size * pos.entry_price), 0);

    res.json({
      success: true,
      stats: {
        users: {
          total_users: totalUsers,
          active_users: activeUsers,
          new_users_24h: newUsers24h
        },
        markets: {
          total_markets: totalMarkets,
          active_markets: activeMarkets
        },
        trading: {
          total_trades: totalTrades,
          total_volume: totalVolume,
          total_fees: totalFees,
          active_traders: activeTraders
        },
        positions: {
          total_positions: totalPositions,
          active_positions: activePositions,
          liquidated_positions: liquidatedPositions,
          total_open_interest: totalOpenInterest
        },
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
    const allUsers = await db.select('users', 'id, wallet_pubkey, username, email, kyc_status, risk_level, total_volume, total_trades, created_at, last_login, is_active');
    
    // Sort by created_at descending and apply pagination
    const sortedUsers = allUsers
      .sort((a, b) => new Date(b.created_at).getTime() - new Date(a.created_at).getTime())
      .slice(offset, offset + limit);

    res.json({
      success: true,
      users: sortedUsers
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
    // Get 24h trading metrics using fluent API
    const allTrades = await db.select('trades', '*');
    const yesterday = new Date(Date.now() - 24 * 60 * 60 * 1000);
    const yesterdayTrades = allTrades.filter(trade => new Date(trade.created_at) >= yesterday);
    
    const totalTrades = yesterdayTrades.length;
    const totalVolume = yesterdayTrades.reduce((sum, trade) => sum + (trade.size * trade.price), 0);
    const winningTrades = yesterdayTrades.filter(trade => trade.pnl > 0).length;
    const totalPnl = yesterdayTrades.reduce((sum, trade) => sum + (trade.pnl || 0), 0);
    const avgPnl = totalTrades > 0 ? totalPnl / totalTrades : 0;
    const activeTraders = new Set(yesterdayTrades.map(trade => trade.user_id)).size;
    
    // Calculate win rate
    const winRate = totalTrades > 0 ? (winningTrades / totalTrades) * 100 : 0;

    res.json({
      success: true,
      metrics: {
        total_trades: totalTrades,
        total_volume: totalVolume,
        winning_trades: winningTrades,
        total_pnl: totalPnl,
        avg_pnl: avgPnl,
        active_traders: activeTraders,
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