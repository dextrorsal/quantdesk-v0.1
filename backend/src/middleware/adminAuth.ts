// Admin Authentication Middleware
// Handles admin-specific authentication and authorization

import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { DatabaseService } from '../services/database';
import { Logger } from '../utils/logger';

interface AdminUser {
  id: string;
  email: string;
  role: 'super_admin' | 'system_admin' | 'support_admin' | 'readonly_admin';
  permissions: string[];
  lastLogin: string;
  isActive: boolean;
}

interface AdminRequest extends Request {
  admin?: AdminUser;
}

const logger = new Logger();
const db = DatabaseService.getInstance();

// Admin role permissions
const ADMIN_PERMISSIONS = {
  super_admin: [
    'system_mode_change',
    'emergency_stop',
    'user_management',
    'system_config',
    'financial_data',
    'audit_logs',
    'security_settings'
  ],
  system_admin: [
    'system_mode_change',
    'system_config',
    'user_management',
    'audit_logs'
  ],
  support_admin: [
    'user_management',
    'audit_logs',
    'system_metrics'
  ],
  readonly_admin: [
    'system_metrics',
    'audit_logs'
  ]
};

// Admin authentication middleware
export const adminAuth = async (req: AdminRequest, res: Response, next: NextFunction) => {
  try {
    const token = req.header('Authorization')?.replace('Bearer ', '');
    
    if (!token) {
      return res.status(401).json({
        error: 'Access denied. No admin token provided.',
        code: 'NO_ADMIN_TOKEN'
      });
    }

    // Verify JWT token
    const decoded = jwt.verify(token, process.env.ADMIN_JWT_SECRET || 'admin_secret') as any;
    
    // Get admin user from database
    const adminUser = await getAdminUser(decoded.adminId);
    
    if (!adminUser) {
      return res.status(401).json({
        error: 'Invalid admin token.',
        code: 'INVALID_ADMIN_TOKEN'
      });
    }

    if (!adminUser.isActive) {
      return res.status(401).json({
        error: 'Admin account is deactivated.',
        code: 'ADMIN_DEACTIVATED'
      });
    }

    // Check IP whitelist (if enabled)
    if (process.env.ADMIN_IP_WHITELIST === 'true') {
      const clientIP = req.ip || req.connection.remoteAddress;
      if (!isIPWhitelisted(clientIP, adminUser.id)) {
        logger.warn(`Admin access denied from IP: ${clientIP}`, { adminId: adminUser.id });
        return res.status(403).json({
          error: 'Access denied. IP not whitelisted.',
          code: 'IP_NOT_WHITELISTED'
        });
      }
    }

    // Add admin user to request
    req.admin = adminUser;
    
    // Log admin access
    await logAdminAccess(adminUser.id, req.method, req.path, req.ip);
    
    next();
  } catch (error) {
    logger.error('Admin authentication error:', error);
    res.status(401).json({
      error: 'Invalid admin token.',
      code: 'ADMIN_AUTH_ERROR'
    });
  }
};

// Check if admin has specific permission
export const requirePermission = (permission: string) => {
  return (req: AdminRequest, res: Response, next: NextFunction) => {
    if (!req.admin) {
      return res.status(401).json({
        error: 'Admin authentication required.',
        code: 'ADMIN_AUTH_REQUIRED'
      });
    }

    const userPermissions = ADMIN_PERMISSIONS[req.admin.role] || [];
    
    if (!userPermissions.includes(permission)) {
      logger.warn(`Admin ${req.admin.id} attempted to access ${permission} without permission`);
      return res.status(403).json({
        error: 'Insufficient permissions.',
        code: 'INSUFFICIENT_PERMISSIONS',
        required: permission,
        userRole: req.admin.role
      });
    }

    next();
  };
};

// Check if admin has any of the specified permissions
export const requireAnyPermission = (permissions: string[]) => {
  return (req: AdminRequest, res: Response, next: NextFunction) => {
    if (!req.admin) {
      return res.status(401).json({
        error: 'Admin authentication required.',
        code: 'ADMIN_AUTH_REQUIRED'
      });
    }

    const userPermissions = ADMIN_PERMISSIONS[req.admin.role] || [];
    const hasPermission = permissions.some(permission => userPermissions.includes(permission));
    
    if (!hasPermission) {
      logger.warn(`Admin ${req.admin.id} attempted to access ${permissions.join(' or ')} without permission`);
      return res.status(403).json({
        error: 'Insufficient permissions.',
        code: 'INSUFFICIENT_PERMISSIONS',
        required: permissions,
        userRole: req.admin.role
      });
    }

    next();
  };
};

// Get admin user from database
async function getAdminUser(adminId: string): Promise<AdminUser | null> {
  try {
    const result = await db.query(
      `SELECT id, email, role, permissions, last_login, is_active, created_at
       FROM admin_users 
       WHERE id = $1`,
      [adminId]
    );

    if (result.rows.length === 0) {
      return null;
    }

    const row = result.rows[0];
    return {
      id: row.id,
      email: row.email,
      role: row.role,
      permissions: row.permissions || [],
      lastLogin: row.last_login,
      isActive: row.is_active
    };
  } catch (error) {
    logger.error('Error fetching admin user:', error);
    return null;
  }
}

// Check if IP is whitelisted for admin
async function isIPWhitelisted(clientIP: string, adminId: string): Promise<boolean> {
  try {
    const result = await db.query(
      `SELECT COUNT(*) as count 
       FROM admin_ip_whitelist 
       WHERE admin_id = $1 AND ip_address = $2 AND is_active = true`,
      [adminId, clientIP]
    );

    return parseInt(result.rows[0].count) > 0;
  } catch (error) {
    logger.error('Error checking IP whitelist:', error);
    return false;
  }
}

// Log admin access
async function logAdminAccess(adminId: string, method: string, path: string, ip: string): Promise<void> {
  try {
    await db.query(
      `INSERT INTO admin_access_logs (admin_id, method, path, ip_address, timestamp) 
       VALUES ($1, $2, $3, $4, NOW())`,
      [adminId, method, path, ip]
    );
  } catch (error) {
    logger.error('Error logging admin access:', error);
  }
}

// Admin login endpoint
export const adminLogin = async (req: Request, res: Response) => {
  try {
    const { email, password, mfaCode } = req.body;

    if (!email || !password) {
      return res.status(400).json({
        error: 'Email and password are required.',
        code: 'MISSING_CREDENTIALS'
      });
    }

    // Get admin user
    const result = await db.query(
      `SELECT id, email, password_hash, role, mfa_secret, is_active, last_login
       FROM admin_users 
       WHERE email = $1`,
      [email]
    );

    if (result.rows.length === 0) {
      return res.status(401).json({
        error: 'Invalid credentials.',
        code: 'INVALID_CREDENTIALS'
      });
    }

    const admin = result.rows[0];

    if (!admin.is_active) {
      return res.status(401).json({
        error: 'Admin account is deactivated.',
        code: 'ADMIN_DEACTIVATED'
      });
    }

    // Verify password
    const bcrypt = require('bcryptjs');
    const isValidPassword = await bcrypt.compare(password, admin.password_hash);
    
    if (!isValidPassword) {
      logger.warn(`Failed admin login attempt for email: ${email}`, { ip: req.ip });
      return res.status(401).json({
        error: 'Invalid credentials.',
        code: 'INVALID_CREDENTIALS'
      });
    }

    // Verify MFA if enabled
    if (admin.mfa_secret && process.env.ADMIN_MFA_REQUIRED === 'true') {
      if (!mfaCode) {
        return res.status(401).json({
          error: 'MFA code is required.',
          code: 'MFA_REQUIRED'
        });
      }

      // TODO: Implement MFA verification
      // const isValidMFA = verifyMFA(admin.mfa_secret, mfaCode);
      // if (!isValidMFA) {
      //   return res.status(401).json({
      //     error: 'Invalid MFA code.',
      //     code: 'INVALID_MFA'
      //   });
      // }
    }

    // Generate JWT token
    const token = jwt.sign(
      { adminId: admin.id, role: admin.role },
      process.env.ADMIN_JWT_SECRET || 'admin_secret',
      { expiresIn: '8h' }
    );

    // Update last login
    await db.query(
      `UPDATE admin_users SET last_login = NOW() WHERE id = $1`,
      [admin.id]
    );

    // Log successful login
    await logAdminAccess(admin.id, 'POST', '/admin/login', req.ip || '');

    res.json({
      success: true,
      token,
      admin: {
        id: admin.id,
        email: admin.email,
        role: admin.role,
        permissions: ADMIN_PERMISSIONS[admin.role] || []
      }
    });

  } catch (error) {
    logger.error('Admin login error:', error);
    res.status(500).json({
      error: 'Internal server error.',
      code: 'LOGIN_ERROR'
    });
  }
};

export { AdminRequest, AdminUser };
