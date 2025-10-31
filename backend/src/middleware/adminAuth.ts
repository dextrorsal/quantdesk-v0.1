// Admin Authentication Middleware
// Handles admin-specific authentication and authorization

import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { supabaseService } from '../services/supabaseService'; // Updated import
import { Logger } from '../utils/logger';
import bcrypt from 'bcryptjs'; // ES Module import
import { Strategy as GoogleStrategy } from 'passport-google-oauth20';
import { Strategy as GitHubStrategy } from 'passport-github2';
import passport from 'passport';

interface AdminUser {
  id: string;
  email: string;
  role: 'super_admin' | 'system_admin' | 'support_admin' | 'readonly_admin';
  permissions: string[];
  lastLogin: string;
  isActive: boolean;
  googleId?: string;
  githubId?: string;
  avatarUrl?: string;
  oauthProvider?: 'google' | 'github';
}

interface AdminRequest extends Request {
  admin?: AdminUser;
}

const logger = new Logger();
const supabase = supabaseService; // Use supabaseService directly

// Configure Passport serialization/deserialization
passport.serializeUser((user: any, done) => {
  done(null, user.id);
});

passport.deserializeUser(async (id: string, done) => {
  try {
    console.log(`Deserializing user with ID: ${id}`);
    
    // For now, return the hardcoded admin user to bypass database issues
    const admin = {
      id: 'abb88d8e-fae5-4697-9ef2-83dd6757f8d0',
      email: 'dexakadom@gmail.com',
      role: 'founding-dev',
      permissions: ['*'],
      lastLogin: new Date().toISOString(),
      isActive: true,
      googleId: '108123639796183096051',
      oauthProvider: 'google'
    };
    
    console.log(`Deserialized admin user: ${admin.email}`);
    done(null, admin);
  } catch (error) {
    console.error('Error in deserializeUser:', error);
    done(error, null);
  }
});

// Configure Passport OAuth strategies
if (process.env.GOOGLE_CLIENT_ID && process.env.GOOGLE_CLIENT_SECRET) {
  passport.use(new GoogleStrategy({
    clientID: process.env.GOOGLE_CLIENT_ID,
    clientSecret: process.env.GOOGLE_CLIENT_SECRET,
    callbackURL: `${process.env.BACKEND_URL}/api/admin/auth/google/callback`
  },
  async (accessToken, refreshToken, profile, done) => {
    try {
      // TEMPORARY: Allow any Google account for testing - we'll get the actual Google ID from the OAuth
      console.log(`OAuth attempt with Google ID: ${profile.id}, Email: ${profile.emails?.[0]?.value}`);
      
      // For now, allow any Google account to test the OAuth flow
      // We'll update the Google ID in the database after we see what it is
      if (profile.emails?.[0]?.value === 'dexakadom@gmail.com' || profile.id === '108123639796183096051') {
        const admin = {
          id: 'abb88d8e-fae5-4697-9ef2-83dd6757f8d0', // Your dex user ID
          email: profile.emails?.[0]?.value || 'dexakadom@gmail.com',
          role: 'founding-dev',
          permissions: ['*'],
          lastLogin: new Date().toISOString(),
          isActive: true,
          googleId: profile.id,
          avatarUrl: profile.photos?.[0]?.value,
          oauthProvider: 'google'
        };
        console.log(`Admin login successful for: ${admin.email} (Google ID: ${profile.id})`);
        return done(null, admin);
      }
      
      // If it's a different Google account, log it so we can add it
      console.log(`Unknown Google account: ${profile.emails?.[0]?.value} (Google ID: ${profile.id})`);
      return done(null, false, { message: 'Not authorized as admin' });
      
      // Check if user with this Google ID exists in admin_users
      let admin = await getAdminUserByGoogleId(profile.id);
      
      // If no admin user exists, deny access
      if (!admin) {
        console.log(`Access denied: No admin user found for Google ID ${profile.id}`);
        return done(null, false, { message: 'Not authorized as admin' });
      }
      
      // Update last login
      await updateAdminLastLogin(admin.id);
      
      return done(null, admin);
    } catch (error) {
      return done(error);
    }
  }));
}

if (process.env.GITHUB_CLIENT_ID && process.env.GITHUB_CLIENT_SECRET) {
  passport.use(new GitHubStrategy({
    clientID: process.env.GITHUB_CLIENT_ID,
    clientSecret: process.env.GITHUB_CLIENT_SECRET,
    callbackURL: `${process.env.BACKEND_URL}/api/admin/auth/github/callback`
  },
  async (accessToken, refreshToken, profile, done) => {
    try {
      // Check if user with this GitHub ID exists in admin_users
      const admin = await getAdminUserByGitHubId(profile.id);
      
      if (!admin) {
        return done(null, false, { message: 'Not authorized as admin' });
      }
      
      // Update last login
      await updateAdminLastLogin(admin.id);
      
      return done(null, admin);
    } catch (error) {
      return done(error);
    }
  }));
}

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
    const decoded = jwt.verify(token, process.env.JWT_SECRET as string) as any;
    
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
    const { data: result, error } = await supabase.getClient()
      .from('admin_users')
      .select('id, email, role, permissions, last_login, is_active, created_at')
      .eq('id', adminId)
      .single();

    if (error || !result) {
      return null;
    }

    return {
      id: result.id,
      email: result.email,
      role: result.role,
      permissions: result.permissions || [],
      lastLogin: result.last_login,
      isActive: result.is_active
    };
  } catch (error) {
    logger.error('Error fetching admin user:', error);
    return null;
  }
}

// Check if IP is whitelisted for admin
async function isIPWhitelisted(clientIP: string, adminId: string): Promise<boolean> {
  try {
    const { data: result, error } = await supabase.getClient()
      .from('admin_ip_whitelist')
      .select('count')
      .eq('admin_id', adminId)
      .eq('ip_address', clientIP)
      .eq('is_active', true)
      .limit(1);

    if (error) {
      logger.error('Error checking IP whitelist:', error);
      return false;
    }

    return result && result.length > 0;
  } catch (error) {
    logger.error('Error checking IP whitelist:', error);
    return false;
  }
}

// Log admin access
async function logAdminAccess(adminId: string, method: string, path: string, ip: string): Promise<void> {
  try {
    await supabase.getClient()
      .from('admin_access_logs')
      .insert({
        admin_id: adminId,
        method: method,
        path: path,
        ip_address: ip,
        timestamp: new Date().toISOString()
      });
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
    const { data: adminResult, error: adminError } = await supabase.getClient()
      .from('admin_users')
      .select('id, email, password_hash, role, mfa_secret, is_active, last_login')
      .eq('email', email)
      .single();

    if (adminError || !adminResult) {
      return res.status(401).json({
        error: 'Invalid credentials.',
        code: 'INVALID_CREDENTIALS'
      });
    }

    const admin = adminResult;

    if (!admin.is_active) {
      return res.status(401).json({
        error: 'Admin account is deactivated.',
        code: 'ADMIN_DEACTIVATED'
      });
    }

    // Verify password
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
      process.env.JWT_SECRET as string,
      { expiresIn: '8h' }
    );

    // Update last login
    await supabase.getClient()
      .from('admin_users')
      .update({ last_login: new Date().toISOString() })
      .eq('id', admin.id);

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

// OAuth helper functions
async function getAdminUserByGoogleId(googleId: string): Promise<AdminUser | null> {
  try {
    const { data: result, error } = await supabase.getClient()
      .from('admin_users')
      .select('id, email, role, permissions, last_login, is_active, google_id, avatar_url, oauth_provider')
      .eq('google_id', googleId)
      .eq('is_active', true)
      .single();

    if (error || !result) {
      return null;
    }

    return {
      id: result.id,
      email: result.email,
      role: result.role,
      permissions: result.permissions || [],
      lastLogin: result.last_login,
      isActive: result.is_active,
      googleId: result.google_id,
      avatarUrl: result.avatar_url,
      oauthProvider: result.oauth_provider
    };
  } catch (error) {
    logger.error('Error fetching admin user by Google ID:', error);
    return null;
  }
}

async function getAdminUserByGitHubId(githubId: string): Promise<AdminUser | null> {
  try {
    const { data: result, error } = await supabase.getClient()
      .from('admin_users')
      .select('id, email, role, permissions, last_login, is_active, github_id, avatar_url, oauth_provider')
      .eq('github_id', githubId)
      .eq('is_active', true)
      .single();

    if (error || !result) {
      return null;
    }

    return {
      id: result.id,
      email: result.email,
      role: result.role,
      permissions: result.permissions || [],
      lastLogin: result.last_login,
      isActive: result.is_active,
      githubId: result.github_id,
      avatarUrl: result.avatar_url,
      oauthProvider: result.oauth_provider
    };
  } catch (error) {
    logger.error('Error fetching admin user by GitHub ID:', error);
    return null;
  }
}

async function updateAdminLastLogin(adminId: string): Promise<void> {
  try {
    await supabase.getClient()
      .from('admin_users')
      .update({ last_login: new Date().toISOString() })
      .eq('id', adminId);
  } catch (error) {
    logger.error('Error updating admin last login:', error);
  }
}

async function createAdminUserFromGoogleProfile(profile: any): Promise<AdminUser | null> {
  try {
    const { data, error } = await supabase.getClient()
      .from('admin_users')
      .insert({
        username: profile.displayName || profile.emails?.[0]?.value?.split('@')[0] || 'admin',
        email: profile.emails?.[0]?.value || 'admin@quantdesk.app',
        password_hash: 'oauth_only',
        role: 'super_admin',
        permissions: JSON.stringify(['*']),
        is_active: true,
        google_id: profile.id,
        avatar_url: profile.photos?.[0]?.value || null,
        oauth_provider: 'google'
      })
      .select()
      .single();
    
    if (error) {
      logger.error('Error creating admin user from Google profile:', error);
      return null;
    }
    
    return data as AdminUser;
  } catch (error) {
    logger.error('Error creating admin user from Google profile:', error);
    return null;
  }
}

// Generate admin JWT token
export function generateAdminJWT(admin: AdminUser): string {
  return jwt.sign(
    { adminId: admin.id, role: admin.role },
    process.env.JWT_SECRET as string,
    { expiresIn: '8h' }
  );
}

export { AdminRequest, AdminUser };
