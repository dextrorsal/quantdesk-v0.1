import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { config } from '../config/environment';
import { Logger } from '../utils/logger';
import { getSupabaseService } from '../services/supabaseService';
// Conditional Redis import
let getSession: any;
if (process.env.NODE_ENV === 'development' && !process.env.REDIS_URL) {
  getSession = () => Promise.resolve(null);
} else {
  getSession = require('../services/redisClient').getSession;
}
import { AdminUser } from './adminAuth'; // Import AdminUser

const logger = new Logger();

declare global {
  namespace Express {
    interface Request {
      userId?: string;        // Database user ID (for RLS mapping)
      walletPubkey?: string;  // Wallet address
      admin?: AdminUser;      // Added admin property
    }
  }
}

// Export AuthenticatedRequest interface for use in routes
export interface AuthenticatedRequest extends Request {
  userId: string;
  walletPubkey: string;
  user?: any; // Add user property for compatibility
  admin?: AdminUser;
}

export const authMiddleware = async (req: Request, res: Response, next: NextFunction): Promise<void | Response> => {
  console.log('🚀 Auth middleware called for:', req.path);
  const authHeader = req.headers.authorization;

  if (!authHeader && !req.cookies?.qd_session) {
    return res.status(401).json({ error: 'Unauthorized', code: 'MISSING_TOKEN' });
  }

  let token = authHeader?.split(' ')[1] || req.cookies.qd_session;

  if (!token) {
    return res.status(401).json({ error: 'Unauthorized', code: 'MISSING_TOKEN' });
  }

  try {
    console.log('🔍 Auth middleware called with token:', token.substring(0, 20) + '...');
    console.log('🔍 Debug: JWT_SECRET from config:', config.JWT_SECRET);
    console.log('🔍 Debug: JWT_SECRET from env:', process.env.JWT_SECRET);
    
    // Decode JWT with flexible payload structure
    const decoded = jwt.verify(token, config.JWT_SECRET as jwt.Secret) as { 
      wallet_pubkey?: string,
      walletAddress?: string, // Legacy field name
      user_id?: string,
      userId?: string, // Legacy field name
      iat: number, 
      exp: number 
    };

    // Normalize wallet address field (handle both wallet_pubkey and walletAddress)
    const walletPubkey = decoded.wallet_pubkey || decoded.walletAddress;
    if (!walletPubkey) {
      console.log('❌ Missing wallet address in JWT payload');
      return res.status(401).json({ error: 'Invalid token format', code: 'MISSING_WALLET_ADDRESS' });
    }

    // Verify session in Redis (optional, but good for active session management)
    // Skip Redis session check in development if Redis is disabled
    if (process.env.NODE_ENV === 'development' && !process.env.REDIS_URL) {
      console.log('⚠️  Skipping Redis session check in development mode');
    } else {
      const session = await getSession(walletPubkey);
      if (!session) {
        return res.status(401).json({ error: 'Unauthorized', code: 'SESSION_EXPIRED' });
      }
    }
    
    // Get user from database to resolve user_id for RLS mapping
    console.log(`🔍 Resolving user for wallet_pubkey: ${walletPubkey}`);
    const supabase = getSupabaseService();
    const user = await supabase.getUserByWallet(walletPubkey);
    console.log(`🔍 User found:`, user);
    
    if (!user) {
      console.log(`❌ User not found for wallet_pubkey: ${walletPubkey}`);
      return res.status(401).json({ error: 'User not found', code: 'USER_NOT_FOUND' });
    }

    // Use user_id from database (not JWT) for RLS mapping to ensure consistency
    const userId = user.id;
    
    // Security check: If JWT contains user_id, verify it matches database user_id
    const jwtUserId = decoded.user_id || decoded.userId;
    if (jwtUserId && jwtUserId !== userId) {
      console.log(`❌ JWT user_id (${jwtUserId}) does not match database user_id (${userId})`);
      return res.status(401).json({ error: 'Token user mismatch', code: 'USER_ID_MISMATCH' });
    }
    
    // Attach user info to request with proper RLS mapping
    req.userId = userId;  // Use database user_id for RLS mapping
    req.walletPubkey = walletPubkey;
    // Some legacy routes expect req.user.id
    (req as any).user = { 
      id: userId,  // This will be used for RLS policies
      wallet_pubkey: walletPubkey 
    };
    
    console.log(`✅ Auth successful - User ID: ${userId}, Wallet: ${walletPubkey}`);
    next();
  } catch (error) {
    console.log('🔍 JWT verification failed:', error.message);
    console.log('🔍 Token being verified:', token.substring(0, 50) + '...');
    console.log('🔍 JWT_SECRET being used:', config.JWT_SECRET);
    logger.error('Authentication error:', error);
    // Ensure a return here after sending a response
    return res.status(401).json({ error: 'Unauthorized', code: 'INVALID_TOKEN' });
  }
};

// Optional auth middleware (doesn't fail if no token)
export const optionalAuthMiddleware = async (
  req: Request,
  _res: Response,
  next: NextFunction
): Promise<void | Response> => {
  // This middleware is currently a no-op as it's not fully integrated
  // and its logic has been removed to avoid confusion and deprecation issues.
  // Full optional auth or admin auth should be handled by dedicated middleware.
  next();
};

// Admin auth middleware (placeholder - actual logic handled by adminAuth.ts)
export const adminAuthMiddleware = async (
  req: Request,
  res: Response,
  next: NextFunction
): Promise<void | Response> => {
  // For now, this acts as a simple pass-through. The actual admin authentication
  // and authorization logic is handled by the `adminAuth` middleware from `adminAuth.ts`.
  // This middleware might be expanded in the future for specific role-based checks.
  next();
};
