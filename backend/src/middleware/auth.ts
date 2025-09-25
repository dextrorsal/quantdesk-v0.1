import { Request, Response, NextFunction } from 'express';
import jwt from 'jsonwebtoken';
import { DatabaseService } from '../services/database';
import { Logger } from '../utils/logger';

const logger = new Logger();

export interface AuthenticatedRequest extends Request {
  user?: {
    id: string;
    walletAddress: string;
    username?: string;
    email?: string;
  };
}

export const authMiddleware = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      res.status(401).json({
        error: 'Authorization header missing or invalid',
        code: 'MISSING_TOKEN'
      });
      return;
    }

    const token = authHeader.substring(7); // Remove 'Bearer ' prefix
    
    if (!token) {
      res.status(401).json({
        error: 'Token is required',
        code: 'MISSING_TOKEN'
      });
      return;
    }

    // Verify JWT token
    const decoded = jwt.verify(token, process.env['JWT_SECRET']!) as any;
    
    if (!decoded.walletAddress) {
      res.status(401).json({
        error: 'Invalid token payload',
        code: 'INVALID_TOKEN'
      });
      return;
    }

    // Get user from database
    const db = DatabaseService.getInstance();
    const user = await db.getUserByWalletAddress(decoded.walletAddress);
    
    if (!user) {
      res.status(401).json({
        error: 'User not found',
        code: 'USER_NOT_FOUND'
      });
      return;
    }

    if (!user.is_active) {
      res.status(401).json({
        error: 'User account is inactive',
        code: 'ACCOUNT_INACTIVE'
      });
      return;
    }

    // Attach user to request (omit undefined optional fields under exactOptionalPropertyTypes)
    const baseUser = {
      id: user.id,
      walletAddress: user.wallet_address,
    } as { id: string; walletAddress: string; username?: string; email?: string };

    if (user.username != null) {
      baseUser.username = user.username;
    }
    if (user.email != null) {
      baseUser.email = user.email;
    }

    req.user = baseUser;

    next();
  } catch (error) {
    logger.error('Authentication error:', error);
    
    if (error instanceof jwt.JsonWebTokenError) {
      res.status(401).json({
        error: 'Invalid token',
        code: 'INVALID_TOKEN'
      });
      return;
    }
    
    if (error instanceof jwt.TokenExpiredError) {
      res.status(401).json({
        error: 'Token expired',
        code: 'TOKEN_EXPIRED'
      });
      return;
    }

    res.status(500).json({
      error: 'Internal server error',
      code: 'INTERNAL_ERROR'
    });
  }
};

// Optional auth middleware (doesn't fail if no token)
export const optionalAuthMiddleware = async (
  req: AuthenticatedRequest,
  _res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    const authHeader = req.headers.authorization;
    
    if (!authHeader || !authHeader.startsWith('Bearer ')) {
      next();
      return;
    }

    const token = authHeader.substring(7);
    
    if (!token) {
      next();
      return;
    }

    // Verify JWT token
    const decoded = jwt.verify(token, process.env['JWT_SECRET']!) as any;
    
    if (!decoded.walletAddress) {
      next();
      return;
    }

    // Get user from database
    const db = DatabaseService.getInstance();
    const user = await db.getUserByWalletAddress(decoded.walletAddress);
    
    if (user && user.is_active) {
      const baseUser = {
        id: user.id,
        walletAddress: user.wallet_address,
      } as { id: string; walletAddress: string; username?: string; email?: string };

      if (user.username != null) {
        baseUser.username = user.username;
      }
      if (user.email != null) {
        baseUser.email = user.email;
      }

      req.user = baseUser;
    }

    next();
  } catch (error) {
    // For optional auth, we just continue without user
    next();
  }
};

// Admin auth middleware
export const adminAuthMiddleware = async (
  req: AuthenticatedRequest,
  res: Response,
  next: NextFunction
): Promise<void> => {
  try {
    // First check if user is authenticated
    await authMiddleware(req, res, () => {});
    
    if (!req.user) {
      return; // authMiddleware already sent response
    }

    // Check if user is admin
    const db = DatabaseService.getInstance();
    const user = await db.getUserById(req.user.id);
    
    if (!user || user.risk_level !== 'admin') {
      res.status(403).json({
        error: 'Admin access required',
        code: 'ADMIN_REQUIRED'
      });
      return;
    }

    next();
  } catch (error) {
    logger.error('Admin authentication error:', error);
    res.status(500).json({
      error: 'Internal server error',
      code: 'INTERNAL_ERROR'
    });
  }
};
