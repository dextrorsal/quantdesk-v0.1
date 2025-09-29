import express, { Request, Response } from 'express';
import jwt from 'jsonwebtoken';
import { DatabaseService } from '../services/database';
import { Logger } from '../utils/logger';
import { config } from '../config/environment';
import { asyncHandler } from '../middleware/errorHandler';

const router = express.Router();
const logger = new Logger();
const db = DatabaseService.getInstance();

// Authenticate user with wallet signature
router.post('/authenticate', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const { walletAddress, signature, message } = req.body;

  if (!walletAddress || !signature || !message) {
    res.status(400).json({
      error: 'Missing required fields',
      code: 'MISSING_FIELDS'
    });
    return;
  }

  try {
    // Verify wallet signature (simplified - in production, use proper verification)
    // For now, we'll just check if the wallet address is valid
    if (!walletAddress.match(/^[1-9A-HJ-NP-Za-km-z]{32,44}$/)) {
      res.status(400).json({
        error: 'Invalid wallet address',
        code: 'INVALID_WALLET'
      });
      return;
    }

    // Get or create user
    let user = await db.getUserByWalletAddress(walletAddress);
    if (!user) {
      user = await db.createUser(walletAddress);
      logger.info(`New user created: ${walletAddress}`);
    } else {
      // Update last login
      await db.updateUser(user.id, { last_login: new Date() });
    }

    // Generate JWT token
    const token = jwt.sign(
      { 
        walletAddress: user.wallet_address,
        userId: user.id,
        tier: 'standard' // Default tier
      },
      config.JWT_SECRET as jwt.Secret,
      { expiresIn: config.JWT_EXPIRES_IN as any }
    );

    res.json({
      success: true,
      token,
      user: {
        id: user.id,
        walletAddress: user.wallet_address,
        username: user.username,
        email: user.email,
        kycStatus: user.kyc_status,
        riskLevel: user.risk_level,
        totalVolume: user.total_volume,
        totalTrades: user.total_trades
      }
    });
    return;

  } catch (error) {
    logger.error('Authentication error:', error);
    res.status(500).json({
      error: 'Authentication failed',
      code: 'AUTH_ERROR'
    });
    return;
  }
}));

// Refresh token
router.post('/refresh', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const { token } = req.body;

  if (!token) {
    res.status(400).json({
      error: 'Token is required',
      code: 'MISSING_TOKEN'
    });
    return;
  }

  try {
    const decoded = jwt.verify(token, config.JWT_SECRET as jwt.Secret) as any;
    
    // Generate new token
    const newToken = jwt.sign(
      { 
        walletAddress: decoded.walletAddress,
        userId: decoded.userId,
        tier: decoded.tier
      },
      config.JWT_SECRET as jwt.Secret,
      { expiresIn: config.JWT_EXPIRES_IN as any }
    );

    res.json({
      success: true,
      token: newToken
    });
    return;

  } catch (error) {
    logger.error('Token refresh error:', error);
    res.status(401).json({
      error: 'Invalid token',
      code: 'INVALID_TOKEN'
    });
    return;
  }
}));

// Get user profile
router.get('/profile', asyncHandler(async (req: Request, res: Response): Promise<void> => {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({
      error: 'Authorization header missing',
      code: 'MISSING_AUTH_HEADER'
    });
    return;
  }

  const token = authHeader.substring(7);
  
  try {
    const decoded = jwt.verify(token, config.JWT_SECRET as jwt.Secret) as any;
    const user = await db.getUserByWalletAddress(decoded.walletAddress);
    
    if (!user) {
      res.status(404).json({
        error: 'User not found',
        code: 'USER_NOT_FOUND'
      });
      return;
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
    return;

  } catch (error) {
    logger.error('Profile fetch error:', error);
    res.status(401).json({
      error: 'Invalid token',
      code: 'INVALID_TOKEN'
    });
    return;
  }
}));

export default router;
