import jwt from 'jsonwebtoken';
import { config } from '../config/environment';
import { Logger } from '../utils/logger';
import { SupabaseDatabaseService } from './supabaseDatabase';

const logger = new Logger();

export interface JWTTokenPayload {
  wallet_pubkey?: string;
  walletAddress?: string; // Legacy field name
  user_id?: string;
  userId?: string; // Legacy field name
  email?: string;
  iat: number;
  exp: number;
}

export interface JWTValidationResult {
  user_id?: string;
  wallet_address?: string;
  email?: string;
  valid: boolean;
  error?: string;
}

export class JWTService {
  private static instance: JWTService;
  private readonly db: SupabaseDatabaseService;
  private readonly secret: string;

  private constructor(database: SupabaseDatabaseService, secret: string) {
    this.db = database;
    this.secret = secret;
  }

  public static getInstance(database?: SupabaseDatabaseService, secret?: string): JWTService {
    if (!JWTService.instance) {
      const db = database || SupabaseDatabaseService.getInstance();
      const jwtSecret = secret || String(config.JWT_SECRET);
      JWTService.instance = new JWTService(db, jwtSecret);
    }
    return JWTService.instance;
  }

  /**
   * Create a JWT token for a user
   */
  public createToken(payload: {
    wallet_pubkey: string;
    user_id?: string;
    email?: string;
    expiresIn?: string;
  }): string {
    const tokenPayload = {
      wallet_pubkey: payload.wallet_pubkey,
      user_id: payload.user_id,
      email: payload.email,
      iat: Math.floor(Date.now() / 1000),
    };

    return jwt.sign(tokenPayload, this.secret, { expiresIn: payload.expiresIn || '24h' } as jwt.SignOptions);
  }

  /**
   * Validate a JWT token and return user information
   */
  public async validateToken(token: string): Promise<JWTValidationResult | null> {
    try {
      // Verify the token signature and expiration
      const decoded = jwt.verify(token, this.secret) as JWTTokenPayload;

      // Normalize wallet address field (handle both wallet_pubkey and walletAddress)
      const walletAddress = decoded.wallet_pubkey || decoded.walletAddress;
      if (!walletAddress) {
        logger.error('Missing wallet address in JWT payload');
        return null;
      }

      // Get user from database to verify they exist
      const user = await this.db.getUserByWallet(walletAddress);
      if (!user) {
        logger.error(`User not found for wallet: ${walletAddress}`);
        return null;
      }

      return {
        user_id: user.id,
        wallet_address: walletAddress,
        email: user.email,
        valid: true
      };

    } catch (error) {
      logger.error('JWT validation failed:', error);
      return null;
    }
  }

  /**
   * Decode a JWT token without verification (for debugging)
   */
  public decodeToken(token: string): JWTTokenPayload | null {
    try {
      return jwt.decode(token) as JWTTokenPayload;
    } catch (error) {
      logger.error('JWT decode failed:', error);
      return null;
    }
  }

  /**
   * Verify token signature without database lookup
   */
  public verifyTokenSignature(token: string): JWTTokenPayload | null {
    try {
      return jwt.verify(token, this.secret) as JWTTokenPayload;
    } catch (error) {
      logger.error('JWT signature verification failed:', error);
      return null;
    }
  }

  /**
   * Check if token is expired
   */
  public isTokenExpired(token: string): boolean {
    try {
      const decoded = jwt.decode(token) as JWTTokenPayload;
      if (!decoded || !decoded.exp) {
        return true;
      }
      return Date.now() >= decoded.exp * 1000;
    } catch (error) {
      logger.error('Token expiration check failed:', error);
      return true;
    }
  }

  /**
   * Refresh a token (create new token with same payload)
   */
  public async refreshToken(token: string): Promise<string | null> {
    try {
      const decoded = jwt.verify(token, this.secret) as JWTTokenPayload;
      const walletAddress = decoded.wallet_pubkey || decoded.walletAddress;
      
      if (!walletAddress) {
        return null;
      }

      // Verify user still exists
      const user = await this.db.getUserByWallet(walletAddress);
      if (!user) {
        return null;
      }

      // Create new token
      return this.createToken({
        wallet_pubkey: walletAddress,
        user_id: user.id,
        email: user.email,
        expiresIn: '24h'
      });

    } catch (error) {
      logger.error('Token refresh failed:', error);
      return null;
    }
  }

  /**
   * Extract user ID from token without database lookup
   */
  public extractUserId(token: string): string | null {
    try {
      const decoded = jwt.decode(token) as JWTTokenPayload;
      return decoded?.user_id || decoded?.userId || null;
    } catch (error) {
      logger.error('User ID extraction failed:', error);
      return null;
    }
  }

  /**
   * Extract wallet address from token without database lookup
   */
  public extractWalletAddress(token: string): string | null {
    try {
      const decoded = jwt.decode(token) as JWTTokenPayload;
      return decoded?.wallet_pubkey || decoded?.walletAddress || null;
    } catch (error) {
      logger.error('Wallet address extraction failed:', error);
      return null;
    }
  }
}
