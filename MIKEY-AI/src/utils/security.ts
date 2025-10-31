import crypto from 'crypto';
import { config } from '../config';

/**
 * Security utilities for the Solana DeFi Trading Intelligence AI
 * Implements best practices for API key management, data encryption, and validation
 */

export class SecurityUtils {
  private static readonly ENCRYPTION_ALGORITHM = 'aes-256-gcm';
  private static readonly IV_LENGTH = 16;

  /**
   * Validate environment variables for security
   */
  static validateEnvironment(): void {
    const requiredVars = [
      'SOLANA_PRIVATE_KEY',
      'OPENAI_API_KEY',
      'JWT_SECRET',
      'ENCRYPTION_KEY'
    ];

    const missingVars = requiredVars.filter(varName => !process.env[varName]);
    
    if (missingVars.length > 0) {
      throw new Error(`Missing required environment variables: ${missingVars.join(', ')}`);
    }

    // Validate encryption key length
    if (process.env['ENCRYPTION_KEY'] && process.env['ENCRYPTION_KEY'].length !== 32) {
      throw new Error('ENCRYPTION_KEY must be exactly 32 characters long');
    }

    // Validate JWT secret strength
    if (process.env['JWT_SECRET'] && process.env['JWT_SECRET'].length < 32) {
      throw new Error('JWT_SECRET must be at least 32 characters long');
    }
  }

  /**
   * Encrypt sensitive data
   */
  static encrypt(text: string): string {
    const key = Buffer.from(config.api.encryptionKey, 'utf8');
    const iv = crypto.randomBytes(this.IV_LENGTH);
    const cipher = crypto.createCipher(this.ENCRYPTION_ALGORITHM, key);
    
    let encrypted = cipher.update(text, 'utf8', 'hex');
    encrypted += cipher.final('hex');
    
    return iv.toString('hex') + ':' + encrypted;
  }

  /**
   * Decrypt sensitive data
   */
  static decrypt(encryptedText: string): string {
    const key = Buffer.from(config.api.encryptionKey, 'utf8');
    const parts = encryptedText.split(':');
    
    if (parts.length !== 2) {
      throw new Error('Invalid encrypted data format');
    }

    const iv = Buffer.from(parts[0], 'hex');
    const encrypted = parts[1];

    const decipher = crypto.createDecipher(this.ENCRYPTION_ALGORITHM, key);

    let decrypted = decipher.update(encrypted, 'hex', 'utf8');
    decrypted += decipher.final('utf8');

    return decrypted;
  }

  /**
   * Hash sensitive data (one-way)
   */
  static hash(text: string, salt?: string): string {
    const actualSalt = salt || crypto.randomBytes(16).toString('hex');
    return crypto.pbkdf2Sync(text, actualSalt, 10000, 64, 'sha512').toString('hex');
  }

  /**
   * Generate secure random string
   */
  static generateSecureRandom(length: number = 32): string {
    return crypto.randomBytes(length).toString('hex');
  }

  /**
   * Validate Solana address format
   */
  static isValidSolanaAddress(address: string): boolean {
    try {
      // Basic validation - Solana addresses are base58 encoded and 32-44 characters
      const base58Regex = /^[1-9A-HJ-NP-Za-km-z]{32,44}$/;
      return base58Regex.test(address);
    } catch {
      return false;
    }
  }

  /**
   * Sanitize user input
   */
  static sanitizeInput(input: string): string {
    return input
      .replace(/[<>]/g, '') // Remove potential HTML tags
      .replace(/['"]/g, '') // Remove quotes
      .replace(/[;\\]/g, '') // Remove potential SQL injection characters
      .trim()
      .substring(0, 1000); // Limit length
  }

  /**
   * Validate API key format
   */
  static isValidApiKey(apiKey: string, expectedPrefix?: string): boolean {
    if (!apiKey || apiKey.length < 20) {
      return false;
    }

    if (expectedPrefix && !apiKey.startsWith(expectedPrefix)) {
      return false;
    }

    // Check for common patterns that might indicate a fake key
    const suspiciousPatterns = [
      /^sk-[a-zA-Z0-9]{20,}$/, // OpenAI pattern
      /^[a-zA-Z0-9]{32,}$/, // Generic pattern
    ];

    return suspiciousPatterns.some(pattern => pattern.test(apiKey));
  }

  /**
   * Mask sensitive data for logging
   */
  static maskSensitiveData(data: string, visibleChars: number = 4): string {
    if (data.length <= visibleChars * 2) {
      return '*'.repeat(data.length);
    }
    
    const start = data.substring(0, visibleChars);
    const end = data.substring(data.length - visibleChars);
    const middle = '*'.repeat(data.length - visibleChars * 2);
    
    return start + middle + end;
  }

  /**
   * Rate limiting helper
   */
  static generateRateLimitKey(identifier: string, windowMs: number): string {
    const window = Math.floor(Date.now() / windowMs);
    return `rate_limit:${identifier}:${window}`;
  }

  /**
   * Validate request origin
   */
  static isValidOrigin(origin: string): boolean {
    const allowedOrigins = config.api.corsOrigins;
    return allowedOrigins.includes(origin) || allowedOrigins.includes('*');
  }

  /**
   * Generate secure session token
   */
  static generateSessionToken(): string {
    return crypto.randomBytes(32).toString('hex');
  }

  /**
   * Validate private key format
   */
  static isValidPrivateKey(privateKey: string): boolean {
    try {
      // Check if it's a valid base58 encoded private key
      const base58Regex = /^[1-9A-HJ-NP-Za-km-z]{64,88}$/;
      return base58Regex.test(privateKey);
    } catch {
      return false;
    }
  }

  /**
   * Secure comparison to prevent timing attacks
   */
  static secureCompare(a: string, b: string): boolean {
    if (a.length !== b.length) {
      return false;
    }

    let result = 0;
    for (let i = 0; i < a.length; i++) {
      result |= a.charCodeAt(i) ^ b.charCodeAt(i);
    }

    return result === 0;
  }
}

/**
 * Security middleware for Express
 */
export const securityMiddleware = {
  /**
   * Validate API key middleware
   */
  validateApiKey: (req: any, res: any, next: any) => {
    const apiKey = req.headers.authorization?.replace('Bearer ', '');
    
    if (!apiKey) {
      return res.status(401).json({
        success: false,
        error: {
          code: 'MISSING_API_KEY',
          message: 'API key is required'
        }
      });
    }

    if (!SecurityUtils.isValidApiKey(apiKey)) {
      return res.status(401).json({
        success: false,
        error: {
          code: 'INVALID_API_KEY',
          message: 'Invalid API key format'
        }
      });
    }

    next();
  },

  /**
   * Rate limiting middleware
   */
  rateLimit: (req: any, res: any, next: any) => {
    // This would integrate with Redis for actual rate limiting
    // For now, just pass through
    next();
  },

  /**
   * CORS validation middleware
   */
  validateCors: (req: any, res: any, next: any) => {
    const origin = req.headers.origin;
    
    if (origin && !SecurityUtils.isValidOrigin(origin)) {
      return res.status(403).json({
        success: false,
        error: {
          code: 'INVALID_ORIGIN',
          message: 'Origin not allowed'
        }
      });
    }

    next();
  }
};
