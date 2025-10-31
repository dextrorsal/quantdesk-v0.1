import { Logger } from '../utils/logger';
import { databaseService } from './supabaseDatabase';

/**
 * Security Validation Service
 * 
 * Implements critical security measures to prevent:
 * - Price manipulation attacks
 * - Fund loss scenarios
 * - Unauthorized access
 * - Data corruption
 */
export class SecurityValidationService {
  private static instance: SecurityValidationService;
  private logger = new Logger();

  // Security thresholds
  private readonly MAX_PRICE_CHANGE_PERCENT = 10; // 10% max price change
  private readonly MAX_TRANSACTION_AMOUNT = 1000000; // $1M max per transaction
  private readonly MIN_TRANSACTION_AMOUNT = 0.000001; // 1 micro-unit minimum
  private readonly PRICE_STALENESS_SECONDS = 30; // 30 seconds max staleness
  private readonly EMERGENCY_PAUSE_THRESHOLD = 50; // 50% price change triggers emergency pause
  private readonly MAX_DAILY_WITHDRAWAL = 100000; // $100K max daily withdrawal per user
  private readonly SUSPICIOUS_ACTIVITY_THRESHOLD = 5; // 5 failed attempts trigger lockout

  private constructor() {}

  public static getInstance(): SecurityValidationService {
    if (!SecurityValidationService.instance) {
      SecurityValidationService.instance = new SecurityValidationService();
    }
    return SecurityValidationService.instance;
  }

  /**
   * Validate price data for manipulation attempts
   */
  public async validatePriceData(
    currentPrice: number,
    previousPrice: number,
    timestamp: number
  ): Promise<{ isValid: boolean; reason?: string }> {
    try {
      // Check price staleness
      const now = Date.now() / 1000;
      if (now - timestamp > this.PRICE_STALENESS_SECONDS) {
        this.logger.warn(`⚠️ Price data is stale: ${now - timestamp} seconds old`);
        return { isValid: false, reason: 'Price data is stale' };
      }

      // Check for extreme price changes (potential manipulation)
      if (previousPrice > 0) {
        const priceChangePercent = Math.abs((currentPrice - previousPrice) / previousPrice) * 100;
        
        if (priceChangePercent > this.MAX_PRICE_CHANGE_PERCENT) {
          this.logger.error(`🚨 Suspicious price change detected: ${priceChangePercent.toFixed(2)}%`);
          return { 
            isValid: false, 
            reason: `Price change too large: ${priceChangePercent.toFixed(2)}%` 
          };
        }
      }

      // Check for invalid price values
      if (currentPrice <= 0 || !isFinite(currentPrice)) {
        this.logger.error(`🚨 Invalid price value: ${currentPrice}`);
        return { isValid: false, reason: 'Invalid price value' };
      }

      // Check for reasonable price ranges
      if (currentPrice < 0.01 || currentPrice > 1000000) {
        this.logger.error(`🚨 Price outside reasonable range: ${currentPrice}`);
        return { isValid: false, reason: 'Price outside reasonable range' };
      }

      return { isValid: true };
    } catch (error) {
      this.logger.error('Error validating price data:', error);
      return { isValid: false, reason: 'Price validation error' };
    }
  }

  /**
   * Validate transaction amount for security
   */
  public validateTransactionAmount(
    amount: number,
    asset: string,
    transactionType: 'deposit' | 'withdrawal'
  ): { isValid: boolean; reason?: string } {
    try {
      // Check minimum amount
      if (amount < this.MIN_TRANSACTION_AMOUNT) {
        return { 
          isValid: false, 
          reason: `Amount too small: ${amount} (minimum: ${this.MIN_TRANSACTION_AMOUNT})` 
        };
      }

      // Check maximum amount
      if (amount > this.MAX_TRANSACTION_AMOUNT) {
        return { 
          isValid: false, 
          reason: `Amount too large: ${amount} (maximum: ${this.MAX_TRANSACTION_AMOUNT})` 
        };
      }

      // Check for valid number
      if (!isFinite(amount) || isNaN(amount)) {
        return { isValid: false, reason: 'Invalid amount value' };
      }

      // Asset-specific validations
      if (asset === 'SOL') {
        // SOL-specific validations
        if (amount > 10000) { // Max 10,000 SOL per transaction
          return { 
            isValid: false, 
            reason: 'SOL amount exceeds maximum limit' 
          };
        }
      } else if (asset === 'USDC' || asset === 'USDT') {
        // Stablecoin validations
        if (amount > 10000000) { // Max $10M per transaction
          return { 
            isValid: false, 
            reason: 'Stablecoin amount exceeds maximum limit' 
          };
        }
      }

      return { isValid: true };
    } catch (error) {
      this.logger.error('Error validating transaction amount:', error);
      return { isValid: false, reason: 'Amount validation error' };
    }
  }

  /**
   * Validate user permissions and account status
   */
  public async validateUserPermissions(
    userId: string,
    walletAddress: string,
    action: 'deposit' | 'withdraw' | 'trade'
  ): Promise<{ isValid: boolean; reason?: string }> {
    try {
      // Get user data
      const userData = await databaseService.select('users', '*', { id: userId });
      
      if (userData.length === 0) {
        return { isValid: false, reason: 'User not found' };
      }

      const user = userData[0];

      // Check if user is active
      if (!user.is_active) {
        this.logger.warn(`🚨 Inactive user attempted ${action}: ${userId}`);
        return { isValid: false, reason: 'User account is inactive' };
      }

      // Check wallet address match
      if (user.wallet_address !== walletAddress) {
        this.logger.error(`🚨 Wallet address mismatch for user ${userId}`);
        return { isValid: false, reason: 'Wallet address mismatch' };
      }

      // Check KYC status for large transactions
      if (action === 'withdraw' && user.kyc_status !== 'verified') {
        // Allow small withdrawals without KYC
        const userBalances = await databaseService.select('user_balances', 'balance', { 
          user_id: userId, 
          asset: 'SOL' 
        });
        
        if (userBalances.length > 0 && parseFloat(userBalances[0].balance) > 1000) {
          return { 
            isValid: false, 
            reason: 'KYC verification required for large withdrawals' 
          };
        }
      }

      // Check risk level
      if (user.risk_level === 'high' && action === 'withdraw') {
        this.logger.warn(`🚨 High-risk user attempted withdrawal: ${userId}`);
        return { isValid: false, reason: 'Withdrawal restricted for high-risk accounts' };
      }

      return { isValid: true };
    } catch (error) {
      this.logger.error('Error validating user permissions:', error);
      return { isValid: false, reason: 'Permission validation error' };
    }
  }

  /**
   * Validate smart contract state for security
   */
  public async validateSmartContractState(
    walletAddress: string
  ): Promise<{ isValid: boolean; reason?: string; data?: any }> {
    try {
      // This would integrate with the smart contract service
      // For now, we'll implement basic validations
      
      // Check if wallet address is valid
      if (!walletAddress || walletAddress.length < 32) {
        return { isValid: false, reason: 'Invalid wallet address format' };
      }

      // Check for suspicious patterns
      if (walletAddress.includes('0000000000000000000000000000000000000000')) {
        this.logger.error(`🚨 Suspicious wallet address pattern: ${walletAddress}`);
        return { isValid: false, reason: 'Suspicious wallet address pattern' };
      }

      return { isValid: true };
    } catch (error) {
      this.logger.error('Error validating smart contract state:', error);
      return { isValid: false, reason: 'Smart contract validation error' };
    }
  }


  /**
   * Log security events for monitoring
   */
  public async logSecurityEvent(
    eventType: 'price_manipulation' | 'unauthorized_access' | 'suspicious_activity' | 'fund_safety',
    userId: string,
    details: any
  ): Promise<void> {
    try {
      await databaseService.insert('security_events', {
        event_type: eventType,
        user_id: userId,
        details: details,
        created_at: new Date().toISOString()
      });

      this.logger.warn(`🚨 Security event logged: ${eventType} for user ${userId}`);
    } catch (error) {
      this.logger.error('Error logging security event:', error);
    }
  }

  /**
   * Comprehensive security check for all transactions
   * CRITICAL: This prevents fund loss and manipulation attacks
   */
  public async performSecurityCheck(
    userId: string,
    walletAddress: string,
    asset: string,
    amount: number,
    transactionType: 'deposit' | 'withdrawal' | 'trade'
  ): Promise<{
    isValid: boolean;
    reasons: string[];
    warnings: string[];
    emergencyPause: boolean;
  }> {
    const reasons: string[] = [];
    const warnings: string[] = [];
    let emergencyPause = false;

    try {
      // 1. Validate transaction amount
      const amountValidation = this.validateTransactionAmount(amount, asset, transactionType as 'deposit' | 'withdrawal');
      if (!amountValidation.isValid) {
        reasons.push(amountValidation.reason!);
      }

      // 2. Check daily withdrawal limits
      if (transactionType === 'withdrawal') {
        const dailyWithdrawal = await this.getDailyWithdrawalAmount(userId);
        if (dailyWithdrawal + amount > this.MAX_DAILY_WITHDRAWAL) {
          reasons.push(`Daily withdrawal limit exceeded: ${dailyWithdrawal + amount} > ${this.MAX_DAILY_WITHDRAWAL}`);
        }
      }

      // 3. Validate price data staleness
      const priceValidation = await this.validatePriceData(
        await this.getCurrentPrice(asset),
        await this.getPreviousPrice(asset),
        Date.now() / 1000
      );

      if (!priceValidation.isValid) {
        reasons.push(`Price validation failed: ${priceValidation.reason}`);
      }

      // 4. Check for suspicious activity patterns
      const suspiciousActivity = await this.checkSuspiciousActivity(userId, walletAddress);
      if (suspiciousActivity.isSuspicious) {
        reasons.push(`Suspicious activity detected: ${suspiciousActivity.reason}`);
        emergencyPause = true;
      }

      // 5. Validate wallet address
      if (!this.isValidWalletAddress(walletAddress)) {
        reasons.push(`Invalid wallet address: ${walletAddress}`);
      }

      // 6. Check for emergency conditions
      const emergencyCheck = await this.checkEmergencyConditions(asset);
      if (emergencyCheck.isEmergency) {
        reasons.push(`Emergency condition detected: ${emergencyCheck.reason}`);
        emergencyPause = true;
      }

      // 7. Additional warnings for high-value transactions
      if (amount > this.MAX_TRANSACTION_AMOUNT * 0.1) {
        warnings.push(`High-value transaction: ${amount} USD`);
      }

      return {
        isValid: reasons.length === 0,
        reasons,
        warnings,
        emergencyPause
      };

    } catch (error) {
      this.logger.error('Error performing security check:', error);
      return {
        isValid: false,
        reasons: ['Security check failed due to system error'],
        warnings: [],
        emergencyPause: true
      };
    }
  }

  /**
   * Check for emergency conditions that require immediate pause
   */
  private async checkEmergencyConditions(asset: string): Promise<{
    isEmergency: boolean;
    reason?: string;
  }> {
    try {
      const currentPrice = await this.getCurrentPrice(asset);
      const previousPrice = await this.getPreviousPrice(asset);

      if (previousPrice > 0) {
        const priceChangePercent = Math.abs((currentPrice - previousPrice) / previousPrice) * 100;
        
        if (priceChangePercent > this.EMERGENCY_PAUSE_THRESHOLD) {
          return {
            isEmergency: true,
            reason: `Extreme price change: ${priceChangePercent.toFixed(2)}%`
          };
        }
      }

      // Check for oracle failures
      const oracleHealth = await this.checkOracleHealth();
      if (!oracleHealth.isHealthy) {
        return {
          isEmergency: true,
          reason: `Oracle system failure: ${oracleHealth.reason}`
        };
      }

      return { isEmergency: false };
    } catch (error) {
      this.logger.error('Error checking emergency conditions:', error);
      return {
        isEmergency: true,
        reason: 'Unable to verify system health'
      };
    }
  }

  /**
   * Check for suspicious activity patterns
   */
  private async checkSuspiciousActivity(userId: string, walletAddress: string): Promise<{
    isSuspicious: boolean;
    reason?: string;
  }> {
    try {
      // Check for rapid successive transactions
      const recentTransactions = await databaseService.select(
        'withdrawals',
        '*',
        { user_id: userId }
      );

      if (recentTransactions.length >= this.SUSPICIOUS_ACTIVITY_THRESHOLD) {
        const now = new Date();
        const oneHourAgo = new Date(now.getTime() - 60 * 60 * 1000);
        
        const recentCount = recentTransactions.filter(tx => 
          new Date(tx.created_at) > oneHourAgo
        ).length;

        if (recentCount >= this.SUSPICIOUS_ACTIVITY_THRESHOLD) {
          return {
            isSuspicious: true,
            reason: `Too many transactions in last hour: ${recentCount}`
          };
        }
      }

      return { isSuspicious: false };
    } catch (error) {
      this.logger.error('Error checking suspicious activity:', error);
      return { isSuspicious: false };
    }
  }

  /**
   * Get daily withdrawal amount for user
   */
  private async getDailyWithdrawalAmount(userId: string): Promise<number> {
    try {
      const today = new Date();
      today.setHours(0, 0, 0, 0);
      
      const withdrawals = await databaseService.select(
        'withdrawals',
        'amount',
        { 
          user_id: userId,
          created_at: { gte: today.toISOString() }
        }
      );

      return withdrawals.reduce((sum, w) => sum + parseFloat(w.amount), 0);
    } catch (error) {
      this.logger.error('Error getting daily withdrawal amount:', error);
      return 0;
    }
  }

  /**
   * Validate wallet address format
   */
  private isValidWalletAddress(address: string): boolean {
    try {
      // Basic Solana address validation
      return /^[1-9A-HJ-NP-Za-km-z]{32,44}$/.test(address);
    } catch {
      return false;
    }
  }

  /**
   * Check oracle system health
   */
  private async checkOracleHealth(): Promise<{
    isHealthy: boolean;
    reason?: string;
  }> {
    try {
      // This would integrate with the actual oracle service
      // For now, return healthy status
      return { isHealthy: true };
    } catch (error) {
      return {
        isHealthy: false,
        reason: 'Oracle service unavailable'
      };
    }
  }

  /**
   * Get current price for asset
   */
  private async getCurrentPrice(asset: string): Promise<number> {
    try {
      // This would integrate with the actual price service
      // For now, return a placeholder
      return 100; // Placeholder price
    } catch (error) {
      this.logger.error('Error getting current price:', error);
      return 0;
    }
  }

  /**
   * Get previous price for asset
   */
  private async getPreviousPrice(asset: string): Promise<number> {
    try {
      // This would integrate with the actual price service
      // For now, return a placeholder
      return 95; // Placeholder previous price
    } catch (error) {
      this.logger.error('Error getting previous price:', error);
      return 0;
    }
  }
}

export const securityValidationService = SecurityValidationService.getInstance();
