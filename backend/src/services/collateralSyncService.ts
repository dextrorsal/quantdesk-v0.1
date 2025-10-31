import { Connection, PublicKey } from '@solana/web3.js';
import { databaseService } from './supabaseDatabase';
import { Logger } from '../utils/logger';
import { backendSmartContractService } from './backendSmartContractService';

/**
 * Collateral Synchronization Service
 * Ensures database stays in sync with on-chain collateral data
 * 
 * CRITICAL: This service prevents fund loss by maintaining data consistency
 */
export class CollateralSyncService {
  private static instance: CollateralSyncService;
  private logger = new Logger();
  private connection: Connection;

  private constructor() {
    this.connection = new Connection(
      process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com',
      'confirmed'
    );
  }

  public static getInstance(): CollateralSyncService {
    if (!CollateralSyncService.instance) {
      CollateralSyncService.instance = new CollateralSyncService();
    }
    return CollateralSyncService.instance;
  }

  /**
   * Sync user's collateral data between on-chain and database
   * This is called after every deposit/withdrawal transaction
   */
  public async syncUserCollateral(userId: string, walletAddress: string): Promise<void> {
    try {
      this.logger.info(`🔄 Syncing collateral for user ${userId} (${walletAddress})`);

      // Get on-chain collateral data
      const onChainData = await this.getOnChainCollateralData(walletAddress);
      
      // Get database collateral data
      const dbData = await this.getDatabaseCollateralData(userId);

      // Perform comprehensive validation before sync
      const validationResult = await this.validateDataConsistency(onChainData, dbData);
      
      if (!validationResult.isValid) {
        this.logger.error(`❌ Data consistency validation failed:`, validationResult.reasons);
        
        // Log critical security event
        await this.logSecurityEvent('data_inconsistency', userId, {
          walletAddress,
          onChainData,
          dbData,
          reasons: validationResult.reasons
        });

        // Trigger emergency procedures if critical discrepancy
        if (validationResult.isCritical) {
          await this.triggerEmergencyProcedures(userId, walletAddress, validationResult);
          return;
        }
      }

      // Compare and sync if needed
      await this.performSync(userId, walletAddress, onChainData, dbData);

      this.logger.info(`✅ Collateral sync completed for user ${userId}`);
    } catch (error) {
      this.logger.error(`❌ Collateral sync failed for user ${userId}:`, error);
      throw error;
    }
  }

  /**
   * Validate data consistency between on-chain and database
   * CRITICAL: This prevents fund loss from sync issues
   */
  private async validateDataConsistency(
    onChainData: any,
    dbData: any
  ): Promise<{
    isValid: boolean;
    isCritical: boolean;
    reasons: string[];
  }> {
    const reasons: string[] = [];
    let isCritical = false;

    try {
      // 1. Check for significant balance discrepancies
      const balanceDifference = Math.abs(onChainData.totalCollateral - dbData.totalCollateral);
      const maxAllowedDifference = Math.max(onChainData.totalCollateral, dbData.totalCollateral) * 0.01; // 1% tolerance

      if (balanceDifference > maxAllowedDifference) {
        const reason = `Significant balance discrepancy: On-chain ${onChainData.totalCollateral}, DB ${dbData.totalCollateral}, Diff ${balanceDifference}`;
        reasons.push(reason);
        
        // Mark as critical if difference is > 5%
        if (balanceDifference > Math.max(onChainData.totalCollateral, dbData.totalCollateral) * 0.05) {
          isCritical = true;
        }
      }

      // 2. Check for negative balances
      if (onChainData.totalCollateral < 0 || dbData.totalCollateral < 0) {
        reasons.push('Negative balance detected');
        isCritical = true;
      }

      // 3. Check for unrealistic values
      if (onChainData.totalCollateral > 10000000 || dbData.totalCollateral > 10000000) {
        reasons.push('Unrealistic balance value detected (>$10M)');
        isCritical = true;
      }

      // 4. Check timestamp consistency
      const timeDifference = Math.abs(
        new Date(onChainData.lastUpdated).getTime() - 
        new Date(dbData.lastUpdated).getTime()
      );
      
      if (timeDifference > 300000) { // 5 minutes
        reasons.push(`Timestamp inconsistency: ${timeDifference}ms difference`);
      }

      return {
        isValid: reasons.length === 0,
        isCritical,
        reasons
      };

    } catch (error) {
      this.logger.error('Error validating data consistency:', error);
      return {
        isValid: false,
        isCritical: true,
        reasons: ['Data validation failed due to system error']
      };
    }
  }

  /**
   * Trigger emergency procedures for critical discrepancies
   */
  private async triggerEmergencyProcedures(
    userId: string,
    walletAddress: string,
    validationResult: any
  ): Promise<void> {
    try {
      this.logger.error(`🚨 EMERGENCY: Triggering procedures for user ${userId}`);

      // 1. Lock user account temporarily
      await this.lockUserAccount(userId, 'data_inconsistency');

      // 2. Create emergency ticket
      await this.createEmergencyTicket(userId, walletAddress, validationResult);

      // 3. Notify administrators
      await this.notifyAdministrators(userId, walletAddress, validationResult);

      // 4. Preserve evidence
      await this.preserveEvidence(userId, walletAddress, validationResult);

    } catch (error) {
      this.logger.error('Error triggering emergency procedures:', error);
    }
  }

  /**
   * Lock user account for security
   */
  private async lockUserAccount(userId: string, reason: string): Promise<void> {
    try {
      await databaseService.update('users', 
        { 
          is_locked: true, 
          lock_reason: reason,
          locked_at: new Date().toISOString()
        },
        { id: userId }
      );
      
      this.logger.warn(`🔒 User account locked: ${userId} (${reason})`);
    } catch (error) {
      this.logger.error('Error locking user account:', error);
    }
  }

  /**
   * Create emergency ticket for manual review
   */
  private async createEmergencyTicket(
    userId: string,
    walletAddress: string,
    validationResult: any
  ): Promise<void> {
    try {
      await databaseService.insert('emergency_tickets', {
        user_id: userId,
        wallet_address: walletAddress,
        issue_type: 'data_inconsistency',
        severity: 'critical',
        details: JSON.stringify(validationResult),
        status: 'open',
        created_at: new Date().toISOString()
      });
      
      this.logger.info(`🎫 Emergency ticket created for user ${userId}`);
    } catch (error) {
      this.logger.error('Error creating emergency ticket:', error);
    }
  }

  /**
   * Notify administrators of critical issues
   */
  private async notifyAdministrators(
    userId: string,
    walletAddress: string,
    validationResult: any
  ): Promise<void> {
    try {
      // This would integrate with notification service
      this.logger.error(`📧 ADMIN ALERT: Critical data inconsistency for user ${userId}`);
      this.logger.error(`Wallet: ${walletAddress}`);
      this.logger.error(`Issues: ${validationResult.reasons.join(', ')}`);
    } catch (error) {
      this.logger.error('Error notifying administrators:', error);
    }
  }

  /**
   * Preserve evidence for investigation
   */
  private async preserveEvidence(
    userId: string,
    walletAddress: string,
    validationResult: any
  ): Promise<void> {
    try {
      await databaseService.insert('security_evidence', {
        user_id: userId,
        wallet_address: walletAddress,
        evidence_type: 'data_inconsistency',
        data: JSON.stringify(validationResult),
        created_at: new Date().toISOString()
      });
      
      this.logger.info(`📁 Evidence preserved for user ${userId}`);
    } catch (error) {
      this.logger.error('Error preserving evidence:', error);
    }
  }

  /**
   * Log security event
   */
  private async logSecurityEvent(
    eventType: string,
    userId: string,
    details: any
  ): Promise<void> {
    try {
      await databaseService.insert('security_events', {
        event_type: eventType,
        user_id: userId,
        details: JSON.stringify(details),
        severity: 'high',
        created_at: new Date().toISOString()
      });
    } catch (error) {
      this.logger.error('Error logging security event:', error);
    }
  }

  /**
   * Get collateral data from on-chain smart contract
   */
  private async getOnChainCollateralData(walletAddress: string): Promise<{
    totalCollateral: number;
    solAmount: number;
    usdValue: number;
    lastUpdated: number;
  }> {
    try {
      // Get user account state from smart contract
      const accountState = await backendSmartContractService.getUserAccountState(walletAddress);
      
      // Get SOL collateral balance
      const solBalance = await backendSmartContractService.getSOLCollateralBalance(walletAddress);

      return {
        totalCollateral: accountState.totalCollateral,
        solAmount: solBalance,
        usdValue: accountState.totalCollateral, // Already in USD from smart contract
        lastUpdated: Date.now()
      };
    } catch (error) {
      this.logger.error('Error fetching on-chain collateral data:', error);
      throw error;
    }
  }

  /**
   * Get collateral data from database
   */
  private async getDatabaseCollateralData(userId: string): Promise<{
    totalCollateral: number;
    solAmount: number;
    usdValue: number;
    lastUpdated: string;
  }> {
    try {
      const result = await databaseService.select('user_balances', '*', { 
        user_id: userId, 
        asset: 'SOL' 
      });

      if (result.length === 0) {
        return {
          totalCollateral: 0,
          solAmount: 0,
          usdValue: 0,
          lastUpdated: new Date().toISOString()
        };
      }

      const balance = result[0];
      return {
        totalCollateral: parseFloat(balance.balance),
        solAmount: parseFloat(balance.balance),
        usdValue: parseFloat(balance.balance), // Assuming 1:1 for now
        lastUpdated: balance.updated_at
      };
    } catch (error) {
      this.logger.error('Error fetching database collateral data:', error);
      throw error;
    }
  }

  /**
   * Perform synchronization between on-chain and database data
   */
  private async performSync(
    userId: string, 
    walletAddress: string, 
    onChainData: any, 
    dbData: any
  ): Promise<void> {
    const tolerance = 0.01; // 1 cent tolerance for floating point differences
    
    // Check if data is out of sync
    const collateralDiff = Math.abs(onChainData.totalCollateral - dbData.totalCollateral);
    const solDiff = Math.abs(onChainData.solAmount - dbData.solAmount);

    if (collateralDiff > tolerance || solDiff > tolerance) {
      this.logger.warn(`⚠️ Data mismatch detected for user ${userId}:`, {
        onChain: onChainData,
        database: dbData,
        collateralDiff,
        solDiff
      });

      // Update database to match on-chain data (source of truth)
      await this.updateDatabaseCollateral(userId, onChainData);

      // Log the sync for audit purposes
      await this.logSyncEvent(userId, walletAddress, onChainData, dbData);
    }
  }

  /**
   * Update database collateral to match on-chain data
   */
  private async updateDatabaseCollateral(userId: string, onChainData: any): Promise<void> {
    try {
      await databaseService.upsert('user_balances', {
        user_id: userId,
        asset: 'SOL',
        balance: onChainData.solAmount,
        locked_balance: 0, // Reset locked balance on sync
        updated_at: new Date().toISOString()
      });

      this.logger.info(`📝 Updated database collateral for user ${userId}: ${onChainData.solAmount} SOL`);
    } catch (error) {
      this.logger.error('Error updating database collateral:', error);
      throw error;
    }
  }

  /**
   * Log sync events for audit trail
   */
  private async logSyncEvent(
    userId: string, 
    walletAddress: string, 
    onChainData: any, 
    dbData: any
  ): Promise<void> {
    try {
      await databaseService.insert('sync_events', {
        user_id: userId,
        wallet_address: walletAddress,
        event_type: 'collateral_sync',
        on_chain_data: onChainData,
        database_data: dbData,
        created_at: new Date().toISOString()
      });
    } catch (error) {
      this.logger.error('Error logging sync event:', error);
      // Don't throw - this is just for audit purposes
    }
  }

  /**
   * Validate collateral consistency across all users
   * This can be run periodically to ensure data integrity
   */
  public async validateAllCollateral(): Promise<{
    totalUsers: number;
    syncedUsers: number;
    mismatchedUsers: number;
    errors: string[];
  }> {
    const results = {
      totalUsers: 0,
      syncedUsers: 0,
      mismatchedUsers: 0,
      errors: [] as string[]
    };

    try {
      // Get all users with SOL balances
      const users = await databaseService.select('user_balances', 'user_id, balance', { 
        asset: 'SOL' 
      });

      results.totalUsers = users.length;

      for (const user of users) {
        try {
          // Get user's wallet address
          const userData = await databaseService.select('users', 'wallet_address', { 
            id: user.user_id 
          });

          if (userData.length === 0) {
            results.errors.push(`User ${user.user_id} not found`);
            continue;
          }

          const walletAddress = userData[0].wallet_address;
          
          // Sync this user's collateral
          await this.syncUserCollateral(user.user_id, walletAddress);
          results.syncedUsers++;

        } catch (error) {
          results.mismatchedUsers++;
          results.errors.push(`User ${user.user_id}: ${error.message}`);
        }
      }

      this.logger.info(`🔍 Collateral validation completed:`, results);
      return results;

    } catch (error) {
      this.logger.error('Error during collateral validation:', error);
      throw error;
    }
  }

  /**
   * Emergency sync - force update all collateral data
   * Use this if there's a major data inconsistency
   */
  public async emergencySync(): Promise<void> {
    this.logger.warn('🚨 Starting emergency collateral sync...');
    
    const results = await this.validateAllCollateral();
    
    if (results.errors.length > 0) {
      this.logger.error('❌ Emergency sync completed with errors:', results.errors);
      throw new Error(`Emergency sync failed: ${results.errors.join(', ')}`);
    }

    this.logger.info('✅ Emergency sync completed successfully');
  }
}

export const collateralSyncService = CollateralSyncService.getInstance();
