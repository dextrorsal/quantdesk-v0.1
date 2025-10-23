import { Connection, ParsedTransactionWithMeta } from '@solana/web3.js';
import { Logger } from '../utils/logger';
import { RPCLoadBalancer } from './rpcLoadBalancer';

export interface TransactionVerificationResult {
  isValid: boolean;
  error?: string;
  transaction?: ParsedTransactionWithMeta;
  logs?: string[];
  accounts?: string[];
  programIds?: string[];
}

export interface DepositVerificationData {
  userWallet: string;
  amount: number;
  asset: string;
  expectedProgramId?: string;
}

export interface AccountCreationVerificationData {
  userWallet: string;
  accountIndex: number;
  expectedProgramId?: string;
}

class TransactionVerificationService {
  private static instance: TransactionVerificationService;
  private readonly connection: Connection;
  private readonly logger: Logger;
  private readonly rpcLoadBalancer: RPCLoadBalancer;

  private constructor() {
    this.connection = new Connection(
      process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com',
      'confirmed'
    );
    this.logger = new Logger();
    this.rpcLoadBalancer = RPCLoadBalancer.getInstance();
  }

  public static getInstance(): TransactionVerificationService {
    if (!TransactionVerificationService.instance) {
      TransactionVerificationService.instance = new TransactionVerificationService();
    }
    return TransactionVerificationService.instance;
  }

  /**
   * Verify a transaction signature on Solana blockchain
   */
  public async verifyTransaction(
    signature: string,
    maxRetries: number = 3
  ): Promise<TransactionVerificationResult> {
    try {
      this.logger.info(`🔍 Verifying transaction: ${signature}`);

      // Get transaction details with retry logic
      const transaction = await this.rpcLoadBalancer.executeWithRetry(
        async (connection) => {
          const tx = await connection.getTransaction(signature, {
            commitment: 'confirmed',
            maxSupportedTransactionVersion: 0
          });
          return tx;
        },
        maxRetries
      );

      if (!transaction) {
        this.logger.error(`❌ Transaction not found: ${signature}`);
        return {
          isValid: false,
          error: 'Transaction not found on blockchain'
        };
      }

      // Check if transaction was successful
      if (transaction.meta?.err) {
        this.logger.error(`❌ Transaction failed: ${JSON.stringify(transaction.meta.err)}`);
        return {
          isValid: false,
          error: `Transaction failed: ${JSON.stringify(transaction.meta.err)}`,
          transaction: transaction as unknown as ParsedTransactionWithMeta,
          logs: transaction.meta.logMessages || []
        };
      }

      this.logger.info(`✅ Transaction verified successfully: ${signature}`);
      return {
        isValid: true,
        transaction: transaction as unknown as ParsedTransactionWithMeta,
        logs: transaction.meta?.logMessages || [],
        accounts: transaction.transaction.message.staticAccountKeys?.map(key => key.toString()) || [],
        programIds: transaction.transaction.message.compiledInstructions?.map(ix => 
          transaction.transaction.message.staticAccountKeys?.[ix.programIdIndex]?.toString()
        ).filter(Boolean) || []
      };

    } catch (error) {
      this.logger.error(`❌ Error verifying transaction ${signature}:`, error);
      return {
        isValid: false,
        error: `Verification failed: ${error.message}`
      };
    }
  }

  /**
   * Verify a deposit transaction specifically
   */
  public async verifyDepositTransaction(
    signature: string,
    verificationData: DepositVerificationData
  ): Promise<TransactionVerificationResult> {
    try {
      this.logger.info(`💰 Verifying deposit transaction: ${signature}`);

      const result = await this.verifyTransaction(signature);
      if (!result.isValid) {
        return result;
      }

      const transaction = result.transaction!;
      const { userWallet } = verificationData;

      // Extract account keys
      const accountKeys = transaction.transaction.message.accountKeys?.map(key => key.toString()) || [];
      
      // Check if user wallet is involved
      if (!accountKeys.includes(userWallet)) {
        this.logger.error(`❌ User wallet ${userWallet} not found in transaction accounts`);
        return {
          isValid: false,
          error: 'User wallet not found in transaction',
          transaction,
          accounts: accountKeys
        };
      }

      // Check if expected program is involved (if specified)
      if (verificationData.expectedProgramId && !accountKeys.includes(verificationData.expectedProgramId)) {
        this.logger.error(`❌ Expected program ${verificationData.expectedProgramId} not found in transaction`);
        return {
          isValid: false,
          error: 'Expected program not found in transaction',
          transaction,
          accounts: accountKeys
        };
      }

      // Check transaction logs for deposit-related messages
      const logs = result.logs || [];
      const depositLogs = logs.filter(log => 
        log.toLowerCase().includes('deposit') || 
        log.toLowerCase().includes('collateral') ||
        log.toLowerCase().includes('transfer')
      );

      if (depositLogs.length === 0) {
        this.logger.warn(`⚠️ No deposit-related logs found in transaction`);
      }

      // Check pre and post balances for changes
      const preBalances = transaction.meta?.preBalances || [];
      const postBalances = transaction.meta?.postBalances || [];
      
      if (preBalances.length !== postBalances.length) {
        this.logger.error(`❌ Balance arrays length mismatch`);
        return {
          isValid: false,
          error: 'Balance verification failed',
          transaction,
          logs
        };
      }

      // Find significant balance changes
      const balanceChanges = preBalances.map((pre, index) => ({
        account: accountKeys[index],
        preBalance: pre,
        postBalance: postBalances[index],
        change: postBalances[index] - pre
      })).filter(change => Math.abs(change.change) > 0);

      this.logger.info(`📊 Balance changes detected:`, balanceChanges);

      this.logger.info(`✅ Deposit transaction verified: ${signature}`);
      return {
        isValid: true,
        transaction,
        logs,
        accounts: accountKeys,
        programIds: result.programIds
      };

    } catch (error) {
      this.logger.error(`❌ Error verifying deposit transaction:`, error);
      return {
        isValid: false,
        error: `Deposit verification failed: ${error.message}`
      };
    }
  }

  /**
   * Verify an account creation transaction specifically
   */
  public async verifyAccountCreationTransaction(
    signature: string,
    verificationData: AccountCreationVerificationData
  ): Promise<TransactionVerificationResult> {
    try {
      this.logger.info(`👤 Verifying account creation transaction: ${signature}`);

      const result = await this.verifyTransaction(signature);
      if (!result.isValid) {
        return result;
      }

      const transaction = result.transaction!;
      const { userWallet } = verificationData;

      // Extract account keys
      const accountKeys = transaction.transaction.message.accountKeys?.map(key => key.toString()) || [];
      
      // Check if user wallet is involved
      if (!accountKeys.includes(userWallet)) {
        this.logger.error(`❌ User wallet ${userWallet} not found in transaction accounts`);
        return {
          isValid: false,
          error: 'User wallet not found in transaction',
          transaction,
          accounts: accountKeys
        };
      }

      // Check if expected program is involved (if specified)
      if (verificationData.expectedProgramId && !accountKeys.includes(verificationData.expectedProgramId)) {
        this.logger.error(`❌ Expected program ${verificationData.expectedProgramId} not found in transaction`);
        return {
          isValid: false,
          error: 'Expected program not found in transaction',
          transaction,
          accounts: accountKeys
        };
      }

      // Check transaction logs for account creation messages
      const logs = result.logs || [];
      const accountCreationLogs = logs.filter(log => 
        log.toLowerCase().includes('create') || 
        log.toLowerCase().includes('initialize') ||
        log.toLowerCase().includes('account')
      );

      if (accountCreationLogs.length === 0) {
        this.logger.warn(`⚠️ No account creation logs found in transaction`);
      }

      // Check for new account creation (rent exemption)
      const preBalances = transaction.meta?.preBalances || [];
      const postBalances = transaction.meta?.postBalances || [];
      
      // Look for accounts that didn't exist before (preBalance = 0, postBalance > 0)
      const newAccounts = preBalances.map((pre, index) => ({
        account: accountKeys[index],
        preBalance: pre,
        postBalance: postBalances[index],
        isNew: pre === 0 && postBalances[index] > 0
      })).filter(acc => acc.isNew);

      this.logger.info(`🆕 New accounts created:`, newAccounts);

      this.logger.info(`✅ Account creation transaction verified: ${signature}`);
      return {
        isValid: true,
        transaction,
        logs,
        accounts: accountKeys,
        programIds: result.programIds
      };

    } catch (error) {
      this.logger.error(`❌ Error verifying account creation transaction:`, error);
      return {
        isValid: false,
        error: `Account creation verification failed: ${error.message}`
      };
    }
  }

  /**
   * Verify a trading transaction (order placement, execution, etc.)
   */
  public async verifyTradingTransaction(
    signature: string,
    expectedProgramId?: string
  ): Promise<TransactionVerificationResult> {
    try {
      this.logger.info(`📈 Verifying trading transaction: ${signature}`);

      const result = await this.verifyTransaction(signature);
      if (!result.isValid) {
        return result;
      }

      const transaction = result.transaction!;

      // Extract account keys
      const accountKeys = transaction.transaction.message.accountKeys?.map(key => key.toString()) || [];
      
      // Check if expected program is involved (if specified)
      if (expectedProgramId && !accountKeys.includes(expectedProgramId)) {
        this.logger.error(`❌ Expected program ${expectedProgramId} not found in transaction`);
        return {
          isValid: false,
          error: 'Expected program not found in transaction',
          transaction,
          accounts: accountKeys
        };
      }

      // Check transaction logs for trading-related messages
      const logs = result.logs || [];
      const tradingLogs = logs.filter(log => 
        log.toLowerCase().includes('order') || 
        log.toLowerCase().includes('trade') ||
        log.toLowerCase().includes('position') ||
        log.toLowerCase().includes('fill')
      );

      if (tradingLogs.length === 0) {
        this.logger.warn(`⚠️ No trading-related logs found in transaction`);
      }

      this.logger.info(`✅ Trading transaction verified: ${signature}`);
      return {
        isValid: true,
        transaction,
        logs,
        accounts: accountKeys,
        programIds: result.programIds
      };

    } catch (error) {
      this.logger.error(`❌ Error verifying trading transaction:`, error);
      return {
        isValid: false,
        error: `Trading verification failed: ${error.message}`
      };
    }
  }

  /**
   * Get transaction details for debugging
   */
  public async getTransactionDetails(signature: string): Promise<any> {
    try {
      const transaction = await this.rpcLoadBalancer.executeWithRetry(
        async (connection) => {
          return await connection.getTransaction(signature, {
            commitment: 'confirmed',
            maxSupportedTransactionVersion: 0
          });
        }
      );

      if (!transaction) {
        return null;
      }

      return {
        signature,
        slot: transaction.slot,
        blockTime: transaction.blockTime,
        fee: transaction.meta?.fee,
        success: !transaction.meta?.err,
        error: transaction.meta?.err,
        logs: transaction.meta?.logMessages || [],
        accounts: transaction.transaction.message.staticAccountKeys?.map(key => key.toString()) || [],
        instructions: transaction.transaction.message.compiledInstructions?.map(ix => ({
          programId: transaction.transaction.message.staticAccountKeys?.[ix.programIdIndex]?.toString(),
          accounts: ix.accountKeyIndexes?.map(accIndex => 
            transaction.transaction.message.staticAccountKeys?.[accIndex]?.toString()
          ) || [],
          data: ix.data
        })) || []
      };

    } catch (error) {
      this.logger.error(`❌ Error getting transaction details:`, error);
      return null;
    }
  }

  /**
   * Health check for the verification service
   */
  public async healthCheck(): Promise<boolean> {
    try {
      // Try to get the latest blockhash as a simple health check
      await this.rpcLoadBalancer.executeWithRetry(
        async (connection) => {
          return await connection.getLatestBlockhash();
        }
      );
      return true;
    } catch (error) {
      this.logger.error('❌ Transaction verification service health check failed:', error);
      return false;
    }
  }
}

export const transactionVerificationService = TransactionVerificationService.getInstance();
