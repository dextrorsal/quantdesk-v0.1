import { Connection, PublicKey, Transaction, Keypair } from '@solana/web3.js';
import { config } from '@/config';
import { SecurityUtils } from '@/utils/security';
import { systemLogger, errorLogger } from '@/utils/logger';
import { SolanaConfig, WalletData, TransactionData } from '@/types';

/**
 * Secure Solana service for blockchain interactions
 * Implements best practices for connection management and data handling
 */
export class SolanaService {
  private connection!: Connection;
  private keypair: Keypair | null = null;
  private config: SolanaConfig;

  constructor() {
    this.config = config.solana;
    this.initializeConnection();
    this.initializeKeypair();
  }

  /**
   * Initialize secure Solana connection
   */
  private initializeConnection(): void {
    try {
      this.connection = new Connection(this.config.rpcUrl, {
        commitment: 'confirmed',
        wsEndpoint: this.config.wsUrl,
        httpHeaders: {
          'User-Agent': 'Solana-DeFi-AI/1.0.0'
        }
      });

      systemLogger.databaseConnection('connected');
    } catch (error) {
      errorLogger.solanaError(error as Error, { action: 'connection_init' });
      throw new Error('Failed to initialize Solana connection');
    }
  }

  /**
   * Initialize keypair from environment variable
   */
  private initializeKeypair(): void {
    try {
      if (!this.config.privateKey) {
        systemLogger.startup('1.0.0', config.dev.nodeEnv);
        return;
      }

      // Validate private key format
      if (!SecurityUtils.isValidPrivateKey(this.config.privateKey)) {
        throw new Error('Invalid private key format');
      }

      // Convert base58 private key to keypair
      const privateKeyBytes = Buffer.from(this.config.privateKey, 'base64');
      this.keypair = Keypair.fromSecretKey(privateKeyBytes);

      systemLogger.startup('1.0.0', config.dev.nodeEnv);
    } catch (error) {
      errorLogger.solanaError(error as Error, { action: 'keypair_init' });
      throw new Error('Failed to initialize Solana keypair');
    }
  }

  /**
   * Get connection instance
   */
  getConnection(): Connection {
    return this.connection;
  }

  /**
   * Get keypair instance
   */
  getKeypair(): Keypair | null {
    return this.keypair;
  }

  /**
   * Get wallet public key
   */
  getPublicKey(): PublicKey | null {
    return this.keypair?.publicKey || null;
  }

  /**
   * Get wallet balance in SOL
   */
  async getBalance(address?: string): Promise<number> {
    try {
      const publicKey = address ? new PublicKey(address) : this.getPublicKey();
      
      if (!publicKey) {
        throw new Error('No public key available');
      }

      const balance = await this.connection.getBalance(publicKey);
      return balance / 1e9; // Convert lamports to SOL
    } catch (error) {
      errorLogger.solanaError(error as Error, { 
        action: 'get_balance',
        address: address ? SecurityUtils.maskSensitiveData(address) : 'current_wallet'
      });
      throw error;
    }
  }

  /**
   * Get wallet data with portfolio information
   */
  async getWalletInfo(address: string): Promise<WalletData> {
    try {
      const publicKey = new PublicKey(address);
      
      // Validate address format
      if (!SecurityUtils.isValidSolanaAddress(address)) {
        throw new Error('Invalid Solana address format');
      }

      const balance = await this.getBalance(address);
      
      // Get token accounts (simplified - in production, you'd want more detailed token data)
      const tokenAccounts = await this.connection.getParsedTokenAccountsByOwner(
        publicKey,
        { programId: new PublicKey('TokenkegQfeZyiNwAJbNbGKPFXCWuBvf9Ss623VQ5DA') }
      );

      const portfolio: { [token: string]: { amount: number; value: number; percentage: number } } = {};
      
      // Add SOL to portfolio
      portfolio['SOL'] = {
        amount: balance,
        value: balance, // Simplified - would need price data
        percentage: 100 // Simplified
      };

      // Process token accounts
      for (const account of tokenAccounts.value) {
        const tokenData = account.account.data.parsed.info;
        const mint = tokenData.mint;
        const amount = tokenData.tokenAmount.uiAmount || 0;
        
        if (amount > 0) {
          portfolio[mint] = {
            amount,
            value: amount, // Simplified - would need price data
            percentage: 0 // Would calculate based on total value
          };
        }
      }

      return {
        address,
        totalValue: balance, // Simplified
        portfolio,
        recentActivity: {
          transactions24h: 0, // Would need to query transaction history
          volume24h: 0,
          largestTransaction: 0
        }
      };
    } catch (error) {
      errorLogger.solanaError(error as Error, { 
        action: 'get_wallet_data',
        address: SecurityUtils.maskSensitiveData(address)
      });
      throw error;
    }
  }

  /**
   * Get recent transactions for a wallet
   */
  async getRecentTransactions(address: string, limit: number = 50): Promise<TransactionData[]> {
    try {
      const publicKey = new PublicKey(address);
      
      // Get recent transaction signatures
      const signatures = await this.connection.getSignaturesForAddress(publicKey, {
        limit
      });

      const transactions: TransactionData[] = [];

      // Process each signature
      for (const signature of signatures) {
        try {
          const tx = await this.connection.getTransaction(signature.signature, {
            commitment: 'confirmed',
            maxSupportedTransactionVersion: 0
          });

          if (tx) {
            const transactionData: TransactionData = {
              transactionId: signature.signature,
              timestamp: new Date(signature.blockTime! * 1000),
              type: 'transfer', // Simplified - would need more analysis
              gasFee: tx.meta?.fee || 0,
              walletAddress: address
            };

            transactions.push(transactionData);
          }
        } catch (txError) {
          // Skip failed transaction parsing
          continue;
        }
      }

      return transactions;
    } catch (error) {
      errorLogger.solanaError(error as Error, { 
        action: 'get_recent_transactions',
        address: SecurityUtils.maskSensitiveData(address)
      });
      throw error;
    }
  }

  /**
   * Monitor wallet for new transactions
   */
  async startWalletMonitoring(address: string, _callback: (tx: TransactionData) => void): Promise<() => void> {
    try {
      const publicKey = new PublicKey(address);
      
      // Subscribe to account changes
      const subscriptionId = this.connection.onAccountChange(
        publicKey,
        async (_accountInfo) => {
          // In a real implementation, you'd analyze the account changes
          // and determine if it's a new transaction
          console.log('Account change detected for:', SecurityUtils.maskSensitiveData(address));
        },
        'confirmed'
      );

      console.log(`Started monitoring wallet: ${SecurityUtils.maskSensitiveData(address)}`);
      
      // Return cleanup function
      return (): void => {
        this.connection.removeAccountChangeListener(subscriptionId);
      };
    } catch (error) {
      errorLogger.solanaError(error as Error, { 
        action: 'start_wallet_monitoring',
        address: SecurityUtils.maskSensitiveData(address)
      });
      throw error;
    }
  }

  /**
   * Get network health status
   */
  async getNetworkHealth(): Promise<{
    isHealthy: boolean;
    slot: number;
    blockHeight: number;
    epoch: number;
    cluster: string;
  }> {
    try {
      const [slot, blockHeight, epochInfo] = await Promise.all([
        this.connection.getSlot(),
        this.connection.getBlockHeight(),
        this.connection.getEpochInfo()
      ]);

      return {
        isHealthy: true,
        slot,
        blockHeight,
        epoch: epochInfo.epoch,
        cluster: this.config.cluster
      };
    } catch (error) {
      errorLogger.solanaError(error as Error, { action: 'get_network_health' });
      return {
        isHealthy: false,
        slot: 0,
        blockHeight: 0,
        epoch: 0,
        cluster: this.config.cluster
      };
    }
  }

  /**
   * Validate transaction before sending
   */
  async validateTransaction(transaction: Transaction): Promise<boolean> {
    try {
      // Check if transaction is properly signed
      if (!transaction.signature) {
        return false;
      }

      // Check transaction size
      const serialized = transaction.serialize();
      if (serialized.length > 1232) { // Solana transaction size limit
        return false;
      }

      // Simulate transaction
      const simulation = await this.connection.simulateTransaction(transaction);
      return !simulation.value.err;
    } catch (error) {
      errorLogger.solanaError(error as Error, { action: 'validate_transaction' });
      return false;
    }
  }

  /**
   * Send transaction with retry logic
   */
  async sendTransaction(transaction: Transaction, maxRetries: number = 3): Promise<string> {
    try {
      if (!this.keypair) {
        throw new Error('No keypair available for signing');
      }

      // Sign transaction
      transaction.sign(this.keypair);

      // Validate transaction
      const isValid = await this.validateTransaction(transaction);
      if (!isValid) {
        throw new Error('Transaction validation failed');
      }

      // Send with retry logic
      let lastError: Error | null = null;
      
      for (let attempt = 1; attempt <= maxRetries; attempt++) {
        try {
          const signature = await this.connection.sendRawTransaction(
            transaction.serialize(),
            {
              skipPreflight: false,
              preflightCommitment: 'confirmed'
            }
          );

          // Wait for confirmation
          await this.connection.confirmTransaction(signature, 'confirmed');
          
          return signature;
        } catch (error) {
          lastError = error as Error;
          
          if (attempt < maxRetries) {
            // Wait before retry
            await new Promise(resolve => setTimeout(resolve, 1000 * attempt));
          }
        }
      }

      throw lastError || new Error('Transaction failed after retries');
    } catch (error) {
      errorLogger.solanaError(error as Error, { action: 'send_transaction' });
      throw error;
    }
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    try {
      // Close any active subscriptions
      // In a real implementation, you'd track and close all subscriptions
      
      systemLogger.shutdown('Service cleanup');
    } catch (error) {
      errorLogger.solanaError(error as Error, { action: 'cleanup' });
    }
  }
}

// Export singleton instance
export const solanaService = new SolanaService();
