import { 
  Connection, 
  PublicKey, 
  Transaction, 
  TransactionInstruction,
  SystemProgram,
  SYSVAR_RENT_PUBKEY,
  AccountInfo
} from '@solana/web3.js';
import { DatabaseService } from './database';
import { Logger } from '../utils/logger';
import { config } from '../config/environment';

const logger = new Logger();

export interface SolanaAccount {
  publicKey: PublicKey;
  accountInfo: AccountInfo<Buffer>;
}

export interface TransactionResult {
  signature: string;
  success: boolean;
  error?: string;
}

export class SolanaService {
  private static instance: SolanaService;
  private connection: Connection;
  private db: DatabaseService;
  private programId: PublicKey;

  private constructor() {
    this.connection = new Connection(config.RPC_URL, 'confirmed');
    this.db = DatabaseService.getInstance();
    this.programId = new PublicKey(config.PROGRAM_ID);
  }

  public static getInstance(): SolanaService {
    if (!SolanaService.instance) {
      SolanaService.instance = new SolanaService();
    }
    return SolanaService.instance;
  }

  /**
   * Get account information
   */
  public async getAccountInfo(address: string): Promise<AccountInfo<Buffer> | null> {
    try {
      const publicKey = new PublicKey(address);
      const accountInfo = await this.connection.getAccountInfo(publicKey);
      return accountInfo;
    } catch (error) {
      logger.error(`Error getting account info for ${address}:`, error);
      return null;
    }
  }

  /**
   * Get multiple account information
   */
  public async getMultipleAccountsInfo(addresses: string[]): Promise<(AccountInfo<Buffer> | null)[]> {
    try {
      const publicKeys = addresses.map(addr => new PublicKey(addr));
      const accountsInfo = await this.connection.getMultipleAccountsInfo(publicKeys);
      return accountsInfo;
    } catch (error) {
      logger.error('Error getting multiple accounts info:', error);
      return [];
    }
  }

  /**
   * Get program accounts
   */
  public async getProgramAccounts(filters?: any[]): Promise<SolanaAccount[]> {
    try {
      const accounts = await this.connection.getProgramAccounts(this.programId, {
        filters: filters || []
      });

      return accounts.map(account => ({
        publicKey: account.pubkey,
        accountInfo: account.account
      }));
    } catch (error) {
      logger.error('Error getting program accounts:', error);
      return [];
    }
  }

  /**
   * Get market accounts
   */
  public async getMarketAccounts(): Promise<SolanaAccount[]> {
    try {
      const accounts = await this.getProgramAccounts([
        {
          dataSize: 200 // Adjust based on your Market struct size
        }
      ]);

      return accounts.filter(account => {
        // Filter for market accounts based on your program logic
        // This is a simplified example
        return account.accountInfo.data.length > 100;
      });
    } catch (error) {
      logger.error('Error getting market accounts:', error);
      return [];
    }
  }

  /**
   * Get position accounts for a user
   */
  public async getUserPositions(userWallet: string): Promise<SolanaAccount[]> {
    try {
      const userPublicKey = new PublicKey(userWallet);
      
      const accounts = await this.getProgramAccounts([
        {
          dataSize: 150 // Adjust based on your Position struct size
        }
      ]);

      return accounts.filter(account => {
        // Filter for user's position accounts
        // This would need to be implemented based on your program's PDA structure
        return true; // Simplified for now
      });
    } catch (error) {
      logger.error(`Error getting positions for user ${userWallet}:`, error);
      return [];
    }
  }

  /**
   * Get order accounts for a user
   */
  public async getUserOrders(userWallet: string): Promise<SolanaAccount[]> {
    try {
      const userPublicKey = new PublicKey(userWallet);
      
      const accounts = await this.getProgramAccounts([
        {
          dataSize: 200 // Adjust based on your Order struct size
        }
      ]);

      return accounts.filter(account => {
        // Filter for user's order accounts
        return true; // Simplified for now
      });
    } catch (error) {
      logger.error(`Error getting orders for user ${userWallet}:`, error);
      return [];
    }
  }

  /**
   * Sync on-chain data with database
   */
  public async syncOnChainData(): Promise<void> {
    try {
      logger.info('Starting on-chain data sync...');

      // Sync markets
      await this.syncMarkets();

      // Sync positions
      await this.syncPositions();

      // Sync orders
      await this.syncOrders();

      logger.info('On-chain data sync completed');
    } catch (error) {
      logger.error('Error syncing on-chain data:', error);
    }
  }

  /**
   * Sync market data from blockchain
   */
  private async syncMarkets(): Promise<void> {
    try {
      const marketAccounts = await this.getMarketAccounts();
      
      for (const account of marketAccounts) {
        // Parse market data from account
        const marketData = this.parseMarketAccount(account);
        if (marketData) {
          // Update database
          await this.updateMarketInDatabase(marketData);
        }
      }
    } catch (error) {
      logger.error('Error syncing markets:', error);
    }
  }

  /**
   * Sync position data from blockchain
   */
  private async syncPositions(): Promise<void> {
    try {
      // Get all users from database
      const users = await this.db.query('SELECT wallet_address FROM users');
      
      for (const user of users.rows) {
        const positions = await this.getUserPositions(user.wallet_address);
        
        for (const position of positions) {
          const positionData = this.parsePositionAccount(position);
          if (positionData) {
            await this.updatePositionInDatabase(positionData);
          }
        }
      }
    } catch (error) {
      logger.error('Error syncing positions:', error);
    }
  }

  /**
   * Sync order data from blockchain
   */
  private async syncOrders(): Promise<void> {
    try {
      // Get all users from database
      const users = await this.db.query('SELECT wallet_address FROM users');
      
      for (const user of users.rows) {
        const orders = await this.getUserOrders(user.wallet_address);
        
        for (const order of orders) {
          const orderData = this.parseOrderAccount(order);
          if (orderData) {
            await this.updateOrderInDatabase(orderData);
          }
        }
      }
    } catch (error) {
      logger.error('Error syncing orders:', error);
    }
  }

  /**
   * Parse market account data
   */
  private parseMarketAccount(account: SolanaAccount): any {
    try {
      // This would parse the account data based on your Market struct
      // For now, return a simplified structure
      return {
        account: account.publicKey.toString(),
        data: account.accountInfo.data
      };
    } catch (error) {
      logger.error('Error parsing market account:', error);
      return null;
    }
  }

  /**
   * Parse position account data
   */
  private parsePositionAccount(account: SolanaAccount): any {
    try {
      // This would parse the account data based on your Position struct
      return {
        account: account.publicKey.toString(),
        data: account.accountInfo.data
      };
    } catch (error) {
      logger.error('Error parsing position account:', error);
      return null;
    }
  }

  /**
   * Parse order account data
   */
  private parseOrderAccount(account: SolanaAccount): any {
    try {
      // This would parse the account data based on your Order struct
      return {
        account: account.publicKey.toString(),
        data: account.accountInfo.data
      };
    } catch (error) {
      logger.error('Error parsing order account:', error);
      return null;
    }
  }

  /**
   * Update market in database
   */
  private async updateMarketInDatabase(marketData: any): Promise<void> {
    try {
      // Update market data in database
      // This would map the on-chain data to your database schema
      logger.info(`Updating market ${marketData.account} in database`);
    } catch (error) {
      logger.error('Error updating market in database:', error);
    }
  }

  /**
   * Update position in database
   */
  private async updatePositionInDatabase(positionData: any): Promise<void> {
    try {
      // Update position data in database
      logger.info(`Updating position ${positionData.account} in database`);
    } catch (error) {
      logger.error('Error updating position in database:', error);
    }
  }

  /**
   * Update order in database
   */
  private async updateOrderInDatabase(orderData: any): Promise<void> {
    try {
      // Update order data in database
      logger.info(`Updating order ${orderData.account} in database`);
    } catch (error) {
      logger.error('Error updating order in database:', error);
    }
  }

  /**
   * Get transaction history for an account
   */
  public async getTransactionHistory(address: string, limit: number = 10): Promise<any[]> {
    try {
      const publicKey = new PublicKey(address);
      const signatures = await this.connection.getSignaturesForAddress(publicKey, { limit });
      
      const transactions = [];
      for (const sig of signatures) {
        const tx = await this.connection.getTransaction(sig.signature);
        if (tx) {
          transactions.push({
            signature: sig.signature,
            slot: sig.slot,
            blockTime: sig.blockTime,
            transaction: tx
          });
        }
      }
      
      return transactions;
    } catch (error) {
      logger.error(`Error getting transaction history for ${address}:`, error);
      return [];
    }
  }

  /**
   * Get current slot
   */
  public async getCurrentSlot(): Promise<number> {
    try {
      return await this.connection.getSlot();
    } catch (error) {
      logger.error('Error getting current slot:', error);
      return 0;
    }
  }

  /**
   * Get cluster info
   */
  public async getClusterInfo(): Promise<any> {
    try {
      const info = await this.connection.getClusterInfo();
      return info;
    } catch (error) {
      logger.error('Error getting cluster info:', error);
      return null;
    }
  }

  /**
   * Health check
   */
  public async healthCheck(): Promise<boolean> {
    try {
      const slot = await this.getCurrentSlot();
      return slot > 0;
    } catch (error) {
      logger.error('Solana health check failed:', error);
      return false;
    }
  }

  /**
   * Start periodic sync
   */
  public startPeriodicSync(intervalMs: number = 30000): void {
    setInterval(async () => {
      try {
        await this.syncOnChainData();
      } catch (error) {
        logger.error('Error in periodic sync:', error);
      }
    }, intervalMs);

    logger.info(`Started periodic sync every ${intervalMs}ms`);
  }
}
