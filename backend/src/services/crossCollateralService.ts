import { Logger } from '../utils/logger';
import { mcpSupabaseService } from './mcpSupabaseService';

export interface CollateralAccount {
  id: string;
  user_id: string;
  asset_type: CollateralType;
  amount: number;
  value_usd: number;
  last_updated: Date;
  is_active: boolean;
  // Additional fields
  price_usd?: number;
  utilization_rate?: number;
  available_amount?: number;
}

export enum CollateralType {
  SOL = 'SOL',
  USDC = 'USDC',
  BTC = 'BTC',
  ETH = 'ETH',
  USDT = 'USDT',
  AVAX = 'AVAX',
  MATIC = 'MATIC',
  ARB = 'ARB',
  OP = 'OP',
  DOGE = 'DOGE',
  ADA = 'ADA',
  DOT = 'DOT',
  LINK = 'LINK'
}

export interface CrossCollateralPosition {
  id: string;
  user_id: string;
  market_id: string;
  size: number;
  side: PositionSide;
  leverage: number;
  entry_price: number;
  margin: number;
  unrealized_pnl: number;
  created_at: Date;
  // Cross-collateralization fields
  collateral_accounts: string[];
  total_collateral_value: number;
  collateral_utilization: number;
  health_factor: number;
}

export enum PositionSide {
  LONG = 'long',
  SHORT = 'short'
}

export interface CollateralSwapRequest {
  from_asset: CollateralType;
  to_asset: CollateralType;
  amount: number;
  user_id: string;
}

export interface CollateralSwapResult {
  success: boolean;
  from_amount: number;
  to_amount: number;
  exchange_rate: number;
  fee: number;
  transaction_id?: string;
  error?: string;
}

export class CrossCollateralService {
  private static instance: CrossCollateralService;
  private db: typeof mcpSupabaseService;
  private logger: Logger;

  // Collateral configuration
  private readonly COLLATERAL_CONFIG = {
    [CollateralType.SOL]: { 
      max_ltv: 0.8, // 80% loan-to-value
      liquidation_threshold: 0.85, // 85% liquidation threshold
      price_feed_id: 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG' // SOL/USD
    },
    [CollateralType.USDC]: { 
      max_ltv: 0.95, // 95% loan-to-value
      liquidation_threshold: 0.97, // 97% liquidation threshold
      price_feed_id: 'Gnt27xtC473ZT2Mw5u8wZ68Z3gULkSTb5DuxJy7eJotD' // USDC/USD
    },
    [CollateralType.BTC]: { 
      max_ltv: 0.85, // 85% loan-to-value
      liquidation_threshold: 0.9, // 90% liquidation threshold
      price_feed_id: 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J' // BTC/USD
    },
    [CollateralType.ETH]: { 
      max_ltv: 0.85, // 85% loan-to-value
      liquidation_threshold: 0.9, // 90% liquidation threshold
      price_feed_id: 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB' // ETH/USD
    },
    [CollateralType.USDT]: { 
      max_ltv: 0.95, // 95% loan-to-value
      liquidation_threshold: 0.97, // 97% liquidation threshold
      price_feed_id: '3vxLXJqLqF3JG5TCbYycbKWRBbCJQLxQmBGCkyqEEefL' // USDT/USD
    },
    [CollateralType.AVAX]: { 
      max_ltv: 0.75, // 75% loan-to-value
      liquidation_threshold: 0.8, // 80% liquidation threshold
      price_feed_id: 'FVb5h1VmHPfVb1RfqZckchq18GpRv4i4F8GFQznz7Fc3' // AVAX/USD
    },
    [CollateralType.MATIC]: { 
      max_ltv: 0.7, // 70% loan-to-value
      liquidation_threshold: 0.75, // 75% liquidation threshold
      price_feed_id: '7KVswB9vkCgeM3SHP7aGDijvdRAHK8P5wi9JXViCrtYh' // MATIC/USD
    },
    [CollateralType.ARB]: { 
      max_ltv: 0.7, // 70% loan-to-value
      liquidation_threshold: 0.75, // 75% liquidation threshold
      price_feed_id: '5SSkXsEKQepHHAewytPVwdej4epN1nxgLVM84L4KXgy7' // ARB/USD
    },
    [CollateralType.OP]: { 
      max_ltv: 0.7, // 70% loan-to-value
      liquidation_threshold: 0.75, // 75% liquidation threshold
      price_feed_id: '4o4CUwzFwLqG5KnbTWMQcyJ4yJkKa7AF3b7C2ufv7w3Z' // OP/USD
    },
    [CollateralType.DOGE]: { 
      max_ltv: 0.6, // 60% loan-to-value
      liquidation_threshold: 0.65, // 65% liquidation threshold
      price_feed_id: '3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh' // DOGE/USD
    },
    [CollateralType.ADA]: { 
      max_ltv: 0.6, // 60% loan-to-value
      liquidation_threshold: 0.65, // 65% liquidation threshold
      price_feed_id: '3pyn4S5DunTAm8QivsJczt1WqzeWxAjUKf4y6Jk8sY7n' // ADA/USD
    },
    [CollateralType.DOT]: { 
      max_ltv: 0.65, // 65% loan-to-value
      liquidation_threshold: 0.7, // 70% liquidation threshold
      price_feed_id: 'EcV1X1gY2yb4kXfHEqfeNhn9JCHfSyXAFU9w6cmHqF8n' // DOT/USD
    },
    [CollateralType.LINK]: { 
      max_ltv: 0.65, // 65% loan-to-value
      liquidation_threshold: 0.7, // 70% liquidation threshold
      price_feed_id: '8A2v9YoF8VxUY6LPp42Rf7D6wGSDi9jP6gT6n2m3K7Yh' // LINK/USD
    }
  };

  private constructor() {
    this.db = mcpSupabaseService;
    this.logger = new Logger();
  }

  public static getInstance(): CrossCollateralService {
    if (!CrossCollateralService.instance) {
      CrossCollateralService.instance = new CrossCollateralService();
    }
    return CrossCollateralService.instance;
  }

  /**
   * Initialize a collateral account for a user
   */
  public async initializeCollateralAccount(
    userId: string, 
    assetType: CollateralType, 
    initialAmount: number
  ): Promise<CollateralAccount> {
    try {
      // Validate input
      if (initialAmount <= 0) {
        throw new Error('Initial amount must be positive');
      }

      // Get current price for the asset
      const currentPrice = await this.getAssetPrice(assetType);
      const valueUsd = initialAmount * currentPrice;

      // Create collateral account in database
      const collateralAccount = await this.createCollateralAccountInDatabase({
        user_id: userId,
        asset_type: assetType,
        amount: initialAmount,
        value_usd: valueUsd,
        last_updated: new Date(),
        is_active: true,
        price_usd: currentPrice,
        utilization_rate: 0,
        available_amount: initialAmount
      });

      this.logger.info(`✅ Collateral account initialized: ${initialAmount} ${assetType} ($${valueUsd.toFixed(2)}) for user ${userId}`);
      return collateralAccount;

    } catch (error) {
      this.logger.error('❌ Error initializing collateral account:', error);
      throw error;
    }
  }

  /**
   * Add collateral to an existing account
   */
  public async addCollateral(
    accountId: string, 
    amount: number, 
    userId: string
  ): Promise<CollateralAccount> {
    try {
      // Validate input
      if (amount <= 0) {
        throw new Error('Amount must be positive');
      }

      // Get existing collateral account
      const existingAccount = await this.getCollateralAccountById(accountId);
      if (!existingAccount) {
        throw new Error('Collateral account not found');
      }

      if (existingAccount.user_id !== userId) {
        throw new Error('Unauthorized to modify this collateral account');
      }

      // Get current price
      const currentPrice = await this.getAssetPrice(existingAccount.asset_type);
      const additionalValueUsd = amount * currentPrice;

      // Update collateral account
      const updatedAccount = await this.updateCollateralAccount(accountId, {
        amount: existingAccount.amount + amount,
        value_usd: existingAccount.value_usd + additionalValueUsd,
        last_updated: new Date(),
        price_usd: currentPrice,
        available_amount: existingAccount.available_amount! + amount
      });

      this.logger.info(`✅ Added ${amount} ${existingAccount.asset_type} ($${additionalValueUsd.toFixed(2)}) to collateral account ${accountId}`);
      return updatedAccount;

    } catch (error) {
      this.logger.error('❌ Error adding collateral:', error);
      throw error;
    }
  }

  /**
   * Remove collateral from an account
   */
  public async removeCollateral(
    accountId: string, 
    amount: number, 
    userId: string
  ): Promise<CollateralAccount> {
    try {
      // Validate input
      if (amount <= 0) {
        throw new Error('Amount must be positive');
      }

      // Get existing collateral account
      const existingAccount = await this.getCollateralAccountById(accountId);
      if (!existingAccount) {
        throw new Error('Collateral account not found');
      }

      if (existingAccount.user_id !== userId) {
        throw new Error('Unauthorized to modify this collateral account');
      }

      if (existingAccount.available_amount! < amount) {
        throw new Error('Insufficient available collateral');
      }

      // Check if removing this amount would violate collateral requirements
      await this.validateCollateralRemoval(existingAccount, amount);

      // Get current price
      const currentPrice = await this.getAssetPrice(existingAccount.asset_type);
      const removedValueUsd = amount * currentPrice;

      // Update collateral account
      const updatedAccount = await this.updateCollateralAccount(accountId, {
        amount: existingAccount.amount - amount,
        value_usd: existingAccount.value_usd - removedValueUsd,
        last_updated: new Date(),
        price_usd: currentPrice,
        available_amount: existingAccount.available_amount! - amount
      });

      this.logger.info(`✅ Removed ${amount} ${existingAccount.asset_type} ($${removedValueUsd.toFixed(2)}) from collateral account ${accountId}`);
      return updatedAccount;

    } catch (error) {
      this.logger.error('❌ Error removing collateral:', error);
      throw error;
    }
  }

  /**
   * Get user's total collateral portfolio
   */
  public async getUserCollateralPortfolio(userId: string): Promise<{
    total_value_usd: number;
    total_available_usd: number;
    total_utilized_usd: number;
    utilization_rate: number;
    health_factor: number;
    collateral_accounts: CollateralAccount[];
  }> {
    try {
      const collateralAccounts = await this.getUserCollateralAccounts(userId);
      
      let totalValueUsd = 0;
      let totalAvailableUsd = 0;
      let totalUtilizedUsd = 0;

      for (const account of collateralAccounts) {
        totalValueUsd += account.value_usd;
        totalAvailableUsd += account.available_amount!;
        totalUtilizedUsd += account.value_usd - account.available_amount!;
      }

      const utilizationRate = totalValueUsd > 0 ? totalUtilizedUsd / totalValueUsd : 0;
      const healthFactor = totalValueUsd > 0 ? totalValueUsd / (totalUtilizedUsd + 1) : 0;

      return {
        total_value_usd: totalValueUsd,
        total_available_usd: totalAvailableUsd,
        total_utilized_usd: totalUtilizedUsd,
        utilization_rate: utilizationRate,
        health_factor: healthFactor,
        collateral_accounts: collateralAccounts
      };

    } catch (error) {
      this.logger.error('❌ Error getting user collateral portfolio:', error);
      throw error;
    }
  }

  /**
   * Swap collateral between different assets
   */
  public async swapCollateral(request: CollateralSwapRequest): Promise<CollateralSwapResult> {
    try {
      // Validate swap request
      if (request.from_asset === request.to_asset) {
        throw new Error('Cannot swap to the same asset');
      }

      if (request.amount <= 0) {
        throw new Error('Swap amount must be positive');
      }

      // Get user's collateral accounts
      const fromAccount = await this.getUserCollateralAccountByAsset(request.user_id, request.from_asset);
      if (!fromAccount) {
        throw new Error(`No ${request.from_asset} collateral account found`);
      }

      if (fromAccount.available_amount! < request.amount) {
        throw new Error(`Insufficient ${request.from_asset} collateral`);
      }

      // Get current prices
      const fromPrice = await this.getAssetPrice(request.from_asset);
      const toPrice = await this.getAssetPrice(request.to_asset);

      // Calculate swap amounts
      const fromValueUsd = request.amount * fromPrice;
      const exchangeRate = fromPrice / toPrice;
      const toAmount = fromValueUsd / toPrice;
      const fee = fromValueUsd * 0.001; // 0.1% fee
      const toAmountAfterFee = toAmount - (fee / toPrice);

      // Execute the swap
      await this.removeCollateral(fromAccount.id, request.amount, request.user_id);
      
      // Add to destination account or create new one
      const toAccount = await this.getUserCollateralAccountByAsset(request.user_id, request.to_asset);
      if (toAccount) {
        await this.addCollateral(toAccount.id, toAmountAfterFee, request.user_id);
      } else {
        await this.initializeCollateralAccount(request.user_id, request.to_asset, toAmountAfterFee);
      }

      this.logger.info(`✅ Collateral swap completed: ${request.amount} ${request.from_asset} → ${toAmountAfterFee.toFixed(6)} ${request.to_asset}`);

      return {
        success: true,
        from_amount: request.amount,
        to_amount: toAmountAfterFee,
        exchange_rate: exchangeRate,
        fee: fee,
        transaction_id: `swap_${Date.now()}`
      };

    } catch (error) {
      this.logger.error('❌ Error swapping collateral:', error);
      return {
        success: false,
        from_amount: 0,
        to_amount: 0,
        exchange_rate: 0,
        fee: 0,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Calculate maximum borrowable amount across all collateral
   */
  public async calculateMaxBorrowableAmount(userId: string): Promise<number> {
    try {
      const portfolio = await this.getUserCollateralPortfolio(userId);
      let maxBorrowableUsd = 0;

      for (const account of portfolio.collateral_accounts) {
        const config = this.COLLATERAL_CONFIG[account.asset_type];
        const maxBorrowableForAsset = account.value_usd * config.max_ltv;
        maxBorrowableUsd += maxBorrowableForAsset;
      }

      return maxBorrowableUsd;

    } catch (error) {
      this.logger.error('❌ Error calculating max borrowable amount:', error);
      throw error;
    }
  }

  /**
   * Update collateral values using current market prices
   */
  public async updateCollateralValues(): Promise<void> {
    try {
      const collateralAccounts = await this.getAllCollateralAccounts();
      
      for (const account of collateralAccounts) {
        try {
          const currentPrice = await this.getAssetPrice(account.asset_type);
          const newValueUsd = account.amount * currentPrice;
          
          await this.updateCollateralAccount(account.id, {
            value_usd: newValueUsd,
            price_usd: currentPrice,
            last_updated: new Date()
          });

          this.logger.info(`💰 Updated ${account.asset_type} collateral value: $${newValueUsd.toFixed(2)}`);
        } catch (error) {
          this.logger.error(`❌ Error updating ${account.asset_type} collateral value:`, error);
        }
      }

    } catch (error) {
      this.logger.error('❌ Error updating collateral values:', error);
      throw error;
    }
  }

  // Private helper methods

  private async getAssetPrice(assetType: CollateralType): Promise<number> {
    try {
      // In production, this would fetch from Pyth or other oracle
      // For now, return mock prices
      const mockPrices = {
        [CollateralType.SOL]: 100,
        [CollateralType.USDC]: 1,
        [CollateralType.BTC]: 45000,
        [CollateralType.ETH]: 3000,
        [CollateralType.USDT]: 1
      };

      return mockPrices[assetType];

    } catch (error) {
      this.logger.error('❌ Error fetching asset price:', error);
      throw error;
    }
  }

  private async createCollateralAccountInDatabase(accountData: Partial<CollateralAccount>): Promise<CollateralAccount> {
    try {
      const result = await this.db.executeQuery(`
        INSERT INTO collateral_accounts (
          user_id, asset_type, amount, value_usd, last_updated, is_active,
          price_usd, utilization_rate, available_amount
        ) VALUES (
          $1, $2, $3, $4, $5, $6, $7, $8, $9
        ) RETURNING *
      `, [
        accountData.user_id, accountData.asset_type, accountData.amount,
        accountData.value_usd, accountData.last_updated, accountData.is_active,
        accountData.price_usd, accountData.utilization_rate, accountData.available_amount
      ]);

      return this.mapDbCollateralAccountToCollateralAccount(result[0]);

    } catch (error) {
      this.logger.error('❌ Error creating collateral account in database:', error);
      throw error;
    }
  }

  public async getCollateralAccountById(accountId: string): Promise<CollateralAccount | null> {
    try {
      const accounts = await this.db.executeQuery(
        'SELECT * FROM collateral_accounts WHERE id = $1',
        [accountId]
      );

      if (accounts.length === 0) {
        return null;
      }

      return this.mapDbCollateralAccountToCollateralAccount(accounts[0]);

    } catch (error) {
      this.logger.error('❌ Error fetching collateral account by ID:', error);
      throw error;
    }
  }

  private async getUserCollateralAccounts(userId: string): Promise<CollateralAccount[]> {
    try {
      const accounts = await this.db.executeQuery(
        'SELECT * FROM collateral_accounts WHERE user_id = $1 AND is_active = true ORDER BY asset_type',
        [userId]
      );

      return accounts.map(this.mapDbCollateralAccountToCollateralAccount);

    } catch (error) {
      this.logger.error('❌ Error fetching user collateral accounts:', error);
      throw error;
    }
  }

  private async getUserCollateralAccountByAsset(userId: string, assetType: CollateralType): Promise<CollateralAccount | null> {
    try {
      const accounts = await this.db.executeQuery(
        'SELECT * FROM collateral_accounts WHERE user_id = $1 AND asset_type = $2 AND is_active = true',
        [userId, assetType]
      );

      if (accounts.length === 0) {
        return null;
      }

      return this.mapDbCollateralAccountToCollateralAccount(accounts[0]);

    } catch (error) {
      this.logger.error('❌ Error fetching user collateral account by asset:', error);
      throw error;
    }
  }

  private async getAllCollateralAccounts(): Promise<CollateralAccount[]> {
    try {
      const accounts = await this.db.executeQuery(
        'SELECT * FROM collateral_accounts WHERE is_active = true'
      );

      return accounts.map(this.mapDbCollateralAccountToCollateralAccount);

    } catch (error) {
      this.logger.error('❌ Error fetching all collateral accounts:', error);
      throw error;
    }
  }

  private async updateCollateralAccount(accountId: string, updates: Partial<CollateralAccount>): Promise<CollateralAccount> {
    try {
      const result = await this.db.executeQuery(`
        UPDATE collateral_accounts 
        SET amount = COALESCE($1, amount),
            value_usd = COALESCE($2, value_usd),
            last_updated = COALESCE($3, last_updated),
            price_usd = COALESCE($4, price_usd),
            utilization_rate = COALESCE($5, utilization_rate),
            available_amount = COALESCE($6, available_amount)
        WHERE id = $7
        RETURNING *
      `, [
        updates.amount, updates.value_usd, updates.last_updated,
        updates.price_usd, updates.utilization_rate, updates.available_amount,
        accountId
      ]);

      return this.mapDbCollateralAccountToCollateralAccount(result[0]);

    } catch (error) {
      this.logger.error('❌ Error updating collateral account:', error);
      throw error;
    }
  }

  private async validateCollateralRemoval(account: CollateralAccount, amount: number): Promise<void> {
    // In production, this would check if removing this amount would violate
    // collateral requirements for existing positions
    // For now, we'll do a simple check
    if (account.available_amount! < amount) {
      throw new Error('Insufficient available collateral');
    }
  }

  private mapDbCollateralAccountToCollateralAccount(dbAccount: any): CollateralAccount {
    return {
      id: dbAccount.id,
      user_id: dbAccount.user_id,
      asset_type: dbAccount.asset_type,
      amount: parseFloat(dbAccount.amount),
      value_usd: parseFloat(dbAccount.value_usd),
      last_updated: new Date(dbAccount.last_updated),
      is_active: dbAccount.is_active,
      price_usd: dbAccount.price_usd ? parseFloat(dbAccount.price_usd) : undefined,
      utilization_rate: dbAccount.utilization_rate ? parseFloat(dbAccount.utilization_rate) : undefined,
      available_amount: dbAccount.available_amount ? parseFloat(dbAccount.available_amount) : undefined
    };
  }
}

export const crossCollateralService = CrossCollateralService.getInstance();
