import axios from 'axios';
import { Wallet } from '@coral-xyz/anchor';
import { SmartContractService } from './smartContractService';

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

export interface CollateralAccount {
  id: string;
  user_id: string;
  asset_type: CollateralType;
  amount: number;
  value_usd: number;
  is_active: boolean;
  created_at: string;
  updated_at: string;
  // Smart contract fields
  account_address?: string;
  token_account?: string;
  vault_address?: string;
}

export interface CollateralPortfolio {
  total_collateral_value_usd: number;
  total_available_collateral_usd: number;
  collateral_utilization_rate: number;
  collateral_accounts: CollateralAccount[];
  // Smart contract fields
  user_account_address?: string;
  account_health?: number;
  liquidation_price?: number;
}

export interface CollateralSwapRequest {
  user_id: string;
  from_asset: CollateralType;
  to_asset: CollateralType;
  amount: number;
}

class CrossCollateralService {
  private static instance: CrossCollateralService;
  private readonly baseURL: string;
  private readonly smartContractService: SmartContractService;

  private constructor() {
    this.baseURL = import.meta.env.VITE_API_URL || 'http://localhost:3001';
    this.smartContractService = SmartContractService.getInstance();
  }

  public static getInstance(): CrossCollateralService {
    if (!CrossCollateralService.instance) {
      CrossCollateralService.instance = new CrossCollateralService();
    }
    return CrossCollateralService.instance;
  }

  /**
   * Initialize a collateral account for a user (backend + on-chain)
   */
  async initializeCollateralAccount(
    userId: string,
    assetType: CollateralType,
    initialAmount: number,
    wallet?: Wallet
  ): Promise<CollateralAccount> {
    try {
      // If wallet provided, also create on-chain account
      if (wallet) {
        console.log('🔗 Creating user account via smart contract service...');
        await this.smartContractService.createUserAccount(wallet);
        
        if (assetType === CollateralType.SOL) {
          console.log('💰 Depositing SOL via smart contract service...');
          await this.smartContractService.depositNativeSOL(wallet, initialAmount * 1e9);
        }
      }

      // Create backend collateral account
      const response = await axios.post(`${this.baseURL}/api/cross-collateral/initialize`, {
        user_id: userId,
        asset_type: assetType,
        initial_amount: initialAmount
      });

      if (response.data.success) {
        return response.data.data;
      } else {
        throw new Error(response.data.error || 'Failed to initialize collateral account');
      }
    } catch (error) {
      console.error('Error initializing collateral account:', error);
      throw error;
    }
  }

  /**
   * Add collateral to an existing account
   */
  async addCollateral(
    accountId: string,
    amount: number,
    userId: string,
    wallet?: Wallet
  ): Promise<CollateralAccount> {
    try {
      // If wallet provided, also deposit on-chain
      if (wallet) {
        console.log('💰 Depositing additional SOL via smart contract service...');
        await this.smartContractService.depositNativeSOL(wallet, amount * 1e9);
      }

      // Update backend collateral account
      const response = await axios.post(`${this.baseURL}/api/cross-collateral/add`, {
        account_id: accountId,
        amount: amount,
        user_id: userId
      });

      if (response.data.success) {
        return response.data.data;
      } else {
        throw new Error(response.data.error || 'Failed to add collateral');
      }
    } catch (error) {
      console.error('Error adding collateral:', error);
      throw error;
    }
  }

  /**
   * Remove collateral from an account
   */
  async removeCollateral(
    accountId: string,
    amount: number,
    userId: string
  ): Promise<CollateralAccount> {
    try {
      const response = await axios.post(`${this.baseURL}/api/cross-collateral/remove`, {
        account_id: accountId,
        amount: amount,
        user_id: userId
      });

      if (response.data.success) {
        return response.data.data;
      } else {
        throw new Error(response.data.error || 'Failed to remove collateral');
      }
    } catch (error) {
      console.error('Error removing collateral:', error);
      throw error;
    }
  }

  /**
   * Get user's collateral portfolio (backend + on-chain data)
   */
  async getUserCollateralPortfolio(userId: string, wallet?: Wallet): Promise<CollateralPortfolio> {
    try {
      // Get backend portfolio
      const response = await axios.get(`${this.baseURL}/api/cross-collateral/portfolio/${userId}`);

      if (response.data.success) {
        const portfolio = response.data.data;

        // If wallet provided, get on-chain data
        if (wallet) {
          try {
            console.log('🔍 Fetching on-chain user account data...');
            const userAccount = await this.smartContractService.getUserAccount(wallet.publicKey.toString());
            
            // Merge on-chain data with backend data
            portfolio.user_account_address = userAccount.userAccountPda;
            portfolio.account_health = userAccount.accountHealth;
            portfolio.liquidation_price = userAccount.liquidationPrice;
            
            console.log('✅ On-chain data merged:', {
              accountHealth: portfolio.account_health,
              liquidationPrice: portfolio.liquidation_price
            });
          } catch (error) {
            console.warn('Could not fetch on-chain user account:', error);
          }
        }

        return portfolio;
      } else {
        throw new Error(response.data.error || 'Failed to fetch collateral portfolio');
      }
    } catch (error) {
      console.error('Error fetching collateral portfolio:', error);
      throw error;
    }
  }

  /**
   * Swap collateral between different assets
   */
  async swapCollateral(request: CollateralSwapRequest): Promise<any> {
    try {
      const response = await axios.post(`${this.baseURL}/api/cross-collateral/swap`, request);

      if (response.data.success) {
        return response.data.data;
      } else {
        throw new Error(response.data.error || 'Failed to swap collateral');
      }
    } catch (error) {
      console.error('Error swapping collateral:', error);
      throw error;
    }
  }

  /**
   * Calculate maximum borrowable amount for a user
   */
  async calculateMaxBorrowableAmount(userId: string): Promise<number> {
    try {
      const response = await axios.get(`${this.baseURL}/api/cross-collateral/max-borrowable/${userId}`);

      if (response.data.success) {
        return response.data.data.max_borrowable_usd;
      } else {
        throw new Error(response.data.error || 'Failed to calculate max borrowable amount');
      }
    } catch (error) {
      console.error('Error calculating max borrowable amount:', error);
      throw error;
    }
  }

  /**
   * Get supported collateral types and their configurations
   */
  async getSupportedTypes(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/api/cross-collateral/types`);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching supported types:', error);
      throw error;
    }
  }

  /**
   * Get user account from smart contract
   */
  async getUserAccount(wallet: Wallet): Promise<any> {
    console.log('🔍 Delegating getUserAccount to smart contract service...');
    return await this.smartContractService.getUserAccount(wallet.publicKey.toString());
  }
}

export const crossCollateralService = CrossCollateralService.getInstance();
