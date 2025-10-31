import { Logger } from '../utils/logger';
import { SupabaseDatabaseService } from './supabaseDatabase';

export interface TradingAccount {
  id: string;
  name: string;
  accountIndex: number;
  isActive: boolean;
  createdAt: Date;
}

export interface UserBalance {
  asset: string;
  balance: number;
  lockedBalance: number;
  availableBalance: number;
  tradingAccountId?: string;
}

export interface AccountState {
  walletConnected: boolean;
  accountCreated: boolean;
  hasDeposits: boolean;
  canTrade: boolean;
  tradingAccounts: TradingAccount[];
  balances: UserBalance[];
  totalBalance: number;
  availableBalance: number;
  riskLevel: 'safe' | 'warning' | 'danger';
  accountHealth: number; // 0-100%
  liquidationPrice?: number;
}

export interface AccountStateResponse {
  success: boolean;
  state: AccountState;
  message?: string;
}

export class AccountStateService {
  private static instance: AccountStateService;
  private db: SupabaseDatabaseService;
  private logger: Logger;

  private constructor() {
    this.db = SupabaseDatabaseService.getInstance();
    this.logger = new Logger();
  }

  public static getInstance(): AccountStateService {
    if (!AccountStateService.instance) {
      AccountStateService.instance = new AccountStateService();
    }
    return AccountStateService.instance;
  }

  /**
   * Get complete account state for a user
   */
  async getUserAccountState(userId: string): Promise<AccountStateResponse> {
    try {
      this.logger.info(`Getting account state for user: ${userId}`);

      // Get user's trading accounts
      const tradingAccounts = await this.getTradingAccounts(userId);
      
      // Get user's balances
      const balances = await this.getUserBalances(userId);
      
      // Calculate total and available balances
      const totalBalance = balances.reduce((sum, balance) => sum + balance.balance, 0);
      const availableBalance = balances.reduce((sum, balance) => sum + balance.availableBalance, 0);
      
      // Determine account state
      const accountCreated = tradingAccounts.length > 0;
      const hasDeposits = totalBalance > 0;
      const canTrade = accountCreated && hasDeposits && availableBalance > 0;
      
      // Calculate risk level and health
      const { riskLevel, accountHealth, liquidationPrice } = await this.calculateRiskMetrics(userId, balances);

      const state: AccountState = {
        walletConnected: true, // If we're here, wallet is connected
        accountCreated,
        hasDeposits,
        canTrade,
        tradingAccounts,
        balances,
        totalBalance,
        availableBalance,
        riskLevel,
        accountHealth,
        liquidationPrice
      };

      this.logger.info(`Account state calculated for user ${userId}:`, {
        accountCreated,
        hasDeposits,
        canTrade,
        totalBalance,
        availableBalance,
        riskLevel
      });

      return {
        success: true,
        state,
        message: this.getStateMessage(state)
      };

    } catch (error) {
      this.logger.error('Error getting account state:', error);
      throw error;
    }
  }

  /**
   * Get user's trading accounts
   */
  private async getTradingAccounts(userId: string): Promise<TradingAccount[]> {
    try {
      const result = await this.db.select('trading_accounts', 'id, name, account_index, is_active, created_at', { 
        master_account_id: userId, 
        is_active: true 
      });

      return result.map(row => ({
        id: row.id,
        name: row.name,
        accountIndex: row.account_index,
        isActive: row.is_active,
        createdAt: row.created_at
      }));
    } catch (error) {
      this.logger.error('Error getting trading accounts:', error);
      return [];
    }
  }

  /**
   * Get user's balances across all accounts
   */
  private async getUserBalances(userId: string): Promise<UserBalance[]> {
    try {
      const result = await this.db.select('user_balances', 'asset, balance, locked_balance, available_balance, trading_account_id', { 
        user_id: userId, 
        balance: { gt: 0 } 
      });

      return result.map(row => ({
        asset: row.asset,
        balance: parseFloat(row.balance),
        lockedBalance: parseFloat(row.locked_balance),
        availableBalance: parseFloat(row.available_balance),
        tradingAccountId: row.trading_account_id
      }));
    } catch (error) {
      this.logger.error('Error getting user balances:', error);
      return [];
    }
  }

  /**
   * Calculate risk metrics for the account
   */
  private async calculateRiskMetrics(userId: string, balances: UserBalance[]): Promise<{
    riskLevel: 'safe' | 'warning' | 'danger';
    accountHealth: number;
    liquidationPrice?: number;
  }> {
    try {
      // Get user's open positions
      const positionsResult = await this.db.select('positions', 'symbol, side, size, entry_price, current_price, leverage, margin_used', { 
        user_id: userId, 
        status: 'open' 
      });

      const positions = positionsResult;
      
      if (positions.length === 0) {
        return {
          riskLevel: 'safe',
          accountHealth: 100,
          liquidationPrice: undefined
        };
      }

      // Calculate total margin used
      const totalMarginUsed = positions.reduce((sum, pos) => sum + parseFloat(pos.margin_used), 0);
      const totalBalance = balances.reduce((sum, balance) => sum + balance.balance, 0);
      
      // Calculate margin ratio
      const marginRatio = totalBalance > 0 ? (totalMarginUsed / totalBalance) * 100 : 0;
      
      // Determine risk level
      let riskLevel: 'safe' | 'warning' | 'danger';
      let accountHealth: number;
      
      if (marginRatio < 50) {
        riskLevel = 'safe';
        accountHealth = 100 - marginRatio;
      } else if (marginRatio < 80) {
        riskLevel = 'warning';
        accountHealth = 100 - marginRatio;
      } else {
        riskLevel = 'danger';
        accountHealth = Math.max(0, 100 - marginRatio);
      }

      // Calculate liquidation price (simplified)
      const liquidationPrice = this.calculateLiquidationPrice(positions);

      return {
        riskLevel,
        accountHealth: Math.max(0, Math.min(100, accountHealth)),
        liquidationPrice
      };

    } catch (error) {
      this.logger.error('Error calculating risk metrics:', error);
      return {
        riskLevel: 'safe',
        accountHealth: 100
      };
    }
  }

  /**
   * Calculate liquidation price for positions
   */
  private calculateLiquidationPrice(positions: any[]): number | undefined {
    if (positions.length === 0) return undefined;
    
    // Simplified liquidation price calculation
    // In a real implementation, this would be more complex
    const totalNotional = positions.reduce((sum, pos) => {
      return sum + (parseFloat(pos.size) * parseFloat(pos.current_price));
    }, 0);
    
    const totalMargin = positions.reduce((sum, pos) => {
      return sum + parseFloat(pos.margin_used);
    }, 0);
    
    // Simplified liquidation price calculation
    return totalNotional > 0 ? totalMargin / totalNotional : undefined;
  }

  /**
   * Get appropriate message for account state
   */
  private getStateMessage(state: AccountState): string {
    if (!state.accountCreated) {
      return 'Please create a trading account to start trading';
    }
    
    if (!state.hasDeposits) {
      return 'Please deposit funds to start trading';
    }
    
    if (!state.canTrade) {
      return 'Insufficient balance to trade';
    }
    
    if (state.riskLevel === 'danger') {
      return 'Account at risk of liquidation';
    }
    
    if (state.riskLevel === 'warning') {
      return 'Account margin is high, consider reducing positions';
    }
    
    return 'Account ready for trading';
  }

  /**
   * Check if user can perform a specific action
   */
  async canUserPerformAction(userId: string, action: 'deposit' | 'withdraw' | 'trade' | 'create_account'): Promise<boolean> {
    try {
      const stateResponse = await this.getUserAccountState(userId);
      const state = stateResponse.state;

      switch (action) {
        case 'create_account':
          return !state.accountCreated;
        
        case 'deposit':
          return state.accountCreated;
        
        case 'withdraw':
          return state.accountCreated && state.hasDeposits;
        
        case 'trade':
          return state.canTrade;
        
        default:
          return false;
      }
    } catch (error) {
      this.logger.error('Error checking user action permission:', error);
      return false;
    }
  }

  /**
   * Get account state summary for dashboard
   */
  async getAccountSummary(userId: string): Promise<{
    totalBalance: number;
    availableBalance: number;
    openPositions: number;
    accountHealth: number;
    riskLevel: string;
  }> {
    try {
      const stateResponse = await this.getUserAccountState(userId);
      const state = stateResponse.state;

      // Get open positions count
      const openPositions = await this.db.count('positions', { 
        user_id: userId, 
        status: 'open' 
      });

      return {
        totalBalance: state.totalBalance,
        availableBalance: state.availableBalance,
        openPositions,
        accountHealth: state.accountHealth,
        riskLevel: state.riskLevel
      };
    } catch (error) {
      this.logger.error('Error getting account summary:', error);
      throw error;
    }
  }
}

// Export singleton instance
export const accountStateService = AccountStateService.getInstance();
