import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode, useMemo } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { SmartContractService, UserAccountState, CollateralAccount, Position, Order, CollateralType, OrderType, PositionSide } from '../services/smartContractService';

// Account Context Types
interface AccountContextType {
  // Wallet
  wallet: any; // Wallet from useWallet hook
  
  // State
  accountState: UserAccountState | null;
  collateralAccounts: CollateralAccount[];
  positions: Position[];
  orders: Order[];
  loading: boolean;
  error: string | null;

  // Actions
  fetchAccountState: () => Promise<void>;
  createAccount: () => Promise<string>;
  depositCollateral: (assetType: CollateralType, amount: number) => Promise<string>;
  placeOrder: (market: string, orderType: OrderType, side: PositionSide, size: number, price: number, leverage: number) => Promise<string>;
  refreshData: () => Promise<void>;

  // Computed properties
  canDeposit: boolean;
  canTrade: boolean;
  totalBalance: number;
  accountHealth: number;
  isAtRisk: boolean;
}

// Create context
const AccountContext = createContext<AccountContextType | undefined>(undefined);

// Account Provider Component
interface AccountProviderProps {
  children: ReactNode;
}

export const AccountProvider: React.FC<AccountProviderProps> = ({ children }) => {
  const { connected, publicKey, wallet } = useWallet();
  
  // State
  const [accountState, setAccountState] = useState<UserAccountState | null>(null);
  const [collateralAccounts, setCollateralAccounts] = useState<CollateralAccount[]>([]);
  const [positions, setPositions] = useState<Position[]>([]);
  const [orders, setOrders] = useState<Order[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  // Helper method to populate collateral accounts with smart contract data
  const populateCollateralAccounts = useCallback(async (walletAddress: string, accountState: UserAccountState) => {
    try {
      console.log('🔍 AccountContext: Populating collateral accounts...');
      
      const collateralAccounts: CollateralAccount[] = [];
      
      // Get SOL collateral balance from smart contract
      if (accountState.totalCollateral > 0) {
        const solCollateralBalance = await SmartContractService.getInstance().getSOLCollateralBalance(walletAddress);
        
        if (solCollateralBalance > 0) {
          // Get current SOL price from oracle service
          let solPrice = 190; // Default fallback price (align with smartContractService)
          try {
            // Try to get real SOL price from backend oracle
            const response = await fetch(`${import.meta.env.VITE_API_URL || 'http://localhost:3002'}/api/oracle/price/SOL`);
            if (response.ok) {
              const priceData = await response.json();
              solPrice = priceData.price || 190;
              console.log('✅ AccountContext: Got real SOL price from oracle:', solPrice);
            }
          } catch (oracleError) {
            console.warn('⚠️ AccountContext: Could not get SOL price from oracle, using fallback:', oracleError);
          }
          
          collateralAccounts.push({
            assetType: 'SOL',
            amount: solCollateralBalance,
            valueUsd: solCollateralBalance * solPrice,
            isActive: true,
          });
          
          console.log('✅ AccountContext: Added SOL collateral account:', {
            amount: solCollateralBalance,
            valueUsd: solCollateralBalance * solPrice
          });
        }
      }
      
      setCollateralAccounts(collateralAccounts);
      console.log('✅ AccountContext: Collateral accounts populated:', collateralAccounts);
      
    } catch (error) {
      console.error('❌ AccountContext: Error populating collateral accounts:', error);
      setCollateralAccounts([]);
    }
  }, []);

  // Fetch account state from smart contract
  const fetchAccountState = useCallback(async () => {
    if (!connected || !publicKey) {
      console.log('🔍 AccountContext: Wallet not connected, clearing state');
      setAccountState(null);
      setCollateralAccounts([]);
      setPositions([]);
      setOrders([]);
      return;
    }

    console.log('🔍 AccountContext: Fetching account state for:', publicKey.toString());
    setLoading(true);
    setError(null);

    try {
      const walletAddress = publicKey.toString();
      
      // Use quick check with timeout protection
      console.log('🔍 AccountContext: Checking if account exists...');
      const quickStatePromise = SmartContractService.getInstance().getQuickAccountState(walletAddress);
      const timeoutPromise = new Promise((_, reject) => 
        setTimeout(() => reject(new Error('Account check timeout')), 10000)
      );
      
      const quickState = await Promise.race([quickStatePromise, timeoutPromise]) as any;
      console.log('✅ AccountContext: Quick state result:', quickState);
      
      if (quickState.exists) {
        // User has account - get detailed state with timeout
        console.log('🔍 AccountContext: Account exists, fetching detailed state...');
        const detailedStatePromise = SmartContractService.getInstance().getUserAccountState(walletAddress);
        const detailedTimeoutPromise = new Promise((_, reject) => 
          setTimeout(() => reject(new Error('Detailed state timeout')), 15000)
        );
        
        try {
          const state = await Promise.race([detailedStatePromise, detailedTimeoutPromise]) as any;
          console.log('✅ AccountContext: Detailed state fetched:', state);
          setAccountState(state);
          
          // Populate collateral accounts with actual smart contract data
          await populateCollateralAccounts(walletAddress, state);
        } catch (detailedError) {
          console.warn('⚠️ AccountContext: Could not get detailed state, using quick state:', detailedError);
          // Fallback to quick state - try to get actual collateral amount
          let actualCollateral = 0;
          if (quickState.hasCollateral) {
            try {
              console.log('🔍 AccountContext: Getting actual collateral amount for fallback...');
              actualCollateral = await SmartContractService.getInstance().getSOLCollateralBalance(walletAddress);
              console.log('✅ AccountContext: Actual collateral amount:', actualCollateral, 'SOL');
            } catch (collateralError) {
              console.warn('⚠️ AccountContext: Could not get actual collateral amount:', collateralError);
              actualCollateral = 0;
            }
          }
          
          const fallbackState: UserAccountState = {
            exists: true,
            canDeposit: true,
            canTrade: quickState.canTrade,
            totalCollateral: actualCollateral,
            initialMarginRequirement: 0,
            maintenanceMarginRequirement: 0,
            availableMargin: actualCollateral,
            accountHealth: actualCollateral > 0 ? 100 : 0,
            liquidationPrice: 0,
            liquidationThreshold: 0,
            maxLeverage: 100,
            totalPositions: 0,
            maxPositions: 10,
            totalOrders: 0,
            totalFundingPaid: 0,
            totalFundingReceived: 0,
            totalFeesPaid: 0,
            totalRebatesEarned: 0,
            isActive: true,
          };
          console.log('✅ AccountContext: Using fallback state with actual collateral:', fallbackState);
          setAccountState(fallbackState);
          
          // Populate collateral accounts with fallback data
          await populateCollateralAccounts(walletAddress, fallbackState);
        }
      } else {
        // User doesn't have account yet
        console.log('🔍 AccountContext: No account exists, setting default state');
        const defaultState: UserAccountState = {
          exists: false,
          canDeposit: false,
          canTrade: false,
          totalCollateral: 0,
          initialMarginRequirement: 0,
          maintenanceMarginRequirement: 0,
          availableMargin: 0,
          accountHealth: 0,
          liquidationPrice: 0,
          liquidationThreshold: 0,
          maxLeverage: 0,
          totalPositions: 0,
          maxPositions: 0,
          totalOrders: 0,
          totalFundingPaid: 0,
          totalFundingReceived: 0,
          totalFeesPaid: 0,
          totalRebatesEarned: 0,
          isActive: false,
        };
        setAccountState(defaultState);
        
        // Clear these only when account doesn't exist
        setCollateralAccounts([]);
        setPositions([]);
        setOrders([]);
      }

    } catch (err) {
      console.error('❌ AccountContext: Error fetching account state:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch account state');
      
      // Set default state on error
      const errorState: UserAccountState = {
        exists: false,
        canDeposit: false,
        canTrade: false,
        totalCollateral: 0,
        initialMarginRequirement: 0,
        maintenanceMarginRequirement: 0,
        availableMargin: 0,
        accountHealth: 0,
        liquidationPrice: 0,
        liquidationThreshold: 0,
        maxLeverage: 0,
        totalPositions: 0,
        maxPositions: 0,
        totalOrders: 0,
        totalFundingPaid: 0,
        totalFundingReceived: 0,
        totalFeesPaid: 0,
        totalRebatesEarned: 0,
        isActive: false,
      };
      console.log('🔍 AccountContext: Setting error state:', errorState);
      setAccountState(errorState);
      
      // Clear collateral accounts on error
      setCollateralAccounts([]);
      setPositions([]);
      setOrders([]);
    } finally {
      setLoading(false);
      console.log('🏁 AccountContext: fetchAccountState completed');
    }
  }, [connected, publicKey, populateCollateralAccounts]);

  // Create account
  const createAccount = useCallback(async (): Promise<string> => {
    if (!wallet) {
      throw new Error('Wallet not connected');
    }

    console.log('🔍 AccountContext: Creating account...');
    setLoading(true);
    setError(null);

    try {
      // Create account using smart contract
      const signature = await SmartContractService.getInstance().createUserAccount(wallet);
      console.log('✅ AccountContext: Account created with signature:', signature);
      
      // Refresh account state after creation
      console.log('🔍 AccountContext: Refreshing account state after creation...');
      await fetchAccountState();
      
      return signature;
    } catch (err) {
      console.error('❌ AccountContext: Account creation failed:', err);
      setError(err instanceof Error ? err.message : 'Failed to create account');
      throw err;
    } finally {
      setLoading(false);
      console.log('🏁 AccountContext: createAccount completed');
    }
  }, [wallet, fetchAccountState]);

  // Deposit collateral
  const depositCollateral = useCallback(async (
    assetType: CollateralType,
    amount: number
  ): Promise<string> => {
    if (!wallet) {
      throw new Error('Wallet not connected');
    }

    console.log('🔍 AccountContext: Depositing collateral:', { assetType, amount });
    setLoading(true);
    setError(null);

    try {
      // First, initialize collateral account if it doesn't exist
      try {
        console.log('🔍 AccountContext: Initializing collateral account...');
        await SmartContractService.getInstance().initializeCollateralAccount(wallet, assetType);
      } catch (err) {
        // Account might already exist, continue
        console.log('ℹ️ AccountContext: Collateral account might already exist:', err);
      }

      // Add collateral
      console.log('🔍 AccountContext: Adding collateral...');
      const signature = await SmartContractService.getInstance().addCollateral(wallet, assetType, amount);
      console.log('✅ AccountContext: Collateral added with signature:', signature);
      
      // Refresh account state after deposit
      console.log('🔍 AccountContext: Refreshing account state after deposit...');
      await fetchAccountState();
      
      return signature;
    } catch (err) {
      console.error('❌ AccountContext: Error depositing collateral:', err);
      setError(err instanceof Error ? err.message : 'Failed to deposit collateral');
      throw err;
    } finally {
      setLoading(false);
      console.log('🏁 AccountContext: depositCollateral completed');
    }
  }, [wallet, fetchAccountState]);

  // Place order
  const placeOrder = useCallback(async (
    market: string,
    orderType: OrderType,
    side: PositionSide,
    size: number,
    price: number,
    leverage: number
  ): Promise<string> => {
    if (!wallet) {
      throw new Error('Wallet not connected');
    }

    setLoading(true);
    setError(null);

    try {
      const signature = await SmartContractService.getInstance().placeOrder(
        wallet,
        market,
        orderType,
        side,
        size,
        price,
        leverage
      );
      
      // Refresh account state after order placement
      await fetchAccountState();
      
      return signature;
    } catch (err) {
      console.error('Error placing order:', err);
      setError(err instanceof Error ? err.message : 'Failed to place order');
      throw err;
    } finally {
      setLoading(false);
    }
  }, [wallet, fetchAccountState]);

  // Refresh all data
  const refreshData = useCallback(async () => {
    await fetchAccountState();
  }, [fetchAccountState]);

  // Computed properties
  const canDeposit = accountState?.canDeposit ?? false;
  const canTrade = accountState?.canTrade ?? false;
  const totalBalance = collateralAccounts.reduce((sum, account) => sum + account.valueUsd, 0);
  const accountHealth = accountState?.accountHealth ?? 0;
  const isAtRisk = accountHealth < 5000; // Less than 50% health

  // Effect to fetch account state when wallet connects/disconnects
  useEffect(() => {
    fetchAccountState();
  }, [fetchAccountState]);

  // Listen for account state refresh events from deposit/withdraw modals
  useEffect(() => {
    const handleRefreshEvent = (event: CustomEvent) => {
      console.log('🔄 AccountContext: Received refreshAccountState event:', event.detail);
      // Add a small delay to ensure blockchain state is updated
      setTimeout(() => {
        console.log('🔄 AccountContext: Refreshing account state after deposit...');
        fetchAccountState();
      }, 2000);
    };

    window.addEventListener('refreshAccountState', handleRefreshEvent as EventListener);
    return () => {
      window.removeEventListener('refreshAccountState', handleRefreshEvent as EventListener);
    };
  }, [fetchAccountState]);

  // Effect: persist selected wallet name for UX continuity
  useEffect(() => {
    try {
      if (wallet?.adapter?.name) {
        localStorage.setItem('quantdesk_selected_wallet', wallet.adapter.name);
      }
    } catch {}
  }, [wallet]);

  // Context value
  const contextValue: AccountContextType = useMemo(() => ({
    // Wallet
    wallet,
    
    // State
    accountState,
    collateralAccounts,
    positions,
    orders,
    loading,
    error,

    // Actions
    fetchAccountState,
    createAccount,
    depositCollateral,
    placeOrder,
    refreshData,

    // Computed properties
    canDeposit,
    canTrade,
    totalBalance,
    accountHealth,
    isAtRisk,
  }), [
    wallet,
    accountState,
    collateralAccounts,
    positions,
    orders,
    loading,
    error,
    fetchAccountState,
    createAccount,
    depositCollateral,
    placeOrder,
    refreshData,
    canDeposit,
    canTrade,
    totalBalance,
    accountHealth,
    isAtRisk,
  ]);

  return (
    <AccountContext.Provider value={contextValue}>
      {children}
    </AccountContext.Provider>
  );
};

// Hook to use account context
export const useAccount = (): AccountContextType => {
  const context = useContext(AccountContext);
  if (context === undefined) {
    throw new Error('useAccount must be used within an AccountProvider');
  }
  return context;
};

// Account state helpers
export const useAccountState = () => {
  const { accountState, loading, error } = useAccount();
  return { accountState, loading, error };
};

export const useAccountActions = () => {
  const { createAccount, depositCollateral, placeOrder, refreshData } = useAccount();
  return { createAccount, depositCollateral, placeOrder, refreshData };
};

export const useAccountData = () => {
  const { collateralAccounts, positions, orders, totalBalance, accountHealth, isAtRisk } = useAccount();
  return { collateralAccounts, positions, orders, totalBalance, accountHealth, isAtRisk };
};
