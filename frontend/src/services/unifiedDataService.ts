import { Connection, PublicKey } from '@solana/web3.js';
import { Wallet } from '@solana/wallet-adapter-react';
import { Program, BN } from '@coral-xyz/anchor';
import axios from 'axios';
import { smartContractService } from './smartContractService';
import { portfolioAnalyticsService } from './portfolioAnalyticsService';

// Consolidated Data Service
// This service unifies all data operations and eliminates overlapping responsibilities
// between BalanceService, CrossCollateralService, and SmartContractService

export interface UnifiedUserData {
  // Basic account info
  walletAddress: string;
  userAccountExists: boolean;
  canDeposit: boolean;
  canTrade: boolean;
  
  // Collateral data (from smart contracts)
  totalCollateral: number;
  collateralBreakdown: {
    SOL: number;
    USDC: number;
    BTC: number;
    ETH: number;
    [key: string]: number;
  };
  
  // Portfolio data (from backend analytics)
  portfolioValue: number;
  totalPnL: number;
  totalPnLPercent: number;
  positions: Array<{
    symbol: string;
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnL: number;
    leverage: number;
    riskScore: number;
  }>;
  
  // Risk metrics
  accountHealth: number;
  liquidationPrice?: number;
  maxLeverage: number;
  availableMargin: number;
  
  // Performance metrics
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  
  // Token balances (from wallet)
  tokenBalances: Array<{
    symbol: string;
    amount: number;
    value: number;
    price: number;
  }>;
  
  // Metadata
  lastUpdated: string;
  dataSource: 'backend' | 'smart-contract' | 'wallet' | 'mixed';
}

export interface DataServiceConfig {
  enableBackendData: boolean;
  enableSmartContractData: boolean;
  enableWalletData: boolean;
  fallbackToMock: boolean;
  cacheTimeout: number; // milliseconds
}

class UnifiedDataService {
  private static instance: UnifiedDataService;
  private config: DataServiceConfig;
  private cache: Map<string, { data: UnifiedUserData; timestamp: number }> = new Map();

  private constructor() {
    this.config = {
      enableBackendData: true,
      enableSmartContractData: true,
      enableWalletData: true,
      fallbackToMock: true,
      cacheTimeout: 30000, // 30 seconds
    };
  }

  public static getInstance(): UnifiedDataService {
    if (!UnifiedDataService.instance) {
      UnifiedDataService.instance = new UnifiedDataService();
    }
    return UnifiedDataService.instance;
  }

  /**
   * Get comprehensive user data from all sources
   * This is the main method that consolidates all data operations
   */
  async getUserData(walletAddress: string): Promise<UnifiedUserData> {
    const cacheKey = `user_${walletAddress}`;
    const cached = this.cache.get(cacheKey);
    
    // Return cached data if still valid
    if (cached && Date.now() - cached.timestamp < this.config.cacheTimeout) {
      console.log('📦 Returning cached user data');
      return cached.data;
    }

    console.log('🔍 Fetching fresh user data for:', walletAddress);
    
    try {
      const userData = await this.fetchUnifiedUserData(walletAddress);
      
      // Cache the result
      this.cache.set(cacheKey, {
        data: userData,
        timestamp: Date.now()
      });
      
      return userData;
    } catch (error) {
      console.error('❌ Error fetching unified user data:', error);
      
      if (this.config.fallbackToMock) {
        return this.getMockUserData(walletAddress);
      }
      
      throw error;
    }
  }

  /**
   * Fetch data from all enabled sources and merge them
   */
  private async fetchUnifiedUserData(walletAddress: string): Promise<UnifiedUserData> {
    const promises: Promise<any>[] = [];
    const dataSources: string[] = [];

    // Smart contract data (always enabled for core functionality)
    if (this.config.enableSmartContractData) {
      promises.push(this.fetchSmartContractData(walletAddress));
      dataSources.push('smart-contract');
    }

    // Backend portfolio data
    if (this.config.enableBackendData) {
      promises.push(this.fetchBackendData(walletAddress));
      dataSources.push('backend');
    }

    // Wallet token balances
    if (this.config.enableWalletData) {
      promises.push(this.fetchWalletData(walletAddress));
      dataSources.push('wallet');
    }

    // Execute all fetches in parallel
    const results = await Promise.allSettled(promises);
    
    // Merge results
    const unifiedData: UnifiedUserData = {
      walletAddress,
      userAccountExists: false,
      canDeposit: false,
      canTrade: false,
      totalCollateral: 0,
      collateralBreakdown: {},
      portfolioValue: 0,
      totalPnL: 0,
      totalPnLPercent: 0,
      positions: [],
      accountHealth: 0,
      maxLeverage: 0,
      availableMargin: 0,
      sharpeRatio: 0,
      maxDrawdown: 0,
      winRate: 0,
      tokenBalances: [],
      lastUpdated: new Date().toISOString(),
      dataSource: 'mixed'
    };

    // Process smart contract results
    const smartContractResult = results[0];
    if (smartContractResult.status === 'fulfilled') {
      const scData = smartContractResult.value;
      unifiedData.userAccountExists = scData.exists;
      unifiedData.canDeposit = scData.canDeposit;
      unifiedData.canTrade = scData.canTrade;
      unifiedData.totalCollateral = scData.totalCollateral;
      unifiedData.accountHealth = scData.accountHealth;
      unifiedData.maxLeverage = scData.maxLeverage;
      unifiedData.availableMargin = scData.availableMargin;
    }

    // Process backend results
    if (this.config.enableBackendData) {
      const backendResult = results[1];
      if (backendResult.status === 'fulfilled') {
        const backendData = backendResult.value;
        unifiedData.portfolioValue = backendData.totalValue;
        unifiedData.totalPnL = backendData.totalPnL;
        unifiedData.totalPnLPercent = backendData.totalPnLPercent;
        unifiedData.positions = backendData.positions;
        unifiedData.sharpeRatio = backendData.sharpeRatio;
        unifiedData.maxDrawdown = backendData.maxDrawdown;
        unifiedData.winRate = backendData.winRate;
      }
    }

    // Process wallet results
    if (this.config.enableWalletData) {
      const walletResult = results[this.config.enableBackendData ? 2 : 1];
      if (walletResult.status === 'fulfilled') {
        unifiedData.tokenBalances = walletResult.value;
      }
    }

    // Determine data source
    if (dataSources.length === 1) {
      unifiedData.dataSource = dataSources[0] as any;
    }

    return unifiedData;
  }

  /**
   * Fetch data from smart contracts
   */
  private async fetchSmartContractData(walletAddress: string): Promise<any> {
    try {
      const accountState = await smartContractService.getUserAccountState(walletAddress);
      const collateralBalance = await smartContractService.getSOLCollateralBalance(walletAddress);
      
      return {
        exists: accountState.exists,
        canDeposit: accountState.canDeposit,
        canTrade: accountState.canTrade,
        totalCollateral: collateralBalance,
        accountHealth: accountState.accountHealth,
        maxLeverage: accountState.maxLeverage,
        availableMargin: accountState.availableMargin,
      };
    } catch (error) {
      console.error('Error fetching smart contract data:', error);
      throw error;
    }
  }

  /**
   * Fetch data from backend APIs
   */
  private async fetchBackendData(walletAddress: string): Promise<any> {
    try {
      const analytics = await portfolioAnalyticsService.getPortfolioAnalytics();
      return {
        totalValue: analytics.totalValue,
        totalPnL: analytics.totalPnL,
        totalPnLPercent: analytics.totalPnLPercent,
        positions: analytics.positions,
        sharpeRatio: analytics.sharpeRatio,
        maxDrawdown: analytics.maxDrawdown,
        winRate: analytics.winRate,
      };
    } catch (error) {
      console.error('Error fetching backend data:', error);
      throw error;
    }
  }

  /**
   * Fetch token balances from wallet
   */
  private async fetchWalletData(walletAddress: string): Promise<any[]> {
    try {
      // This would integrate with the existing balance service
      // For now, return empty array as placeholder
      return [];
    } catch (error) {
      console.error('Error fetching wallet data:', error);
      throw error;
    }
  }

  /**
   * Get mock data for fallback
   */
  private getMockUserData(walletAddress: string): UnifiedUserData {
    return {
      walletAddress,
      userAccountExists: true,
      canDeposit: true,
      canTrade: true,
      totalCollateral: 1.5,
      collateralBreakdown: {
        SOL: 1.5,
        USDC: 0,
        BTC: 0,
        ETH: 0,
      },
      portfolioValue: 125430.50,
      totalPnL: 2450,
      totalPnLPercent: 1.99,
      positions: [
        {
          symbol: 'BTC',
          size: 0.5,
          entryPrice: 45000,
          currentPrice: 47000,
          unrealizedPnL: 1000,
          leverage: 10,
          riskScore: 0.7,
        },
      ],
      accountHealth: 85,
      maxLeverage: 100,
      availableMargin: 5000,
      sharpeRatio: 1.2,
      maxDrawdown: 0.15,
      winRate: 0.65,
      tokenBalances: [],
      lastUpdated: new Date().toISOString(),
      dataSource: 'mixed',
    };
  }

  /**
   * Clear cache for a specific user or all users
   */
  clearCache(walletAddress?: string): void {
    if (walletAddress) {
      this.cache.delete(`user_${walletAddress}`);
    } else {
      this.cache.clear();
    }
  }

  /**
   * Update configuration
   */
  updateConfig(newConfig: Partial<DataServiceConfig>): void {
    this.config = { ...this.config, ...newConfig };
  }

  /**
   * Get current configuration
   */
  getConfig(): DataServiceConfig {
    return { ...this.config };
  }
}

export const unifiedDataService = UnifiedDataService.getInstance();
