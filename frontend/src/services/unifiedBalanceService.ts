/**
 * Unified Balance Service
 * Following Drift Protocol & industry best practices for perp DEX balance management
 * 
 * Architecture:
 * - Fetch wallet SOL balance on-chain via RPC
 * - Fetch deposited collateral from program accounts using getProgramAccounts
 * - Use Anchor discriminator for proper deserialization
 * - Cache results to minimize RPC calls
 */

import { Connection, PublicKey } from '@solana/web3.js';
import { AnchorProvider, Program } from '@coral-xyz/anchor';
import { balanceService } from './balanceService';
import { smartContractService } from './smartContractService';

export interface UnifiedBalance {
  // On-chain balances
  walletSOL: number;           // SOL in wallet (not deposited)
  depositedSOL: number;         // SOL deposited as collateral
  totalSOL: number;             // walletSOL + depositedSOL
  
  // USD values
  walletSOLValueUSD: number;
  depositedSOLValueUSD: number;
  totalValueUSD: number;
  
  // Account states
  hasUserAccount: boolean;
  canDeposit: boolean;
  canTrade: boolean;
  accountHealth: number;
  
  // Program account addresses
  userAccountAddress?: string;
  collateralAccountAddress?: string;
}

class UnifiedBalanceService {
  private static instance: UnifiedBalanceService;
  private connection: Connection;
  private cache: Map<string, UnifiedBalance> = new Map();
  private cacheTime = 5000; // 5 seconds cache

  private constructor() {
    this.connection = new Connection(
      import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
      'confirmed'
    );
  }

  public static getInstance(): UnifiedBalanceService {
    if (!UnifiedBalanceService.instance) {
      UnifiedBalanceService.instance = new UnifiedBalanceService();
    }
    return UnifiedBalanceService.instance;
  }

  /**
   * Get comprehensive balance - Drift Protocol style
   * Fetches both wallet balance AND program-stored collateral
   */
  async getUnifiedBalance(walletAddress: PublicKey): Promise<UnifiedBalance> {
    const cacheKey = walletAddress.toString();
    const cached = this.cache.get(cacheKey);
    
    if (cached && Date.now() - (cached as any)._timestamp < this.cacheTime) {
      return cached;
    }

    console.log('🔍 Fetching unified balance for:', walletAddress.toString());

    try {
      // 1. Get wallet SOL balance (on-chain via RPC)
      const walletSOL = await balanceService.getNativeSOLBalance(walletAddress);
      
      // 2. Get deposited collateral (from program accounts)
      const [depositedSOL, accountState] = await Promise.all([
        smartContractService.getSOLCollateralBalance(walletAddress.toString()),
        smartContractService.getQuickAccountState(walletAddress.toString()).catch(() => null)
      ]);

      // 3. Derive program account PDAs
      const userAccountAddress = await this.deriveUserAccountAddress(walletAddress);
      const collateralAccountAddress = await this.deriveCollateralAccountAddress(walletAddress);

      // 4. Get SOL price for USD conversion
      const solPrice = await this.getSOLPrice();

      // 5. Calculate USD values
      const walletSOLValueUSD = walletSOL * solPrice;
      const depositedSOLValueUSD = depositedSOL * solPrice;
      const totalValueUSD = walletSOLValueUSD + depositedSOLValueUSD;

      // 6. Get account state
      const hasUserAccount = accountState?.exists || false;
      const canDeposit = accountState?.canDeposit || false;
      const canTrade = accountState?.canTrade || false;
      const accountHealth = accountState?.accountHealth || 0;

      const unifiedBalance: UnifiedBalance = {
        walletSOL,
        depositedSOL,
        totalSOL: walletSOL + depositedSOL,
        walletSOLValueUSD,
        depositedSOLValueUSD,
        totalValueUSD,
        hasUserAccount,
        canDeposit,
        canTrade,
        accountHealth,
        userAccountAddress,
        collateralAccountAddress,
      };

      // Cache the result
      (unifiedBalance as any)._timestamp = Date.now();
      this.cache.set(cacheKey, unifiedBalance);

      console.log('✅ Unified balance fetched:', unifiedBalance);
      return unifiedBalance;
    } catch (error) {
      console.error('❌ Error fetching unified balance:', error);
      return {
        walletSOL: 0,
        depositedSOL: 0,
        totalSOL: 0,
        walletSOLValueUSD: 0,
        depositedSOLValueUSD: 0,
        totalValueUSD: 0,
        hasUserAccount: false,
        canDeposit: false,
        canTrade: false,
        accountHealth: 0,
      };
    }
  }

  /**
   * Derive UserAccount PDA address
   */
  private async deriveUserAccountAddress(walletAddress: PublicKey): Promise<string> {
    try {
      const [userAccount] = await PublicKey.findProgramAddress(
        [Buffer.from('user'), walletAddress.toBuffer()],
        smartContractService.getProgramId()
      );
      return userAccount.toString();
    } catch (error) {
      console.error('Error deriving user account address:', error);
      return 'N/A';
    }
  }

  /**
   * Derive CollateralAccount PDA address for SOL
   */
  private async deriveCollateralAccountAddress(walletAddress: PublicKey): Promise<string> {
    try {
      const [collateralAccount] = await PublicKey.findProgramAddress(
        [Buffer.from('collateral'), walletAddress.toBuffer(), Buffer.from('SOL')],
        smartContractService.getProgramId()
      );
      return collateralAccount.toString();
    } catch (error) {
      console.error('Error deriving collateral account address:', error);
      return 'N/A';
    }
  }

  /**
   * Get current SOL price from oracle
   */
  private async getSOLPrice(): Promise<number> {
    try {
      const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002';
      const response = await fetch(`${apiUrl}/api/oracle/price/SOL`);
      if (response.ok) {
        const data = await response.json();
        return data.price || 100; // Fallback to $100 if no price
      }
    } catch (error) {
      console.warn('⚠️ Could not fetch SOL price:', error);
    }
    return 100; // Fallback
  }

  /**
   * Clear cache for a specific wallet
   */
  clearCache(walletAddress: PublicKey) {
    this.cache.delete(walletAddress.toString());
  }

  /**
   * Clear all cache
   */
  clearAllCache() {
    this.cache.clear();
  }
}

export const unifiedBalanceService = UnifiedBalanceService.getInstance();

