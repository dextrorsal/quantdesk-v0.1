//! Secure Wallet Management for QuantDesk Backend
//! This module provides secure wallet creation and management using environment variables

import { Keypair } from '@solana/web3.js';
import bs58 from 'bs58';
import { logger } from './logger';

export interface WalletConfig {
  solanaPrivateKey: string;
  keeperPrivateKey?: string;
  adminPrivateKey?: string;
}

export class SecureWalletManager {
  private static instance: SecureWalletManager;
  private wallets: Map<string, Keypair> = new Map();

  private constructor() {}

  public static getInstance(): SecureWalletManager {
    if (!SecureWalletManager.instance) {
      SecureWalletManager.instance = new SecureWalletManager();
    }
    return SecureWalletManager.instance;
  }

  /**
   * Create a wallet from Base58 encoded private key
   * @param privateKeyEnv - Environment variable containing Base58 private key
   * @param walletName - Name for the wallet (for caching)
   * @returns Keypair instance
   */
  public createWalletFromEnv(privateKeyEnv: string, walletName: string): Keypair {
    try {
      // Check if wallet is already cached
      if (this.wallets.has(walletName)) {
        return this.wallets.get(walletName)!;
      }

      // Validate environment variable
      if (!privateKeyEnv || privateKeyEnv === 'your_base58_private_key_here') {
        throw new Error(`Invalid private key for wallet: ${walletName}. Please set the environment variable.`);
      }

      // Decode Base58 private key
      const privateKeyBytes = bs58.decode(privateKeyEnv);
      
      // Validate key length (should be 64 bytes for Ed25519)
      if (privateKeyBytes.length !== 64) {
        throw new Error(`Invalid private key length for wallet: ${walletName}. Expected 64 bytes, got ${privateKeyBytes.length}`);
      }

      // Create keypair
      const keypair = Keypair.fromSecretKey(privateKeyBytes);
      
      // Cache the wallet
      this.wallets.set(walletName, keypair);
      
      logger.info(`✅ Wallet created successfully: ${walletName} (${keypair.publicKey.toString()})`);
      
      return keypair;
    } catch (error) {
      logger.error(`❌ Failed to create wallet ${walletName}:`, error);
      throw new Error(`Wallet creation failed for ${walletName}: ${error.message}`);
    }
  }

  /**
   * Get the main operational wallet
   */
  public getMainWallet(): Keypair {
    const privateKey = process.env.SOLANA_PRIVATE_KEY;
    if (!privateKey) {
      throw new Error('SOLANA_PRIVATE_KEY not found in environment variables');
    }
    return this.createWalletFromEnv(privateKey, 'main');
  }

  /**
   * Get the keeper wallet for liquidation operations
   */
  public getKeeperWallet(): Keypair {
    const privateKey = process.env.KEEPER_PRIVATE_KEY;
    if (!privateKey) {
      throw new Error('KEEPER_PRIVATE_KEY not found in environment variables');
    }
    return this.createWalletFromEnv(privateKey, 'keeper');
  }

  /**
   * Get the admin wallet for administrative operations
   */
  public getAdminWallet(): Keypair {
    const privateKey = process.env.ADMIN_PRIVATE_KEY;
    if (!privateKey) {
      throw new Error('ADMIN_PRIVATE_KEY not found in environment variables');
    }
    return this.createWalletFromEnv(privateKey, 'admin');
  }

  /**
   * Validate all required wallets are available
   */
  public validateWallets(): void {
    try {
      this.getMainWallet();
      this.getKeeperWallet();
      this.getAdminWallet();
      logger.info('✅ All required wallets validated successfully');
    } catch (error) {
      logger.error('❌ Wallet validation failed:', error);
      throw error;
    }
  }

  /**
   * Get wallet public key as string
   */
  public getWalletAddress(walletName: string): string {
    const wallet = this.wallets.get(walletName);
    if (!wallet) {
      throw new Error(`Wallet ${walletName} not found`);
    }
    return wallet.publicKey.toString();
  }

  /**
   * Clear cached wallets (for testing)
   */
  public clearCache(): void {
    this.wallets.clear();
    logger.info('🧹 Wallet cache cleared');
  }
}

/**
 * Utility function to create wallet from environment variable
 * @param privateKeyEnv - Environment variable containing Base58 private key
 * @returns Keypair instance
 */
export function createWalletFromEnv(privateKeyEnv: string): Keypair {
  const walletManager = SecureWalletManager.getInstance();
  return walletManager.createWalletFromEnv(privateKeyEnv, 'temp');
}

/**
 * Utility function to validate private key format
 * @param privateKey - Base58 encoded private key
 * @returns boolean indicating if key is valid
 */
export function validatePrivateKey(privateKey: string): boolean {
  try {
    const decoded = bs58.decode(privateKey);
    return decoded.length === 64;
  } catch {
    return false;
  }
}

/**
 * Generate a new keypair (for testing only)
 * @returns New Keypair instance
 */
export function generateTestWallet(): Keypair {
  logger.warn('⚠️  Generating test wallet - use only for testing!');
  return Keypair.generate();
}

export default SecureWalletManager;
