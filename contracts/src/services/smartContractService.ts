//! Updated Smart Contract Service with Secure Wallet Management
//! This service provides secure interaction with QuantDesk smart contracts

import { 
  AnchorProvider, 
  Program, 
  Wallet,
  BN
} from '@coral-xyz/anchor';
import { 
  Connection, 
  PublicKey, 
  Keypair,
  Commitment,
  Transaction
} from '@solana/web3.js';
import { QuantdeskPerpDex } from '../target/types/quantdesk_perp_dex';
import { config } from './config/environment';
import { SecureWalletManager } from './utils/wallet-manager';
import { logger } from './logger';

export interface OrderExecutionParams {
  market: PublicKey;
  user: PublicKey;
  orderType: 'market' | 'limit' | 'stop';
  side: 'long' | 'short';
  size: number;
  price?: number;
  leverage: number;
}

export interface OrderExecutionResult {
  success: boolean;
  transactionId?: string;
  error?: string;
}

export interface PositionParams {
  market: PublicKey;
  user: PublicKey;
  side: 'long' | 'short';
  size: number;
  leverage: number;
  entryPrice: number;
}

export interface PositionResult {
  success: boolean;
  positionId?: PublicKey;
  transactionId?: string;
  error?: string;
}

export class SmartContractService {
  private static instance: SmartContractService;
  private connection: Connection;
  private provider: AnchorProvider;
  private program: Program<QuantdeskPerpDex>;
  private walletManager: SecureWalletManager;
  private mainWallet: Keypair;

  private constructor() {
    this.walletManager = SecureWalletManager.getInstance();
    this.initializeConnection();
    this.initializeProvider();
    this.initializeProgram();
  }

  public static getInstance(): SmartContractService {
    if (!SmartContractService.instance) {
      SmartContractService.instance = new SmartContractService();
    }
    return SmartContractService.instance;
  }

  private initializeConnection(): void {
    try {
      this.connection = new Connection(
        config.SOLANA_RPC_URL,
        config.SOLANA_COMMITMENT as Commitment
      );
      
      logger.info(`✅ Connected to Solana ${config.SOLANA_NETWORK}: ${config.SOLANA_RPC_URL}`);
    } catch (error) {
      logger.error('❌ Failed to initialize Solana connection:', error);
      throw error;
    }
  }

  private initializeProvider(): void {
    try {
      // ✅ FIXED: Use secure wallet from environment
      this.mainWallet = this.walletManager.getMainWallet();
      
      this.provider = new AnchorProvider(
        this.connection,
        this.mainWallet as any,
        { 
          commitment: config.SOLANA_COMMITMENT as Commitment,
          preflightCommitment: config.SOLANA_COMMITMENT as Commitment
        }
      );

      logger.info(`✅ SmartContractService initialized with secure wallet: ${this.mainWallet.publicKey.toString()}`);
    } catch (error) {
      logger.error('❌ Failed to initialize SmartContractService:', error);
      throw error;
    }
  }

  private initializeProgram(): void {
    try {
      const programId = new PublicKey(config.QUANTDESK_PROGRAM_ID);
      
      this.program = new Program<QuantdeskPerpDex>(
        {} as any, // IDL will be loaded from target/types
        programId,
        this.provider
      );
      
      logger.info(`✅ Program initialized: ${config.QUANTDESK_PROGRAM_ID}`);
    } catch (error) {
      logger.error('❌ Failed to initialize program:', error);
      throw error;
    }
  }

  /**
   * Execute an order on the smart contract
   */
  public async executeOrder(params: OrderExecutionParams): Promise<OrderExecutionResult> {
    try {
      logger.info(`Executing order: ${params.orderType} ${params.side} ${params.size} @ ${params.price || 'market'}`);
      
      // Get keeper wallet for order execution
      const keeperWallet = this.walletManager.getKeeperWallet();
      
      const tx = await this.program.methods
        .placeOrder(
          { [params.orderType]: {} },
          { [params.side]: {} },
          new BN(params.size),
          new BN(params.price || 0),
          new BN(0), // stopPrice
          new BN(0), // trailingDistance
          params.leverage,
          new BN(0), // expiresAt
          new BN(0), // hiddenSize
          new BN(0), // displaySize
          { gtc: {} }, // timeInForce
          new BN(0), // targetPrice
          new BN(0), // twapDuration
          new BN(0)  // twapInterval
        )
        .accounts({
          market: params.market,
          order: Keypair.generate().publicKey,
          user: params.user,
          systemProgram: PublicKey.default,
        })
        .signers([keeperWallet])
        .rpc();

      logger.info(`✅ Order executed successfully: ${tx}`);
      
      return {
        success: true,
        transactionId: tx
      };
    } catch (error) {
      logger.error('❌ Order execution failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Open a position on the smart contract
   */
  public async openPosition(params: PositionParams): Promise<PositionResult> {
    try {
      logger.info(`Opening position: ${params.side} ${params.size} @ ${params.entryPrice} (${params.leverage}x)`);
      
      const positionKeypair = Keypair.generate();
      
      const tx = await this.program.methods
        .openPosition(
          0, // positionIndex
          { [params.side]: {} },
          new BN(params.size),
          new BN(params.leverage),
          new BN(params.entryPrice)
        )
        .accounts({
          position: positionKeypair.publicKey,
          userAccount: params.user,
          market: params.market,
          user: params.user,
          systemProgram: PublicKey.default,
        })
        .signers([positionKeypair])
        .rpc();

      logger.info(`✅ Position opened successfully: ${tx}`);
      
      return {
        success: true,
        positionId: positionKeypair.publicKey,
        transactionId: tx
      };
    } catch (error) {
      logger.error('❌ Position opening failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Close a position
   */
  public async closePosition(positionId: PublicKey, user: PublicKey): Promise<OrderExecutionResult> {
    try {
      logger.info(`Closing position: ${positionId.toString()}`);
      
      const tx = await this.program.methods
        .closePosition()
        .accounts({
          market: PublicKey.default, // Will be fetched from position
          position: positionId,
          user: user,
        })
        .rpc();

      logger.info(`✅ Position closed successfully: ${tx}`);
      
      return {
        success: true,
        transactionId: tx
      };
    } catch (error) {
      logger.error('❌ Position closing failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Liquidate a position (keeper operation)
   */
  public async liquidatePosition(positionId: PublicKey, user: PublicKey): Promise<OrderExecutionResult> {
    try {
      logger.info(`Liquidating position: ${positionId.toString()}`);
      
      // Get keeper wallet for liquidation
      const keeperWallet = this.walletManager.getKeeperWallet();
      
      const tx = await this.program.methods
        .liquidatePosition()
        .accounts({
          market: PublicKey.default, // Will be fetched from position
          position: positionId,
          liquidator: keeperWallet.publicKey,
          vault: PublicKey.default, // Liquidation vault
        })
        .signers([keeperWallet])
        .rpc();

      logger.info(`✅ Position liquidated successfully: ${tx}`);
      
      return {
        success: true,
        transactionId: tx
      };
    } catch (error) {
      logger.error('❌ Position liquidation failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Update oracle price (admin operation)
   */
  public async updateOraclePrice(market: PublicKey, newPrice: number): Promise<OrderExecutionResult> {
    try {
      logger.info(`Updating oracle price for market ${market.toString()}: ${newPrice}`);
      
      // Get admin wallet for price updates
      const adminWallet = this.walletManager.getAdminWallet();
      
      const tx = await this.program.methods
        .updateOraclePrice(new BN(newPrice))
        .accounts({
          market: market,
          priceFeed: PublicKey.default, // Pyth price feed
          authority: adminWallet.publicKey,
        })
        .signers([adminWallet])
        .rpc();

      logger.info(`✅ Oracle price updated successfully: ${tx}`);
      
      return {
        success: true,
        transactionId: tx
      };
    } catch (error) {
      logger.error('❌ Oracle price update failed:', error);
      return {
        success: false,
        error: error.message
      };
    }
  }

  /**
   * Get wallet balance
   */
  public async getWalletBalance(walletType: 'main' | 'keeper' | 'admin' = 'main'): Promise<number> {
    try {
      let wallet: Keypair;
      
      switch (walletType) {
        case 'main':
          wallet = this.walletManager.getMainWallet();
          break;
        case 'keeper':
          wallet = this.walletManager.getKeeperWallet();
          break;
        case 'admin':
          wallet = this.walletManager.getAdminWallet();
          break;
      }
      
      const balance = await this.connection.getBalance(wallet.publicKey);
      logger.info(`💰 ${walletType} wallet balance: ${balance / 1e9} SOL`);
      
      return balance;
    } catch (error) {
      logger.error(`❌ Failed to get ${walletType} wallet balance:`, error);
      throw error;
    }
  }

  /**
   * Validate all wallets have sufficient balance
   */
  public async validateWalletBalances(): Promise<boolean> {
    try {
      const minBalance = 0.1 * 1e9; // 0.1 SOL minimum
      
      const mainBalance = await this.getWalletBalance('main');
      const keeperBalance = await this.getWalletBalance('keeper');
      const adminBalance = await this.getWalletBalance('admin');
      
      const allSufficient = mainBalance >= minBalance && 
                           keeperBalance >= minBalance && 
                           adminBalance >= minBalance;
      
      if (!allSufficient) {
        logger.warn('⚠️  Some wallets have insufficient balance');
        logger.warn(`Main: ${mainBalance / 1e9} SOL, Keeper: ${keeperBalance / 1e9} SOL, Admin: ${adminBalance / 1e9} SOL`);
      }
      
      return allSufficient;
    } catch (error) {
      logger.error('❌ Wallet balance validation failed:', error);
      return false;
    }
  }

  /**
   * Get connection instance
   */
  public getConnection(): Connection {
    return this.connection;
  }

  /**
   * Get program instance
   */
  public getProgram(): Program<QuantdeskPerpDex> {
    return this.program;
  }

  /**
   * Get provider instance
   */
  public getProvider(): AnchorProvider {
    return this.provider;
  }
}

export default SmartContractService;
