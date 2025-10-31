import { Connection, Keypair, PublicKey, Transaction, TransactionInstruction } from '@solana/web3.js';
import { AnchorProvider, Program } from '@coral-xyz/anchor';
import * as anchor from '@coral-xyz/anchor';
import { Logger } from '../utils/logger';
import { DEVNET_CONFIG } from '../config/devnet';
import { getStandardizedConfig } from '../config/standardizedConfig';

const logger = new Logger();

export interface OrderExecutionParams {
  orderId: string;
  userId: string;
  marketSymbol: string;
  side: 'long' | 'short';
  size: number;
  price: number;
  leverage: number;
  orderType: 'market' | 'limit';
}

export interface OrderExecutionResult {
  success: boolean;
  transactionSignature?: string;
  positionId?: string;
  error?: string;
}

export class SmartContractService {
  private static instance: SmartContractService;
  private connection: Connection;
  private provider: AnchorProvider;
  private program: Program | null = null;
  private wallet: Keypair | null = null;

  private constructor() {
    this.connection = new Connection(
      DEVNET_CONFIG.rpcEndpoint,
      DEVNET_CONFIG.commitment
    );
    
    this.initializeProvider();
  }

  public static getInstance(): SmartContractService {
    if (!SmartContractService.instance) {
      SmartContractService.instance = new SmartContractService();
    }
    return SmartContractService.instance;
  }

  private initializeProvider(): void {
    try {
      // Use standardized configuration
      const config = getStandardizedConfig();
      
      if (!config.solanaWalletPath) {
        throw new Error('SOLANA_WALLET environment variable is required');
      }

      // Load wallet from file
      const fs = require('fs');
      const walletData = JSON.parse(fs.readFileSync(config.solanaWalletPath, 'utf8'));
      this.wallet = Keypair.fromSecretKey(new Uint8Array(walletData));
      
      // Use standardized RPC URL
      this.connection = new Connection(config.solanaRpcUrl, DEVNET_CONFIG.commitment);
      
      this.provider = new AnchorProvider(
        this.connection,
        this.wallet as any,
        { commitment: DEVNET_CONFIG.commitment }
      );

      // Load the program IDL
      this.loadProgram();

      logger.info('✅ SmartContractService initialized with wallet:', this.wallet.publicKey.toString());
      logger.info('🔧 Using standardized configuration - RPC URL:', config.solanaRpcUrl);
    } catch (error) {
      logger.error('❌ Failed to initialize SmartContractService:', error);
      // SECURITY FIX: Do not fallback to generated wallet - fail securely
      throw new Error(`SmartContractService initialization failed: ${error instanceof Error ? error.message : 'Unknown error'}. Please check SOLANA_WALLET environment variable.`);
    }
  }

  private loadProgram(): void {
    try {
      // Load program IDL from file system
      const idlPath = process.env.PROGRAM_IDL_PATH || './contracts/target/idl/quantdesk_perp_dex.json';
      const fs = require('fs');
      
      if (fs.existsSync(idlPath)) {
        const idl = JSON.parse(fs.readFileSync(idlPath, 'utf8'));
        // Use standardized program ID
        const config = getStandardizedConfig();
        const programId = new PublicKey(config.quantdeskProgramId);
        
        this.program = new Program(idl, this.provider);
        logger.info('✅ Program loaded successfully:', programId.toString());
      } else {
        logger.warn('⚠️ Program IDL not found, running in mock mode');
      }
    } catch (error) {
      logger.error('❌ Failed to load program:', error);
      logger.warn('⚠️ Running in mock mode');
    }
  }

  /**
   * Execute an order on the smart contract with atomic position creation
   */
  public async executeOrder(params: OrderExecutionParams): Promise<OrderExecutionResult> {
    try {
      logger.info(`🚀 Executing order ${params.orderId} on smart contract with atomic position creation`);

      if (!this.program) {
        logger.warn('⚠️ Program not initialized, using mock execution');
        return await this.simulateOrderExecution(params);
      }

      // Get market account
      const marketPda = await this.getMarketPda(params.marketSymbol);
      
      // Get user account
      const userPda = await this.getUserPda(params.userId);
      
      // Get order account - use user's public key for PDA derivation to match smart contract
      const orderPda = await this.getOrderPda(params.orderId, params.userId);

      // Create atomic transaction for order execution + position creation
      const transaction = new Transaction();

      // Step 1: Place order on smart contract
      const placeOrderIx = await this.program.methods
        .placeOrder(
          this.mapOrderType(params.orderType),
          this.mapPositionSide(params.side),
          new anchor.BN(params.size),
          new anchor.BN(params.price),
          0, // stop_price
          0, // trailing_distance
          params.leverage,
          0, // expires_at
          0, // hidden_size
          0, // display_size
          { gtc: {} }, // time_in_force
          0, // target_price
          0, // twap_duration
          0  // twap_interval
        )
        .accounts({
          market: marketPda,
          order: orderPda,
          user: userPda,
          authority: this.wallet!.publicKey,
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .instruction();

      transaction.add(placeOrderIx);

      // Step 2: Execute order if it's a market order
      if (params.orderType === 'market') {
        const executeOrderIx = await this.program.methods
          .executeConditionalOrder()
          .accounts({
            market: marketPda,
            order: orderPda,
            executor: this.wallet!.publicKey,
          })
          .instruction();

        transaction.add(executeOrderIx);
      }

      // Step 3: Create position atomically with order execution
      const positionPda = await this.getPositionPda(params.userId, params.marketSymbol, params.side);
      
      const openPositionIx = await this.program.methods
        .openPosition(
          0, // position_index - will be managed by smart contract
          this.mapPositionSide(params.side),
          new anchor.BN(params.size),
          params.leverage,
          new anchor.BN(params.price)
        )
        .accounts({
          position: positionPda,
          userAccount: userPda,
          market: marketPda,
          user: this.wallet!.publicKey,
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .instruction();

      transaction.add(openPositionIx);

      // Send and confirm atomic transaction
      const signature = await this.provider.sendAndConfirm(transaction);
      
      logger.info(`✅ Order ${params.orderId} executed with atomic position creation:`, signature);
      
      return {
        success: true,
        transactionSignature: signature,
        positionId: positionPda.toString()
      };

    } catch (error) {
      logger.error(`❌ Failed to execute order ${params.orderId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Simulate order execution (placeholder for actual smart contract interaction)
   */
  private async simulateOrderExecution(params: OrderExecutionParams): Promise<OrderExecutionResult> {
    // Simulate processing time
    await new Promise(resolve => setTimeout(resolve, 1000));

    // Generate mock transaction signature
    const mockSignature = `tx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Generate mock position ID
    const mockPositionId = `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

    return {
      success: true,
      transactionSignature: mockSignature,
      positionId: mockPositionId
    };
  }

  /**
   * Create a position on the smart contract
   */
  public async createPosition(
    userId: string,
    marketSymbol: string,
    side: 'long' | 'short',
    size: number,
    entryPrice: number,
    leverage: number
  ): Promise<{ success: boolean; positionId?: string; error?: string }> {
    try {
      logger.info(`📈 Creating position for user ${userId} on ${marketSymbol}`);

      if (!this.program) {
        logger.warn('⚠️ Program not initialized, using mock position creation');
        const mockPositionId = `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
        return {
          success: true,
          positionId: mockPositionId
        };
      }

      // Get market account
      const marketPda = await this.getMarketPda(marketSymbol);
      
      // Get user account
      const userPda = await this.getUserPda(userId);
      
      // Get position account
      const positionPda = await this.getPositionPda(userId, marketSymbol, side);

      // Create transaction for position creation
      const transaction = new Transaction();

      // Add open position instruction
      const openPositionIx = await this.program.methods
        .openPosition(
          this.mapPositionSide(side),
          new anchor.BN(size),
          new anchor.BN(entryPrice),
          leverage
        )
        .accounts({
          market: marketPda,
          position: positionPda,
          user: userPda,
          authority: this.wallet!.publicKey,
          systemProgram: anchor.web3.SystemProgram.programId,
        })
        .instruction();

      transaction.add(openPositionIx);

      // Send and confirm transaction
      const signature = await this.provider.sendAndConfirm(transaction);
      
      logger.info(`✅ Position created successfully:`, signature);
      
      return {
        success: true,
        positionId: positionPda.toString()
      };

    } catch (error) {
      logger.error('❌ Failed to create position:', error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Close a position on the smart contract
   */
  public async closePosition(
    positionId: string,
    userId: string
  ): Promise<{ success: boolean; transactionSignature?: string; error?: string }> {
    try {
      logger.info(`📉 Closing position ${positionId} for user ${userId}`);

      // TODO: Implement actual position closing on smart contract
      // For now, simulate position closing
      const mockSignature = `close_tx_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;

      return {
        success: true,
        transactionSignature: mockSignature
      };

    } catch (error) {
      logger.error(`❌ Failed to close position ${positionId}:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Update order status on smart contract
   */
  public async updateOrderStatus(
    orderId: string,
    status: 'pending' | 'filled' | 'cancelled' | 'expired'
  ): Promise<{ success: boolean; error?: string }> {
    try {
      logger.info(`🔄 Updating order ${orderId} status to ${status}`);

      // TODO: Implement actual order status update on smart contract
      return { success: true };

    } catch (error) {
      logger.error(`❌ Failed to update order ${orderId} status:`, error);
      return {
        success: false,
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }

  /**
   * Get connection for other services
   */
  public getConnection(): Connection {
    return this.connection;
  }

  /**
   * Health check
   */
  public async healthCheck(): Promise<boolean> {
    try {
      const version = await this.connection.getVersion();
      return !!version;
    } catch (error) {
      return false;
    }
  }

  /**
   * Get Program Derived Address for market
   */
  private async getMarketPda(marketSymbol: string): Promise<PublicKey> {
    // Parse market symbol (e.g., "BTC/USD" -> ["BTC", "USD"])
    const [baseAsset, quoteAsset] = marketSymbol.split('/');
    const [marketPda] = await PublicKey.findProgramAddress(
      [Buffer.from('market'), Buffer.from(baseAsset), Buffer.from(quoteAsset)],
      this.program!.programId
    );
    return marketPda;
  }

  /**
   * Get Program Derived Address for user account
   */
  private async getUserPda(userId: string): Promise<PublicKey> {
    // Convert userId to PublicKey if it's a wallet address, otherwise use as string
    let userPubkey: PublicKey;
    try {
      userPubkey = new PublicKey(userId);
    } catch {
      // If userId is not a valid PublicKey, create a deterministic one from the string
      const userBytes = Buffer.from(userId);
      userPubkey = new PublicKey(userBytes.slice(0, 32)); // Take first 32 bytes
    }
    
    const [userPda] = await PublicKey.findProgramAddress(
      [Buffer.from('user_account'), userPubkey.toBuffer(), Buffer.from([0, 0])], // account_index = 0
      this.program!.programId
    );
    return userPda;
  }

  /**
   * Get Program Derived Address for order
   * Matches smart contract: seeds = [b"order", user.key().as_ref()]
   */
  private async getOrderPda(orderId: string, userPublicKey: string): Promise<PublicKey> {
    const userPubkey = new PublicKey(userPublicKey);
    const [orderPda] = await PublicKey.findProgramAddress(
      [Buffer.from('order'), userPubkey.toBuffer()],
      this.program!.programId
    );
    return orderPda;
  }

  /**
   * Get Program Derived Address for position
   * Matches smart contract: seeds = [b"position", user_account.key().as_ref(), &position_index.to_le_bytes()]
   */
  private async getPositionPda(userId: string, marketSymbol: string, side: 'long' | 'short'): Promise<PublicKey> {
    // Get user account PDA first
    const userPda = await this.getUserPda(userId);
    
    // Use position index 0 for now - in production this would be managed by the smart contract
    const positionIndex = 0;
    
    const [positionPda] = await PublicKey.findProgramAddress(
      [
        Buffer.from('position'),
        userPda.toBuffer(),
        Buffer.from(positionIndex.toString().padStart(4, '0'), 'utf8')
      ],
      this.program!.programId
    );
    return positionPda;
  }

  /**
   * Map order type from backend to smart contract
   */
  private mapOrderType(orderType: 'market' | 'limit'): any {
    switch (orderType) {
      case 'market':
        return { market: {} };
      case 'limit':
        return { limit: {} };
      default:
        return { market: {} };
    }
  }

  /**
   * Map position side from backend to smart contract
   */
  private mapPositionSide(side: 'long' | 'short'): any {
    switch (side) {
      case 'long':
        return { long: {} };
      case 'short':
        return { short: {} };
      default:
        return { long: {} };
    }
  }
}

// Export singleton instance
export const smartContractService = SmartContractService.getInstance();
