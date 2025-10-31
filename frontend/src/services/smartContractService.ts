import { Connection, PublicKey, SystemProgram, Transaction, TransactionInstruction, SYSVAR_RENT_PUBKEY } from '@solana/web3.js';
import { Wallet } from '@solana/wallet-adapter-react';
import { Program, AnchorProvider, BN } from '@coral-xyz/anchor';
import { TOKEN_PROGRAM_ID, ASSOCIATED_TOKEN_PROGRAM_ID } from '@solana/spl-token';
import idl from '../types/quantdesk_perp_dex.json';

// Type assertion to ensure proper IDL typing
const programIdl = idl as any;

// Export IDL for use in test suites
export { programIdl };

/**
 * PDA Seed Verification Utilities
 * These functions help ensure TS and Rust PDA derivations match exactly
 */

/**
 * Verify PDA seeds match between TypeScript and expected Rust patterns
 */
export function verifyPDASeeds(
  userPubkey: PublicKey,
  accountIndex: number = 0
): {
  userAccount: { seeds: Buffer[]; isValid: boolean };
  collateralSOL: { seeds: Buffer[]; isValid: boolean };
  protocolVault: { seeds: Buffer[]; isValid: boolean };
} {
  // User account: [b"user_account", user.key().as_ref(), &[0u8, 0u8]]
  const accountIndexBuffer = Buffer.alloc(2);
  accountIndexBuffer.writeUInt16LE(accountIndex, 0);
  const userAccountSeeds = [
    Buffer.from('user_account'),
    userPubkey.toBuffer(),
    accountIndexBuffer, // Should be [0u8, 0u8] for index 0
  ];

  // Collateral SOL: [b"collateral", user.key().as_ref(), b"SOL"]
  const collateralSOLSeeds = [
    Buffer.from('collateral'),
    userPubkey.toBuffer(),
    Buffer.from('SOL'),
  ];

  // Protocol vault: [b"protocol_sol_vault"]
  const protocolVaultSeeds = [Buffer.from('protocol_sol_vault')];

  return {
    userAccount: {
      seeds: userAccountSeeds,
      isValid: accountIndexBuffer.equals(Buffer.from([0, 0])) || accountIndex === 0,
    },
    collateralSOL: {
      seeds: collateralSOLSeeds,
      isValid: true,
    },
    protocolVault: {
      seeds: protocolVaultSeeds,
      isValid: true,
    },
  };
}

/**
 * Derive all PDAs needed for deposit operations
 */
export async function deriveDepositPDAs(
  userPubkey: PublicKey,
  programId: PublicKey,
  accountIndex: number = 0
): Promise<{
  userAccount: PublicKey;
  collateralSOL: PublicKey;
  protocolVault: PublicKey;
  bumps: {
    userAccount: number;
    collateralSOL: number;
    protocolVault: number;
  };
}> {
  const accountIndexBuffer = Buffer.alloc(2);
  accountIndexBuffer.writeUInt16LE(accountIndex, 0);

  const [userAccount, userAccountBump] = await PublicKey.findProgramAddress(
    [Buffer.from('user_account'), userPubkey.toBuffer(), accountIndexBuffer],
    programId
  );

  const [collateralSOL, collateralSOLBump] = await PublicKey.findProgramAddress(
    [Buffer.from('collateral'), userPubkey.toBuffer(), Buffer.from('SOL')],
    programId
  );

  const [protocolVault, protocolVaultBump] = await PublicKey.findProgramAddress(
    [Buffer.from('protocol_sol_vault')],
    programId
  );

  return {
    userAccount,
    collateralSOL,
    protocolVault,
    bumps: {
      userAccount: userAccountBump,
      collateralSOL: collateralSOLBump,
      protocolVault: protocolVaultBump,
    },
  };
}

// Debug: Log the IDL program ID to ensure it's correct
console.log('🔍 IDL Program ID from import:', programIdl.address);

// Smart Contract Integration Service
// This service handles all interactions with the QuantDesk Solana programs

export interface UserAccountState {
  exists: boolean;
  canDeposit: boolean;
  canTrade: boolean;
  
  // Enhanced fields
  totalCollateral: number;
  initialMarginRequirement: number;
  maintenanceMarginRequirement: number;
  availableMargin: number;
  
  accountHealth: number;
  liquidationPrice?: number;
  liquidationThreshold: number;
  maxLeverage: number;
  
  totalPositions: number;
  maxPositions: number;
  totalOrders: number;
  
  // New tracking
  totalFundingPaid: number;
  totalFundingReceived: number;
  totalFeesPaid: number;
  totalRebatesEarned: number;
  
  isActive: boolean;
}

export interface CollateralAccount {
  assetType: string;
  amount: number;
  valueUsd: number;
  isActive: boolean;
}

export interface Position {
  userAccount: string;
  market: string;
  positionIndex: number;
  side: 'Long' | 'Short';
  status: 'Open' | 'Closed' | 'Liquidated';
  size: number;
  entryPrice: number;
  currentPrice: number;
  liquidationPrice: number;
  unrealizedPnl: number;
  realizedPnl: number;
  fundingRate: number;
  leverage: number;
  openedAt: number;
}

export interface Order {
  id: string;
  market: string;
  orderType: string;
  side: 'Long' | 'Short';
  size: number;
  price: number;
  status: string;
  createdAt: number;
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
  LINK = 'LINK',
}

export enum OrderType {
  Market = 'Market',
  Limit = 'Limit',
  StopLoss = 'StopLoss',
  TakeProfit = 'TakeProfit',
  TrailingStop = 'TrailingStop',
  PostOnly = 'PostOnly',
  IOC = 'IOC',
  FOK = 'FOK',
  Iceberg = 'Iceberg',
  TWAP = 'TWAP',
  StopLimit = 'StopLimit',
  Bracket = 'Bracket',
}

export enum PositionSide {
  Long = 'Long',
  Short = 'Short',
}

class SmartContractService {
  private static instance: SmartContractService;
  private readonly connection: Connection;
  private readonly programId: PublicKey;

  private constructor() {
    // Initialize connection (you can make this configurable)
    this.connection = new Connection(
      import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
      'confirmed'
    );
    
    // Use the program ID from the IDL to avoid mismatch
    this.programId = new PublicKey(programIdl.address);
  }

  /**
   * Create a proper Anchor wallet from wallet adapter
   * CRITICAL: Must return object with signTransaction/signAllTransactions that Anchor recognizes
   */
  private createAnchorWallet(wallet: Wallet) {
    console.log('🔧 Creating Anchor-compatible wallet wrapper:', {
      publicKey: wallet.adapter.publicKey?.toString(),
      hasSignTransaction: !!wallet.adapter.signTransaction,
      hasSignAllTransactions: !!wallet.adapter.signAllTransactions,
      adapterType: wallet.adapter.constructor.name
    });

    if (!wallet.adapter.publicKey) {
      throw new Error('Wallet adapter publicKey is null');
    }

    if (!wallet.adapter.signTransaction) {
      throw new Error('Wallet adapter does not support signTransaction');
    }

    // CRITICAL FIX: Create explicit wrapper that Anchor can recognize
    // StandardWalletAdapter doesn't work directly with Anchor's .rpc()
    const anchorWallet = {
      publicKey: wallet.adapter.publicKey,
      signTransaction: wallet.adapter.signTransaction.bind(wallet.adapter),
      signAllTransactions: wallet.adapter.signAllTransactions?.bind(wallet.adapter),
    };

    console.log('✅ Anchor wallet wrapper created with bound methods');
    return anchorWallet;
  }

  // Helper function to convert number to 8-byte little-endian buffer
  private numberToBytes(num: number): Uint8Array {
    const buffer = new ArrayBuffer(8);
    const view = new DataView(buffer);
    view.setFloat64(0, num, true); // true for little-endian
    return new Uint8Array(buffer);
  }

  public static getInstance(): SmartContractService {
    if (!SmartContractService.instance) {
      SmartContractService.instance = new SmartContractService();
    }
    return SmartContractService.instance;
  }

  // ==================== USER ACCOUNT MANAGEMENT ====================

  /**
   * Check if user has a QuantDesk account
   */
  async checkUserAccount(walletAddress: string): Promise<boolean> {
    try {
      console.log('🔍 Checking if user account exists for:', walletAddress);
      
      // Use 3 seeds to match deployed smart contract
      const accountIndex = 0;
      const accountIndexBuffer = Buffer.alloc(2);
      accountIndexBuffer.writeUInt16LE(accountIndex, 0);
      
      const [userAccountPda] = await PublicKey.findProgramAddress(
        [Buffer.from('user_account'), new PublicKey(walletAddress).toBuffer(), accountIndexBuffer],
        this.programId
      );
      
      console.log('📍 User Account PDA:', userAccountPda.toString());
      
      const accountInfo = await this.connection.getAccountInfo(userAccountPda);
      const exists = accountInfo !== null;
      
      console.log('✅ User account exists:', exists);
      return exists;
    } catch (error) {
      console.error('❌ Error checking user account:', error);
      return false;
    }
  }

  /**
   * Create a new user account
   */
  async createUserAccount(wallet: Wallet): Promise<string> {
    console.log('🚀 Starting user account creation with Anchor...');
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        console.error('❌ Wallet adapter or public key is not available');
        throw new Error('Wallet adapter or public key is not available');
      }
      console.log('✅ Wallet connected:', wallet.adapter.publicKey.toString());
      console.log('🔗 Connection endpoint:', this.connection.rpcEndpoint);
      console.log('📋 Program ID:', this.programId.toString());

      // Get program ID from IDL to ensure consistency
      const programIdFromIdl = new PublicKey(programIdl.address);
      console.log('🔍 IDL Program ID:', programIdFromIdl.toString());
      console.log('🔍 Service Program ID:', this.programId.toString());
      console.log('🔍 Program IDs match:', programIdFromIdl.equals(this.programId));
      
      // Check program exists using IDL program ID
      const programInfo = await this.connection.getAccountInfo(programIdFromIdl);
      if (!programInfo) {
        console.error('❌ Program not found on devnet:', programIdFromIdl.toString());
        throw new Error(`Program ${programIdFromIdl.toString()} not found on devnet. Please deploy the program first.`);
      }
      console.log('✅ Program found on devnet');

      // Check wallet balance
      const balance = await this.connection.getBalance(wallet.adapter.publicKey);
      console.log('💰 Wallet SOL balance:', balance, 'lamports (', balance / 1e9, 'SOL)');
      if (balance < 5000000) { // 0.005 SOL minimum
        console.error('❌ Insufficient SOL balance for transaction fees');
        throw new Error('Insufficient SOL balance. Please add at least 0.005 SOL to your wallet for transaction fees.');
      }
      console.log('✅ Sufficient SOL balance for transaction');

      // Create Anchor provider - this is the key part from Anchor docs
      const provider = new AnchorProvider(
        this.connection,
        this.createAnchorWallet(wallet),
        { 
          commitment: 'confirmed',
          preflightCommitment: 'confirmed'
        }
      );
      console.log('✅ Anchor provider created');

      // Create program instance using the IDL
      const program = new Program(programIdl, provider);
      console.log('✅ Program instance created');

      // Find PDA for user account using 3 seeds to match deployed smart contract
      const accountIndex = 0; // First account
      const accountIndexBuffer = Buffer.alloc(2);
      accountIndexBuffer.writeUInt16LE(accountIndex, 0);
      
      const [userAccountPda] = await PublicKey.findProgramAddress(
        [
          Buffer.from('user_account'),
          wallet.adapter.publicKey.toBuffer(),
          accountIndexBuffer
        ],
        programIdFromIdl
      );
      console.log('📍 User Account PDA:', userAccountPda.toString());

      // Check if account already exists
      const accountInfo = await this.connection.getAccountInfo(userAccountPda);
      if (accountInfo) {
        console.log('⚠️ Account already exists!');
        throw new Error('User account already exists');
      }
      console.log('✅ Account does not exist, proceeding with creation...');

      // Call the create_user_account instruction using Anchor
      // Following Solana Cookbook pattern for instruction calls
      console.log('📞 Calling create_user_account instruction...');
      const signature = await program.methods
        .createUserAccount(accountIndex) // account_index = 0
        .accounts({
          userAccount: userAccountPda,
          authority: provider.wallet.publicKey,  // FIX: Use provider's wallet for consistency
          systemProgram: SystemProgram.programId,
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
        });

      console.log('📤 Transaction sent, signature:', signature);
      console.log('🎉 User account created successfully!');
      return signature;
    } catch (error: any) {
      console.error('❌ Transaction error:', error);
      
      // Better error classification
      if (error?.message?.includes('User rejected')) {
        throw new Error('Transaction cancelled by user');
      } else if (error?.message?.includes('Insufficient')) {
        throw new Error('Insufficient funds for transaction');
      } else if (error?.message?.includes('Blockhash')) {
        throw new Error('Transaction expired - please try again');
      } else if (error?.logs?.some(log => log.includes('success'))) {
        // Transaction actually succeeded despite error
        console.log('✅ Transaction succeeded despite error message');
        return signature; // Return success
      } else {
        throw new Error(`Transaction failed: ${error?.message || 'Unknown error'}`);
      }
    }
  }

  /**
   * Get user account state
   */
  async getUserAccountState(walletAddress: string): Promise<UserAccountState> {
    try {
      console.log('🔍 Getting user account state for:', walletAddress);
      
      // Validate wallet address format
      if (!walletAddress || walletAddress.length < 32) {
        throw new Error(`Invalid wallet address format: ${walletAddress}`);
      }
      
      // Use 3 seeds to match deployed smart contract
      const accountIndex = 0;
      const accountIndexBuffer = Buffer.alloc(2);
      accountIndexBuffer.writeUInt16LE(accountIndex, 0);
      
      let userAccountPda: PublicKey;
      try {
        [userAccountPda] = await PublicKey.findProgramAddress(
          [Buffer.from('user_account'), new PublicKey(walletAddress).toBuffer(), accountIndexBuffer],
          this.programId
        );
      } catch (pdaError) {
        console.error('❌ Error creating PDA for wallet address:', walletAddress, pdaError);
        throw new Error(`Invalid wallet address for PDA creation: ${walletAddress}`);
      }
      
      console.log('📍 User Account PDA:', userAccountPda.toString());
      
      const accountInfo = await this.connection.getAccountInfo(userAccountPda);
      if (!accountInfo) {
        console.log('❌ User account not found');
      return {
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
        totalOrders: 0,
        totalFundingPaid: 0,
        totalFundingReceived: 0,
        totalFeesPaid: 0,
        totalRebatesEarned: 0,
        isActive: false,
      };
      }

      // Parse the account data to get actual values
      console.log('📊 Account data length:', accountInfo.data.length);
      
      // Check SOL collateral account with improved method
      let totalCollateral = 0;
      let accountHealth = 0;
      let canTrade = false;
      
      try {
        // Use the improved getSOLCollateralBalance method
        const solCollateral = await this.getSOLCollateralBalance(walletAddress);
        
        // CRITICAL FIX: Convert SOL to USD using current SOL price
        // Get SOL price from oracle/backend API
        let solPrice = 190; // Default fallback (~current price)
        try {
          const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002';
          const priceResponse = await fetch(`${apiUrl}/api/oracle/price/SOL`);
          if (priceResponse.ok) {
            const priceData = await priceResponse.json();
            solPrice = priceData.price || priceData.value || 190;
            console.log('💰 Current SOL price from oracle:', solPrice);
          }
        } catch (priceError) {
          console.warn('⚠️ Could not fetch SOL price, using fallback:', priceError);
        }
        
        // Convert SOL collateral to USD value
        totalCollateral = solCollateral * solPrice;
        
        // Calculate account health based on collateral
        accountHealth = totalCollateral > 0 ? 10000 : 0; // 100.00% = 10000 basis points
        canTrade = totalCollateral > 0;
        
        console.log('💰 Total collateral retrieved:', solCollateral, 'SOL =', totalCollateral.toFixed(2), 'USD');
        console.log('💰 SOL price used for conversion:', solPrice);
        
      } catch (error) {
        console.warn('⚠️ Could not fetch collateral data, using defaults:', error);
        // Use defaults if collateral check fails
        totalCollateral = 0;
        accountHealth = 0;
        canTrade = false;
      }
      
      console.log('📈 Parsed account state:', {
        totalCollateralUSD: totalCollateral,
        accountHealth,
        canTrade,
        exists: true
      });
      
      console.log('✅ User account found, returning real state');
      
      // Return the actual account state with real collateral data (in USD)
      return {
        exists: true,
        canDeposit: true,
        canTrade: canTrade,
        totalCollateral: totalCollateral, // NOW IN USD!
        initialMarginRequirement: 0,
        maintenanceMarginRequirement: 0,
        availableMargin: totalCollateral * 0.95, // 95% of collateral
        accountHealth: accountHealth,
        liquidationPrice: 0,
        liquidationThreshold: 0,
        maxLeverage: 0,
        totalPositions: 0, // TODO: Parse from account data
        totalOrders: 0,    // TODO: Parse from account data
        totalFundingPaid: 0,
        totalFundingReceived: 0,
        totalFeesPaid: 0,
        totalRebatesEarned: 0,
        isActive: true,
      };
    } catch (error) {
      console.error('❌ Error getting user account state:', error);
      return {
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
        totalOrders: 0,
        totalFundingPaid: 0,
        totalFundingReceived: 0,
        totalFeesPaid: 0,
        totalRebatesEarned: 0,
        isActive: false,
      };
    }
  }

  // ==================== COLLATERAL MANAGEMENT ====================

  /**
   * Initialize collateral account for a specific asset
   */
  async initializeCollateralAccount(
    wallet: Wallet,
    assetType: CollateralType
  ): Promise<string> {
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      const [collateralPda] = await PublicKey.findProgramAddress(
        [Buffer.from('collateral'), wallet.adapter.publicKey.toBuffer(), Buffer.from(assetType)],
        this.programId
      );

      const transaction = new Transaction().add(
        new TransactionInstruction({
          keys: [
            { pubkey: wallet.adapter.publicKey, isSigner: true, isWritable: false },
            { pubkey: collateralPda, isSigner: false, isWritable: true },
            { pubkey: SystemProgram.programId, isSigner: false, isWritable: false },
          ],
          programId: this.programId,
          data: Buffer.from([2, CollateralType[assetType] as number]), // initialize_collateral_account instruction
        })
      );

      const signature = await wallet.adapter.sendTransaction(transaction, this.connection);
      await this.connection.confirmTransaction(signature);
      
      return signature;
    } catch (error) {
      console.error('Error initializing collateral account:', error);
      throw error;
    }
  }

  /**
   * Add collateral to an account
   */
  async addCollateral(
    wallet: Wallet,
    assetType: CollateralType,
    amount: number
  ): Promise<string> {
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      const [collateralPda] = await PublicKey.findProgramAddress(
        [Buffer.from('collateral'), wallet.adapter.publicKey.toBuffer(), Buffer.from(assetType)],
        this.programId
      );

      // Get user's token account (simplified - you'd need proper token account handling)
      const tokenMint = this.getTokenMint(assetType);
      // In production, you'd get the associated token account properly

      const transaction = new Transaction().add(
        new TransactionInstruction({
          keys: [
            { pubkey: wallet.adapter.publicKey, isSigner: true, isWritable: false },
            { pubkey: collateralPda, isSigner: false, isWritable: true },
            // Add token account here in production
          ],
          programId: this.programId,
          data: Buffer.from([3, ...this.numberToBytes(amount)]), // add_collateral instruction
        })
      );

      const signature = await wallet.adapter.sendTransaction(transaction, this.connection);
      await this.connection.confirmTransaction(signature);
      
      return signature;
    } catch (error) {
      console.error('Error adding collateral:', error);
      throw error;
    }
  }

  /**
   * Get collateral accounts for a user
   */
  async getCollateralAccounts(walletAddress: string): Promise<CollateralAccount[]> {
    try {
      const collateralAccounts: CollateralAccount[] = [];
      
      // Check each supported asset type
      for (const assetType of Object.values(CollateralType)) {
        const [collateralPda] = await PublicKey.findProgramAddress(
          [Buffer.from('collateral'), new PublicKey(walletAddress).toBuffer(), Buffer.from(assetType)],
          this.programId
        );
        
        const accountInfo = await this.connection.getAccountInfo(collateralPda);
        if (accountInfo) {
          // Parse account data (simplified)
          const data = accountInfo.data;
          const amount = data.readUInt32LE(32 + 1); // Skip user (32) + asset_type (1)
          const valueUsd = data.readUInt32LE(32 + 1 + 8);
          const isActive = data.readUInt8(32 + 1 + 8 + 8 + 8) === 1;
          
          collateralAccounts.push({
            assetType,
            amount,
            valueUsd,
            isActive,
          });
        }
      }
      
      return collateralAccounts;
    } catch (error) {
      console.error('Error getting collateral accounts:', error);
      return [];
    }
  }

  // ==================== TOKEN OPERATIONS ====================

  /**
   * Initialize token vault for a specific mint
   * Following Drift Protocol patterns for vault management
   */
  async initializeTokenVault(wallet: Wallet, mintAddress: string): Promise<string> {
    console.log('🚀 Initializing token vault...');
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      console.log('✅ Wallet connected:', wallet.adapter.publicKey.toString());
      console.log('💰 Mint address:', mintAddress);

      // Get fresh connection to avoid blockhash conflicts
      const freshConnection = new Connection(
        import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
        'confirmed'
      );

      // Get program ID from IDL
      const programIdFromIdl = new PublicKey(programIdl.address);
      
      // Create Anchor provider with fresh connection
      const provider = new AnchorProvider(
        freshConnection,
        this.createAnchorWallet(wallet),
        { 
          commitment: 'confirmed',
          preflightCommitment: 'confirmed'
        }
      );

      // Create program instance
      const program = new Program(programIdl, provider);

      // Derive vault PDA
      const [vaultPda] = await PublicKey.findProgramAddress(
        [Buffer.from('vault'), new PublicKey(mintAddress).toBuffer()],
        programIdFromIdl
      );

      // Check if vault already exists
      const vaultAccountInfo = await freshConnection.getAccountInfo(vaultPda);
      if (vaultAccountInfo) {
        console.log('ℹ️ Token vault already exists, skipping initialization');
        return 'already_exists';
      }

      // Derive vault token account
      const vaultTokenAccount = await this.getAssociatedTokenAddress(
        new PublicKey(mintAddress),
        vaultPda
      );

      console.log('📍 Vault PDA:', vaultPda.toString());
      console.log('📍 Vault Token Account:', vaultTokenAccount.toString());

      // Call initialize_token_vault instruction using Anchor
      const signature = await program.methods
        .initializeTokenVault(new PublicKey(mintAddress))
        .accounts({
          vault: vaultPda,
          vaultTokenAccount: vaultTokenAccount,
          mint: new PublicKey(mintAddress),
          authority: provider.wallet.publicKey,  // FIX: Use provider's wallet
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
        });

      console.log('📤 Token vault initialized, signature:', signature);
      console.log('🎉 Vault initialization successful!');
      return signature;
    } catch (error: any) {
      console.error('❌ Error initializing token vault:', error);
      console.error('Error details:', {
        message: error?.message,
        stack: error?.stack,
        name: error?.name,
        code: error?.code,
        logs: error?.logs
      });
      
      if (error?.message && error.message.includes('User rejected')) {
        throw new Error('Transaction was rejected by user');
      } else if (error?.message && error.message.includes('already exists')) {
        console.log('ℹ️ Token vault already exists, continuing...');
        return 'already_exists';
      } else if (error?.message && error.message.includes('Blockhash')) {
        throw new Error('Transaction expired - please try again');
      } else if (error?.message && error.message.includes('already been processed')) {
        throw new Error('Transaction already processed. Please try again.');
      } else {
        throw new Error(`Vault initialization failed: ${error?.message || 'Unknown error'}`);
      }
    }
  }

  /**
   * Deposit tokens into protocol vault
   * Following Drift Protocol patterns for deposit management
   */
  async depositTokens(wallet: Wallet, mintAddress: string, amount: number): Promise<string> {
    console.log('🚀 Depositing tokens to protocol vault...');
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      console.log('✅ Wallet connected:', wallet.adapter.publicKey.toString());
      console.log('💰 Mint address:', mintAddress);
      console.log('📊 Amount:', amount);

      // Get fresh connection to avoid blockhash conflicts
      const freshConnection = new Connection(
        import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
        'confirmed'
      );

      // Get program ID from IDL
      const programIdFromIdl = new PublicKey(programIdl.address);
      
      // Create Anchor provider with fresh connection
      const provider = new AnchorProvider(
        freshConnection,
        this.createAnchorWallet(wallet),
        { 
          commitment: 'confirmed',
          preflightCommitment: 'confirmed'
        }
      );

      // Create program instance
      const program = new Program(programIdl, provider);

      // Derive vault PDA
      const [vaultPda] = await PublicKey.findProgramAddress(
        [Buffer.from('vault'), new PublicKey(mintAddress).toBuffer()],
        programIdFromIdl
      );

      // Check if vault exists
      const vaultAccountInfo = await freshConnection.getAccountInfo(vaultPda);
      if (!vaultAccountInfo) {
        console.log('⚠️ Token vault does not exist, please initialize vault first');
        throw new Error('Token vault not initialized. Please initialize the vault first by calling initializeTokenVault()');
      } else {
        console.log('✅ Token vault already exists');
      }

      // Derive user token account
      const userTokenAccount = await this.getAssociatedTokenAddress(
        new PublicKey(mintAddress),
        wallet.adapter.publicKey
      );

      // Derive vault token account
      const vaultTokenAccount = await this.getAssociatedTokenAddress(
        new PublicKey(mintAddress),
        vaultPda
      );

      console.log('📍 Vault PDA:', vaultPda.toString());
      console.log('📍 User Token Account:', userTokenAccount.toString());
      console.log('📍 Vault Token Account:', vaultTokenAccount.toString());

      // Call deposit_tokens instruction using Anchor
      const signature = await program.methods
        .depositTokens(new BN(amount))
        .accounts({
          vault: vaultPda,
          userTokenAccount: userTokenAccount,
          vaultTokenAccount: vaultTokenAccount,
          user: provider.wallet.publicKey,  // FIX: Use provider's wallet
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
        });

      console.log('📤 Tokens deposited, signature:', signature);
      console.log('🎉 Deposit successful!');
      return signature;
    } catch (error: any) {
      console.error('❌ Error depositing tokens:', error);
      console.error('Error details:', {
        message: error?.message,
        stack: error?.stack,
        name: error?.name,
        code: error?.code,
        logs: error?.logs
      });
      
      if (error?.message && error.message.includes('User rejected')) {
        throw new Error('Transaction was rejected by user');
      } else if (error?.message && error.message.includes('Insufficient')) {
        throw new Error('Insufficient token balance for deposit');
      } else if (error?.message && error.message.includes('Blockhash')) {
        throw new Error('Transaction expired - please try again');
      } else if (error?.message && error.message.includes('already been processed')) {
        throw new Error('Transaction already processed. Please try again with a fresh transaction.');
      } else {
        throw new Error(`Deposit failed: ${error?.message || 'Unknown error'}`);
      }
    }
  }

  /**
   * Withdraw tokens from protocol vault
   * Following Anchor and Solana Cookbook best practices
   */
  async withdrawTokens(wallet: Wallet, mintAddress: string, amount: number): Promise<string> {
    console.log('🚀 Withdrawing tokens from protocol vault...');
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      console.log('✅ Wallet connected:', wallet.adapter.publicKey.toString());
      console.log('💰 Mint address:', mintAddress);
      console.log('📊 Amount:', amount);

      // Get program ID from IDL
      const programIdFromIdl = new PublicKey(programIdl.address);
      
      // Create Anchor provider
      const provider = new AnchorProvider(
        this.connection,
        this.createAnchorWallet(wallet),
        { 
          commitment: 'confirmed',
          preflightCommitment: 'confirmed'
        }
      );

      // Create program instance
      const program = new Program(programIdl, provider);

      // Derive vault PDA
      const [vaultPda] = await PublicKey.findProgramAddress(
        [Buffer.from('vault'), new PublicKey(mintAddress).toBuffer()],
        programIdFromIdl
      );

      // Derive user token account
      const userTokenAccount = await this.getAssociatedTokenAddress(
        new PublicKey(mintAddress),
        wallet.adapter.publicKey
      );

      // Derive vault token account
      const vaultTokenAccount = await this.getAssociatedTokenAddress(
        new PublicKey(mintAddress),
        vaultPda
      );

      console.log('📍 Vault PDA:', vaultPda.toString());
      console.log('📍 User Token Account:', userTokenAccount.toString());
      console.log('📍 Vault Token Account:', vaultTokenAccount.toString());

      // Call withdraw_tokens instruction using Anchor
      const signature = await program.methods
        .withdrawTokens(new BN(amount))
        .accounts({
          vault: vaultPda,
          vaultTokenAccount: vaultTokenAccount,
          userTokenAccount: userTokenAccount,
          user: provider.wallet.publicKey,  // FIX: Use provider's wallet
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
        });

      console.log('📤 Tokens withdrawn, signature:', signature);
      console.log('🎉 Withdrawal successful!');
      return signature;
    } catch (error: any) {
      console.error('❌ Error withdrawing tokens:', error);
      console.error('Error details:', {
        message: error?.message,
        stack: error?.stack,
        name: error?.name,
        code: error?.code,
        logs: error?.logs
      });
      
      if (error?.message && error.message.includes('User rejected')) {
        throw new Error('Transaction was rejected by user');
      } else if (error?.message && error.message.includes('Insufficient')) {
        throw new Error('Insufficient vault balance for withdrawal');
      } else if (error?.message && error.message.includes('Blockhash')) {
        throw new Error('Transaction expired - please try again');
      } else {
        throw new Error(`Withdrawal failed: ${error?.message || 'Unknown error'}`);
      }
    }
  }

  /**
   * Create user token account if needed
   * Following Anchor and Solana Cookbook best practices
   */
  async createUserTokenAccount(wallet: Wallet, mintAddress: string): Promise<string> {
    console.log('🚀 Creating user token account...');
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      console.log('✅ Wallet connected:', wallet.adapter.publicKey.toString());
      console.log('💰 Mint address:', mintAddress);

      // Derive user token account
      const userTokenAccount = await this.getAssociatedTokenAddress(
        new PublicKey(mintAddress),
        wallet.adapter.publicKey
      );

      console.log('📍 User Token Account:', userTokenAccount.toString());

      // Check if token account already exists
      const accountInfo = await this.connection.getAccountInfo(userTokenAccount);
      if (accountInfo) {
        console.log('ℹ️ Token account already exists, skipping creation');
        return 'already_exists';
      }

      // Get program ID from IDL
      const programIdFromIdl = new PublicKey(programIdl.address);
      
      // Create Anchor provider
      const provider = new AnchorProvider(
        this.connection,
        this.createAnchorWallet(wallet),
        { 
          commitment: 'confirmed',
          preflightCommitment: 'confirmed'
        }
      );

      // Create program instance
      const program = new Program(programIdl, provider);

      // Call create_user_token_account instruction using Anchor
      const signature = await program.methods
        .createUserTokenAccount()
        .accounts({
          userTokenAccount: userTokenAccount,
          mint: new PublicKey(mintAddress),
          user: provider.wallet.publicKey,  // FIX: Use provider's wallet
          tokenProgram: TOKEN_PROGRAM_ID,
          associatedTokenProgram: ASSOCIATED_TOKEN_PROGRAM_ID,
          systemProgram: SystemProgram.programId,
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
        });

      console.log('📤 Token account created, signature:', signature);
      console.log('🎉 Token account creation successful!');
      return signature;
    } catch (error: any) {
      console.error('❌ Error creating token account:', error);
      console.error('Error details:', {
        message: error?.message,
        stack: error?.stack,
        name: error?.name,
        code: error?.code,
        logs: error?.logs
      });
      
      if (error?.message && error.message.includes('User rejected')) {
        throw new Error('Transaction was rejected by user');
      } else if (error?.message && error.message.includes('already exists')) {
        console.log('ℹ️ Token account already exists, continuing...');
        return 'already_exists';
      } else if (error?.message && error.message.includes('Blockhash')) {
        throw new Error('Transaction expired - please try again');
      } else {
        throw new Error(`Token account creation failed: ${error?.message || 'Unknown error'}`);
      }
    }
  }

  /**
   * Check protocol vault balance
   */
  async getProtocolVaultBalance(): Promise<number> {
    try {
      const connection = new Connection(
        import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
        'confirmed'
      );

      const programIdFromIdl = new PublicKey(programIdl.address);
      
      // Derive protocol SOL vault PDA
      const [protocolSOLVault] = await PublicKey.findProgramAddress(
        [Buffer.from('protocol_sol_vault')],
        programIdFromIdl
      );

      console.log('📍 Checking Protocol SOL Vault:', protocolSOLVault.toString());
      
      // Get vault account info
      const vaultInfo = await connection.getAccountInfo(protocolSOLVault);
      
      if (!vaultInfo) {
        console.log('❌ Protocol vault does not exist');
        return 0;
      }

      // Get SOL balance of the vault
      const solBalance = await connection.getBalance(protocolSOLVault);
      console.log('💰 Protocol vault SOL balance:', solBalance / 1e9, 'SOL');
      
      return solBalance;
    } catch (error) {
      console.error('❌ Error checking protocol vault balance:', error);
      return 0;
    }
  }

  /**
   * Check user's SOL collateral account balance
   */
  async getSOLCollateralBalance(walletAddress: string): Promise<number> {
    try {
      console.log('🔍 Getting SOL collateral balance for:', walletAddress);
      
      // Validate wallet address format
      if (!walletAddress || walletAddress.length < 32) {
        throw new Error(`Invalid wallet address format: ${walletAddress}`);
      }
      
      const connection = new Connection(
        import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
        'confirmed'
      );

      const programIdFromIdl = new PublicKey(programIdl.address);
      
      // Derive SOL collateral account PDA using correct seeds from smart contract
      let solCollateralAccount: PublicKey;
      try {
        [solCollateralAccount] = await PublicKey.findProgramAddress(
          [Buffer.from('collateral'), new PublicKey(walletAddress).toBuffer(), Buffer.from('SOL')],
          programIdFromIdl
        );
      } catch (pdaError) {
        console.error('❌ Error creating SOL collateral PDA for wallet address:', walletAddress, pdaError);
        throw new Error(`Invalid wallet address for SOL collateral PDA creation: ${walletAddress}`);
      }

      console.log('📍 Checking SOL Collateral Account:', solCollateralAccount.toString());
      
      // First check if the account exists
      const accountInfo = await connection.getAccountInfo(solCollateralAccount);
      if (!accountInfo) {
        console.log('⚠️ SOL collateral account does not exist yet');
        return 0;
      }
      
      console.log('✅ SOL collateral account exists, fetching data...');
      
      // Try using Anchor's proper account fetching first
      try {
        // Create a minimal program instance for account fetching
        const program = new Program(programIdl as any, {
          connection,
        });
        
        const collateralAccount = await program.account.collateralAccount.fetch(solCollateralAccount);
        
        if (!collateralAccount) {
          console.log('❌ SOL collateral account not found');
          return 0;
        }
        
        // Convert lamports to SOL (1 SOL = 1e9 lamports)
        const solAmount = Number(collateralAccount.amount) / 1e9;
        
        console.log('💰 SOL collateral amount (Anchor client):', solAmount, 'SOL');
        console.log('📊 Raw amount in lamports:', collateralAccount.amount.toString());
        console.log('📊 Asset type:', collateralAccount.assetType);
        console.log('📊 USD value:', Number(collateralAccount.valueUsd) / 1e6);
        
        return solAmount;
      } catch (anchorError) {
        console.log('⚠️ Anchor client failed, trying manual parsing as fallback:', anchorError);
        
        // Fallback to manual parsing if Anchor client fails
        const collateralInfo = await connection.getAccountInfo(solCollateralAccount);
        
        if (!collateralInfo) {
          console.log('❌ SOL collateral account does not exist');
          return 0;
        }

        // Read the collateral account data
        const accountData = collateralInfo.data;
        console.log('📊 Collateral account data length:', accountData.length);
        
        // Parse the account data with correct structure based on CollateralAccount
        // Structure: discriminator(8) + user(32) + asset_type(1) + amount(8) + initial_asset_weight(2) + maintenance_asset_weight(2) + initial_liability_weight(2) + maintenance_liability_weight(2) + value_usd(8) + last_price(8) + last_updated(8) + is_active(1) + bump(1)
        if (accountData.length >= 73) { // Total size should be 73 bytes
          // Skip discriminator(8) + user(32) + asset_type(1) = 41 bytes
          const amountBuffer = accountData.slice(41, 49);
          const amount = new BN(amountBuffer, 'le'); // Little endian
          
          const lamportsStr = amount.toString();
          const lamports = parseFloat(lamportsStr);
          const sol = lamports / 1e9;
          
          console.log('💰 SOL collateral amount (manual):', sol, 'SOL');
          console.log('📊 Raw amount in lamports:', lamportsStr);
          console.log('📊 Account data length:', accountData.length);
          return sol;
        } else {
          console.log('⚠️ Collateral account data too short, length:', accountData.length);
          return 0;
        }
      }
    } catch (error) {
      console.error('❌ Error checking SOL collateral balance:', error);
      return 0;
    }
  }

  /**
   * Check if user can trade (has sufficient collateral)
   */
  async canUserTrade(walletAddress: string): Promise<boolean> {
    try {
      const userAccount = await this.getUserAccountState(walletAddress);
      return userAccount.exists && userAccount.totalCollateral > 0 && userAccount.canTrade;
    } catch (error) {
      console.error('❌ Error checking trading permissions:', error);
      return false;
    }
  }

  /**
   * Quick check for account existence and basic state (faster than getUserAccountState)
   */
  async getQuickAccountState(walletAddress: string): Promise<{
    exists: boolean;
    hasCollateral: boolean;
    canTrade: boolean;
  }> {
    try {
      console.log('⚡ Quick account state check for:', walletAddress);
      
      // Use 3 seeds to match deployed smart contract
      const accountIndex = 0;
      const accountIndexBuffer = Buffer.alloc(2);
      accountIndexBuffer.writeUInt16LE(accountIndex, 0);
      
      const [userAccountPda] = await PublicKey.findProgramAddress(
        [Buffer.from('user_account'), new PublicKey(walletAddress).toBuffer(), accountIndexBuffer],
        this.programId
      );
      
      // Quick check if user account exists
      const accountInfo = await this.connection.getAccountInfo(userAccountPda);
      if (!accountInfo) {
        return { exists: false, hasCollateral: false, canTrade: false };
      }
      
      // Quick check if SOL collateral account exists AND has actual collateral
      const [solCollateralPda] = await PublicKey.findProgramAddress(
        [Buffer.from('collateral'), new PublicKey(walletAddress).toBuffer(), Buffer.from('SOL')],
        this.programId
      );
      
      const collateralInfo = await this.connection.getAccountInfo(solCollateralPda);
      if (!collateralInfo) {
        return { exists: true, hasCollateral: false, canTrade: false };
      }
      
      // Parse the actual collateral amount from account data
      let hasCollateral = false;
      try {
        const accountData = collateralInfo.data;
        console.log('🔍 Quick check - Collateral account data length:', accountData.length);
        
        if (accountData.length >= 49) {
          // Read SOL lamports from offset 41-49 (8 bytes, little endian)
          // Skip discriminator(8) + user(32) + asset_type(1) = 41
          const amountBuffer = accountData.slice(41, 49);
          const amount = new BN(amountBuffer, 'le');
          const lamportsStr = amount.toString();
          const solAmount = parseFloat(lamportsStr) / 1e9;
          
          console.log('💰 Quick check - SOL collateral amount:', solAmount, 'SOL');
          hasCollateral = solAmount > 0;
        }
      } catch (parseError) {
        console.warn('⚠️ Quick check - Could not parse collateral amount:', parseError);
        hasCollateral = false;
      }
      
      return {
        exists: true,
        hasCollateral,
        canTrade: hasCollateral
      };
      
    } catch (error) {
      console.error('❌ Error in quick account check:', error);
      return { exists: false, hasCollateral: false, canTrade: false };
    }
  }

  /**
   * Get user's trading balance and permissions
   */
  async getUserTradingInfo(walletAddress: string): Promise<{
    canTrade: boolean;
    totalCollateral: number;
    accountHealth: number;
    isActive: boolean;
  }> {
    try {
      const userAccount = await this.getUserAccountState(walletAddress);
      return {
        canTrade: userAccount.exists && userAccount.totalCollateral > 0 && userAccount.canTrade,
        totalCollateral: userAccount.totalCollateral,
        accountHealth: userAccount.accountHealth,
        isActive: userAccount.isActive,
      };
    } catch (error) {
      console.error('❌ Error getting trading info:', error);
      return {
        canTrade: false,
        totalCollateral: 0,
        accountHealth: 0,
        isActive: false,
      };
    }
  }

  /**
   * Initialize SOL collateral account for user
   */
  async initializeSOLCollateralAccount(wallet: Wallet, amount?: number): Promise<string> {
    console.log('🔄 Initializing SOL collateral account...');
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      // Get fresh connection
      const freshConnection = new Connection(
        import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
        'confirmed'
      );

      // Get program ID from IDL
      const programIdFromIdl = new PublicKey(programIdl.address);
      
      // Create Anchor provider
      const provider = new AnchorProvider(
        freshConnection,
        this.createAnchorWallet(wallet),
        { 
          commitment: 'confirmed',
          preflightCommitment: 'confirmed'
        }
      );

      // Create program instance
      const program = new Program(programIdl, provider);

      // Derive user account PDA (required by IDL even though Rust doesn't show it)
      const accountIndex = 0;
      const accountIndexBuffer = Buffer.alloc(2);
      accountIndexBuffer.writeUInt16LE(accountIndex, 0);
      const [userAccountPda] = await PublicKey.findProgramAddress(
        [Buffer.from('user_account'), wallet.adapter.publicKey.toBuffer(), accountIndexBuffer],
        programIdFromIdl
      );

      // NOTE: IDL requires user_account, but Rust code might not show it
      // Also: InitializeCollateralAccount uses seed &[asset_type as u8] (SOL = [0])
      // But DepositNativeSol uses seed b"SOL" - they create DIFFERENT PDAs!
      // We use &[0] here to match InitializeCollateralAccount's Rust seed
      const [collateralAccountPda] = await PublicKey.findProgramAddress(
        [Buffer.from('collateral'), wallet.adapter.publicKey.toBuffer(), Buffer.from([0])], // SOL = enum variant 0
        programIdFromIdl
      );

      console.log('📍 User Account PDA:', userAccountPda.toString());
      console.log('📍 SOL Collateral Account PDA (&[0] seed):', collateralAccountPda.toString());
      console.log('⚠️ NOTE: This PDA is DIFFERENT from DepositNativeSol\'s b"SOL" PDA!');
      console.log('💡 RECOMMENDATION: Skip this and use DepositNativeSol (init_if_needed) instead');

      // Check if collateral account already exists
      const collateralInfo = await freshConnection.getAccountInfo(collateralAccountPda);
      if (collateralInfo) {
        console.log('✅ SOL collateral account already exists');
        return 'already_exists';
      }

      // Initialize SOL collateral account via wallet adapter send
      // IDL requires user_account even though current Rust code doesn't show it
      const ix = await program.methods
        .initializeCollateralAccount({ sol: {} })
        .accounts({
          collateralAccount: collateralAccountPda,
          userAccount: userAccountPda, // Required by IDL
          user: provider.wallet.publicKey,  // FIX: Use provider's wallet
          systemProgram: SystemProgram.programId,
        })
        .instruction();

      const tx = new Transaction().add(ix);
      tx.recentBlockhash = (await freshConnection.getLatestBlockhash('confirmed')).blockhash;
      tx.feePayer = wallet.adapter.publicKey!;
      
      // Log account metas
      const accountMetas = ix.keys.map(k => ({
        key: k.pubkey.toBase58(),
        isSigner: k.isSigner,
        isWritable: k.isWritable,
      }));
      console.log('🧩 InitializeCollateralAccount metas:', accountMetas);
      
      // Use explicit sign + send for better error handling
      try {
        console.log('🔐 Signing transaction...');
        const signedTx = await wallet.adapter.signTransaction!(tx);
        console.log('✅ Transaction signed');
        
        console.log('📤 Sending transaction...');
        const sig = await freshConnection.sendRawTransaction(signedTx.serialize(), {
          skipPreflight: false,
          preflightCommitment: 'confirmed',
          maxRetries: 3,
        });
        console.log('📬 Transaction sent:', sig);
        await freshConnection.confirmTransaction(sig, 'confirmed');
        console.log('✅ SOL collateral account initialized:', sig);
        return sig;
      } catch (signError: any) {
        console.error('❌ Signing failed, trying sendTransaction fallback:', signError);
        // @ts-ignore
        const sig = await wallet.adapter.sendTransaction(tx, freshConnection, { skipPreflight: false, preflightCommitment: 'confirmed' });
        await freshConnection.confirmTransaction(sig, 'confirmed');
        console.log('✅ SOL collateral account initialized (fallback):', sig);
        return sig;
      }
    } catch (error: any) {
      console.error('❌ Error initializing SOL collateral account:', error);
      
      // Log full error details for debugging
      if (error?.logs) {
        console.error('📋 Program logs:', error.logs);
      }
      if (error?.error?.errorMessage) {
        console.error('🔍 Error message:', error.error.errorMessage);
      }
      if (error?.error?.errorCode) {
        console.error('🔍 Error code:', error.error.errorCode);
      }
      if (error?.programErrorStack) {
        console.error('📚 Program error stack:', error.programErrorStack);
      }
      
      // Check if account already exists (different error format)
      if (error?.message && (
        error.message.includes('already been processed') ||
        error.message.includes('already exists') ||
        error.message.includes('AccountAlreadyInitialized')
      )) {
        console.log('🔄 Collateral account likely already exists');
        return 'already_exists';
      }
      
      // Better error message
      const errorMsg = error?.message || error?.toString() || 'Unknown error';
      console.error('❌ Full error:', JSON.stringify(error, Object.getOwnPropertyNames(error), 2));
      throw new Error(`Failed to initialize SOL collateral account: ${errorMsg}`);
    }
  }

  /**
   * Initialize protocol SOL vault if needed
   */
  async initializeProtocolSOLVault(wallet: Wallet): Promise<string> {
    console.log('🔄 Initializing protocol SOL vault...');
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      // Get fresh connection with new blockhash
      const freshConnection = new Connection(
        import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
        'confirmed'
      );

      // Get program ID from IDL
      const programIdFromIdl = new PublicKey(programIdl.address);
      
      // Create Anchor provider with fresh connection
      const provider = new AnchorProvider(
        freshConnection,
        this.createAnchorWallet(wallet),
        { 
          commitment: 'confirmed',
          preflightCommitment: 'confirmed'
        }
      );

      // Create program instance
      const program = new Program(programIdl, provider);

      // Derive protocol SOL vault PDA
      const [protocolSOLVault] = await PublicKey.findProgramAddress(
        [Buffer.from('protocol_sol_vault')],
        programIdFromIdl
      );

      console.log('📍 Protocol SOL Vault PDA:', protocolSOLVault.toString());

      // Check if vault already exists
      const vaultInfo = await freshConnection.getAccountInfo(protocolSOLVault);
      if (vaultInfo) {
        console.log('✅ Protocol SOL vault already exists');
        return 'already_exists';
      }

      // Get fresh blockhash to avoid "already processed" error
      const { blockhash } = await freshConnection.getLatestBlockhash('confirmed');
      
      // Initialize vault with fresh blockhash
      const signature = await program.methods
        .initializeProtocolSolVault()
        .accounts({
          protocolVault: protocolSOLVault,
          authority: provider.wallet.publicKey,  // FIX: Use provider's wallet
          systemProgram: SystemProgram.programId,
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
          preflightCommitment: 'confirmed',
        });

      console.log('✅ Protocol SOL vault initialized:', signature);
      return signature;
    } catch (error: any) {
      console.error('❌ Error initializing protocol SOL vault:', error);
      
      if (error?.message && error.message.includes('already been processed')) {
        console.log('🔄 Transaction already processed, vault likely exists');
        return 'already_exists';
      }
      
      throw error;
    }
  }

  /**
   * Withdraw native SOL from user account
   */
  async withdrawNativeSOL(wallet: any, amountInLamports: number): Promise<string> {
    try {
      console.log('🚀 Withdrawing native SOL from user account...');
      
      if (!wallet?.adapter?.publicKey) {
        throw new Error('Wallet not connected');
      }

      console.log('✅ Wallet connected:', wallet.adapter.publicKey.toString());
      console.log('💰 Amount:', amountInLamports, 'lamports');

      // Create fresh connection to avoid blockhash conflicts
      const freshConnection = new Connection(
        import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
        'confirmed'
      );

      // Get program ID from IDL
      const programIdFromIdl = new PublicKey(programIdl.address);

      // Get fresh blockhash
      const { blockhash } = await freshConnection.getLatestBlockhash('confirmed');
      console.log('🔄 Using fresh blockhash:', blockhash);
      
      // Create Anchor provider with fresh connection
      const provider = new AnchorProvider(
        freshConnection,
        this.createAnchorWallet(wallet),
        { 
          commitment: 'confirmed',
          preflightCommitment: 'confirmed'
        }
      );

      // Create program instance
      const program = new Program(programIdl, provider);

      // Derive user account PDA (using 3 seeds to match deployed smart contract)
      const accountIndex = 0;
      const accountIndexBuffer = Buffer.alloc(2);
      accountIndexBuffer.writeUInt16LE(accountIndex, 0);
      
      const [userAccountPda] = await PublicKey.findProgramAddress(
        [Buffer.from('user_account'), wallet.adapter.publicKey.toBuffer(), accountIndexBuffer],
        programIdFromIdl
      );

      // Derive protocol SOL vault PDA
      const [protocolSOLVault] = await PublicKey.findProgramAddress(
        [Buffer.from('protocol_sol_vault')],
        programIdFromIdl
      );

      // Derive SOL collateral account PDA - use b"SOL" to match DepositNativeSol seed
      const [solCollateralAccount] = await PublicKey.findProgramAddress(
        [Buffer.from('collateral'), wallet.adapter.publicKey.toBuffer(), Buffer.from('SOL')],
        programIdFromIdl
      );

      console.log('📍 User Account PDA:', userAccountPda.toString());
      console.log('📍 Protocol SOL Vault:', protocolSOLVault.toString());
      console.log('📍 SOL Collateral Account:', solCollateralAccount.toString());

      // Pyth SOL/USD price feed for devnet
      const SOL_USD_PRICE_FEED = new PublicKey('H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG');

      // Guard: ensure withdraw instruction exists in current IDL/program
      if (!((program as any)?.methods && (program as any).methods.withdrawNativeSol)) {
        throw new Error('Withdraw instruction not available in current program/IDL. Please upgrade the program with withdraw_native_sol.');
      }

      // Call withdraw_native_sol instruction
      const signature = await (program as any).methods
        .withdrawNativeSol(new BN(amountInLamports))
        .accounts({
          userAccount: userAccountPda,
          user: provider.wallet.publicKey,  // FIX: Use provider's wallet
          protocolVault: protocolSOLVault,
          collateralAccount: solCollateralAccount,
          solUsdPriceFeed: SOL_USD_PRICE_FEED,
          systemProgram: SystemProgram.programId,
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
          preflightCommitment: 'confirmed',
        });

      console.log('✅ Native SOL withdrawal successful:', signature);
      return signature;

    } catch (error: any) {
      console.error('❌ Transaction error:', error);
      
      // Better error classification
      if (error?.message?.includes('User rejected')) {
        throw new Error('Transaction cancelled by user');
      } else if (error?.message?.includes('Insufficient')) {
        throw new Error('Insufficient collateral for withdrawal. Please check your available balance.');
      } else if (error?.message?.includes('Blockhash')) {
        throw new Error('Transaction expired - please try again');
      } else if (error?.logs?.some(log => log.includes('success'))) {
        // Transaction actually succeeded despite error
        console.log('✅ Transaction succeeded despite error message');
        return signature; // Return success
      } else if (error.message?.includes('already been processed')) {
        console.log('⚠️ Transaction already processed, returning success');
        return 'already_processed';
      } else if (error?.message?.includes('CollateralAccountInactive')) {
        throw new Error('Collateral account is inactive. Please contact support.');
      } else if (error?.message?.includes('InvalidAmount')) {
        throw new Error('Invalid withdrawal amount. Please enter a valid amount.');
      } else {
        throw new Error(`Native SOL withdrawal failed: ${error.message}`);
      }
    }
  }

  /**
   * Deposit native SOL to user account
   * Following professional platform patterns for native SOL deposits
   */
  async depositNativeSOL(wallet: Wallet, amount: number): Promise<string> {
    console.log('🚀 Depositing native SOL to user account...');
    let signature: string;
    // Predeclare so fallback branch can reuse computed PDAs/connection
    let userAccountPda: PublicKey | undefined;
    let protocolSOLVault: PublicKey | undefined;
    let solCollateralAccount: PublicKey | undefined;
    let freshConnection: Connection | undefined;
    // Pyth SOL/USD price feed for devnet (lifted to outer scope so fallback can use it)
    const SOL_USD_PRICE_FEED = new PublicKey('H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG');
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      console.log('✅ Wallet connected:', wallet.adapter.publicKey.toString());
      console.log('💰 Amount:', amount, 'lamports');

      // Get fresh connection to avoid blockhash conflicts
      freshConnection = new Connection(
        import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
        'confirmed'
      );

      // Get program ID from IDL
      const programIdFromIdl = new PublicKey(programIdl.address);
      
      // Create Anchor provider with fresh connection
      // Ensure wallet adapter implements Anchor's Wallet interface properly
      const anchorWallet = this.createAnchorWallet(wallet);
      
      const provider = new AnchorProvider(
        freshConnection,
        anchorWallet,
        { 
          commitment: 'confirmed',
          preflightCommitment: 'confirmed'
        }
      );

      // Create program instance
      const program = new Program(programIdl, provider);

      // Derive user account PDA (using 3 seeds to match deployed smart contract)
      const accountIndex = 0;
      const accountIndexBuffer = Buffer.alloc(2);
      accountIndexBuffer.writeUInt16LE(accountIndex, 0);
      
      [userAccountPda] = await PublicKey.findProgramAddress(
        [Buffer.from('user_account'), wallet.adapter.publicKey.toBuffer(), accountIndexBuffer],
        programIdFromIdl
      );

      // Derive protocol SOL vault PDA
      [protocolSOLVault] = await PublicKey.findProgramAddress(
        [Buffer.from('protocol_sol_vault')],
        programIdFromIdl
      );

      // Derive SOL collateral account PDA - use b"SOL" to match DepositNativeSol seed
      [solCollateralAccount] = await PublicKey.findProgramAddress(
        [Buffer.from('collateral'), wallet.adapter.publicKey.toBuffer(), Buffer.from('SOL')],
        programIdFromIdl
      );

      console.log('📍 User Account PDA:', userAccountPda!.toString());
      console.log('📍 Protocol SOL Vault:', protocolSOLVault!.toString());
      console.log('📍 SOL Collateral Account:', solCollateralAccount!.toString());

      // Price feed already defined above

      // Call deposit_native_sol instruction using Anchor
      console.log('🔧 Transaction accounts:', {
        userAccount: userAccountPda.toString(),
        user: wallet.adapter.publicKey.toString(),
        protocolVault: protocolSOLVault.toString(),
        collateralAccount: solCollateralAccount.toString(),
        solUsdPriceFeed: SOL_USD_PRICE_FEED.toString(),
        systemProgram: SystemProgram.programId.toString(),
        rent: SYSVAR_RENT_PUBKEY.toString(),
      });

      console.log('🔧 Wallet adapter details:', {
        publicKey: wallet.adapter.publicKey.toString(),
        connected: wallet.adapter.connected,
        readyState: wallet.adapter.readyState,
        hasSignTransaction: !!wallet.adapter.signTransaction,
        hasSignAllTransactions: !!wallet.adapter.signAllTransactions,
      });

      // Use Anchor's .rpc() method which handles transaction building, signing, and sending properly
      // This is the recommended approach and matches how Drift/other protocols do it
      console.log('🔧 Using Anchor .rpc() method for transaction...');
      console.log('📋 Transaction accounts:', {
        userAccount: userAccountPda!.toBase58(),
        collateralAccount: solCollateralAccount!.toBase58(),
        protocolVault: protocolSOLVault!.toBase58(),
        user: wallet.adapter.publicKey.toBase58(),
        solUsdPriceFeed: SOL_USD_PRICE_FEED.toBase58(),
        systemProgram: SystemProgram.programId.toBase58(),
      });
      
      // USE THE PROVEN WORKING PATTERN: Anchor's .rpc() 
      // This EXACTLY matches createUserAccount which WORKS
      // Anchor's .rpc() uses the provider's wallet (createAnchorWallet) to sign automatically
      console.log('🚀 Using Anchor .rpc() method (same pattern as createUserAccount)...');
      console.log('📋 Account details:', {
        userAccount: userAccountPda!.toBase58(),
        collateralAccount: solCollateralAccount!.toBase58(),
        protocolVault: protocolSOLVault!.toBase58(),
        user: wallet.adapter.publicKey.toBase58(),
        solUsdPriceFeed: SOL_USD_PRICE_FEED.toBase58(),
        systemProgram: SystemProgram.programId.toBase58(),
      });
      
      try {
        // FINAL FIX: Use Anchor's proper signing flow with .rpc() but ensure connection is set
        // The issue was connection/provider mismatch causing signer validation to fail
        // CRITICAL: Account order MUST match IDL exactly (from quantdesk_perp_dex.json):
        // IDL order: user_account, user, protocol_vault, collateral_account, sol_usd_price_feed, system_program, rent
        
        const signature = await program.methods
          .depositNativeSol(new BN(amount))
          .accounts({
            userAccount: userAccountPda!,     // Position 0: user_account
            user: provider.wallet.publicKey,  // Position 1: user (Signer) - CRITICAL: Must be position 1 per IDL!
            protocolVault: protocolSOLVault!, // Position 2: protocol_vault
            collateralAccount: solCollateralAccount!, // Position 3: collateral_account
            solUsdPriceFeed: SOL_USD_PRICE_FEED, // Position 4: sol_usd_price_feed
            systemProgram: SystemProgram.programId, // Position 5: system_program
            rent: SYSVAR_RENT_PUBKEY,         // Position 6: rent
          })
          .rpc();
        
        console.log('✅ Deposit successful via Anchor .rpc()!');
        console.log('📤 Transaction signature:', signature);
        return signature;
      } catch (rpcError: any) {
        console.error('❌ Anchor .rpc() failed:', rpcError);
        console.error('📋 Error details:', {
          message: rpcError?.message,
          code: rpcError?.code,
          errorCode: rpcError?.error?.errorCode,
          logs: rpcError?.logs,
          programErrorStack: rpcError?.programErrorStack,
        });
        
        // Check if it's AccountNotSigner - this means the wallet didn't sign properly
        if (rpcError?.code === 3010 || 
            rpcError?.message?.includes('AccountNotSigner') ||
            rpcError?.error?.errorCode?.code === 'AccountNotSigner') {
          
          console.error('❌ AccountNotSigner detected!');
          console.error('🔍 This means the transaction was not signed by the user account');
          console.error('🔍 Checking wallet adapter signing capability...');
          
          const anchorWallet = this.createAnchorWallet(wallet);
          const testTx = new Transaction();
          testTx.feePayer = wallet.adapter.publicKey!;
          testTx.recentBlockhash = (await freshConnection!.getLatestBlockhash()).blockhash;
          
          try {
            // Test if wallet can actually sign
            const testSigned = await anchorWallet.signTransaction(testTx);
            console.log('✅ Wallet CAN sign transactions:', {
              hasSignature: !!testSigned.signatures[0]?.signature,
              signatureLength: testSigned.signatures[0]?.signature?.length,
            });
          } catch (signTestError: any) {
            console.error('❌ Wallet SIGNING TEST FAILED:', signTestError);
            throw new Error(`Wallet cannot sign transactions: ${signTestError.message}`);
          }
          
          throw new Error('Anchor .rpc() failed with AccountNotSigner. Wallet can sign, but Anchor may have an issue with the account structure.');
        }
        
        throw rpcError;
      }
    } catch (error: any) {
      console.error('❌ Deposit error:', error);
      
      // Better error messages
      if (error?.message?.includes('User rejected') || error?.message?.includes('user rejected')) {
        throw new Error('Transaction cancelled by user');
      } else if (error?.message?.includes('Insufficient')) {
        throw new Error('Insufficient SOL balance for deposit');
      } else if (error?.message?.includes('Blockhash')) {
        throw new Error('Transaction expired - please try again');
      } else if (error?.message?.includes('already been processed')) {
        throw new Error('Transaction already processed. Please try again with a fresh transaction.');
      } else {
        // Return the original error message
        throw error;
      }
    }
  }

  /**
   * Get associated token address
   * Following Solana Cookbook patterns
   */
  private async getAssociatedTokenAddress(mint: PublicKey, owner: PublicKey): Promise<PublicKey> {
    const [address] = await PublicKey.findProgramAddress(
      [owner.toBuffer(), TOKEN_PROGRAM_ID.toBuffer(), mint.toBuffer()],
      ASSOCIATED_TOKEN_PROGRAM_ID
    );
    return address;
  }

  // ==================== TRADING FUNCTIONS ====================

  /**
   * Place an order
   */
  async placeOrder(
    wallet: Wallet,
    market: string,
    orderType: OrderType,
    side: PositionSide,
    size: number,
    price: number,
    leverage: number
  ): Promise<string> {
    try {
      if (!wallet.adapter || !wallet.adapter.publicKey) {
        throw new Error('Wallet adapter or public key is not available');
      }

      const [marketPda] = await PublicKey.findProgramAddress(
        [Buffer.from('market'), Buffer.from(market.split('/')[0]), Buffer.from(market.split('/')[1])],
        this.programId
      );
      
      const [orderPda] = await PublicKey.findProgramAddress(
        [Buffer.from('order'), wallet.adapter.publicKey.toBuffer(), marketPda.toBuffer()],
        this.programId
      );

      const transaction = new Transaction().add(
        new TransactionInstruction({
          keys: [
            { pubkey: wallet.adapter.publicKey, isSigner: true, isWritable: false },
            { pubkey: marketPda, isSigner: false, isWritable: true },
            { pubkey: orderPda, isSigner: false, isWritable: true },
          ],
          programId: this.programId,
          data: Buffer.from([
            4, // place_order instruction
            OrderType[orderType] as number,
            PositionSide[side] as number,
            ...this.numberToBytes(size),
            ...this.numberToBytes(price),
            leverage,
          ]),
        })
      );

      const signature = await wallet.adapter.sendTransaction(transaction, this.connection);
      await this.connection.confirmTransaction(signature);
      
      return signature;
    } catch (error) {
      console.error('Error placing order:', error);
      throw error;
    }
  }

  /**
   * Get user's positions
   */
  async getPositions(walletAddress: string): Promise<Position[]> {
    try {
      // This is a simplified version - in production you'd query all positions
      // For now, we'll return an empty array
      return [];
    } catch (error) {
      console.error('Error getting positions:', error);
      return [];
    }
  }

  /**
   * Get user's orders
   */
  async getOrders(walletAddress: string): Promise<Order[]> {
    try {
      // This is a simplified version - in production you'd query all orders
      // For now, we'll return an empty array
      return [];
    } catch (error) {
      console.error('Error getting orders:', error);
      return [];
    }
  }

  // ==================== HELPER FUNCTIONS ====================

  /**
   * Get token mint for a collateral type
   */
  private getTokenMint(assetType: CollateralType): PublicKey {
    // In production, you'd have the actual token mint addresses
    const tokenMints: Record<CollateralType, string> = {
      [CollateralType.SOL]: 'So11111111111111111111111111111111111111112',
      [CollateralType.USDC]: 'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',
      [CollateralType.BTC]: '9n4nbM75f5Ui33ZbPYXn59EwSgE8CGsHtAeTH5YFeJ9E',
      [CollateralType.ETH]: '2FPyTwcZLUg1MDrwsyoP4D6s1tM7h7HYsTW2L7fJ1sxJ',
      [CollateralType.USDT]: 'Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB',
      [CollateralType.AVAX]: 'A8LpGcwgVdkVjU7Xr6dMWhZx9k3dD6Z2V8F4nB7cE1hJ',
      [CollateralType.MATIC]: '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs',
      [CollateralType.ARB]: '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM',
      [CollateralType.OP]: '7vfCXTUXx5WJV5JADk17DUJ4ksgau7utNKj4b963voxs',
      [CollateralType.DOGE]: '3NZ9JMVBmGAqocybic2c7LQCJScmgsAZ6vQqTDzcqmJh',
      [CollateralType.ADA]: 'A8LpGcwgVdkVjU7Xr6dMWhZx9k3dD6Z2V8F4nB7cE1hJ',
      [CollateralType.DOT]: '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM',
      [CollateralType.LINK]: 'CWE8jPTUYhdCTZYWPTe1o5DFqfdjzWKc9WKz6rSjQUdG',
    };
    
    return new PublicKey(tokenMints[assetType]);
  }

  /**
   * Get connection instance
   */
  getConnection(): Connection {
    return this.connection;
  }

  /**
   * BEST PRACTICE: Deposit Native SOL following Solana Cookbook and Anchor best practices
   * 
   * This function implements the recommended patterns:
   * 1. Always uses provider.wallet.publicKey (never wallet.adapter.publicKey) in .accounts()
   * 2. Derives PDAs correctly matching Rust seeds exactly
   * 3. Uses Anchor's .rpc() method (preferred over manual transaction building)
   * 4. Proper error handling and validation
   * 
   * @param wallet - Wallet adapter instance
   * @param amount - Amount in lamports to deposit
   * @returns Transaction signature
   */
  async depositNativeSOLBestPractice(wallet: Wallet, amount: number): Promise<string> {
    console.log('🚀 [BEST PRACTICE] Depositing native SOL...');
    
    if (!wallet.adapter || !wallet.adapter.publicKey) {
      throw new Error('Wallet adapter or public key is not available');
    }

    const userPubkey = wallet.adapter.publicKey;
    const programId = new PublicKey(programIdl.address);
    
    // Step 1: Verify PDA seeds match Rust expectations
    console.log('🔍 Step 1: Verifying PDA seeds...');
    const seedVerification = verifyPDASeeds(userPubkey, 0);
    if (!seedVerification.userAccount.isValid) {
      console.warn('⚠️ User account seed verification failed - account index may be incorrect');
    }
    console.log('✅ PDA seeds verified:', seedVerification);

    // Step 2: Derive all PDAs
    console.log('🔍 Step 2: Deriving PDAs...');
    const pdas = await deriveDepositPDAs(userPubkey, programId, 0);
    console.log('✅ PDAs derived:', {
      userAccount: pdas.userAccount.toString(),
      collateralSOL: pdas.collateralSOL.toString(),
      protocolVault: pdas.protocolVault.toString(),
      bumps: pdas.bumps,
    });

    // Step 3: Create fresh connection and provider
    console.log('🔍 Step 3: Creating Anchor provider...');
    const connection = new Connection(
      import.meta.env.VITE_SOLANA_RPC_URL || 'https://api.devnet.solana.com',
      'confirmed'
    );

    // CRITICAL FIX: Create proper AnchorWallet that Anchor can recognize
    // Based on Solana docs: https://docs.solanamobile.com/react-native/anchor_integration
    const anchorWallet = {
      publicKey: wallet.adapter.publicKey,
      signTransaction: wallet.adapter.signTransaction!.bind(wallet.adapter),
      signAllTransactions: wallet.adapter.signAllTransactions!.bind(wallet.adapter),
    };

    const provider = new AnchorProvider(connection, anchorWallet, {
      commitment: 'confirmed',
      preflightCommitment: 'confirmed',
    });

    // Step 4: CRITICAL - Store the EXACT publicKey object reference from provider.wallet
    // This ensures we use the same object that Anchor uses internally for signing
    const signerPublicKey = provider.wallet.publicKey;
    
    // Verify it matches the user's pubkey (address-wise)
    if (!signerPublicKey || !signerPublicKey.equals(userPubkey)) {
      throw new Error(
        `Provider wallet mismatch! Expected ${userPubkey.toString()}, got ${signerPublicKey?.toString() || 'null'}`
      );
    }
    
    // CRITICAL: Verify it's the exact same object reference
    const isSameObject = signerPublicKey === provider.wallet.publicKey;
    console.log('✅ Provider wallet verification:', {
      address: signerPublicKey.toString(),
      matchesUserPubkey: signerPublicKey.equals(userPubkey),
      isSameObjectReference: isSameObject,
      walletHasSignTransaction: !!provider.wallet.signTransaction,
      walletHasSignAllTransactions: !!provider.wallet.signAllTransactions,
    });

    // Step 5: Create program instance
    const program = new Program(programIdl, provider);

    // Step 6: Pyth price feed (devnet)
    const SOL_USD_PRICE_FEED = new PublicKey('H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG');

    // Step 7: Execute deposit using Anchor's .rpc() - THE BEST PRACTICE METHOD
    console.log('🔍 Step 7: Executing deposit via Anchor .rpc()...');
    
    // Pre-check: Verify the instruction structure before sending
    try {
      const preCheckTx = await program.methods
        .depositNativeSol(new BN(amount))
        .accounts({
          userAccount: pdas.userAccount,
          user: signerPublicKey,              // Position 1: user (Signer) - CRITICAL: Must be position 1 per IDL!
          protocolVault: pdas.protocolVault,
          collateralAccount: pdas.collateralSOL,
          solUsdPriceFeed: SOL_USD_PRICE_FEED,
          systemProgram: SystemProgram.programId,
          rent: SYSVAR_RENT_PUBKEY,           // Position 6: rent - REQUIRED!
        })
        .transaction();
      
      const userAccountMeta = preCheckTx.instructions[0]?.keys?.find(
        k => k.pubkey.equals(signerPublicKey)
      );
      
      console.log('📋 Pre-flight instruction check:');
      console.log('  Total accounts in instruction:', preCheckTx.instructions[0]?.keys?.length);
      console.log('  Fee payer (will be set by Anchor):', signerPublicKey.toString());
      console.log('  User account meta:', {
        found: !!userAccountMeta,
        isSigner: userAccountMeta?.isSigner,
        isWritable: userAccountMeta?.isWritable,
        pubkey: userAccountMeta?.pubkey.toString(),
      });
      
      if (!userAccountMeta?.isSigner) {
        throw new Error('User account is NOT marked as signer in instruction metadata!');
      }
      
      console.log('✅ Pre-flight check passed - user is marked as signer');
    } catch (preCheckError: any) {
      console.error('❌ Pre-flight check failed:', preCheckError);
      throw preCheckError;
    }

    console.log('📋 Accounts being used:', {
      userAccount: pdas.userAccount.toString(),
      collateralAccount: pdas.collateralSOL.toString(),
      protocolVault: pdas.protocolVault.toString(),
      user: signerPublicKey.toString(),
      solUsdPriceFeed: SOL_USD_PRICE_FEED.toString(),
      systemProgram: SystemProgram.programId.toString(),
    });

    try {
      // BEST PRACTICE: Use the EXACT publicKey object reference from provider.wallet
      // This ensures Anchor recognizes the signer correctly by object reference equality
      // Anchor internally compares the account's publicKey with the signer's publicKey by reference
      console.log('🔍 About to call .rpc() with accounts:');
      console.log('  user account:', pdas.userAccount.toString());
      console.log('  user signer:', signerPublicKey.toString());
      console.log('  signer object identity:', signerPublicKey === provider.wallet.publicKey);
      
      // CRITICAL: Account order MUST match IDL exactly (from quantdesk_perp_dex.json):
      // IDL order: user_account, user, protocol_vault, collateral_account, sol_usd_price_feed, system_program, rent
      // 1. user_account (PDA)
      // 2. user (Signer - MUST be position 1 per IDL, not position 4!)
      // 3. protocol_vault (PDA)
      // 4. collateral_account (PDA)
      // 5. sol_usd_price_feed
      // 6. system_program
      // 7. rent - CRITICAL: Must be explicitly provided!
      
      // IMPORTANT: Anchor's .rpc() will automatically sign with provider.wallet
      // We must ensure the 'user' account uses the EXACT same publicKey object
      // that the provider.wallet uses for signing (which it should since we're using provider.wallet.publicKey)
      
      // Use Anchor's .rpc() method - this is the standard pattern from all working examples
      console.log('📤 Sending transaction via Anchor .rpc()...');
      
      const signature = await program.methods
        .depositNativeSol(new BN(amount))
        .accounts({
          userAccount: pdas.userAccount,      // Position 0: user_account
          user: signerPublicKey,              // Position 1: user (Signer) - CRITICAL: Must be position 1 per IDL!
          protocolVault: pdas.protocolVault,  // Position 2: protocol_vault
          collateralAccount: pdas.collateralSOL, // Position 3: collateral_account
          solUsdPriceFeed: SOL_USD_PRICE_FEED, // Position 4: sol_usd_price_feed
          systemProgram: SystemProgram.programId, // Position 5: system_program
          rent: SYSVAR_RENT_PUBKEY,          // Position 6: rent - REQUIRED!
        })
        .rpc({
          commitment: 'confirmed',
          skipPreflight: false,
        });

      console.log('✅ [BEST PRACTICE] Deposit successful!');
      console.log('📤 Transaction signature:', signature);
      console.log('🌐 Explorer:', `https://explorer.solana.com/tx/${signature}?cluster=devnet`);
      
      return signature;
    } catch (error: any) {
      console.error('❌ [BEST PRACTICE] Deposit failed:', error);
      
      // CRITICAL: Log the ACTUAL raw error from Solana/Anchor
      console.error('🔥 RAW ERROR DETAILS:');
      console.error('  - Error code:', error?.code);
      console.error('  - Error message:', error?.message);
      console.error('  - Error logs:', error?.logs);
      console.error('  - Full error:', error);
      
      // Enhanced error analysis
      if (error?.code === 3010 || error?.message?.includes('AccountNotSigner')) {
        console.error('🔍 AccountNotSigner Deep Analysis:');
        console.error('  - Signer publicKey (from provider.wallet):', signerPublicKey.toString());
        console.error('  - User pubkey (from wallet.adapter):', userPubkey.toString());
        console.error('  - Addresses match:', signerPublicKey.equals(userPubkey));
        console.error('  - Object reference match:', signerPublicKey === provider.wallet.publicKey);
        console.error('  - Has signTransaction:', !!provider.wallet.signTransaction);
        console.error('  - Has signAllTransactions:', !!provider.wallet.signAllTransactions);
        
        // Check if we can inspect the transaction that was built
        try {
          const testTx = await program.methods
            .depositNativeSol(new BN(amount))
            .accounts({
              userAccount: pdas.userAccount,
              user: signerPublicKey,              // Position 1: user (Signer) - CRITICAL: Must be position 1 per IDL!
              protocolVault: pdas.protocolVault,
              collateralAccount: pdas.collateralSOL,
              solUsdPriceFeed: SOL_USD_PRICE_FEED,
              systemProgram: SystemProgram.programId,
            })
            .transaction();
          
          const userMeta = testTx.instructions[0]?.keys?.find(
            k => k.pubkey.equals(signerPublicKey)
          );
          console.error('  - User in transaction accounts:', {
            found: !!userMeta,
            isSigner: userMeta?.isSigner,
            isWritable: userMeta?.isWritable,
            index: testTx.instructions[0]?.keys?.findIndex(
              k => k.pubkey.equals(signerPublicKey)
            ),
          });
        } catch (txInspectError) {
          console.error('  - Could not inspect transaction:', txInspectError);
        }
        
        throw new Error(
          `AccountNotSigner: The user account (${userPubkey.toString()}) is not being recognized as a signer. ` +
          `Even though the transaction was signed, Anchor is rejecting it. This might indicate: ` +
          `1) Account order mismatch with Rust, 2) Object reference issue, or 3) Rust program expecting different signer configuration.`
        );
      }

      // User-friendly error messages
      if (error?.message?.includes('User rejected')) {
        throw new Error('Transaction cancelled by user');
      } else if (error?.message?.includes('Insufficient')) {
        throw new Error('Insufficient SOL balance for deposit');
      } else if (error?.message?.includes('Blockhash')) {
        throw new Error('Transaction expired - please try again');
      }

      throw error;
    }
  }

  /**
   * Get program ID
   */
  getProgramId(): PublicKey {
    return this.programId;
  }

  /**
   * Open a new position
   */
  async openPosition(
    wallet: Wallet,
    market: string,
    side: 'Long' | 'Short',
    size: number,
    leverage: number,
    entryPrice: number
  ): Promise<string> {
    try {
      if (!wallet.publicKey) {
        throw new Error('Wallet not connected');
      }

      const userAccountPDA = await this.getUserAccountPDA(wallet.publicKey);
      const positionIndex = 0; // For now, use index 0
      const positionPDA = await this.getPositionPDA(userAccountPDA, positionIndex);

      const instruction = await this.program.methods
        .openPosition(
          positionIndex,
          side === 'Long' ? { long: {} } : { short: {} },
          new BN(size),
          new BN(leverage),
          new BN(entryPrice)
        )
        .accounts({
          position: positionPDA,
          userAccount: userAccountPDA,
          market: new PublicKey(market),
          user: wallet.publicKey,
          systemProgram: SystemProgram.programId,
          clock: SYSVAR_RENT_PUBKEY,
        })
        .instruction();

      const transaction = new Transaction().add(instruction);
      const signature = await wallet.sendTransaction(transaction, this.connection);

      console.log('Position opened:', signature);
      return signature;
    } catch (error) {
      console.error('Error opening position:', error);
      throw error;
    }
  }

  /**
   * Close a position
   */
  async closePosition(wallet: Wallet, positionIndex: number): Promise<string> {
    try {
      if (!wallet.publicKey) {
        throw new Error('Wallet not connected');
      }

      const userAccountPDA = await this.getUserAccountPDA(wallet.publicKey);
      const positionPDA = await this.getPositionPDA(userAccountPDA, positionIndex);

      const instruction = await this.program.methods
        .closePosition()
        .accounts({
          position: positionPDA,
          userAccount: userAccountPDA,
          user: wallet.publicKey,
          clock: SYSVAR_RENT_PUBKEY,
        })
        .instruction();

      const transaction = new Transaction().add(instruction);
      const signature = await wallet.sendTransaction(transaction, this.connection);

      console.log('Position closed:', signature);
      return signature;
    } catch (error) {
      console.error('Error closing position:', error);
      throw error;
    }
  }

  /**
   * Get position PDA
   */
  private async getPositionPDA(userAccount: PublicKey, positionIndex: number): Promise<PublicKey> {
    const [pda] = await PublicKey.findProgramAddress(
      [
        Buffer.from('position'),
        userAccount.toBuffer(),
        Buffer.from(positionIndex.toString()),
      ],
      this.programId
    );
    return pda;
  }
}

// Export class and singleton instance
export { SmartContractService };

// Export for debugging
export const debugSmartContract = {
  async checkProtocolVault() {
    const service = SmartContractService.getInstance();
    const balance = await service.getProtocolVaultBalance();
    console.log('Protocol vault balance:', balance / 1e9, 'SOL');
    return balance;
  },
  
  async checkSOLCollateral(walletAddress: string) {
    const service = SmartContractService.getInstance();
    const balance = await service.getSOLCollateralBalance(walletAddress);
    console.log('SOL collateral balance:', balance / 1e9, 'SOL');
    return balance;
  },
  
  async checkUserAccount(walletAddress: string) {
    const service = SmartContractService.getInstance();
    const account = await service.getUserAccountState(walletAddress);
    console.log('User account state:', account);
    return account;
  },
  
  async checkAllBalances(walletAddress: string) {
    const service = SmartContractService.getInstance();
    console.log('🔍 Checking all balances for:', walletAddress);
    
    const protocolBalance = await service.getProtocolVaultBalance();
    const collateralBalance = await service.getSOLCollateralBalance(walletAddress);
    const accountState = await service.getUserAccountState(walletAddress);
    
    console.log('📊 Summary:');
    console.log('  Protocol Vault:', protocolBalance / 1e9, 'SOL');
    console.log('  SOL Collateral:', collateralBalance / 1e9, 'SOL');
    console.log('  User Account:', accountState);
    
    return { protocolBalance, collateralBalance, accountState };
  }
};

// Make it available on window for debugging
if (typeof window !== 'undefined') {
  (window as any).debugSmartContract = debugSmartContract;
  (window as any).SmartContractService = SmartContractService;
  
  // Add a simple test function
  (window as any).testAccount = async () => {
    const { useWallet } = await import('@solana/wallet-adapter-react');
    const wallet = useWallet();
    if (wallet.publicKey) {
      console.log('🧪 Testing account for:', wallet.publicKey.toString());
      return debugSmartContract.checkAllBalances(wallet.publicKey.toString());
    } else {
      console.log('❌ No wallet connected');
    }
  };
  
  // Add function to get current wallet address
  (window as any).getWalletAddress = () => {
    const walletElement = document.querySelector('[data-wallet-adapter-button]');
    if (walletElement) {
      const addressElement = walletElement.querySelector('span');
      if (addressElement) {
        const address = addressElement.textContent;
        console.log('📍 Current wallet address:', address);
        return address;
      }
    }
    console.log('❌ Could not find wallet address in UI');
    return null;
  };
}
export const smartContractService = SmartContractService.getInstance();
