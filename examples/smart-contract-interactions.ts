/**
 * QuantDesk Smart Contract Interaction Examples
 * 
 * This file demonstrates reusable patterns for interacting with Solana smart contracts.
 * These examples are open source and can be used by the community.
 */

import { 
  Connection, 
  PublicKey, 
  Keypair, 
  Transaction, 
  SystemProgram,
  LAMPORTS_PER_SOL 
} from '@solana/web3.js';
import { 
  Program, 
  AnchorProvider, 
  BN,
  web3 
} from '@coral-xyz/anchor';
import { TOKEN_PROGRAM_ID, createMint, getOrCreateAssociatedTokenAccount } from '@solana/spl-token';

// Example: Program Client
export class QuantDeskProgramClient {
  private program: Program;
  private provider: AnchorProvider;

  constructor(
    connection: Connection,
    wallet: Keypair,
    programId: PublicKey,
    idl: any
  ) {
    this.provider = new AnchorProvider(
      connection,
      new Wallet(wallet),
      { preflightCommitment: 'confirmed' }
    );
    this.program = new Program(idl, programId, this.provider);
  }

  // Example: Create User Account
  async createUserAccount(userKeypair: Keypair): Promise<string> {
    const [userAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('user'), userKeypair.publicKey.toBuffer()],
      this.program.programId
    );

    const tx = await this.program.methods
      .createUserAccount()
      .accounts({
        user: userKeypair.publicKey,
        userAccount: userAccountPda,
        systemProgram: SystemProgram.programId,
      })
      .signers([userKeypair])
      .rpc();

    return tx;
  }

  // Example: Deposit Collateral
  async depositCollateral(
    userKeypair: Keypair,
    amount: number,
    mint: PublicKey
  ): Promise<string> {
    const [userAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('user'), userKeypair.publicKey.toBuffer()],
      this.program.programId
    );

    const [collateralAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('collateral'), userAccountPda.toBuffer(), mint.toBuffer()],
      this.program.programId
    );

    const userTokenAccount = await getOrCreateAssociatedTokenAccount(
      this.provider.connection,
      userKeypair,
      mint,
      userKeypair.publicKey
    );

    const tx = await this.program.methods
      .depositCollateral(new BN(amount))
      .accounts({
        user: userKeypair.publicKey,
        userAccount: userAccountPda,
        collateralAccount: collateralAccountPda,
        userTokenAccount: userTokenAccount.address,
        mint: mint,
        tokenProgram: TOKEN_PROGRAM_ID,
      })
      .signers([userKeypair])
      .rpc();

    return tx;
  }

  // Example: Create Position
  async createPosition(
    userKeypair: Keypair,
    marketId: PublicKey,
    side: 'long' | 'short',
    size: number,
    price: number
  ): Promise<string> {
    const [userAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('user'), userKeypair.publicKey.toBuffer()],
      this.program.programId
    );

    const [positionPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('position'), userAccountPda.toBuffer(), marketId.toBuffer()],
      this.program.programId
    );

    const tx = await this.program.methods
      .createPosition(
        marketId,
        side === 'long' ? { long: {} } : { short: {} },
        new BN(size),
        new BN(price)
      )
      .accounts({
        user: userKeypair.publicKey,
        userAccount: userAccountPda,
        position: positionPda,
        market: marketId,
        systemProgram: SystemProgram.programId,
      })
      .signers([userKeypair])
      .rpc();

    return tx;
  }

  // Example: Close Position
  async closePosition(
    userKeypair: Keypair,
    marketId: PublicKey
  ): Promise<string> {
    const [userAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('user'), userKeypair.publicKey.toBuffer()],
      this.program.programId
    );

    const [positionPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('position'), userAccountPda.toBuffer(), marketId.toBuffer()],
      this.program.programId
    );

    const tx = await this.program.methods
      .closePosition()
      .accounts({
        user: userKeypair.publicKey,
        userAccount: userAccountPda,
        position: positionPda,
        market: marketId,
      })
      .signers([userKeypair])
      .rpc();

    return tx;
  }

  // Example: Get User Account Data
  async getUserAccount(userPublicKey: PublicKey): Promise<any> {
    const [userAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('user'), userPublicKey.toBuffer()],
      this.program.programId
    );

    try {
      const account = await this.program.account.userAccount.fetch(userAccountPda);
      return account;
    } catch (error) {
      console.error('Failed to fetch user account:', error);
      return null;
    }
  }

  // Example: Get Position Data
  async getPosition(userPublicKey: PublicKey, marketId: PublicKey): Promise<any> {
    const [userAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('user'), userPublicKey.toBuffer()],
      this.program.programId
    );

    const [positionPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('position'), userAccountPda.toBuffer(), marketId.toBuffer()],
      this.program.programId
    );

    try {
      const account = await this.program.account.position.fetch(positionPda);
      return account;
    } catch (error) {
      console.error('Failed to fetch position:', error);
      return null;
    }
  }
}

// Example: Market Data Client
export class MarketDataClient {
  private connection: Connection;
  private program: Program;

  constructor(connection: Connection, program: Program) {
    this.connection = connection;
    this.program = program;
  }

  // Example: Get Market Price
  async getMarketPrice(marketId: PublicKey): Promise<number> {
    try {
      const marketAccount = await this.program.account.market.fetch(marketId);
      return marketAccount.currentPrice.toNumber();
    } catch (error) {
      console.error('Failed to fetch market price:', error);
      throw error;
    }
  }

  // Example: Get Market Data
  async getMarketData(marketId: PublicKey): Promise<{
    price: number;
    volume: number;
    openInterest: number;
    fundingRate: number;
  }> {
    try {
      const marketAccount = await this.program.account.market.fetch(marketId);
      
      return {
        price: marketAccount.currentPrice.toNumber(),
        volume: marketAccount.volume24h.toNumber(),
        openInterest: marketAccount.openInterest.toNumber(),
        fundingRate: marketAccount.fundingRate.toNumber()
      };
    } catch (error) {
      console.error('Failed to fetch market data:', error);
      throw error;
    }
  }

  // Example: Get All Markets
  async getAllMarkets(): Promise<Array<{
    id: PublicKey;
    symbol: string;
    price: number;
    volume: number;
  }>> {
    try {
      const markets = await this.program.account.market.all();
      
      return markets.map(market => ({
        id: market.publicKey,
        symbol: market.account.symbol,
        price: market.account.currentPrice.toNumber(),
        volume: market.account.volume24h.toNumber()
      }));
    } catch (error) {
      console.error('Failed to fetch all markets:', error);
      throw error;
    }
  }
}

// Example: Transaction Builder
export class TransactionBuilder {
  private connection: Connection;
  private program: Program;

  constructor(connection: Connection, program: Program) {
    this.connection = connection;
    this.program = program;
  }

  // Example: Build Deposit Transaction
  async buildDepositTransaction(
    userKeypair: Keypair,
    amount: number,
    mint: PublicKey
  ): Promise<Transaction> {
    const [userAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('user'), userKeypair.publicKey.toBuffer()],
      this.program.programId
    );

    const [collateralAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('collateral'), userAccountPda.toBuffer(), mint.toBuffer()],
      this.program.programId
    );

    const userTokenAccount = await getOrCreateAssociatedTokenAccount(
      this.connection,
      userKeypair,
      mint,
      userKeypair.publicKey
    );

    const tx = await this.program.methods
      .depositCollateral(new BN(amount))
      .accounts({
        user: userKeypair.publicKey,
        userAccount: userAccountPda,
        collateralAccount: collateralAccountPda,
        userTokenAccount: userTokenAccount.address,
        mint: mint,
        tokenProgram: TOKEN_PROGRAM_ID,
      })
      .transaction();

    return tx;
  }

  // Example: Build Position Transaction
  async buildPositionTransaction(
    userKeypair: Keypair,
    marketId: PublicKey,
    side: 'long' | 'short',
    size: number,
    price: number
  ): Promise<Transaction> {
    const [userAccountPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('user'), userKeypair.publicKey.toBuffer()],
      this.program.programId
    );

    const [positionPda] = PublicKey.findProgramAddressSync(
      [Buffer.from('position'), userAccountPda.toBuffer(), marketId.toBuffer()],
      this.program.programId
    );

    const tx = await this.program.methods
      .createPosition(
        marketId,
        side === 'long' ? { long: {} } : { short: {} },
        new BN(size),
        new BN(price)
      )
      .accounts({
        user: userKeypair.publicKey,
        userAccount: userAccountPda,
        position: positionPda,
        market: marketId,
        systemProgram: SystemProgram.programId,
      })
      .transaction();

    return tx;
  }
}

// Example: Wallet Helper
export class WalletHelper {
  private connection: Connection;

  constructor(connection: Connection) {
    this.connection = connection;
  }

  // Example: Create New Wallet
  static createWallet(): Keypair {
    return Keypair.generate();
  }

  // Example: Fund Wallet
  async fundWallet(keypair: Keypair, amount: number = 1): Promise<string> {
    const signature = await this.connection.requestAirdrop(
      keypair.publicKey,
      amount * LAMPORTS_PER_SOL
    );

    await this.connection.confirmTransaction(signature);
    return signature;
  }

  // Example: Get Balance
  async getBalance(publicKey: PublicKey): Promise<number> {
    const balance = await this.connection.getBalance(publicKey);
    return balance / LAMPORTS_PER_SOL;
  }

  // Example: Get Token Balance
  async getTokenBalance(
    publicKey: PublicKey,
    mint: PublicKey
  ): Promise<number> {
    try {
      const tokenAccounts = await this.connection.getParsedTokenAccountsByOwner(
        publicKey,
        { mint: mint }
      );

      if (tokenAccounts.value.length === 0) {
        return 0;
      }

      const tokenAccount = tokenAccounts.value[0];
      return tokenAccount.account.data.parsed.info.tokenAmount.uiAmount;
    } catch (error) {
      console.error('Failed to get token balance:', error);
      return 0;
    }
  }
}

// Example: Event Listener
export class EventListener {
  private connection: Connection;
  private program: Program;

  constructor(connection: Connection, program: Program) {
    this.connection = connection;
    this.program = program;
  }

  // Example: Listen for Position Events
  async listenForPositionEvents(
    callback: (event: any) => void
  ): Promise<void> {
    this.program.addEventListener('PositionCreated', (event) => {
      callback({
        type: 'PositionCreated',
        data: event
      });
    });

    this.program.addEventListener('PositionClosed', (event) => {
      callback({
        type: 'PositionClosed',
        data: event
      });
    });

    this.program.addEventListener('PositionLiquidated', (event) => {
      callback({
        type: 'PositionLiquidated',
        data: event
      });
    });
  }

  // Example: Listen for Market Events
  async listenForMarketEvents(
    callback: (event: any) => void
  ): Promise<void> {
    this.program.addEventListener('PriceUpdated', (event) => {
      callback({
        type: 'PriceUpdated',
        data: event
      });
    });

    this.program.addEventListener('FundingRateUpdated', (event) => {
      callback({
        type: 'FundingRateUpdated',
        data: event
      });
    });
  }
}

// Example: Utility Functions
export class ContractUtils {
  // Example: Calculate Position PnL
  static calculatePnL(
    entryPrice: number,
    currentPrice: number,
    size: number,
    side: 'long' | 'short'
  ): number {
    if (side === 'long') {
      return (currentPrice - entryPrice) * size;
    } else {
      return (entryPrice - currentPrice) * size;
    }
  }

  // Example: Calculate Liquidation Price
  static calculateLiquidationPrice(
    entryPrice: number,
    size: number,
    collateral: number,
    side: 'long' | 'short',
    maintenanceMargin: number = 0.05
  ): number {
    const marginRatio = collateral / (entryPrice * size);
    
    if (side === 'long') {
      return entryPrice * (1 - marginRatio + maintenanceMargin);
    } else {
      return entryPrice * (1 + marginRatio - maintenanceMargin);
    }
  }

  // Example: Convert to BN
  static toBN(value: number, decimals: number = 6): BN {
    return new BN(value * Math.pow(10, decimals));
  }

  // Example: Convert from BN
  static fromBN(value: BN, decimals: number = 6): number {
    return value.toNumber() / Math.pow(10, decimals);
  }
}

// Example: Simple Wallet Implementation
class Wallet {
  constructor(private keypair: Keypair) {}

  get publicKey(): PublicKey {
    return this.keypair.publicKey;
  }

  async signTransaction(tx: Transaction): Promise<Transaction> {
    tx.sign(this.keypair);
    return tx;
  }

  async signAllTransactions(txs: Transaction[]): Promise<Transaction[]> {
    return txs.map(tx => {
      tx.sign(this.keypair);
      return tx;
    });
  }
}

export default {
  QuantDeskProgramClient,
  MarketDataClient,
  TransactionBuilder,
  WalletHelper,
  EventListener,
  ContractUtils
};
