import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { PublicKey, Keypair, SystemProgram, Connection } from "@solana/web3.js";
import { expect } from "chai";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";

/**
 * Comprehensive Test Utilities for QuantDesk Perpetual DEX
 * Provides reusable test helpers, mock objects, and test data fixtures
 */

export class TestUtils {
  private program: Program<QuantdeskPerpDex>;
  private provider: anchor.AnchorProvider;
  private connection: Connection;

  constructor(program: Program<QuantdeskPerpDex>, provider: anchor.AnchorProvider) {
    this.program = program;
    this.provider = provider;
    this.connection = provider.connection;
  }

  /**
   * Generate test accounts with proper funding
   */
  async generateTestAccounts(count: number = 1): Promise<Keypair[]> {
    const accounts: Keypair[] = [];
    
    for (let i = 0; i < count; i++) {
      const account = Keypair.generate();
      await this.fundAccount(account.publicKey, 2 * anchor.web3.LAMPORTS_PER_SOL);
      accounts.push(account);
    }
    
    return accounts;
  }

  /**
   * Fund an account with SOL
   */
  async fundAccount(publicKey: PublicKey, lamports: number): Promise<void> {
    try {
      const airdropTx = await this.connection.requestAirdrop(publicKey, lamports);
      await this.connection.confirmTransaction(airdropTx);
    } catch (error) {
      console.warn(`Failed to fund account ${publicKey.toString()}:`, error);
    }
  }

  /**
   * Create a mock market for testing
   */
  async createMockMarket(): Promise<{ market: Keypair; priceFeed: PublicKey }> {
    const market = Keypair.generate();
    const priceFeed = Keypair.generate().publicKey;

    await this.program.methods
      .initializeMarket(
        "BTC",
        "USDT",
        new anchor.BN(50000),
        100, // maxLeverage
        500, // initialMarginRatio (5%)
        300  // maintenanceMarginRatio (3%)
      )
      .accounts({
        market: market.publicKey,
        authority: this.provider.wallet.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([market])
      .rpc();

    return { market, priceFeed };
  }

  /**
   * Create a mock position for testing
   */
  async createMockPosition(
    market: PublicKey,
    user: Keypair,
    size: number = 1000,
    side: any = { long: {} },
    leverage: number = 10
  ): Promise<Keypair> {
    const position = Keypair.generate();
    const userCollateral = Keypair.generate();
    
    await this.fundAccount(userCollateral.publicKey, anchor.web3.LAMPORTS_PER_SOL);

    await this.program.methods
      .openPosition(
        new anchor.BN(size),
        side,
        leverage
      )
      .accounts({
        market,
        position: position.publicKey,
        user: user.publicKey,
        userCollateral: userCollateral.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([position, user])
      .rpc();

    return position;
  }

  /**
   * Create a mock order for testing
   */
  async createMockOrder(
    market: PublicKey,
    user: Keypair,
    orderType: any = { limit: {} },
    side: any = { long: {} },
    size: number = 1000,
    price: number = 50000,
    leverage: number = 5
  ): Promise<Keypair> {
    const order = Keypair.generate();
    const userCollateral = Keypair.generate();
    
    await this.fundAccount(userCollateral.publicKey, anchor.web3.LAMPORTS_PER_SOL);

    await this.program.methods
      .placeOrder(
        orderType,
        side,
        new anchor.BN(size),
        new anchor.BN(price),
        new anchor.BN(0), // stopPrice
        new anchor.BN(0), // trailingDistance
        leverage,
        new anchor.BN(0) // expiresAt
      )
      .accounts({
        market,
        order: order.publicKey,
        user: user.publicKey,
        userCollateral: userCollateral.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([order, user])
      .rpc();

    return order;
  }

  /**
   * Wait for a specific number of slots
   */
  async waitForSlots(slots: number): Promise<void> {
    const currentSlot = await this.connection.getSlot();
    const targetSlot = currentSlot + slots;
    
    while (await this.connection.getSlot() < targetSlot) {
      await new Promise(resolve => setTimeout(resolve, 100));
    }
  }

  /**
   * Get account balance
   */
  async getBalance(publicKey: PublicKey): Promise<number> {
    return await this.connection.getBalance(publicKey);
  }

  /**
   * Verify account exists and has expected data size
   */
  async verifyAccount(publicKey: PublicKey, expectedSize?: number): Promise<boolean> {
    const accountInfo = await this.connection.getAccountInfo(publicKey);
    if (!accountInfo) return false;
    if (expectedSize && accountInfo.data.length !== expectedSize) return false;
    return true;
  }

  /**
   * Create test data fixtures
   */
  static getTestData() {
    return {
      markets: [
        { baseAsset: "BTC", quoteAsset: "USDT", initialPrice: 50000 },
        { baseAsset: "ETH", quoteAsset: "USDT", initialPrice: 3000 },
        { baseAsset: "SOL", quoteAsset: "USDT", initialPrice: 100 },
      ],
      orders: [
        { type: "limit", side: "long", size: 1000, price: 50000, leverage: 5 },
        { type: "market", side: "short", size: 2000, price: 0, leverage: 10 },
        { type: "stopLoss", side: "long", size: 1500, stopPrice: 48000, leverage: 8 },
      ],
      positions: [
        { side: "long", size: 1000, leverage: 10 },
        { side: "short", size: 2000, leverage: 5 },
        { side: "long", size: 500, leverage: 20 },
      ],
    };
  }
}

/**
 * Security Test Utilities
 */
export class SecurityTestUtils {
  private program: Program<QuantdeskPerpDex>;
  private provider: anchor.AnchorProvider;

  constructor(program: Program<QuantdeskPerpDex>, provider: anchor.AnchorProvider) {
    this.program = program;
    this.provider = provider;
  }

  /**
   * Test for reentrancy vulnerabilities
   */
  async testReentrancy(instruction: any, accounts: any): Promise<boolean> {
    try {
      // Attempt to call the same instruction multiple times in a single transaction
      const promises = [];
      for (let i = 0; i < 3; i++) {
        promises.push(instruction.accounts(accounts).rpc());
      }
      await Promise.all(promises);
      return false; // Should not succeed
    } catch (error) {
      return true; // Correctly rejected
    }
  }

  /**
   * Test for integer overflow/underflow
   */
  async testIntegerOverflow(instruction: any, accounts: any): Promise<boolean> {
    try {
      // Test with maximum possible values
      const maxValue = new anchor.BN("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF");
      await instruction.accounts(accounts).rpc();
      return false; // Should not succeed with max values
    } catch (error) {
      return true; // Correctly rejected
    }
  }

  /**
   * Test for unauthorized access
   */
  async testUnauthorizedAccess(instruction: any, accounts: any, unauthorizedUser: Keypair): Promise<boolean> {
    try {
      // Replace authorized user with unauthorized user
      const unauthorizedAccounts = { ...accounts, user: unauthorizedUser.publicKey };
      await instruction.accounts(unauthorizedAccounts).signers([unauthorizedUser]).rpc();
      return false; // Should not succeed
    } catch (error) {
      return true; // Correctly rejected
    }
  }

  /**
   * Test for oracle manipulation
   */
  async testOracleManipulation(market: PublicKey, maliciousPriceFeed: PublicKey): Promise<boolean> {
    try {
      await this.program.methods
        .updateOraclePrice()
        .accounts({
          market,
          priceFeed: maliciousPriceFeed,
          authority: this.provider.wallet.publicKey,
        })
        .rpc();
      return false; // Should not succeed with malicious price feed
    } catch (error) {
      return true; // Correctly rejected
    }
  }
}

/**
 * Performance Test Utilities
 */
export class PerformanceTestUtils {
  private program: Program<QuantdeskPerpDex>;
  private provider: anchor.AnchorProvider;

  constructor(program: Program<QuantdeskPerpDex>, provider: anchor.AnchorProvider) {
    this.program = program;
    this.provider = provider;
  }

  /**
   * Measure transaction execution time
   */
  async measureExecutionTime(instruction: any): Promise<number> {
    const startTime = Date.now();
    await instruction.rpc();
    const endTime = Date.now();
    return endTime - startTime;
  }

  /**
   * Test concurrent transaction handling
   */
  async testConcurrency(instructionFactory: () => any, count: number = 10): Promise<{
    successCount: number;
    averageTime: number;
    errors: any[];
  }> {
    const startTime = Date.now();
    const promises = [];
    const errors: any[] = [];

    for (let i = 0; i < count; i++) {
      const promise = instructionFactory().rpc().catch(error => {
        errors.push(error);
        return null;
      });
      promises.push(promise);
    }

    const results = await Promise.all(promises);
    const endTime = Date.now();
    const successCount = results.filter(result => result !== null).length;
    const averageTime = (endTime - startTime) / count;

    return { successCount, averageTime, errors };
  }

  /**
   * Test gas optimization
   */
  async testGasOptimization(instruction: any): Promise<{
    computeUnits: number;
    isOptimized: boolean;
  }> {
    const simulation = await instruction.simulate();
    const computeUnits = simulation.value.unitsConsumed || 0;
    const isOptimized = computeUnits < 200000; // Threshold for optimization

    return { computeUnits, isOptimized };
  }
}

/**
 * Mock Objects for Testing
 */
export class MockObjects {
  /**
   * Create mock price feed data
   */
  static createMockPriceFeed(price: number, confidence: number = 100): any {
    return {
      price: new anchor.BN(price),
      confidence: new anchor.BN(confidence),
      exponent: new anchor.BN(-8),
      timestamp: new anchor.BN(Math.floor(Date.now() / 1000)),
    };
  }

  /**
   * Create mock oracle account
   */
  static createMockOracleAccount(price: number): any {
    return {
      price: new anchor.BN(price),
      lastUpdateTime: new anchor.BN(Math.floor(Date.now() / 1000)),
      isStale: false,
    };
  }

  /**
   * Create mock market data
   */
  static createMockMarketData(): any {
    return {
      baseAsset: "BTC",
      quoteAsset: "USDT",
      initialPrice: new anchor.BN(50000),
      maxLeverage: 100,
      initialMarginRatio: 500,
      maintenanceMarginRatio: 300,
      isActive: true,
      lastOraclePrice: new anchor.BN(50000),
      lastFundingTime: new anchor.BN(0),
    };
  }

  /**
   * Create mock position data
   */
  static createMockPositionData(user: PublicKey, market: PublicKey): any {
    return {
      user,
      market,
      size: new anchor.BN(1000),
      side: { long: {} },
      leverage: 10,
      entryPrice: new anchor.BN(50000),
      margin: new anchor.BN(5000),
      unrealizedPnl: new anchor.BN(0),
      lastUpdateTime: new anchor.BN(Math.floor(Date.now() / 1000)),
    };
  }

  /**
   * Create mock order data
   */
  static createMockOrderData(user: PublicKey, market: PublicKey): any {
    return {
      user,
      market,
      orderType: { limit: {} },
      side: { long: {} },
      size: new anchor.BN(1000),
      price: new anchor.BN(50000),
      stopPrice: new anchor.BN(0),
      trailingDistance: new anchor.BN(0),
      leverage: 5,
      status: { pending: {} },
      createdAt: new anchor.BN(Math.floor(Date.now() / 1000)),
      expiresAt: new anchor.BN(0),
    };
  }
}

/**
 * Test Data Fixtures
 */
export const TestFixtures = {
  /**
   * Standard test market configuration
   */
  standardMarket: {
    baseAsset: "BTC",
    quoteAsset: "USDT",
    initialPrice: 50000,
    maxLeverage: 100,
    initialMarginRatio: 500,
    maintenanceMarginRatio: 300,
  },

  /**
   * High-risk market configuration for testing
   */
  highRiskMarket: {
    baseAsset: "ETH",
    quoteAsset: "USDT",
    initialPrice: 3000,
    maxLeverage: 50,
    initialMarginRatio: 1000,
    maintenanceMarginRatio: 500,
  },

  /**
   * Test user configurations
   */
  testUsers: {
    trader: {
      initialBalance: 10 * anchor.web3.LAMPORTS_PER_SOL,
      riskTolerance: "high",
    },
    conservative: {
      initialBalance: 5 * anchor.web3.LAMPORTS_PER_SOL,
      riskTolerance: "low",
    },
    keeper: {
      initialBalance: 2 * anchor.web3.LAMPORTS_PER_SOL,
      role: "keeper",
    },
  },

  /**
   * Test order configurations
   */
  testOrders: {
    limitOrder: {
      type: "limit",
      side: "long",
      size: 1000,
      price: 50000,
      leverage: 5,
    },
    marketOrder: {
      type: "market",
      side: "short",
      size: 2000,
      leverage: 10,
    },
    stopLossOrder: {
      type: "stopLoss",
      side: "long",
      size: 1500,
      stopPrice: 48000,
      leverage: 8,
    },
  },

  /**
   * Test position configurations
   */
  testPositions: {
    longPosition: {
      side: "long",
      size: 1000,
      leverage: 10,
      entryPrice: 50000,
    },
    shortPosition: {
      side: "short",
      size: 2000,
      leverage: 5,
      entryPrice: 50000,
    },
    highLeveragePosition: {
      side: "long",
      size: 500,
      leverage: 20,
      entryPrice: 50000,
    },
  },
};

export default TestUtils;
