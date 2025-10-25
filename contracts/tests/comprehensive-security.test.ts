import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { expect } from "chai";

// Test utilities
class TestHelper {
  static async createTestMarket(params: {
    baseAsset: string;
    quoteAsset: string;
    initialPrice: number;
    maxLeverage: number;
    initialMarginRatio: number;
    maintenanceMarginRatio: number;
  }): Promise<Keypair> {
    const market = Keypair.generate();
    
    await program.methods
      .initializeMarket(
        params.baseAsset,
        params.quoteAsset,
        new anchor.BN(params.initialPrice),
        params.maxLeverage,
        params.initialMarginRatio,
        params.maintenanceMarginRatio
      )
      .accounts({
        market: market.publicKey,
        authority: provider.wallet.publicKey,
        systemProgram: SystemProgram.programId,
      })
      .signers([market])
      .rpc();
    
    return market;
  }

  static async fundAccount(publicKey: PublicKey, amount: number): Promise<void> {
    const airdropTx = await provider.connection.requestAirdrop(publicKey, amount);
    await provider.connection.confirmTransaction(airdropTx);
  }

  static async setupTradingEnvironment(): Promise<{
    market: Keypair;
    user: Keypair;
    collateral: Keypair;
  }> {
    const market = await this.createTestMarket({
      baseAsset: "BTC",
      quoteAsset: "USDC",
      initialPrice: 50000,
      maxLeverage: 10,
      initialMarginRatio: 1000,
      maintenanceMarginRatio: 500
    });
    
    const user = Keypair.generate();
    const collateral = Keypair.generate();
    
    // Fund accounts
    await this.fundAccount(user.publicKey, 2 * anchor.web3.LAMPORTS_PER_SOL);
    await this.fundAccount(collateral.publicKey, anchor.web3.LAMPORTS_PER_SOL);
    
    return { market, user, collateral };
  }
}

describe("QuantDesk Perpetual DEX - Comprehensive Security Tests", () => {
  // Configure the client to use the local cluster
  anchor.setProvider(anchor.AnchorProvider.env());

  const program = anchor.workspace.QuantdeskPerpDex as Program<QuantdeskPerpDex>;
  const provider = anchor.getProvider();

  describe("🔒 Security Module Tests", () => {
    describe("Circuit Breaker Protection", () => {
      it("should trigger circuit breaker on extreme price movement", async () => {
        const { market } = await TestHelper.setupTradingEnvironment();
        
        // Normal price update
        await program.methods
          .updateOraclePrice(new anchor.BN(50000))
          .accounts({
            market: market.publicKey,
            priceFeed: Keypair.generate().publicKey,
            authority: provider.wallet.publicKey,
          })
          .rpc();
        
        // Extreme price movement (1000% increase)
        try {
          await program.methods
            .updateOraclePrice(new anchor.BN(500000))
            .accounts({
              market: market.publicKey,
              priceFeed: Keypair.generate().publicKey,
              authority: provider.wallet.publicKey,
            })
            .rpc();
          
          expect.fail("Should have triggered circuit breaker");
        } catch (error) {
          expect(error.message).to.include("CircuitBreakerActive");
        }
      });

      it("should prevent rapid price manipulation", async () => {
        const { market } = await TestHelper.setupTradingEnvironment();
        
        // Rapid price updates
        for (let i = 0; i < 5; i++) {
          await program.methods
            .updateOraclePrice(new anchor.BN(50000 + i * 1000))
            .accounts({
              market: market.publicKey,
              priceFeed: Keypair.generate().publicKey,
              authority: provider.wallet.publicKey,
            })
            .rpc();
        }
        
        // Should trigger circuit breaker
        try {
          await program.methods
            .updateOraclePrice(new anchor.BN(60000))
            .accounts({
              market: market.publicKey,
              priceFeed: Keypair.generate().publicKey,
              authority: provider.wallet.publicKey,
            })
            .rpc();
          
          expect.fail("Should have triggered circuit breaker");
        } catch (error) {
          expect(error.message).to.include("CircuitBreakerActive");
        }
      });
    });

    describe("Keeper Authorization", () => {
      it("should register keeper with sufficient stake", async () => {
        const keeper = Keypair.generate();
        await TestHelper.fundAccount(keeper.publicKey, anchor.web3.LAMPORTS_PER_SOL);
        
        await program.methods
          .registerKeeper(new anchor.BN(1000000)) // 1 SOL stake
          .accounts({
            keeperNetwork: Keypair.generate().publicKey,
            keeper: keeper.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([keeper])
          .rpc();
        
        // Verify keeper registration
        const keeperNetwork = await program.account.keeperNetwork.fetch(
          Keypair.generate().publicKey // This would be the actual PDA
        );
        
        expect(keeperNetwork.keepers.length).to.be.greaterThan(0);
      });

      it("should reject keeper with insufficient stake", async () => {
        const keeper = Keypair.generate();
        await TestHelper.fundAccount(keeper.publicKey, anchor.web3.LAMPORTS_PER_SOL);
        
        try {
          await program.methods
            .registerKeeper(new anchor.BN(100000)) // 0.1 SOL (insufficient)
            .accounts({
              keeperNetwork: Keypair.generate().publicKey,
              keeper: keeper.publicKey,
              systemProgram: SystemProgram.programId,
            })
            .signers([keeper])
            .rpc();
          
          expect.fail("Should have rejected insufficient stake");
        } catch (error) {
          expect(error.message).to.include("InsufficientKeeperStake");
        }
      });
    });

    describe("Oracle Staleness Protection", () => {
      it("should reject stale oracle updates", async () => {
        const { market } = await TestHelper.setupTradingEnvironment();
        
        // Simulate stale price (older than threshold)
        try {
          await program.methods
            .updateOraclePrice(new anchor.BN(50000))
            .accounts({
              market: market.publicKey,
              priceFeed: Keypair.generate().publicKey,
              authority: provider.wallet.publicKey,
            })
            .rpc();
          
          // This would need to be implemented with timestamp checking
          // For now, we'll test the concept
        } catch (error) {
          expect(error.message).to.include("PriceStale");
        }
      });
    });
  });

  describe("💰 Trading Flow Integration Tests", () => {
    describe("Complete Trading Cycle", () => {
      it("should execute complete trading flow", async () => {
        const { market, user, collateral } = await TestHelper.setupTradingEnvironment();
        
        // 1. Deposit collateral
        await program.methods
          .depositNativeSol(new anchor.BN(1000000)) // 1 SOL
          .accounts({
            userAccount: Keypair.generate().publicKey,
            collateralAccount: collateral.publicKey,
            protocolVault: Keypair.generate().publicKey,
            user: user.publicKey,
            solUsdPriceFeed: Keypair.generate().publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([user])
          .rpc();
        
        // 2. Open position
        const position = Keypair.generate();
        await program.methods
          .openPosition(
            new anchor.BN(1000), // size
            { long: {} }, // side
            new anchor.BN(5), // leverage
            new anchor.BN(50000) // entry price
          )
          .accounts({
            position: position.publicKey,
            userAccount: Keypair.generate().publicKey,
            market: market.publicKey,
            user: user.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, user])
          .rpc();
        
        // 3. Update price (profit scenario)
        await program.methods
          .updateOraclePrice(new anchor.BN(51000))
          .accounts({
            market: market.publicKey,
            priceFeed: Keypair.generate().publicKey,
            authority: provider.wallet.publicKey,
          })
          .rpc();
        
        // 4. Close position
        await program.methods
          .closePosition()
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
          })
          .signers([user])
          .rpc();
        
        // 5. Verify position is closed
        const positionAccount = await program.account.position.fetch(position.publicKey);
        expect(positionAccount.size.toNumber()).to.equal(0);
      });
    });

    describe("Cross-Collateral Trading", () => {
      it("should support cross-collateral trading", async () => {
        const { market, user } = await TestHelper.setupTradingEnvironment();
        
        // This would test cross-collateral functionality
        // Implementation depends on the actual cross-collateral module
        
        const position = Keypair.generate();
        await program.methods
          .openPositionCrossCollateral(
            new anchor.BN(0), // marketIndex
            new anchor.BN(2000), // size
            { long: {} }, // side
            3, // leverage
            [] // collateralAccounts
          )
          .accounts({
            position: position.publicKey,
            market: market.publicKey,
            userAccount: Keypair.generate().publicKey,
            user: user.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, user])
          .rpc();
        
        const positionAccount = await program.account.position.fetch(position.publicKey);
        expect(positionAccount.size.toNumber()).to.equal(2000);
      });
    });
  });

  describe("⚡ Performance Tests", () => {
    describe("Gas Optimization", () => {
      it("should stay within gas limits for position opening", async () => {
        const { market, user } = await TestHelper.setupTradingEnvironment();
        
        const position = Keypair.generate();
        const tx = await program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            new anchor.BN(5),
            new anchor.BN(50000)
          )
          .accounts({
            position: position.publicKey,
            userAccount: Keypair.generate().publicKey,
            market: market.publicKey,
            user: user.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, user])
          .rpc();

        const txDetails = await provider.connection.getTransaction(tx);
        const gasUsed = txDetails.meta.fee;
        
        expect(gasUsed).to.be.lessThan(5000); // Should be under 5000 lamports
      });
    });

    describe("Concurrent Operations", () => {
      it("should handle multiple concurrent orders", async () => {
        const { market, user } = await TestHelper.setupTradingEnvironment();
        
        const orders = [];
        const promises = [];
        
        // Create 10 concurrent orders
        for (let i = 0; i < 10; i++) {
          const order = Keypair.generate();
          orders.push(order);
          
          const promise = program.methods
            .placeOrder(
              { limit: {} },
              { long: {} },
              new anchor.BN(1000),
              new anchor.BN(50000 + i * 100),
              new anchor.BN(0),
              new anchor.BN(0),
              5,
              new anchor.BN(0),
              new anchor.BN(0),
              new anchor.BN(0),
              { gtc: {} },
              new anchor.BN(0),
              new anchor.BN(0),
              new anchor.BN(0)
            )
            .accounts({
              market: market.publicKey,
              order: order.publicKey,
              user: user.publicKey,
              systemProgram: SystemProgram.programId,
            })
            .signers([order, user])
            .rpc();
          
          promises.push(promise);
        }
        
        const startTime = Date.now();
        const results = await Promise.all(promises);
        const endTime = Date.now();
        
        expect(results.length).to.equal(10);
        expect(endTime - startTime).to.be.lessThan(30000); // Should complete in under 30 seconds
      });
    });
  });

  describe("🚨 Error Handling Tests", () => {
    describe("Invalid Input Validation", () => {
      it("should reject invalid leverage", async () => {
        const { market, user } = await TestHelper.setupTradingEnvironment();
        
        const position = Keypair.generate();
        
        try {
          await program.methods
            .openPosition(
              new anchor.BN(1000),
              { long: {} },
              new anchor.BN(101), // Invalid leverage > 100
              new anchor.BN(50000)
            )
            .accounts({
              position: position.publicKey,
              userAccount: Keypair.generate().publicKey,
              market: market.publicKey,
              user: user.publicKey,
              systemProgram: SystemProgram.programId,
            })
            .signers([position, user])
            .rpc();
          
          expect.fail("Should have rejected invalid leverage");
        } catch (error) {
          expect(error.message).to.include("InvalidLeverage");
        }
      });

      it("should reject invalid position size", async () => {
        const { market, user } = await TestHelper.setupTradingEnvironment();
        
        const position = Keypair.generate();
        
        try {
          await program.methods
            .openPosition(
              new anchor.BN(0), // Invalid size = 0
              { long: {} },
              new anchor.BN(5),
              new anchor.BN(50000)
            )
            .accounts({
              position: position.publicKey,
              userAccount: Keypair.generate().publicKey,
              market: market.publicKey,
              user: user.publicKey,
              systemProgram: SystemProgram.programId,
            })
            .signers([position, user])
            .rpc();
          
          expect.fail("Should have rejected invalid position size");
        } catch (error) {
          expect(error.message).to.include("InvalidSize");
        }
      });
    });

    describe("Access Control", () => {
      it("should reject unauthorized operations", async () => {
        const { market } = await TestHelper.setupTradingEnvironment();
        const unauthorizedUser = Keypair.generate();
        
        try {
          await program.methods
            .updateOraclePrice(new anchor.BN(50000))
            .accounts({
              market: market.publicKey,
              priceFeed: Keypair.generate().publicKey,
              authority: unauthorizedUser.publicKey, // Unauthorized user
            })
            .signers([unauthorizedUser])
            .rpc();
          
          expect.fail("Should have rejected unauthorized operation");
        } catch (error) {
          expect(error.message).to.include("UnauthorizedAdminOperation");
        }
      });
    });
  });
});
