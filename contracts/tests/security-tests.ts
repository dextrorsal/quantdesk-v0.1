import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { expect } from "chai";
import TestUtils, { SecurityTestUtils, MockObjects, TestFixtures } from "./utils/test-utils";

/**
 * Comprehensive Security Tests for QuantDesk Perpetual DEX Smart Contracts
 * Tests for common vulnerabilities, attack vectors, and security best practices
 */

describe("QuantDesk Perpetual DEX - Security Tests", () => {
  // Configure the client to use the local cluster
  anchor.setProvider(anchor.AnchorProvider.env());

  const program = anchor.workspace.QuantdeskPerpDex as Program<QuantdeskPerpDex>;
  const provider = anchor.getProvider();
  
  let testUtils: TestUtils;
  let securityUtils: SecurityTestUtils;

  before(async () => {
    testUtils = new TestUtils(program, provider);
    securityUtils = new SecurityTestUtils(program, provider);
  });

  describe("Access Control Security Tests", () => {
    let market: Keypair;
    let user: Keypair;
    let unauthorizedUser: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
      unauthorizedUser = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should prevent unauthorized market initialization", async () => {
      const newMarket = Keypair.generate();
      const config = TestFixtures.standardMarket;

      try {
        await program.methods
          .initializeMarket(
            config.baseAsset,
            config.quoteAsset,
            new anchor.BN(config.initialPrice),
            config.maxLeverage,
            config.initialMarginRatio,
            config.maintenanceMarginRatio
          )
          .accounts({
            market: newMarket.publicKey,
            authority: unauthorizedUser.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([newMarket, unauthorizedUser])
          .rpc();

        expect.fail("Should have thrown an error for unauthorized access");
      } catch (error) {
        expect(error.message).to.include("Unauthorized");
      }
    });

    it("should prevent unauthorized oracle price updates", async () => {
      const priceFeed = Keypair.generate().publicKey;

      try {
        await program.methods
          .updateOraclePrice()
          .accounts({
            market: market.publicKey,
            priceFeed: priceFeed,
            authority: unauthorizedUser.publicKey,
          })
          .signers([unauthorizedUser])
          .rpc();

        expect.fail("Should have thrown an error for unauthorized access");
      } catch (error) {
        expect(error.message).to.include("Unauthorized");
      }
    });

    it("should prevent unauthorized market status changes", async () => {
      try {
        await program.methods
          .setMarketStatus(false)
          .accounts({
            market: market.publicKey,
            authority: unauthorizedUser.publicKey,
          })
          .signers([unauthorizedUser])
          .rpc();

        expect.fail("Should have thrown an error for unauthorized access");
      } catch (error) {
        expect(error.message).to.include("Unauthorized");
      }
    });

    it("should prevent unauthorized position operations", async () => {
      const position = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      try {
        await program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            10
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: unauthorizedUser.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, unauthorizedUser])
          .rpc();

        expect.fail("Should have thrown an error for unauthorized access");
      } catch (error) {
        expect(error.message).to.include("Unauthorized");
      }
    });

    it("should prevent unauthorized order operations", async () => {
      const order = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      try {
        await program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(1000),
            new anchor.BN(50000),
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order.publicKey,
            user: unauthorizedUser.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([order, unauthorizedUser])
          .rpc();

        expect.fail("Should have thrown an error for unauthorized access");
      } catch (error) {
        expect(error.message).to.include("Unauthorized");
      }
    });

    it("should prevent unauthorized liquidation operations", async () => {
      const position = await testUtils.createMockPosition(
        market.publicKey,
        user,
        1000,
        { long: {} },
        20
      );
      const vault = (await testUtils.generateTestAccounts(1))[0];

      try {
        await program.methods
          .liquidatePosition()
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            liquidator: unauthorizedUser.publicKey,
            vault: vault.publicKey,
          })
          .signers([unauthorizedUser])
          .rpc();

        expect.fail("Should have thrown an error for unauthorized access");
      } catch (error) {
        expect(error.message).to.include("UnauthorizedKeeper");
      }
    });
  });

  describe("Input Validation Security Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should reject invalid leverage values", async () => {
      const position = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      // Test leverage = 0
      try {
        await program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            0 // Invalid leverage
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, user])
          .rpc();

        expect.fail("Should have thrown an error for invalid leverage");
      } catch (error) {
        expect(error.message).to.include("InvalidLeverage");
      }

      // Test leverage > 100
      try {
        await program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            101 // Invalid leverage
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, user])
          .rpc();

        expect.fail("Should have thrown an error for invalid leverage");
      } catch (error) {
        expect(error.message).to.include("InvalidLeverage");
      }
    });

    it("should reject invalid position sizes", async () => {
      const position = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      // Test size = 0
      try {
        await program.methods
          .openPosition(
            new anchor.BN(0), // Invalid size
            { long: {} },
            10
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, user])
          .rpc();

        expect.fail("Should have thrown an error for invalid size");
      } catch (error) {
        expect(error.message).to.include("InvalidSize");
      }

      // Test negative size
      try {
        await program.methods
          .openPosition(
            new anchor.BN(-1000), // Invalid size
            { long: {} },
            10
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, user])
          .rpc();

        expect.fail("Should have thrown an error for invalid size");
      } catch (error) {
        expect(error.message).to.include("InvalidSize");
      }
    });

    it("should reject invalid price values", async () => {
      const order = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      // Test price = 0 for limit order
      try {
        await program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(1000),
            new anchor.BN(0), // Invalid price for limit order
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([order, user])
          .rpc();

        expect.fail("Should have thrown an error for invalid price");
      } catch (error) {
        expect(error.message).to.include("InvalidPrice");
      }

      // Test negative price
      try {
        await program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(1000),
            new anchor.BN(-1000), // Invalid price
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([order, user])
          .rpc();

        expect.fail("Should have thrown an error for invalid price");
      } catch (error) {
        expect(error.message).to.include("InvalidPrice");
      }
    });

    it("should reject invalid margin ratios", async () => {
      const newMarket = Keypair.generate();

      // Test initial margin < maintenance margin
      try {
        await program.methods
          .initializeMarket(
            "BTC",
            "USDT",
            new anchor.BN(50000),
            100,
            200, // Initial margin
            300  // Maintenance margin (should be less than initial)
          )
          .accounts({
            market: newMarket.publicKey,
            authority: provider.wallet.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([newMarket])
          .rpc();

        expect.fail("Should have thrown an error for invalid margin ratios");
      } catch (error) {
        expect(error.message).to.include("InvalidMarginRatio");
      }
    });
  });

  describe("Oracle Security Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should reject stale oracle prices", async () => {
      const stalePriceFeed = Keypair.generate().publicKey;

      try {
        await program.methods
          .updateOraclePrice()
          .accounts({
            market: market.publicKey,
            priceFeed: stalePriceFeed,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        expect.fail("Should have thrown an error for stale price");
      } catch (error) {
        expect(error.message).to.include("StalePrice");
      }
    });

    it("should reject oracle price manipulation", async () => {
      const maliciousPriceFeed = Keypair.generate().publicKey;

      try {
        await program.methods
          .updateOraclePrice()
          .accounts({
            market: market.publicKey,
            priceFeed: maliciousPriceFeed,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        expect.fail("Should have thrown an error for price manipulation");
      } catch (error) {
        expect(error.message).to.include("InvalidPriceFeed");
      }
    });

    it("should reject extreme price changes", async () => {
      // Test for circuit breaker on extreme price changes
      const extremePriceFeed = Keypair.generate().publicKey;

      try {
        await program.methods
          .updateOraclePrice()
          .accounts({
            market: market.publicKey,
            priceFeed: extremePriceFeed,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        expect.fail("Should have thrown an error for extreme price change");
      } catch (error) {
        expect(error.message).to.include("PriceChangeTooLarge");
      }
    });

    it("should handle oracle failures gracefully", async () => {
      // Test system behavior when oracle fails
      const position = await testUtils.createMockPosition(
        market.publicKey,
        user,
        1000,
        { long: {} },
        10
      );

      try {
        await program.methods
          .closePosition()
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: (await testUtils.generateTestAccounts(1))[0].publicKey,
          })
          .signers([user])
          .rpc();

        expect.fail("Should have thrown an error for oracle failure");
      } catch (error) {
        expect(error.message).to.include("OracleUnavailable");
      }
    });
  });

  describe("Reentrancy Security Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should prevent reentrancy in position operations", async () => {
      const position = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      const instruction = program.methods
        .openPosition(
          new anchor.BN(1000),
          { long: {} },
          10
        )
        .accounts({
          market: market.publicKey,
          position: position.publicKey,
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        });

      const accounts = {
        market: market.publicKey,
        position: position.publicKey,
        user: user.publicKey,
        userCollateral: userCollateral.publicKey,
        systemProgram: SystemProgram.programId,
      };

      const isSecure = await securityUtils.testReentrancy(instruction, accounts);
      expect(isSecure).to.be.true;
    });

    it("should prevent reentrancy in order operations", async () => {
      const order = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      const instruction = program.methods
        .placeOrder(
          { limit: {} },
          { long: {} },
          new anchor.BN(1000),
          new anchor.BN(50000),
          new anchor.BN(0),
          new anchor.BN(0),
          5,
          new anchor.BN(0)
        )
        .accounts({
          market: market.publicKey,
          order: order.publicKey,
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        });

      const accounts = {
        market: market.publicKey,
        order: order.publicKey,
        user: user.publicKey,
        userCollateral: userCollateral.publicKey,
        systemProgram: SystemProgram.programId,
      };

      const isSecure = await securityUtils.testReentrancy(instruction, accounts);
      expect(isSecure).to.be.true;
    });

    it("should prevent reentrancy in liquidation operations", async () => {
      const position = await testUtils.createMockPosition(
        market.publicKey,
        user,
        1000,
        { long: {} },
        20
      );
      const liquidator = (await testUtils.generateTestAccounts(1))[0];
      const vault = (await testUtils.generateTestAccounts(1))[0];

      const instruction = program.methods
        .liquidatePosition()
        .accounts({
          market: market.publicKey,
          position: position.publicKey,
          liquidator: liquidator.publicKey,
          vault: vault.publicKey,
        });

      const accounts = {
        market: market.publicKey,
        position: position.publicKey,
        liquidator: liquidator.publicKey,
        vault: vault.publicKey,
      };

      const isSecure = await securityUtils.testReentrancy(instruction, accounts);
      expect(isSecure).to.be.true;
    });
  });

  describe("Integer Overflow/Underflow Security Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should prevent integer overflow in position calculations", async () => {
      const position = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      const instruction = program.methods
        .openPosition(
          new anchor.BN("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"), // Max value
          { long: {} },
          10
        )
        .accounts({
          market: market.publicKey,
          position: position.publicKey,
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        });

      const accounts = {
        market: market.publicKey,
        position: position.publicKey,
        user: user.publicKey,
        userCollateral: userCollateral.publicKey,
        systemProgram: SystemProgram.programId,
      };

      const isSecure = await securityUtils.testIntegerOverflow(instruction, accounts);
      expect(isSecure).to.be.true;
    });

    it("should prevent integer overflow in order calculations", async () => {
      const order = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      const instruction = program.methods
        .placeOrder(
          { limit: {} },
          { long: {} },
          new anchor.BN("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"), // Max value
          new anchor.BN("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF"), // Max value
          new anchor.BN(0),
          new anchor.BN(0),
          5,
          new anchor.BN(0)
        )
        .accounts({
          market: market.publicKey,
          order: order.publicKey,
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        });

      const accounts = {
        market: market.publicKey,
        order: order.publicKey,
        user: user.publicKey,
        userCollateral: userCollateral.publicKey,
        systemProgram: SystemProgram.programId,
      };

      const isSecure = await securityUtils.testIntegerOverflow(instruction, accounts);
      expect(isSecure).to.be.true;
    });

    it("should prevent integer underflow in margin calculations", async () => {
      const position = await testUtils.createMockPosition(
        market.publicKey,
        user,
        1000,
        { long: {} },
        10
      );
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      const instruction = program.methods
        .updatePositionMargin(new anchor.BN("0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF").neg())
        .accounts({
          position: position.publicKey,
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
        });

      const accounts = {
        position: position.publicKey,
        user: user.publicKey,
        userCollateral: userCollateral.publicKey,
      };

      const isSecure = await securityUtils.testIntegerOverflow(instruction, accounts);
      expect(isSecure).to.be.true;
    });
  });

  describe("Economic Security Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should prevent flash loan attacks", async () => {
      const position = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      // Simulate flash loan attack by trying to manipulate price and liquidate
      try {
        // Step 1: Open position
        await program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            20 // High leverage
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, user])
          .rpc();

        // Step 2: Attempt to manipulate price (should fail)
        const maliciousPriceFeed = Keypair.generate().publicKey;
        await program.methods
          .updateOraclePrice()
          .accounts({
            market: market.publicKey,
            priceFeed: maliciousPriceFeed,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        // Step 3: Attempt liquidation (should fail due to price manipulation protection)
        const liquidator = (await testUtils.generateTestAccounts(1))[0];
        const vault = (await testUtils.generateTestAccounts(1))[0];

        await program.methods
          .liquidatePosition()
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            liquidator: liquidator.publicKey,
            vault: vault.publicKey,
          })
          .signers([liquidator])
          .rpc();

        expect.fail("Should have thrown an error for flash loan attack");
      } catch (error) {
        expect(error.message).to.include("PriceManipulationDetected");
      }
    });

    it("should prevent sandwich attacks", async () => {
      const order1 = Keypair.generate();
      const order2 = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      // Simulate sandwich attack by placing orders around a large trade
      try {
        // Step 1: Place front-running order
        await program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(1000),
            new anchor.BN(49900), // Slightly below market
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order1.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([order1, user])
          .rpc();

        // Step 2: Place back-running order
        await program.methods
          .placeOrder(
            { limit: {} },
            { short: {} },
            new anchor.BN(1000),
            new anchor.BN(50100), // Slightly above market
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order2.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([order2, user])
          .rpc();

        expect.fail("Should have thrown an error for sandwich attack");
      } catch (error) {
        expect(error.message).to.include("SandwichAttackDetected");
      }
    });

    it("should prevent MEV attacks", async () => {
      const order = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      // Simulate MEV attack by trying to front-run with high gas
      try {
        await program.methods
          .placeOrder(
            { market: {} }, // Market order for immediate execution
            { long: {} },
            new anchor.BN(10000), // Large size
            new anchor.BN(0),
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([order, user])
          .rpc();

        expect.fail("Should have thrown an error for MEV attack");
      } catch (error) {
        expect(error.message).to.include("MEVProtection");
      }
    });
  });

  describe("Circuit Breaker Security Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should trigger circuit breaker on extreme price changes", async () => {
      // Simulate extreme price drop
      const extremePriceFeed = Keypair.generate().publicKey;

      try {
        await program.methods
          .updateOraclePrice()
          .accounts({
            market: market.publicKey,
            priceFeed: extremePriceFeed,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        expect.fail("Should have thrown an error for extreme price change");
      } catch (error) {
        expect(error.message).to.include("CircuitBreakerTriggered");
      }
    });

    it("should trigger circuit breaker on high volume", async () => {
      // Simulate high volume trading
      const orders = [];
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      try {
        // Create multiple large orders
        for (let i = 0; i < 100; i++) {
          const order = Keypair.generate();
          orders.push(order);

          await program.methods
            .placeOrder(
              { market: {} },
              { long: {} },
              new anchor.BN(10000), // Large size
              new anchor.BN(0),
              new anchor.BN(0),
              new anchor.BN(0),
              5,
              new anchor.BN(0)
            )
            .accounts({
              market: market.publicKey,
              order: order.publicKey,
              user: user.publicKey,
              userCollateral: userCollateral.publicKey,
              systemProgram: SystemProgram.programId,
            })
            .signers([order, user])
            .rpc();
        }

        expect.fail("Should have thrown an error for high volume");
      } catch (error) {
        expect(error.message).to.include("VolumeCircuitBreakerTriggered");
      }
    });

    it("should prevent trading during circuit breaker activation", async () => {
      // First trigger circuit breaker
      const extremePriceFeed = Keypair.generate().publicKey;

      try {
        await program.methods
          .updateOraclePrice()
          .accounts({
            market: market.publicKey,
            priceFeed: extremePriceFeed,
            authority: provider.wallet.publicKey,
          })
          .rpc();
      } catch (error) {
        // Circuit breaker triggered, now test trading prevention
      }

      // Now try to place an order (should fail)
      const order = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      try {
        await program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(1000),
            new anchor.BN(50000),
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([order, user])
          .rpc();

        expect.fail("Should have thrown an error for trading during circuit breaker");
      } catch (error) {
        expect(error.message).to.include("TradingHalted");
      }
    });
  });

  describe("Keeper Security Tests", () => {
    let market: Keypair;
    let user: Keypair;
    let keeper: Keypair;
    let unauthorizedKeeper: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
      keeper = (await testUtils.generateTestAccounts(1))[0];
      unauthorizedKeeper = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should prevent unauthorized keeper operations", async () => {
      const position = await testUtils.createMockPosition(
        market.publicKey,
        user,
        1000,
        { long: {} },
        20
      );
      const vault = (await testUtils.generateTestAccounts(1))[0];

      try {
        await program.methods
          .liquidatePosition()
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            liquidator: unauthorizedKeeper.publicKey,
            vault: vault.publicKey,
          })
          .signers([unauthorizedKeeper])
          .rpc();

        expect.fail("Should have thrown an error for unauthorized keeper");
      } catch (error) {
        expect(error.message).to.include("UnauthorizedKeeper");
      }
    });

    it("should prevent keeper from liquidating healthy positions", async () => {
      const healthyPosition = await testUtils.createMockPosition(
        market.publicKey,
        user,
        1000,
        { long: {} },
        5 // Low leverage
      );
      const vault = (await testUtils.generateTestAccounts(1))[0];

      try {
        await program.methods
          .liquidatePosition()
          .accounts({
            market: market.publicKey,
            position: healthyPosition.publicKey,
            liquidator: keeper.publicKey,
            vault: vault.publicKey,
          })
          .signers([keeper])
          .rpc();

        expect.fail("Should have thrown an error for healthy position");
      } catch (error) {
        expect(error.message).to.include("PositionNotLiquidatable");
      }
    });

    it("should prevent keeper from manipulating funding", async () => {
      try {
        // Try to settle funding multiple times in short period
        for (let i = 0; i < 10; i++) {
          await program.methods
            .settleFunding()
            .accounts({
              market: market.publicKey,
              keeper: keeper.publicKey,
            })
            .signers([keeper])
            .rpc();
        }

        expect.fail("Should have thrown an error for funding manipulation");
      } catch (error) {
        expect(error.message).to.include("FundingManipulationDetected");
      }
    });
  });

  describe("Cross-Program Security Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should prevent malicious program calls", async () => {
      const maliciousProgram = Keypair.generate().publicKey;
      const position = Keypair.generate();
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      try {
        await program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            10
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: maliciousProgram, // Malicious program
          })
          .signers([position, user])
          .rpc();

        expect.fail("Should have thrown an error for malicious program");
      } catch (error) {
        expect(error.message).to.include("InvalidProgram");
      }
    });

    it("should prevent account substitution attacks", async () => {
      const position = Keypair.generate();
      const maliciousUserCollateral = Keypair.generate().publicKey;

      try {
        await program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            10
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: maliciousUserCollateral, // Wrong collateral account
            systemProgram: SystemProgram.programId,
          })
          .signers([position, user])
          .rpc();

        expect.fail("Should have thrown an error for account substitution");
      } catch (error) {
        expect(error.message).to.include("InvalidAccount");
      }
    });
  });
});
