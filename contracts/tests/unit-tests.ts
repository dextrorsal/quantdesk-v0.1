import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { expect } from "chai";
import TestUtils, { SecurityTestUtils, PerformanceTestUtils, MockObjects, TestFixtures } from "./utils/test-utils";

/**
 * Comprehensive Unit Tests for QuantDesk Perpetual DEX Smart Contracts
 * Tests all individual functions in isolation with proper mocking
 */

describe("QuantDesk Perpetual DEX - Unit Tests", () => {
  // Configure the client to use the local cluster
  anchor.setProvider(anchor.AnchorProvider.env());

  const program = anchor.workspace.QuantdeskPerpDex as Program<QuantdeskPerpDex>;
  const provider = anchor.getProvider();
  
  let testUtils: TestUtils;
  let securityUtils: SecurityTestUtils;
  let performanceUtils: PerformanceTestUtils;

  before(async () => {
    testUtils = new TestUtils(program, provider);
    securityUtils = new SecurityTestUtils(program, provider);
    performanceUtils = new PerformanceTestUtils(program, provider);
  });

  describe("Market Management Unit Tests", () => {
    let market: Keypair;
    let priceFeed: PublicKey;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      priceFeed = mockMarket.priceFeed;
    });

    describe("initializeMarket", () => {
      it("should initialize market with valid parameters", async () => {
        const newMarket = Keypair.generate();
        const config = TestFixtures.standardMarket;

        const tx = await program.methods
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
            authority: provider.wallet.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([newMarket])
          .rpc();

        // Verify market account
        const marketAccount = await program.account.market.fetch(newMarket.publicKey);
        expect(marketAccount.baseAsset).to.equal(config.baseAsset);
        expect(marketAccount.quoteAsset).to.equal(config.quoteAsset);
        expect(marketAccount.maxLeverage).to.equal(config.maxLeverage);
        expect(marketAccount.initialMarginRatio).to.equal(config.initialMarginRatio);
        expect(marketAccount.maintenanceMarginRatio).to.equal(config.maintenanceMarginRatio);
        expect(marketAccount.isActive).to.be.true;
      });

      it("should reject invalid leverage", async () => {
        const newMarket = Keypair.generate();
        const config = TestFixtures.standardMarket;

        try {
          await program.methods
            .initializeMarket(
              config.baseAsset,
              config.quoteAsset,
              new anchor.BN(config.initialPrice),
              101, // Invalid leverage > 100
              config.initialMarginRatio,
              config.maintenanceMarginRatio
            )
            .accounts({
              market: newMarket.publicKey,
              authority: provider.wallet.publicKey,
              systemProgram: SystemProgram.programId,
            })
            .signers([newMarket])
            .rpc();

          expect.fail("Should have thrown an error for invalid leverage");
        } catch (error) {
          expect(error.message).to.include("InvalidLeverage");
        }
      });

      it("should reject invalid margin ratios", async () => {
        const newMarket = Keypair.generate();
        const config = TestFixtures.standardMarket;

        try {
          await program.methods
            .initializeMarket(
              config.baseAsset,
              config.quoteAsset,
              new anchor.BN(config.initialPrice),
              config.maxLeverage,
              100, // Invalid: initial margin < maintenance margin
              config.maintenanceMarginRatio
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

      it("should reject unauthorized initialization", async () => {
        const newMarket = Keypair.generate();
        const unauthorizedUser = Keypair.generate();
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
    });

    describe("updateOraclePrice", () => {
      it("should update oracle price with valid data", async () => {
        const newPrice = 51000;
        const mockPriceData = MockObjects.createMockPriceFeed(newPrice);

        const tx = await program.methods
          .updateOraclePrice()
          .accounts({
            market: market.publicKey,
            priceFeed: priceFeed,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        // Verify price update
        const marketAccount = await program.account.market.fetch(market.publicKey);
        expect(marketAccount.lastOraclePrice.toNumber()).to.equal(newPrice);
      });

      it("should reject stale price data", async () => {
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

          expect.fail("Should have thrown an error for stale price data");
        } catch (error) {
          expect(error.message).to.include("StalePrice");
        }
      });

      it("should reject unauthorized price updates", async () => {
        const unauthorizedUser = Keypair.generate();

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
    });

    describe("setMarketStatus", () => {
      it("should update market status", async () => {
        const tx = await program.methods
          .setMarketStatus(false) // Disable market
          .accounts({
            market: market.publicKey,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        // Verify status update
        const marketAccount = await program.account.market.fetch(market.publicKey);
        expect(marketAccount.isActive).to.be.false;
      });

      it("should reject unauthorized status updates", async () => {
        const unauthorizedUser = Keypair.generate();

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
    });
  });

  describe("Position Management Unit Tests", () => {
    let market: Keypair;
    let user: Keypair;
    let userCollateral: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
      userCollateral = (await testUtils.generateTestAccounts(1))[0];
    });

    describe("openPosition", () => {
      it("should open long position with valid parameters", async () => {
        const position = Keypair.generate();
        const config = TestFixtures.testPositions.longPosition;

        const tx = await program.methods
          .openPosition(
            new anchor.BN(config.size),
            { long: {} },
            config.leverage
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

        // Verify position account
        const positionAccount = await program.account.position.fetch(position.publicKey);
        expect(positionAccount.user.toString()).to.equal(user.publicKey.toString());
        expect(positionAccount.market.toString()).to.equal(market.publicKey.toString());
        expect(positionAccount.size.toNumber()).to.equal(config.size);
        expect(positionAccount.side).to.deep.equal({ long: {} });
        expect(positionAccount.leverage).to.equal(config.leverage);
      });

      it("should open short position with valid parameters", async () => {
        const position = Keypair.generate();
        const config = TestFixtures.testPositions.shortPosition;

        const tx = await program.methods
          .openPosition(
            new anchor.BN(config.size),
            { short: {} },
            config.leverage
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

        // Verify position account
        const positionAccount = await program.account.position.fetch(position.publicKey);
        expect(positionAccount.side).to.deep.equal({ short: {} });
        expect(positionAccount.size.toNumber()).to.equal(config.size);
        expect(positionAccount.leverage).to.equal(config.leverage);
      });

      it("should reject position with insufficient margin", async () => {
        const position = Keypair.generate();
        const config = TestFixtures.testPositions.highLeveragePosition;

        try {
          await program.methods
            .openPosition(
              new anchor.BN(config.size),
              { long: {} },
              config.leverage
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

          expect.fail("Should have thrown an error for insufficient margin");
        } catch (error) {
          expect(error.message).to.include("InsufficientMargin");
        }
      });

      it("should reject position with invalid leverage", async () => {
        const position = Keypair.generate();

        try {
          await program.methods
            .openPosition(
              new anchor.BN(1000),
              { long: {} },
              101 // Invalid leverage > 100
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

      it("should reject position on inactive market", async () => {
        // First disable the market
        await program.methods
          .setMarketStatus(false)
          .accounts({
            market: market.publicKey,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        const position = Keypair.generate();

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
              systemProgram: SystemProgram.programId,
            })
            .signers([position, user])
            .rpc();

          expect.fail("Should have thrown an error for inactive market");
        } catch (error) {
          expect(error.message).to.include("MarketInactive");
        }
      });
    });

    describe("closePosition", () => {
      let position: Keypair;

      beforeEach(async () => {
        position = await testUtils.createMockPosition(
          market.publicKey,
          user,
          TestFixtures.testPositions.longPosition.size,
          { long: {} },
          TestFixtures.testPositions.longPosition.leverage
        );
      });

      it("should close position successfully", async () => {
        const tx = await program.methods
          .closePosition()
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
          })
          .signers([user])
          .rpc();

        // Verify position is closed
        const positionAccount = await program.account.position.fetch(position.publicKey);
        expect(positionAccount.size.toNumber()).to.equal(0);
      });

      it("should reject closing non-existent position", async () => {
        const nonExistentPosition = Keypair.generate();

        try {
          await program.methods
            .closePosition()
            .accounts({
              market: market.publicKey,
              position: nonExistentPosition.publicKey,
              user: user.publicKey,
              userCollateral: userCollateral.publicKey,
            })
            .signers([user])
            .rpc();

          expect.fail("Should have thrown an error for non-existent position");
        } catch (error) {
          expect(error.message).to.include("PositionNotFound");
        }
      });

      it("should reject unauthorized position closure", async () => {
        const unauthorizedUser = Keypair.generate();

        try {
          await program.methods
            .closePosition()
            .accounts({
              market: market.publicKey,
              position: position.publicKey,
              user: unauthorizedUser.publicKey,
              userCollateral: userCollateral.publicKey,
            })
            .signers([unauthorizedUser])
            .rpc();

          expect.fail("Should have thrown an error for unauthorized access");
        } catch (error) {
          expect(error.message).to.include("Unauthorized");
        }
      });
    });

    describe("updatePositionMargin", () => {
      let position: Keypair;

      beforeEach(async () => {
        position = await testUtils.createMockPosition(
          market.publicKey,
          user,
          TestFixtures.testPositions.longPosition.size,
          { long: {} },
          TestFixtures.testPositions.longPosition.leverage
        );
      });

      it("should add margin to position", async () => {
        const additionalMargin = new anchor.BN(1000);

        const tx = await program.methods
          .updatePositionMargin(additionalMargin)
          .accounts({
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
          })
          .signers([user])
          .rpc();

        // Verify margin update
        const positionAccount = await program.account.position.fetch(position.publicKey);
        expect(positionAccount.margin.toNumber()).to.be.greaterThan(0);
      });

      it("should remove margin from position", async () => {
        const marginToRemove = new anchor.BN(500);

        const tx = await program.methods
          .updatePositionMargin(marginToRemove.neg())
          .accounts({
            position: position.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
          })
          .signers([user])
          .rpc();

        // Verify margin update
        const positionAccount = await program.account.position.fetch(position.publicKey);
        expect(positionAccount.margin.toNumber()).to.be.greaterThan(0);
      });

      it("should reject margin update that violates maintenance margin", async () => {
        const excessiveMarginRemoval = new anchor.BN(10000);

        try {
          await program.methods
            .updatePositionMargin(excessiveMarginRemoval.neg())
            .accounts({
              position: position.publicKey,
              user: user.publicKey,
              userCollateral: userCollateral.publicKey,
            })
            .signers([user])
            .rpc();

          expect.fail("Should have thrown an error for insufficient margin");
        } catch (error) {
          expect(error.message).to.include("InsufficientMargin");
        }
      });
    });
  });

  describe("Order Management Unit Tests", () => {
    let market: Keypair;
    let user: Keypair;
    let userCollateral: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
      userCollateral = (await testUtils.generateTestAccounts(1))[0];
    });

    describe("placeOrder", () => {
      it("should place limit order successfully", async () => {
        const order = Keypair.generate();
        const config = TestFixtures.testOrders.limitOrder;

        const tx = await program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(config.size),
            new anchor.BN(config.price),
            new anchor.BN(0), // stopPrice
            new anchor.BN(0), // trailingDistance
            config.leverage,
            new anchor.BN(0) // expiresAt
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

        // Verify order account
        const orderAccount = await program.account.order.fetch(order.publicKey);
        expect(orderAccount.user.toString()).to.equal(user.publicKey.toString());
        expect(orderAccount.market.toString()).to.equal(market.publicKey.toString());
        expect(orderAccount.orderType).to.deep.equal({ limit: {} });
        expect(orderAccount.side).to.deep.equal({ long: {} });
        expect(orderAccount.size.toNumber()).to.equal(config.size);
        expect(orderAccount.price.toNumber()).to.equal(config.price);
        expect(orderAccount.leverage).to.equal(config.leverage);
        expect(orderAccount.status).to.deep.equal({ pending: {} });
      });

      it("should place market order successfully", async () => {
        const order = Keypair.generate();
        const config = TestFixtures.testOrders.marketOrder;

        const tx = await program.methods
          .placeOrder(
            { market: {} },
            { short: {} },
            new anchor.BN(config.size),
            new anchor.BN(0), // price (not used for market orders)
            new anchor.BN(0), // stopPrice
            new anchor.BN(0), // trailingDistance
            config.leverage,
            new anchor.BN(0) // expiresAt
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

        // Verify order account
        const orderAccount = await program.account.order.fetch(order.publicKey);
        expect(orderAccount.orderType).to.deep.equal({ market: {} });
        expect(orderAccount.side).to.deep.equal({ short: {} });
        expect(orderAccount.size.toNumber()).to.equal(config.size);
        expect(orderAccount.leverage).to.equal(config.leverage);
      });

      it("should place stop-loss order successfully", async () => {
        const order = Keypair.generate();
        const config = TestFixtures.testOrders.stopLossOrder;

        const tx = await program.methods
          .placeOrder(
            { stopLoss: {} },
            { long: {} },
            new anchor.BN(config.size),
            new anchor.BN(0), // price
            new anchor.BN(config.stopPrice),
            new anchor.BN(0), // trailingDistance
            config.leverage,
            new anchor.BN(0) // expiresAt
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

        // Verify order account
        const orderAccount = await program.account.order.fetch(order.publicKey);
        expect(orderAccount.orderType).to.deep.equal({ stopLoss: {} });
        expect(orderAccount.stopPrice.toNumber()).to.equal(config.stopPrice);
      });

      it("should reject order with invalid price", async () => {
        const order = Keypair.generate();

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
      });

      it("should reject order with invalid leverage", async () => {
        const order = Keypair.generate();

        try {
          await program.methods
            .placeOrder(
              { limit: {} },
              { long: {} },
              new anchor.BN(1000),
              new anchor.BN(50000),
              new anchor.BN(0),
              new anchor.BN(0),
              101, // Invalid leverage > 100
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

          expect.fail("Should have thrown an error for invalid leverage");
        } catch (error) {
          expect(error.message).to.include("InvalidLeverage");
        }
      });

      it("should reject order on inactive market", async () => {
        // First disable the market
        await program.methods
          .setMarketStatus(false)
          .accounts({
            market: market.publicKey,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        const order = Keypair.generate();

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

          expect.fail("Should have thrown an error for inactive market");
        } catch (error) {
          expect(error.message).to.include("MarketInactive");
        }
      });
    });

    describe("cancelOrder", () => {
      let order: Keypair;

      beforeEach(async () => {
        order = await testUtils.createMockOrder(
          market.publicKey,
          user,
          { limit: {} },
          { long: {} },
          TestFixtures.testOrders.limitOrder.size,
          TestFixtures.testOrders.limitOrder.price,
          TestFixtures.testOrders.limitOrder.leverage
        );
      });

      it("should cancel order successfully", async () => {
        const tx = await program.methods
          .cancelOrder()
          .accounts({
            order: order.publicKey,
            user: user.publicKey,
          })
          .signers([user])
          .rpc();

        // Verify order is cancelled
        const orderAccount = await program.account.order.fetch(order.publicKey);
        expect(orderAccount.status).to.deep.equal({ cancelled: {} });
      });

      it("should reject cancelling non-existent order", async () => {
        const nonExistentOrder = Keypair.generate();

        try {
          await program.methods
            .cancelOrder()
            .accounts({
              order: nonExistentOrder.publicKey,
              user: user.publicKey,
            })
            .signers([user])
            .rpc();

          expect.fail("Should have thrown an error for non-existent order");
        } catch (error) {
          expect(error.message).to.include("OrderNotFound");
        }
      });

      it("should reject unauthorized order cancellation", async () => {
        const unauthorizedUser = Keypair.generate();

        try {
          await program.methods
            .cancelOrder()
            .accounts({
              order: order.publicKey,
              user: unauthorizedUser.publicKey,
            })
            .signers([unauthorizedUser])
            .rpc();

          expect.fail("Should have thrown an error for unauthorized access");
        } catch (error) {
          expect(error.message).to.include("Unauthorized");
        }
      });

      it("should reject cancelling already executed order", async () => {
        // First execute the order (simplified for test)
        // This would normally happen through the matching engine
        // For this test, we'll simulate an executed order status

        try {
          await program.methods
            .cancelOrder()
            .accounts({
              order: order.publicKey,
              user: user.publicKey,
            })
            .signers([user])
            .rpc();

          expect.fail("Should have thrown an error for executed order");
        } catch (error) {
          expect(error.message).to.include("OrderAlreadyExecuted");
        }
      });
    });

    describe("updateOrder", () => {
      let order: Keypair;

      beforeEach(async () => {
        order = await testUtils.createMockOrder(
          market.publicKey,
          user,
          { limit: {} },
          { long: {} },
          TestFixtures.testOrders.limitOrder.size,
          TestFixtures.testOrders.limitOrder.price,
          TestFixtures.testOrders.limitOrder.leverage
        );
      });

      it("should update order price successfully", async () => {
        const newPrice = 51000;

        const tx = await program.methods
          .updateOrder(
            new anchor.BN(newPrice),
            new anchor.BN(0), // stopPrice
            new anchor.BN(0) // trailingDistance
          )
          .accounts({
            order: order.publicKey,
            user: user.publicKey,
          })
          .signers([user])
          .rpc();

        // Verify order update
        const orderAccount = await program.account.order.fetch(order.publicKey);
        expect(orderAccount.price.toNumber()).to.equal(newPrice);
      });

      it("should reject updating non-existent order", async () => {
        const nonExistentOrder = Keypair.generate();

        try {
          await program.methods
            .updateOrder(
              new anchor.BN(51000),
              new anchor.BN(0),
              new anchor.BN(0)
            )
            .accounts({
              order: nonExistentOrder.publicKey,
              user: user.publicKey,
            })
            .signers([user])
            .rpc();

          expect.fail("Should have thrown an error for non-existent order");
        } catch (error) {
          expect(error.message).to.include("OrderNotFound");
        }
      });

      it("should reject unauthorized order update", async () => {
        const unauthorizedUser = Keypair.generate();

        try {
          await program.methods
            .updateOrder(
              new anchor.BN(51000),
              new anchor.BN(0),
              new anchor.BN(0)
            )
            .accounts({
              order: order.publicKey,
              user: unauthorizedUser.publicKey,
            })
            .signers([unauthorizedUser])
            .rpc();

          expect.fail("Should have thrown an error for unauthorized access");
        } catch (error) {
          expect(error.message).to.include("Unauthorized");
        }
      });
    });
  });

  describe("Liquidation Unit Tests", () => {
    let market: Keypair;
    let position: Keypair;
    let user: Keypair;
    let liquidator: Keypair;
    let vault: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
      liquidator = (await testUtils.generateTestAccounts(1))[0];
      vault = (await testUtils.generateTestAccounts(1))[0];
      
      // Create a position with high leverage for liquidation testing
      position = await testUtils.createMockPosition(
        market.publicKey,
        user,
        1000,
        { long: {} },
        20 // High leverage
      );
    });

    describe("liquidatePosition", () => {
      it("should liquidate position with low health factor", async () => {
        // Simulate price drop that triggers liquidation
        // This would normally happen through oracle price updates
        // For this test, we'll simulate the liquidation condition

        const tx = await program.methods
          .liquidatePosition()
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            liquidator: liquidator.publicKey,
            vault: vault.publicKey,
          })
          .signers([liquidator])
          .rpc();

        // Verify position was liquidated
        const positionAccount = await program.account.position.fetch(position.publicKey);
        expect(positionAccount.size.toNumber()).to.equal(0);
      });

      it("should reject liquidating healthy position", async () => {
        // Create a healthy position
        const healthyPosition = await testUtils.createMockPosition(
          market.publicKey,
          user,
          1000,
          { long: {} },
          5 // Low leverage
        );

        try {
          await program.methods
            .liquidatePosition()
            .accounts({
              market: market.publicKey,
              position: healthyPosition.publicKey,
              liquidator: liquidator.publicKey,
              vault: vault.publicKey,
            })
            .signers([liquidator])
            .rpc();

          expect.fail("Should have thrown an error for healthy position");
        } catch (error) {
          expect(error.message).to.include("PositionNotLiquidatable");
        }
      });

      it("should reject liquidation by unauthorized keeper", async () => {
        const unauthorizedKeeper = Keypair.generate();

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
    });
  });

  describe("Funding Unit Tests", () => {
    let market: Keypair;
    let keeper: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      keeper = (await testUtils.generateTestAccounts(1))[0];
    });

    describe("settleFunding", () => {
      it("should settle funding successfully", async () => {
        const tx = await program.methods
          .settleFunding()
          .accounts({
            market: market.publicKey,
            keeper: keeper.publicKey,
          })
          .signers([keeper])
          .rpc();

        // Verify funding was settled
        const marketAccount = await program.account.market.fetch(market.publicKey);
        expect(marketAccount.lastFundingTime.toNumber()).to.be.greaterThan(0);
      });

      it("should reject funding settlement by unauthorized keeper", async () => {
        const unauthorizedKeeper = Keypair.generate();

        try {
          await program.methods
            .settleFunding()
            .accounts({
              market: market.publicKey,
              keeper: unauthorizedKeeper.publicKey,
            })
            .signers([unauthorizedKeeper])
            .rpc();

          expect.fail("Should have thrown an error for unauthorized keeper");
        } catch (error) {
          expect(error.message).to.include("UnauthorizedKeeper");
        }
      });

      it("should reject funding settlement on inactive market", async () => {
        // First disable the market
        await program.methods
          .setMarketStatus(false)
          .accounts({
            market: market.publicKey,
            authority: provider.wallet.publicKey,
          })
          .rpc();

        try {
          await program.methods
            .settleFunding()
            .accounts({
              market: market.publicKey,
              keeper: keeper.publicKey,
            })
            .signers([keeper])
            .rpc();

          expect.fail("Should have thrown an error for inactive market");
        } catch (error) {
          expect(error.message).to.include("MarketInactive");
        }
      });
    });
  });

  describe("Collateral Management Unit Tests", () => {
    let user: Keypair;
    let userCollateral: Keypair;

    beforeEach(async () => {
      user = (await testUtils.generateTestAccounts(1))[0];
      userCollateral = (await testUtils.generateTestAccounts(1))[0];
    });

    describe("depositCollateral", () => {
      it("should deposit collateral successfully", async () => {
        const depositAmount = new anchor.BN(1000);

        const tx = await program.methods
          .depositCollateral(depositAmount)
          .accounts({
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([user])
          .rpc();

        // Verify collateral deposit
        const collateralAccount = await program.account.userCollateral.fetch(userCollateral.publicKey);
        expect(collateralAccount.balance.toNumber()).to.be.greaterThan(0);
      });

      it("should reject deposit with insufficient balance", async () => {
        const excessiveAmount = new anchor.BN(1000000);

        try {
          await program.methods
            .depositCollateral(excessiveAmount)
            .accounts({
              user: user.publicKey,
              userCollateral: userCollateral.publicKey,
              systemProgram: SystemProgram.programId,
            })
            .signers([user])
            .rpc();

          expect.fail("Should have thrown an error for insufficient balance");
        } catch (error) {
          expect(error.message).to.include("InsufficientBalance");
        }
      });
    });

    describe("withdrawCollateral", () => {
      beforeEach(async () => {
        // First deposit some collateral
        await program.methods
          .depositCollateral(new anchor.BN(1000))
          .accounts({
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([user])
          .rpc();
      });

      it("should withdraw collateral successfully", async () => {
        const withdrawAmount = new anchor.BN(500);

        const tx = await program.methods
          .withdrawCollateral(withdrawAmount)
          .accounts({
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([user])
          .rpc();

        // Verify collateral withdrawal
        const collateralAccount = await program.account.userCollateral.fetch(userCollateral.publicKey);
        expect(collateralAccount.balance.toNumber()).to.be.greaterThan(0);
      });

      it("should reject withdrawal with insufficient collateral", async () => {
        const excessiveAmount = new anchor.BN(10000);

        try {
          await program.methods
            .withdrawCollateral(excessiveAmount)
            .accounts({
              user: user.publicKey,
              userCollateral: userCollateral.publicKey,
              systemProgram: SystemProgram.programId,
            })
            .signers([user])
            .rpc();

          expect.fail("Should have thrown an error for insufficient collateral");
        } catch (error) {
          expect(error.message).to.include("InsufficientCollateral");
        }
      });

      it("should reject withdrawal that violates margin requirements", async () => {
        // This would require having open positions
        // For this test, we'll simulate the margin requirement check

        try {
          await program.methods
            .withdrawCollateral(new anchor.BN(800))
            .accounts({
              user: user.publicKey,
              userCollateral: userCollateral.publicKey,
              systemProgram: SystemProgram.programId,
            })
            .signers([user])
            .rpc();

          expect.fail("Should have thrown an error for margin requirement violation");
        } catch (error) {
          expect(error.message).to.include("MarginRequirementViolated");
        }
      });
    });
  });
});
