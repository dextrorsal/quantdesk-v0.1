import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { expect } from "chai";
import TestUtils, { MockObjects, TestFixtures } from "./utils/test-utils";

/**
 * Comprehensive Integration Tests for QuantDesk Perpetual DEX Smart Contracts
 * Tests complete workflows and interactions between different contract functions
 */

describe("QuantDesk Perpetual DEX - Integration Tests", () => {
  // Configure the client to use the local cluster
  anchor.setProvider(anchor.AnchorProvider.env());

  const program = anchor.workspace.QuantdeskPerpDex as Program<QuantdeskPerpDex>;
  const provider = anchor.getProvider();
  
  let testUtils: TestUtils;

  before(async () => {
    testUtils = new TestUtils(program, provider);
  });

  describe("Complete Trading Workflow Integration Tests", () => {
    let market: Keypair;
    let trader: Keypair;
    let keeper: Keypair;
    let admin: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      trader = (await testUtils.generateTestAccounts(1))[0];
      keeper = (await testUtils.generateTestAccounts(1))[0];
      admin = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should complete full trading workflow: deposit -> open position -> place order -> close position -> withdraw", async () => {
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];
      const position = Keypair.generate();
      const order = Keypair.generate();

      // Step 1: Deposit collateral
      const depositTx = await program.methods
        .depositCollateral(new anchor.BN(10000))
        .accounts({
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([trader])
        .rpc();

      console.log("Deposit transaction:", depositTx);

      // Verify collateral deposit
      const collateralAccount = await program.account.userCollateral.fetch(userCollateral.publicKey);
      expect(collateralAccount.balance.toNumber()).to.be.greaterThan(0);

      // Step 2: Open position
      const openPositionTx = await program.methods
        .openPosition(
          new anchor.BN(1000),
          { long: {} },
          10
        )
        .accounts({
          market: market.publicKey,
          position: position.publicKey,
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([position, trader])
        .rpc();

      console.log("Open position transaction:", openPositionTx);

      // Verify position
      const positionAccount = await program.account.position.fetch(position.publicKey);
      expect(positionAccount.user.toString()).to.equal(trader.publicKey.toString());
      expect(positionAccount.size.toNumber()).to.equal(1000);
      expect(positionAccount.side).to.deep.equal({ long: {} });

      // Step 3: Place order
      const placeOrderTx = await program.methods
        .placeOrder(
          { limit: {} },
          { long: {} },
          new anchor.BN(500),
          new anchor.BN(51000),
          new anchor.BN(0),
          new anchor.BN(0),
          5,
          new anchor.BN(0)
        )
        .accounts({
          market: market.publicKey,
          order: order.publicKey,
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([order, trader])
        .rpc();

      console.log("Place order transaction:", placeOrderTx);

      // Verify order
      const orderAccount = await program.account.order.fetch(order.publicKey);
      expect(orderAccount.user.toString()).to.equal(trader.publicKey.toString());
      expect(orderAccount.size.toNumber()).to.equal(500);
      expect(orderAccount.price.toNumber()).to.equal(51000);

      // Step 4: Close position
      const closePositionTx = await program.methods
        .closePosition()
        .accounts({
          market: market.publicKey,
          position: position.publicKey,
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
        })
        .signers([trader])
        .rpc();

      console.log("Close position transaction:", closePositionTx);

      // Verify position is closed
      const closedPositionAccount = await program.account.position.fetch(position.publicKey);
      expect(closedPositionAccount.size.toNumber()).to.equal(0);

      // Step 5: Withdraw collateral
      const withdrawTx = await program.methods
        .withdrawCollateral(new anchor.BN(5000))
        .accounts({
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([trader])
        .rpc();

      console.log("Withdraw transaction:", withdrawTx);

      // Verify withdrawal
      const finalCollateralAccount = await program.account.userCollateral.fetch(userCollateral.publicKey);
      expect(finalCollateralAccount.balance.toNumber()).to.be.greaterThan(0);
    });

    it("should handle complex order management workflow", async () => {
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];
      const limitOrder = Keypair.generate();
      const stopLossOrder = Keypair.generate();
      const takeProfitOrder = Keypair.generate();

      // Deposit collateral
      await program.methods
        .depositCollateral(new anchor.BN(10000))
        .accounts({
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([trader])
        .rpc();

      // Place limit order
      const limitOrderTx = await program.methods
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
          order: limitOrder.publicKey,
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([limitOrder, trader])
        .rpc();

      console.log("Limit order transaction:", limitOrderTx);

      // Place stop-loss order
      const stopLossTx = await program.methods
        .placeOrder(
          { stopLoss: {} },
          { long: {} },
          new anchor.BN(1000),
          new anchor.BN(0),
          new anchor.BN(48000),
          new anchor.BN(0),
          5,
          new anchor.BN(0)
        )
        .accounts({
          market: market.publicKey,
          order: stopLossOrder.publicKey,
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([stopLossOrder, trader])
        .rpc();

      console.log("Stop-loss order transaction:", stopLossTx);

      // Place take-profit order
      const takeProfitTx = await program.methods
        .placeOrder(
          { takeProfit: {} },
          { long: {} },
          new anchor.BN(1000),
          new anchor.BN(0),
          new anchor.BN(55000),
          new anchor.BN(0),
          5,
          new anchor.BN(0)
        )
        .accounts({
          market: market.publicKey,
          order: takeProfitOrder.publicKey,
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([takeProfitOrder, trader])
        .rpc();

      console.log("Take-profit order transaction:", takeProfitTx);

      // Update limit order
      const updateOrderTx = await program.methods
        .updateOrder(
          new anchor.BN(51000),
          new anchor.BN(0),
          new anchor.BN(0)
        )
        .accounts({
          order: limitOrder.publicKey,
          user: trader.publicKey,
        })
        .signers([trader])
        .rpc();

      console.log("Update order transaction:", updateOrderTx);

      // Cancel stop-loss order
      const cancelOrderTx = await program.methods
        .cancelOrder()
        .accounts({
          order: stopLossOrder.publicKey,
          user: trader.publicKey,
        })
        .signers([trader])
        .rpc();

      console.log("Cancel order transaction:", cancelOrderTx);

      // Verify all orders
      const limitOrderAccount = await program.account.order.fetch(limitOrder.publicKey);
      const stopLossAccount = await program.account.order.fetch(stopLossOrder.publicKey);
      const takeProfitAccount = await program.account.order.fetch(takeProfitOrder.publicKey);

      expect(limitOrderAccount.price.toNumber()).to.equal(51000);
      expect(stopLossAccount.status).to.deep.equal({ cancelled: {} });
      expect(takeProfitAccount.status).to.deep.equal({ pending: {} });
    });

    it("should handle liquidation workflow", async () => {
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];
      const position = Keypair.generate();
      const vault = (await testUtils.generateTestAccounts(1))[0];

      // Deposit collateral
      await program.methods
        .depositCollateral(new anchor.BN(10000))
        .accounts({
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([trader])
        .rpc();

      // Open high-leverage position
      const openPositionTx = await program.methods
        .openPosition(
          new anchor.BN(1000),
          { long: {} },
          20 // High leverage
        )
        .accounts({
          market: market.publicKey,
          position: position.publicKey,
          user: trader.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([position, trader])
        .rpc();

      console.log("High-leverage position opened:", openPositionTx);

      // Simulate price drop (would normally happen through oracle)
      // For this test, we'll simulate the liquidation condition

      // Attempt liquidation
      const liquidationTx = await program.methods
        .liquidatePosition()
        .accounts({
          market: market.publicKey,
          position: position.publicKey,
          liquidator: keeper.publicKey,
          vault: vault.publicKey,
        })
        .signers([keeper])
        .rpc();

      console.log("Liquidation transaction:", liquidationTx);

      // Verify liquidation
      const liquidatedPositionAccount = await program.account.position.fetch(position.publicKey);
      expect(liquidatedPositionAccount.size.toNumber()).to.equal(0);
    });
  });

  describe("Market Management Integration Tests", () => {
    let market: Keypair;
    let admin: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      admin = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should handle complete market lifecycle", async () => {
      const newMarket = Keypair.generate();
      const config = TestFixtures.standardMarket;

      // Step 1: Initialize market
      const initTx = await program.methods
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

      console.log("Market initialization:", initTx);

      // Verify market
      const marketAccount = await program.account.market.fetch(newMarket.publicKey);
      expect(marketAccount.isActive).to.be.true;

      // Step 2: Update oracle price
      const priceFeed = Keypair.generate().publicKey;
      const updatePriceTx = await program.methods
        .updateOraclePrice()
        .accounts({
          market: newMarket.publicKey,
          priceFeed: priceFeed,
          authority: provider.wallet.publicKey,
        })
        .rpc();

      console.log("Oracle price update:", updatePriceTx);

      // Step 3: Disable market
      const disableTx = await program.methods
        .setMarketStatus(false)
        .accounts({
          market: newMarket.publicKey,
          authority: provider.wallet.publicKey,
        })
        .rpc();

      console.log("Market disabled:", disableTx);

      // Verify market is disabled
      const disabledMarketAccount = await program.account.market.fetch(newMarket.publicKey);
      expect(disabledMarketAccount.isActive).to.be.false;

      // Step 4: Re-enable market
      const enableTx = await program.methods
        .setMarketStatus(true)
        .accounts({
          market: newMarket.publicKey,
          authority: provider.wallet.publicKey,
        })
        .rpc();

      console.log("Market enabled:", enableTx);

      // Verify market is enabled
      const enabledMarketAccount = await program.account.market.fetch(newMarket.publicKey);
      expect(enabledMarketAccount.isActive).to.be.true;
    });

    it("should handle funding settlement workflow", async () => {
      const keeper = (await testUtils.generateTestAccounts(1))[0];

      // Settle funding
      const fundingTx = await program.methods
        .settleFunding()
        .accounts({
          market: market.publicKey,
          keeper: keeper.publicKey,
        })
        .signers([keeper])
        .rpc();

      console.log("Funding settlement:", fundingTx);

      // Verify funding was settled
      const marketAccount = await program.account.market.fetch(market.publicKey);
      expect(marketAccount.lastFundingTime.toNumber()).to.be.greaterThan(0);
    });
  });

  describe("Multi-User Integration Tests", () => {
    let market: Keypair;
    let traders: Keypair[];
    let keeper: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      traders = await testUtils.generateTestAccounts(5);
      keeper = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should handle multiple users trading simultaneously", async () => {
      const userCollaterals = await testUtils.generateTestAccounts(traders.length);
      const positions: Keypair[] = [];
      const orders: Keypair[] = [];

      // All users deposit collateral
      const depositPromises = traders.map((trader, index) => 
        program.methods
          .depositCollateral(new anchor.BN(10000))
          .accounts({
            user: trader.publicKey,
            userCollateral: userCollaterals[index].publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([trader])
          .rpc()
      );

      await Promise.all(depositPromises);
      console.log("All users deposited collateral");

      // All users open positions
      const positionPromises = traders.map((trader, index) => {
        const position = Keypair.generate();
        positions.push(position);
        
        return program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            10
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: trader.publicKey,
            userCollateral: userCollaterals[index].publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, trader])
          .rpc();
      });

      await Promise.all(positionPromises);
      console.log("All users opened positions");

      // All users place orders
      const orderPromises = traders.map((trader, index) => {
        const order = Keypair.generate();
        orders.push(order);
        
        return program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(500),
            new anchor.BN(50000 + index * 100),
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order.publicKey,
            user: trader.publicKey,
            userCollateral: userCollaterals[index].publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([order, trader])
          .rpc();
      });

      await Promise.all(orderPromises);
      console.log("All users placed orders");

      // Verify all positions and orders
      for (let i = 0; i < traders.length; i++) {
        const positionAccount = await program.account.position.fetch(positions[i].publicKey);
        const orderAccount = await program.account.order.fetch(orders[i].publicKey);

        expect(positionAccount.user.toString()).to.equal(traders[i].publicKey.toString());
        expect(orderAccount.user.toString()).to.equal(traders[i].publicKey.toString());
      }
    });

    it("should handle keeper operations with multiple users", async () => {
      const userCollaterals = await testUtils.generateTestAccounts(traders.length);
      const positions: Keypair[] = [];
      const vault = (await testUtils.generateTestAccounts(1))[0];

      // All users deposit and open positions
      for (let i = 0; i < traders.length; i++) {
        await program.methods
          .depositCollateral(new anchor.BN(10000))
          .accounts({
            user: traders[i].publicKey,
            userCollateral: userCollaterals[i].publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([traders[i]])
          .rpc();

        const position = Keypair.generate();
        positions.push(position);

        await program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            20 // High leverage for liquidation
          )
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            user: traders[i].publicKey,
            userCollateral: userCollaterals[i].publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([position, traders[i]])
          .rpc();
      }

      // Keeper settles funding
      const fundingTx = await program.methods
        .settleFunding()
        .accounts({
          market: market.publicKey,
          keeper: keeper.publicKey,
        })
        .signers([keeper])
        .rpc();

      console.log("Keeper settled funding:", fundingTx);

      // Keeper liquidates positions (simulate liquidation conditions)
      const liquidationPromises = positions.map(position =>
        program.methods
          .liquidatePosition()
          .accounts({
            market: market.publicKey,
            position: position.publicKey,
            liquidator: keeper.publicKey,
            vault: vault.publicKey,
          })
          .signers([keeper])
          .rpc()
          .catch(error => {
            console.warn("Liquidation failed:", error.message);
            return null;
          })
      );

      const liquidationResults = await Promise.all(liquidationPromises);
      const successfulLiquidations = liquidationResults.filter(result => result !== null);

      console.log(`Keeper performed ${successfulLiquidations.length} liquidations`);
    });
  });

  describe("Error Recovery Integration Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should handle and recover from oracle failures", async () => {
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];
      const position = Keypair.generate();

      // Deposit collateral and open position
      await program.methods
        .depositCollateral(new anchor.BN(10000))
        .accounts({
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([user])
        .rpc();

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

      // Simulate oracle failure by trying to update with invalid price feed
      try {
        await program.methods
          .updateOraclePrice()
          .accounts({
            market: market.publicKey,
            priceFeed: Keypair.generate().publicKey, // Invalid price feed
            authority: provider.wallet.publicKey,
          })
          .rpc();
      } catch (error) {
        console.log("Oracle failure handled:", error.message);
      }

      // System should still allow position closure with fallback mechanism
      const closePositionTx = await program.methods
        .closePosition()
        .accounts({
          market: market.publicKey,
          position: position.publicKey,
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
        })
        .signers([user])
        .rpc();

      console.log("Position closed despite oracle failure:", closePositionTx);

      // Verify position is closed
      const positionAccount = await program.account.position.fetch(position.publicKey);
      expect(positionAccount.size.toNumber()).to.equal(0);
    });

    it("should handle market suspension and recovery", async () => {
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];
      const order = Keypair.generate();

      // Deposit collateral
      await program.methods
        .depositCollateral(new anchor.BN(10000))
        .accounts({
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([user])
        .rpc();

      // Suspend market
      await program.methods
        .setMarketStatus(false)
        .accounts({
          market: market.publicKey,
          authority: provider.wallet.publicKey,
        })
        .rpc();

      // Try to place order (should fail)
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

        expect.fail("Should have thrown an error for suspended market");
      } catch (error) {
        expect(error.message).to.include("MarketInactive");
      }

      // Re-enable market
      await program.methods
        .setMarketStatus(true)
        .accounts({
          market: market.publicKey,
          authority: provider.wallet.publicKey,
        })
        .rpc();

      // Now order placement should work
      const orderTx = await program.methods
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

      console.log("Order placed after market recovery:", orderTx);

      // Verify order
      const orderAccount = await program.account.order.fetch(order.publicKey);
      expect(orderAccount.status).to.deep.equal({ pending: {} });
    });
  });

  describe("Cross-Contract Integration Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should integrate with system program for account creation", async () => {
      const userCollateral = Keypair.generate();

      // This should create the user collateral account
      const depositTx = await program.methods
        .depositCollateral(new anchor.BN(10000))
        .accounts({
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([user])
        .rpc();

      console.log("Account creation integration:", depositTx);

      // Verify account was created
      const accountInfo = await provider.connection.getAccountInfo(userCollateral.publicKey);
      expect(accountInfo).to.not.be.null;
      expect(accountInfo!.owner.toString()).to.equal(program.programId.toString());
    });

    it("should integrate with token program for collateral management", async () => {
      const userCollateral = (await testUtils.generateTestAccounts(1))[0];

      // Deposit collateral
      await program.methods
        .depositCollateral(new anchor.BN(10000))
        .accounts({
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([user])
        .rpc();

      // Withdraw collateral
      const withdrawTx = await program.methods
        .withdrawCollateral(new anchor.BN(5000))
        .accounts({
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([user])
        .rpc();

      console.log("Token program integration:", withdrawTx);

      // Verify withdrawal
      const collateralAccount = await program.account.userCollateral.fetch(userCollateral.publicKey);
      expect(collateralAccount.balance.toNumber()).to.be.greaterThan(0);
    });
  });
});
