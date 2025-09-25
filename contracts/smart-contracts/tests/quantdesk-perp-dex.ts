import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { expect } from "chai";

describe("quantdesk-perp-dex", () => {
  // Configure the client to use the local cluster.
  anchor.setProvider(anchor.AnchorProvider.env());

  const program = anchor.workspace.QuantdeskPerpDex as Program<QuantdeskPerpDex>;
  const provider = anchor.getProvider();

  // Test accounts
  let market: Keypair;
  let user: Keypair;
  let position: Keypair;
  let order: Keypair;
  let priceFeed: PublicKey;

  before(async () => {
    // Generate test accounts
    market = Keypair.generate();
    user = Keypair.generate();
    position = Keypair.generate();
    order = Keypair.generate();
    
    // Mock price feed account (in production, this would be a real Pyth price feed)
    priceFeed = Keypair.generate().publicKey;
    
    // Airdrop SOL to user
    await provider.connection.requestAirdrop(user.publicKey, 2 * anchor.web3.LAMPORTS_PER_SOL);
  });

  describe("Market Operations", () => {
    it("Initializes a new market", async () => {
      const baseAsset = "BTC";
      const quoteAsset = "USDT";
      const initialPrice = 50000;
      const maxLeverage = 100;
      const initialMarginRatio = 500; // 5%
      const maintenanceMarginRatio = 300; // 3%

      const tx = await program.methods
        .initializeMarket(
          baseAsset,
          quoteAsset,
          new anchor.BN(initialPrice),
          maxLeverage,
          initialMarginRatio,
          maintenanceMarginRatio
        )
        .accounts({
          market: market.publicKey,
          authority: provider.wallet.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([market])
        .rpc();

      console.log("Market initialization transaction:", tx);

      // Verify market account
      const marketAccount = await program.account.market.fetch(market.publicKey);
      expect(marketAccount.baseAsset).to.equal(baseAsset);
      expect(marketAccount.quoteAsset).to.equal(quoteAsset);
      expect(marketAccount.maxLeverage).to.equal(maxLeverage);
      expect(marketAccount.initialMarginRatio).to.equal(initialMarginRatio);
      expect(marketAccount.maintenanceMarginRatio).to.equal(maintenanceMarginRatio);
      expect(marketAccount.isActive).to.be.true;
    });

    it("Updates oracle price", async () => {
      const newPrice = 51000;

      const tx = await program.methods
        .updateOraclePrice()
        .accounts({
          market: market.publicKey,
          priceFeed: priceFeed,
          authority: provider.wallet.publicKey,
        })
        .rpc();

      console.log("Oracle price update transaction:", tx);

      // Verify price update
      const marketAccount = await program.account.market.fetch(market.publicKey);
      expect(marketAccount.lastOraclePrice.toNumber()).to.equal(newPrice);
    });
  });

  describe("Position Operations", () => {
    it("Opens a long position", async () => {
      const size = 1000; // 0.001 BTC
      const side = { long: {} };
      const leverage = 10;

      // Create mock user collateral account
      const userCollateral = Keypair.generate();

      const tx = await program.methods
        .openPosition(
          new anchor.BN(size),
          side,
          leverage
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

      console.log("Position opening transaction:", tx);

      // Verify position account
      const positionAccount = await program.account.position.fetch(position.publicKey);
      expect(positionAccount.user.toString()).to.equal(user.publicKey.toString());
      expect(positionAccount.market.toString()).to.equal(market.publicKey.toString());
      expect(positionAccount.size.toNumber()).to.equal(size);
      expect(positionAccount.side).to.deep.equal(side);
      expect(positionAccount.leverage).to.equal(leverage);
    });

    it("Closes a position", async () => {
      const userCollateral = Keypair.generate();

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

      console.log("Position closing transaction:", tx);

      // Verify position is closed
      const positionAccount = await program.account.position.fetch(position.publicKey);
      expect(positionAccount.size.toNumber()).to.equal(0);
    });
  });

  describe("Order Operations", () => {
    it("Places a limit order", async () => {
      const orderType = { limit: {} };
      const side = { long: {} };
      const size = 1000;
      const price = 52000;
      const leverage = 5;
      const expiresAt = Math.floor(Date.now() / 1000) + 3600; // 1 hour from now

      const userCollateral = Keypair.generate();

      const tx = await program.methods
        .placeOrder(
          orderType,
          side,
          new anchor.BN(size),
          new anchor.BN(price),
          new anchor.BN(0), // stopPrice
          new anchor.BN(0), // trailingDistance
          leverage,
          new anchor.BN(expiresAt)
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

      console.log("Order placement transaction:", tx);

      // Verify order account
      const orderAccount = await program.account.order.fetch(order.publicKey);
      expect(orderAccount.user.toString()).to.equal(user.publicKey.toString());
      expect(orderAccount.market.toString()).to.equal(market.publicKey.toString());
      expect(orderAccount.orderType).to.deep.equal(orderType);
      expect(orderAccount.side).to.deep.equal(side);
      expect(orderAccount.size.toNumber()).to.equal(size);
      expect(orderAccount.price.toNumber()).to.equal(price);
      expect(orderAccount.leverage).to.equal(leverage);
      expect(orderAccount.status).to.deep.equal({ pending: {} });
    });

    it("Cancels an order", async () => {
      const tx = await program.methods
        .cancelOrder()
        .accounts({
          order: order.publicKey,
          user: user.publicKey,
        })
        .signers([user])
        .rpc();

      console.log("Order cancellation transaction:", tx);

      // Verify order is cancelled
      const orderAccount = await program.account.order.fetch(order.publicKey);
      expect(orderAccount.status).to.deep.equal({ cancelled: {} });
    });
  });

  describe("Advanced Order Types", () => {
    it("Places a stop-loss order", async () => {
      const stopOrder = Keypair.generate();
      const orderType = { stopLoss: {} };
      const side = { long: {} };
      const size = 1000;
      const stopPrice = 48000;
      const leverage = 5;

      const userCollateral = Keypair.generate();

      const tx = await program.methods
        .placeOrder(
          orderType,
          side,
          new anchor.BN(size),
          new anchor.BN(0), // price
          new anchor.BN(stopPrice),
          new anchor.BN(0), // trailingDistance
          leverage,
          new anchor.BN(0) // expiresAt
        )
        .accounts({
          market: market.publicKey,
          order: stopOrder.publicKey,
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([stopOrder, user])
        .rpc();

      console.log("Stop-loss order placement transaction:", tx);

      // Verify stop-loss order
      const orderAccount = await program.account.order.fetch(stopOrder.publicKey);
      expect(orderAccount.orderType).to.deep.equal(orderType);
      expect(orderAccount.stopPrice.toNumber()).to.equal(stopPrice);
    });

    it("Places a take-profit order", async () => {
      const tpOrder = Keypair.generate();
      const orderType = { takeProfit: {} };
      const side = { long: {} };
      const size = 1000;
      const stopPrice = 55000;
      const leverage = 5;

      const userCollateral = Keypair.generate();

      const tx = await program.methods
        .placeOrder(
          orderType,
          side,
          new anchor.BN(size),
          new anchor.BN(0), // price
          new anchor.BN(stopPrice),
          new anchor.BN(0), // trailingDistance
          leverage,
          new anchor.BN(0) // expiresAt
        )
        .accounts({
          market: market.publicKey,
          order: tpOrder.publicKey,
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([tpOrder, user])
        .rpc();

      console.log("Take-profit order placement transaction:", tx);

      // Verify take-profit order
      const orderAccount = await program.account.order.fetch(tpOrder.publicKey);
      expect(orderAccount.orderType).to.deep.equal(orderType);
      expect(orderAccount.stopPrice.toNumber()).to.equal(stopPrice);
    });
  });

  describe("Funding Operations", () => {
    it("Settles funding", async () => {
      const keeper = Keypair.generate();

      const tx = await program.methods
        .settleFunding()
        .accounts({
          market: market.publicKey,
          keeper: keeper.publicKey,
        })
        .signers([keeper])
        .rpc();

      console.log("Funding settlement transaction:", tx);

      // Verify funding was settled
      const marketAccount = await program.account.market.fetch(market.publicKey);
      expect(marketAccount.lastFundingTime.toNumber()).to.be.greaterThan(0);
    });
  });

  describe("Liquidation Operations", () => {
    it("Liquidates a position", async () => {
      // Create a position with low health factor
      const liquidatedPosition = Keypair.generate();
      const liquidator = Keypair.generate();
      const vault = Keypair.generate();

      // First create a position (simplified for test)
      const userCollateral = Keypair.generate();
      
      const openTx = await program.methods
        .openPosition(
          new anchor.BN(1000),
          { long: {} },
          20 // High leverage
        )
        .accounts({
          market: market.publicKey,
          position: liquidatedPosition.publicKey,
          user: user.publicKey,
          userCollateral: userCollateral.publicKey,
          systemProgram: SystemProgram.programId,
        })
        .signers([liquidatedPosition, user])
        .rpc();

      console.log("Position opened for liquidation test:", openTx);

      // Attempt liquidation
      const tx = await program.methods
        .liquidatePosition()
        .accounts({
          market: market.publicKey,
          position: liquidatedPosition.publicKey,
          liquidator: liquidator.publicKey,
          vault: vault.publicKey,
        })
        .signers([liquidator])
        .rpc();

      console.log("Liquidation transaction:", tx);

      // Verify position was liquidated
      const positionAccount = await program.account.position.fetch(liquidatedPosition.publicKey);
      expect(positionAccount.size.toNumber()).to.equal(0);
    });
  });

  describe("Error Handling", () => {
    it("Rejects invalid leverage", async () => {
      const invalidPosition = Keypair.generate();
      const userCollateral = Keypair.generate();

      try {
        await program.methods
          .openPosition(
            new anchor.BN(1000),
            { long: {} },
            101 // Invalid leverage > 100
          )
          .accounts({
            market: market.publicKey,
            position: invalidPosition.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([invalidPosition, user])
          .rpc();

        expect.fail("Should have thrown an error for invalid leverage");
      } catch (error) {
        expect(error.message).to.include("InvalidLeverage");
      }
    });

    it("Rejects invalid order price", async () => {
      const invalidOrder = Keypair.generate();
      const userCollateral = Keypair.generate();

      try {
        await program.methods
          .placeOrder(
            { market: {} }, // Market order
            { long: {} },
            new anchor.BN(1000),
            new anchor.BN(1000), // Invalid price for market order
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: invalidOrder.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([invalidOrder, user])
          .rpc();

        expect.fail("Should have thrown an error for invalid price");
      } catch (error) {
        expect(error.message).to.include("InvalidPrice");
      }
    });
  });

  describe("Performance Tests", () => {
    it("Handles multiple concurrent orders", async () => {
      const orders = [];
      const promises = [];

      // Create multiple orders concurrently
      for (let i = 0; i < 10; i++) {
        const orderKp = Keypair.generate();
        const userCollateral = Keypair.generate();
        
        const promise = program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(1000),
            new anchor.BN(50000 + i * 100),
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: orderKp.publicKey,
            user: user.publicKey,
            userCollateral: userCollateral.publicKey,
            systemProgram: SystemProgram.programId,
          })
          .signers([orderKp, user])
          .rpc();

        promises.push(promise);
        orders.push(orderKp);
      }

      const results = await Promise.all(promises);
      console.log(`Created ${results.length} orders concurrently`);

      // Verify all orders were created
      for (const orderKp of orders) {
        const orderAccount = await program.account.order.fetch(orderKp.publicKey);
        expect(orderAccount.status).to.deep.equal({ pending: {} });
      }
    });
  });
});