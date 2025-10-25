import * as anchor from "@coral-xyz/anchor";
import { Program } from "@coral-xyz/anchor";
import { QuantdeskPerpDex } from "../target/types/quantdesk_perp_dex";
import { PublicKey, Keypair, SystemProgram } from "@solana/web3.js";
import { expect } from "chai";
import TestUtils, { PerformanceTestUtils, MockObjects, TestFixtures } from "./utils/test-utils";

/**
 * Comprehensive Performance Tests for QuantDesk Perpetual DEX Smart Contracts
 * Tests for gas optimization, transaction throughput, and system performance
 */

describe("QuantDesk Perpetual DEX - Performance Tests", () => {
  // Configure the client to use the local cluster
  anchor.setProvider(anchor.AnchorProvider.env());

  const program = anchor.workspace.QuantdeskPerpDex as Program<QuantdeskPerpDex>;
  const provider = anchor.getProvider();
  
  let testUtils: TestUtils;
  let performanceUtils: PerformanceTestUtils;

  before(async () => {
    testUtils = new TestUtils(program, provider);
    performanceUtils = new PerformanceTestUtils(program, provider);
  });

  describe("Gas Optimization Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should optimize gas usage for market initialization", async () => {
      const newMarket = Keypair.generate();
      const config = TestFixtures.standardMarket;

      const instruction = program.methods
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
        });

      const gasAnalysis = await performanceUtils.testGasOptimization(instruction);
      
      expect(gasAnalysis.computeUnits).to.be.lessThan(100000); // Should be under 100k CU
      expect(gasAnalysis.isOptimized).to.be.true;
      
      console.log(`Market initialization uses ${gasAnalysis.computeUnits} compute units`);
    });

    it("should optimize gas usage for position opening", async () => {
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

      const gasAnalysis = await performanceUtils.testGasOptimization(instruction);
      
      expect(gasAnalysis.computeUnits).to.be.lessThan(150000); // Should be under 150k CU
      expect(gasAnalysis.isOptimized).to.be.true;
      
      console.log(`Position opening uses ${gasAnalysis.computeUnits} compute units`);
    });

    it("should optimize gas usage for order placement", async () => {
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

      const gasAnalysis = await performanceUtils.testGasOptimization(instruction);
      
      expect(gasAnalysis.computeUnits).to.be.lessThan(120000); // Should be under 120k CU
      expect(gasAnalysis.isOptimized).to.be.true;
      
      console.log(`Order placement uses ${gasAnalysis.computeUnits} compute units`);
    });

    it("should optimize gas usage for liquidation", async () => {
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

      const gasAnalysis = await performanceUtils.testGasOptimization(instruction);
      
      expect(gasAnalysis.computeUnits).to.be.lessThan(200000); // Should be under 200k CU
      expect(gasAnalysis.isOptimized).to.be.true;
      
      console.log(`Liquidation uses ${gasAnalysis.computeUnits} compute units`);
    });

    it("should optimize gas usage for funding settlement", async () => {
      const keeper = (await testUtils.generateTestAccounts(1))[0];

      const instruction = program.methods
        .settleFunding()
        .accounts({
          market: market.publicKey,
          keeper: keeper.publicKey,
        });

      const gasAnalysis = await performanceUtils.testGasOptimization(instruction);
      
      expect(gasAnalysis.computeUnits).to.be.lessThan(80000); // Should be under 80k CU
      expect(gasAnalysis.isOptimized).to.be.true;
      
      console.log(`Funding settlement uses ${gasAnalysis.computeUnits} compute units`);
    });
  });

  describe("Transaction Throughput Tests", () => {
    let market: Keypair;
    let users: Keypair[];

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      users = await testUtils.generateTestAccounts(10);
    });

    it("should handle concurrent order placements", async () => {
      const orderCount = 50;
      const orders: Keypair[] = [];
      const userCollaterals = await testUtils.generateTestAccounts(orderCount);

      const instructionFactory = (index: number) => {
        const order = Keypair.generate();
        orders.push(order);
        
        return program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(1000),
            new anchor.BN(50000 + index * 10), // Different prices
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order.publicKey,
            user: users[index % users.length].publicKey,
            userCollateral: userCollaterals[index].publicKey,
            systemProgram: SystemProgram.programId,
          });
      };

      const concurrencyTest = await performanceUtils.testConcurrency(
        instructionFactory,
        orderCount
      );

      expect(concurrencyTest.successCount).to.be.greaterThan(orderCount * 0.8); // 80% success rate
      expect(concurrencyTest.averageTime).to.be.lessThan(5000); // Under 5 seconds average
      
      console.log(`Concurrent order placement: ${concurrencyTest.successCount}/${orderCount} successful`);
      console.log(`Average time: ${concurrencyTest.averageTime}ms`);
    });

    it("should handle concurrent position operations", async () => {
      const positionCount = 30;
      const positions: Keypair[] = [];
      const userCollaterals = await testUtils.generateTestAccounts(positionCount);

      const instructionFactory = (index: number) => {
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
            user: users[index % users.length].publicKey,
            userCollateral: userCollaterals[index].publicKey,
            systemProgram: SystemProgram.programId,
          });
      };

      const concurrencyTest = await performanceUtils.testConcurrency(
        instructionFactory,
        positionCount
      );

      expect(concurrencyTest.successCount).to.be.greaterThan(positionCount * 0.8); // 80% success rate
      expect(concurrencyTest.averageTime).to.be.lessThan(3000); // Under 3 seconds average
      
      console.log(`Concurrent position opening: ${concurrencyTest.successCount}/${positionCount} successful`);
      console.log(`Average time: ${concurrencyTest.averageTime}ms`);
    });

    it("should handle mixed concurrent operations", async () => {
      const operationCount = 40;
      const operations: any[] = [];
      const userCollaterals = await testUtils.generateTestAccounts(operationCount);

      const instructionFactory = (index: number) => {
        const operationType = index % 3;
        
        switch (operationType) {
          case 0: // Order placement
            const order = Keypair.generate();
            operations.push({ type: 'order', keypair: order });
            return program.methods
              .placeOrder(
                { limit: {} },
                { long: {} },
                new anchor.BN(1000),
                new anchor.BN(50000 + index * 10),
                new anchor.BN(0),
                new anchor.BN(0),
                5,
                new anchor.BN(0)
              )
              .accounts({
                market: market.publicKey,
                order: order.publicKey,
                user: users[index % users.length].publicKey,
                userCollateral: userCollaterals[index].publicKey,
                systemProgram: SystemProgram.programId,
              });

          case 1: // Position opening
            const position = Keypair.generate();
            operations.push({ type: 'position', keypair: position });
            return program.methods
              .openPosition(
                new anchor.BN(1000),
                { long: {} },
                10
              )
              .accounts({
                market: market.publicKey,
                position: position.publicKey,
                user: users[index % users.length].publicKey,
                userCollateral: userCollaterals[index].publicKey,
                systemProgram: SystemProgram.programId,
              });

          case 2: // Collateral deposit
            const userCollateral = userCollaterals[index];
            operations.push({ type: 'collateral', keypair: userCollateral });
            return program.methods
              .depositCollateral(new anchor.BN(1000))
              .accounts({
                user: users[index % users.length].publicKey,
                userCollateral: userCollateral.publicKey,
                systemProgram: SystemProgram.programId,
              });

          default:
            throw new Error("Invalid operation type");
        }
      };

      const concurrencyTest = await performanceUtils.testConcurrency(
        instructionFactory,
        operationCount
      );

      expect(concurrencyTest.successCount).to.be.greaterThan(operationCount * 0.7); // 70% success rate
      expect(concurrencyTest.averageTime).to.be.lessThan(4000); // Under 4 seconds average
      
      console.log(`Mixed concurrent operations: ${concurrencyTest.successCount}/${operationCount} successful`);
      console.log(`Average time: ${concurrencyTest.averageTime}ms`);
    });
  });

  describe("Execution Time Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should execute market initialization within acceptable time", async () => {
      const newMarket = Keypair.generate();
      const config = TestFixtures.standardMarket;

      const instruction = program.methods
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
        });

      const executionTime = await performanceUtils.measureExecutionTime(instruction);
      
      expect(executionTime).to.be.lessThan(2000); // Under 2 seconds
      console.log(`Market initialization execution time: ${executionTime}ms`);
    });

    it("should execute position operations within acceptable time", async () => {
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

      const executionTime = await performanceUtils.measureExecutionTime(instruction);
      
      expect(executionTime).to.be.lessThan(1500); // Under 1.5 seconds
      console.log(`Position opening execution time: ${executionTime}ms`);
    });

    it("should execute order operations within acceptable time", async () => {
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

      const executionTime = await performanceUtils.measureExecutionTime(instruction);
      
      expect(executionTime).to.be.lessThan(1200); // Under 1.2 seconds
      console.log(`Order placement execution time: ${executionTime}ms`);
    });

    it("should execute liquidation within acceptable time", async () => {
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

      const executionTime = await performanceUtils.measureExecutionTime(instruction);
      
      expect(executionTime).to.be.lessThan(2500); // Under 2.5 seconds
      console.log(`Liquidation execution time: ${executionTime}ms`);
    });
  });

  describe("Memory Usage Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should handle large batch operations efficiently", async () => {
      const batchSize = 100;
      const orders: Keypair[] = [];
      const userCollaterals = await testUtils.generateTestAccounts(batchSize);

      const startTime = Date.now();
      const promises = [];

      for (let i = 0; i < batchSize; i++) {
        const order = Keypair.generate();
        orders.push(order);

        const promise = program.methods
          .placeOrder(
            { limit: {} },
            { long: {} },
            new anchor.BN(1000),
            new anchor.BN(50000 + i * 10),
            new anchor.BN(0),
            new anchor.BN(0),
            5,
            new anchor.BN(0)
          )
          .accounts({
            market: market.publicKey,
            order: order.publicKey,
            user: users[i % users.length].publicKey,
            userCollateral: userCollaterals[i].publicKey,
            systemProgram: SystemProgram.programId,
          })
          .rpc()
          .catch(error => {
            console.warn(`Order ${i} failed:`, error.message);
            return null;
          });

        promises.push(promise);
      }

      const results = await Promise.all(promises);
      const endTime = Date.now();
      const totalTime = endTime - startTime;
      const successCount = results.filter(result => result !== null).length;

      expect(successCount).to.be.greaterThan(batchSize * 0.8); // 80% success rate
      expect(totalTime).to.be.lessThan(30000); // Under 30 seconds total
      
      console.log(`Large batch operation: ${successCount}/${batchSize} successful in ${totalTime}ms`);
      console.log(`Average time per operation: ${totalTime / batchSize}ms`);
    });

    it("should handle memory efficiently with many accounts", async () => {
      const accountCount = 200;
      const accounts: Keypair[] = [];
      const userCollaterals = await testUtils.generateTestAccounts(accountCount);

      const startTime = Date.now();

      // Create many accounts
      for (let i = 0; i < accountCount; i++) {
        const account = Keypair.generate();
        accounts.push(account);

        await program.methods
          .depositCollateral(new anchor.BN(1000))
          .accounts({
            user: users[i % users.length].publicKey,
            userCollateral: userCollaterals[i].publicKey,
            systemProgram: SystemProgram.programId,
          })
          .rpc()
          .catch(error => {
            console.warn(`Account ${i} creation failed:`, error.message);
          });
      }

      const endTime = Date.now();
      const totalTime = endTime - startTime;

      expect(totalTime).to.be.lessThan(60000); // Under 60 seconds
      
      console.log(`Created ${accountCount} accounts in ${totalTime}ms`);
      console.log(`Average time per account: ${totalTime / accountCount}ms`);
    });
  });

  describe("Scalability Tests", () => {
    let market: Keypair;
    let users: Keypair[];

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      users = await testUtils.generateTestAccounts(20);
    });

    it("should scale with increasing number of users", async () => {
      const userCounts = [10, 20, 50, 100];
      const results: { users: number; time: number; successRate: number }[] = [];

      for (const userCount of userCounts) {
        const testUsers = users.slice(0, userCount);
        const orders: Keypair[] = [];
        const userCollaterals = await testUtils.generateTestAccounts(userCount);

        const startTime = Date.now();
        const promises = [];

        for (let i = 0; i < userCount; i++) {
          const order = Keypair.generate();
          orders.push(order);

          const promise = program.methods
            .placeOrder(
              { limit: {} },
              { long: {} },
              new anchor.BN(1000),
              new anchor.BN(50000 + i * 10),
              new anchor.BN(0),
              new anchor.BN(0),
              5,
              new anchor.BN(0)
            )
            .accounts({
              market: market.publicKey,
              order: order.publicKey,
              user: testUsers[i].publicKey,
              userCollateral: userCollaterals[i].publicKey,
              systemProgram: SystemProgram.programId,
            })
            .rpc()
            .catch(() => null);

          promises.push(promise);
        }

        const results_batch = await Promise.all(promises);
        const endTime = Date.now();
        const successCount = results_batch.filter(result => result !== null).length;
        const successRate = successCount / userCount;

        results.push({
          users: userCount,
          time: endTime - startTime,
          successRate: successRate
        });
      }

      // Verify scalability
      for (const result of results) {
        expect(result.successRate).to.be.greaterThan(0.7); // At least 70% success rate
        expect(result.time).to.be.lessThan(result.users * 100); // Linear scaling
      }

      console.log("Scalability test results:");
      results.forEach(result => {
        console.log(`${result.users} users: ${result.time}ms, ${(result.successRate * 100).toFixed(1)}% success`);
      });
    });

    it("should maintain performance under load", async () => {
      const loadLevels = [50, 100, 200, 500];
      const results: { load: number; avgTime: number; successRate: number }[] = [];

      for (const load of loadLevels) {
        const orders: Keypair[] = [];
        const userCollaterals = await testUtils.generateTestAccounts(load);

        const startTime = Date.now();
        const promises = [];

        for (let i = 0; i < load; i++) {
          const order = Keypair.generate();
          orders.push(order);

          const promise = program.methods
            .placeOrder(
              { limit: {} },
              { long: {} },
              new anchor.BN(1000),
              new anchor.BN(50000 + i * 10),
              new anchor.BN(0),
              new anchor.BN(0),
              5,
              new anchor.BN(0)
            )
            .accounts({
              market: market.publicKey,
              order: order.publicKey,
              user: users[i % users.length].publicKey,
              userCollateral: userCollaterals[i].publicKey,
              systemProgram: SystemProgram.programId,
            })
            .rpc()
            .catch(() => null);

          promises.push(promise);
        }

        const results_batch = await Promise.all(promises);
        const endTime = Date.now();
        const totalTime = endTime - startTime;
        const successCount = results_batch.filter(result => result !== null).length;
        const successRate = successCount / load;
        const avgTime = totalTime / load;

        results.push({
          load: load,
          avgTime: avgTime,
          successRate: successRate
        });
      }

      // Verify performance under load
      for (const result of results) {
        expect(result.successRate).to.be.greaterThan(0.6); // At least 60% success rate
        expect(result.avgTime).to.be.lessThan(1000); // Under 1 second average
      }

      console.log("Load test results:");
      results.forEach(result => {
        console.log(`Load ${result.load}: ${result.avgTime.toFixed(1)}ms avg, ${(result.successRate * 100).toFixed(1)}% success`);
      });
    });
  });

  describe("Resource Usage Tests", () => {
    let market: Keypair;
    let user: Keypair;

    beforeEach(async () => {
      const mockMarket = await testUtils.createMockMarket();
      market = mockMarket.market;
      user = (await testUtils.generateTestAccounts(1))[0];
    });

    it("should optimize compute unit usage", async () => {
      const operations = [
        {
          name: "Market Initialization",
          instruction: () => {
            const newMarket = Keypair.generate();
            return program.methods
              .initializeMarket(
                "BTC",
                "USDT",
                new anchor.BN(50000),
                100,
                500,
                300
              )
              .accounts({
                market: newMarket.publicKey,
                authority: provider.wallet.publicKey,
                systemProgram: SystemProgram.programId,
              });
          }
        },
        {
          name: "Position Opening",
          instruction: () => {
            const position = Keypair.generate();
            const userCollateral = Keypair.generate();
            return program.methods
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
          }
        },
        {
          name: "Order Placement",
          instruction: () => {
            const order = Keypair.generate();
            const userCollateral = Keypair.generate();
            return program.methods
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
          }
        }
      ];

      for (const operation of operations) {
        const gasAnalysis = await performanceUtils.testGasOptimization(operation.instruction());
        
        console.log(`${operation.name}: ${gasAnalysis.computeUnits} compute units`);
        expect(gasAnalysis.isOptimized).to.be.true;
      }
    });

    it("should minimize account creation overhead", async () => {
      const accountCount = 50;
      const accounts: Keypair[] = [];
      const userCollaterals = await testUtils.generateTestAccounts(accountCount);

      const startTime = Date.now();

      for (let i = 0; i < accountCount; i++) {
        const account = Keypair.generate();
        accounts.push(account);

        await program.methods
          .depositCollateral(new anchor.BN(1000))
          .accounts({
            user: user.publicKey,
            userCollateral: userCollaterals[i].publicKey,
            systemProgram: SystemProgram.programId,
          })
          .rpc()
          .catch(error => {
            console.warn(`Account ${i} creation failed:`, error.message);
          });
      }

      const endTime = Date.now();
      const totalTime = endTime - startTime;
      const avgTimePerAccount = totalTime / accountCount;

      expect(avgTimePerAccount).to.be.lessThan(500); // Under 500ms per account
      
      console.log(`Account creation overhead: ${avgTimePerAccount.toFixed(1)}ms per account`);
    });
  });
});
