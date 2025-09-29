#!/usr/bin/env node

/**
 * 🔗 QuantDesk Smart Contract Simulator
 * 
 * Simulates Solana smart contract interactions for:
 * - Account creation on-chain
 * - Token transfers and deposits
 * - Order execution and matching
 * - Position management
 * - Liquidation events
 */

const EventEmitter = require('events');
const chalk = require('chalk').default;
const ora = require('ora').default;

class SmartContractSimulator extends EventEmitter {
  constructor() {
    super();
    this.accounts = new Map();
    this.positions = new Map();
    this.orders = new Map();
    this.liquidityPool = {
      SOL: 1000000,
      USDC: 50000000,
      USDT: 25000000,
      BTC: 1000,
      ETH: 10000
    };
    this.oraclePrices = {
      'BTC-PERP': 45000,
      'ETH-PERP': 3200,
      'SOL-PERP': 95
    };
    this.isRunning = false;
  }

  async start() {
    console.log(chalk.blue.bold('\n🔗 QuantDesk Smart Contract Simulator Started'));
    console.log(chalk.gray('Simulating Solana blockchain interactions...\n'));
    
    this.isRunning = true;
    
    // Start price feed simulation
    this.startPriceFeed();
    
    // Start order matching engine
    this.startOrderMatching();
    
    // Start liquidation monitoring
    this.startLiquidationMonitoring();
    
    console.log(chalk.green('✅ Smart contract simulator is running'));
    console.log(chalk.cyan('   - Price feeds active'));
    console.log(chalk.cyan('   - Order matching engine running'));
    console.log(chalk.cyan('   - Liquidation monitoring active\n'));
  }

  // Simulate creating a trading account on-chain
  async createTradingAccount(walletAddress, accountIndex) {
    const spinner = ora('Creating trading account on-chain...').start();
    
    try {
      // Simulate blockchain transaction
      await this.delay(2000);
      
      const accountId = `account_${walletAddress}_${accountIndex}`;
      const account = {
        id: accountId,
        walletAddress,
        accountIndex,
        margin: 0,
        positions: [],
        orders: [],
        createdAt: new Date(),
        onChain: true
      };
      
      this.accounts.set(accountId, account);
      
      spinner.succeed(chalk.green('✅ Trading account created on-chain!'));
      console.log(chalk.cyan(`   Account ID: ${accountId}`));
      console.log(chalk.cyan(`   Program: QuantDesk Trading Program`));
      console.log(chalk.cyan(`   Transaction: ${this.generateTxHash()}\n`));
      
      this.emit('accountCreated', account);
      return account;
      
    } catch (error) {
      spinner.fail('❌ Account creation failed');
      throw error;
    }
  }

  // Simulate token deposit to smart contract
  async depositTokens(accountId, asset, amount) {
    const spinner = ora(`Depositing ${amount} ${asset} to smart contract...`).start();
    
    try {
      const account = this.accounts.get(accountId);
      if (!account) {
        throw new Error('Account not found');
      }
      
      // Simulate SPL token transfer
      await this.delay(1500);
      
      // Update account balance
      account.margin += amount * this.getAssetPrice(asset);
      
      // Update liquidity pool
      this.liquidityPool[asset] += amount;
      
      spinner.succeed(chalk.green(`✅ Deposited ${amount} ${asset}!`));
      console.log(chalk.cyan(`   From: User Wallet`));
      console.log(chalk.cyan(`   To: QuantDesk Vault`));
      console.log(chalk.cyan(`   Transaction: ${this.generateTxHash()}`));
      console.log(chalk.cyan(`   New Margin: $${account.margin.toFixed(2)}\n`));
      
      this.emit('tokensDeposited', { accountId, asset, amount });
      
    } catch (error) {
      spinner.fail('❌ Token deposit failed');
      throw error;
    }
  }

  // Simulate order placement on-chain
  async placeOrder(accountId, orderData) {
    const spinner = ora('Placing order on-chain...').start();
    
    try {
      const account = this.accounts.get(accountId);
      if (!account) {
        throw new Error('Account not found');
      }
      
      // Simulate order validation
      await this.delay(800);
      
      const order = {
        id: `order_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        accountId,
        ...orderData,
        status: 'pending',
        createdAt: new Date(),
        onChain: true
      };
      
      this.orders.set(order.id, order);
      account.orders.push(order.id);
      
      spinner.succeed(chalk.green('✅ Order placed on-chain!'));
      console.log(chalk.cyan(`   Order ID: ${order.id}`));
      console.log(chalk.cyan(`   Market: ${order.symbol}`));
      console.log(chalk.cyan(`   Side: ${order.side.toUpperCase()}`));
      console.log(chalk.cyan(`   Size: ${order.size}`));
      console.log(chalk.cyan(`   Transaction: ${this.generateTxHash()}\n`));
      
      this.emit('orderPlaced', order);
      return order;
      
    } catch (error) {
      spinner.fail('❌ Order placement failed');
      throw error;
    }
  }

  // Simulate order execution and position creation
  async executeOrder(orderId) {
    const order = this.orders.get(orderId);
    if (!order) return;
    
    const spinner = ora(`Executing ${order.symbol} order...`).start();
    
    try {
      // Simulate order matching
      await this.delay(1200);
      
      const account = this.accounts.get(order.accountId);
      const currentPrice = this.oraclePrices[order.symbol];
      
      // Create or update position
      let position = account.positions.find(p => p.symbol === order.symbol);
      
      if (!position) {
        position = {
          id: `pos_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
          symbol: order.symbol,
          side: order.side,
          size: order.size,
          entryPrice: currentPrice,
          currentPrice,
          leverage: order.leverage,
          margin: order.margin,
          unrealizedPnL: 0,
          createdAt: new Date()
        };
        
        account.positions.push(position);
        this.positions.set(position.id, position);
      } else {
        // Update existing position
        if (position.side === order.side) {
          // Increase position
          const totalValue = (position.size * position.entryPrice) + (order.size * currentPrice);
          position.size += order.size;
          position.entryPrice = totalValue / position.size;
        } else {
          // Close/reverse position
          if (order.size >= position.size) {
            // Close position and potentially open new one
            const realizedPnL = this.calculatePnL(position, currentPrice);
            account.margin += realizedPnL;
            
            if (order.size > position.size) {
              // Open new position in opposite direction
              position.side = order.side;
              position.size = order.size - position.size;
              position.entryPrice = currentPrice;
            } else {
              // Position fully closed
              account.positions = account.positions.filter(p => p.id !== position.id);
              this.positions.delete(position.id);
            }
          } else {
            // Partial close
            position.size -= order.size;
          }
        }
      }
      
      // Update order status
      order.status = 'filled';
      order.filledPrice = currentPrice;
      order.filledAt = new Date();
      
      // Update account margin
      account.margin -= order.margin;
      
      spinner.succeed(chalk.green('✅ Order executed!'));
      console.log(chalk.cyan(`   Filled Price: $${currentPrice}`));
      console.log(chalk.cyan(`   Position Size: ${position.size}`));
      console.log(chalk.cyan(`   Transaction: ${this.generateTxHash()}\n`));
      
      this.emit('orderExecuted', { order, position });
      
    } catch (error) {
      spinner.fail('❌ Order execution failed');
      order.status = 'failed';
      throw error;
    }
  }

  // Simulate liquidation event
  async liquidatePosition(positionId) {
    const position = this.positions.get(positionId);
    if (!position) return;
    
    const spinner = ora(`Liquidating ${position.symbol} position...`).start();
    
    try {
      await this.delay(1000);
      
      const account = this.accounts.get(position.accountId);
      const currentPrice = this.oraclePrices[position.symbol];
      
      // Calculate liquidation
      const liquidationValue = position.size * currentPrice;
      const liquidationFee = liquidationValue * 0.05; // 5% fee
      const remainingValue = liquidationValue - liquidationFee;
      
      // Update account
      account.margin += remainingValue;
      account.positions = account.positions.filter(p => p.id !== positionId);
      
      // Remove position
      this.positions.delete(positionId);
      
      spinner.succeed(chalk.red('⚡ Position liquidated!'));
      console.log(chalk.red(`   Liquidation Price: $${currentPrice}`));
      console.log(chalk.red(`   Liquidation Fee: $${liquidationFee.toFixed(2)}`));
      console.log(chalk.red(`   Remaining Value: $${remainingValue.toFixed(2)}`));
      console.log(chalk.red(`   Transaction: ${this.generateTxHash()}\n`));
      
      this.emit('positionLiquidated', { position, liquidationFee });
      
    } catch (error) {
      spinner.fail('❌ Liquidation failed');
      throw error;
    }
  }

  // Start price feed simulation
  startPriceFeed() {
    setInterval(() => {
      Object.keys(this.oraclePrices).forEach(symbol => {
        // Simulate price volatility
        const change = (Math.random() - 0.5) * 0.02; // ±1% change
        this.oraclePrices[symbol] *= (1 + change);
        
        // Update position PnL
        this.positions.forEach(position => {
          if (position.symbol === symbol) {
            position.currentPrice = this.oraclePrices[symbol];
            position.unrealizedPnL = this.calculatePnL(position, this.oraclePrices[symbol]);
          }
        });
      });
      
      this.emit('pricesUpdated', this.oraclePrices);
    }, 5000); // Update every 5 seconds
  }

  // Start order matching engine
  startOrderMatching() {
    setInterval(() => {
      this.orders.forEach((order, orderId) => {
        if (order.status === 'pending') {
          const currentPrice = this.oraclePrices[order.symbol];
          
          // Simple matching logic
          if (order.type === 'market' || 
              (order.type === 'limit' && 
               ((order.side === 'long' && currentPrice <= order.price) ||
                (order.side === 'short' && currentPrice >= order.price)))) {
            this.executeOrder(orderId);
          }
        }
      });
    }, 2000); // Check every 2 seconds
  }

  // Start liquidation monitoring
  startLiquidationMonitoring() {
    setInterval(() => {
      this.positions.forEach((position, positionId) => {
        const account = this.accounts.get(position.accountId);
        if (!account) return;
        
        // Calculate health factor
        const currentPrice = this.oraclePrices[position.symbol];
        const unrealizedPnL = this.calculatePnL(position, currentPrice);
        const totalValue = account.margin + unrealizedPnL;
        const positionValue = position.size * currentPrice;
        const healthFactor = totalValue / (positionValue / position.leverage);
        
        // Liquidate if health factor < 1.1
        if (healthFactor < 1.1) {
          this.liquidatePosition(positionId);
        }
      });
    }, 3000); // Check every 3 seconds
  }

  // Helper methods
  calculatePnL(position, currentPrice) {
    const priceChange = currentPrice - position.entryPrice;
    return position.side === 'long' 
      ? priceChange * position.size 
      : -priceChange * position.size;
  }

  getAssetPrice(asset) {
    switch (asset) {
      case 'SOL': return this.oraclePrices['SOL-PERP'];
      case 'BTC': return this.oraclePrices['BTC-PERP'];
      case 'ETH': return this.oraclePrices['ETH-PERP'];
      case 'USDC':
      case 'USDT': return 1;
      default: return 1;
    }
  }

  generateTxHash() {
    return Array.from({length: 64}, () => 
      Math.floor(Math.random() * 16).toString(16)
    ).join('');
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  // Get current state
  getState() {
    return {
      accounts: Array.from(this.accounts.values()),
      positions: Array.from(this.positions.values()),
      orders: Array.from(this.orders.values()),
      liquidityPool: this.liquidityPool,
      oraclePrices: this.oraclePrices
    };
  }

  stop() {
    this.isRunning = false;
    console.log(chalk.yellow('\n🛑 Smart contract simulator stopped'));
  }
}

module.exports = SmartContractSimulator;
