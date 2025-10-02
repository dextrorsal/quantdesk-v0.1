#!/usr/bin/env node

/**
 * 🚀 QuantDesk Interactive Trading Demo
 * 
 * This script simulates a complete trading session with:
 * - Wallet connection simulation
 * - Account creation
 * - Token deposits
 * - Real-time trading execution
 * - Position management
 * - Smart contract interaction simulation
 */

const readline = require('readline');
const axios = require('axios');
const chalk = require('chalk').default;
const ora = require('ora').default;

// Configuration
const API_BASE = 'http://localhost:3002/api';
const DEMO_WALLET = 'DemoWallet123456789012345678901234567890';
const DEMO_USER = {
  walletAddress: DEMO_WALLET,
  username: 'DemoTrader',
  email: 'demo@quantdesk.app'
};

// Demo trading data
const MARKETS = {
  'BTC-PERP': { price: 45000, change: 2.5 },
  'ETH-PERP': { price: 3200, change: -1.2 },
  'SOL-PERP': { price: 95, change: 5.8 }
};

const TOKENS = ['SOL', 'USDC', 'USDT', 'BTC', 'ETH'];

class TradingDemo {
  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    this.authToken = null;
    this.user = null;
    this.tradingAccount = null;
    this.positions = [];
    this.balances = {};
    this.orders = [];
  }

  async start() {
    console.log(chalk.blue.bold('\n🚀 Welcome to QuantDesk Trading Demo!'));
    console.log(chalk.gray('Simulating a complete trading session...\n'));

    try {
      await this.connectWallet();
      await this.createAccount();
      await this.depositTokens();
      await this.createTradingAccount();
      await this.showMainMenu();
    } catch (error) {
      console.error(chalk.red('Demo failed:'), error.message);
    }
  }

  async connectWallet() {
    const spinner = ora('Connecting wallet...').start();
    
    try {
      // Simulate wallet connection
      await this.delay(1000);
      
      // Authenticate with backend
      const response = await axios.post(`${API_BASE}/auth/authenticate`, {
        walletAddress: DEMO_WALLET,
        signature: [1, 2, 3, 4, 5], // Mock signature
        message: 'Demo authentication message'
      });

      this.authToken = response.data.token;
      this.user = response.data.user;
      
      spinner.succeed(chalk.green('✅ Wallet connected successfully!'));
      console.log(chalk.cyan(`   User ID: ${this.user.id}`));
      console.log(chalk.cyan(`   Wallet: ${this.user.walletAddress}\n`));
      
    } catch (error) {
      spinner.fail('❌ Wallet connection failed');
      throw error;
    }
  }

  async createAccount() {
    const spinner = ora('Creating user account...').start();
    
    try {
      await this.delay(800);
      spinner.succeed(chalk.green('✅ Account created successfully!'));
      console.log(chalk.cyan(`   Account Type: ${this.user.accountType || 'Standard'}`));
      console.log(chalk.cyan(`   KYC Status: ${this.user.kycStatus || 'Pending'}\n`));
      
    } catch (error) {
      spinner.fail('❌ Account creation failed');
      throw error;
    }
  }

  async depositTokens() {
    const spinner = ora('Depositing demo tokens...').start();
    
    try {
      // Simulate token deposits
      const depositAmounts = {
        'SOL': 100,
        'USDC': 10000,
        'USDT': 5000,
        'BTC': 0.5,
        'ETH': 5
      };

      for (const [token, amount] of Object.entries(depositAmounts)) {
        await this.delay(300);
        
        const response = await axios.post(`${API_BASE}/deposits/deposit`, {
          asset: token,
          amount: amount
        }, {
          headers: { Authorization: `Bearer ${this.authToken}` }
        });

        // Confirm deposit
        await axios.post(`${API_BASE}/deposits/confirm`, {
          depositId: response.data.deposit.id,
          transactionSignature: `demo_tx_${token}_${Date.now()}`
        }, {
          headers: { Authorization: `Bearer ${this.authToken}` }
        });

        this.balances[token] = amount;
      }

      spinner.succeed(chalk.green('✅ Tokens deposited successfully!'));
      this.displayBalances();
      
    } catch (error) {
      spinner.fail('❌ Token deposit failed');
      throw error;
    }
  }

  async createTradingAccount() {
    const spinner = ora('Creating trading account...').start();
    
    try {
      const response = await axios.post(`${API_BASE}/accounts/trading-accounts`, {
        name: 'Main Trading Account'
      }, {
        headers: { Authorization: `Bearer ${this.authToken}` }
      });

      this.tradingAccount = response.data.tradingAccount;
      
      spinner.succeed(chalk.green('✅ Trading account created!'));
      console.log(chalk.cyan(`   Account: ${this.tradingAccount.name}`));
      console.log(chalk.cyan(`   Index: ${this.tradingAccount.accountIndex}\n`));
      
    } catch (error) {
      spinner.fail('❌ Trading account creation failed');
      throw error;
    }
  }

  displayBalances() {
    console.log(chalk.yellow.bold('\n💰 Current Balances:'));
    Object.entries(this.balances).forEach(([token, amount]) => {
      console.log(chalk.white(`   ${token}: ${amount}`));
    });
    console.log();
  }

  displayMarkets() {
    console.log(chalk.yellow.bold('\n📈 Available Markets:'));
    Object.entries(MARKETS).forEach(([symbol, data]) => {
      const changeColor = data.change >= 0 ? chalk.green : chalk.red;
      console.log(chalk.white(`   ${symbol}: $${data.price} ${changeColor(`(${data.change > 0 ? '+' : ''}${data.change}%)`)}`));
    });
    console.log();
  }

  displayPositions() {
    if (this.positions.length === 0) {
      console.log(chalk.yellow('📊 No open positions\n'));
      return;
    }

    console.log(chalk.yellow.bold('\n📊 Open Positions:'));
    this.positions.forEach((pos, index) => {
      const pnlColor = pos.unrealizedPnL >= 0 ? chalk.green : chalk.red;
      console.log(chalk.white(`   ${index + 1}. ${pos.symbol} ${pos.side.toUpperCase()}`));
      console.log(chalk.gray(`      Size: ${pos.size} | Entry: $${pos.entryPrice} | Current: $${pos.currentPrice}`));
      console.log(chalk.gray(`      Leverage: ${pos.leverage}x | PnL: ${pnlColor(`$${pos.unrealizedPnL}`)}`));
    });
    console.log();
  }

  async showMainMenu() {
    while (true) {
      console.clear();
      console.log(chalk.blue.bold('🚀 QuantDesk Trading Demo'));
      console.log(chalk.gray('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━'));
      
      this.displayBalances();
      this.displayMarkets();
      this.displayPositions();

      console.log(chalk.cyan.bold('📋 Trading Options:'));
      console.log(chalk.white('   1. 📈 Place Market Order'));
      console.log(chalk.white('   2. 📊 Place Limit Order'));
      console.log(chalk.white('   3. 📋 View Orders'));
      console.log(chalk.white('   4. 💰 Deposit More Tokens'));
      console.log(chalk.white('   5. 🏦 Create Another Trading Account'));
      console.log(chalk.white('   6. 📊 View Portfolio Analytics'));
      console.log(chalk.white('   7. 🔄 Simulate Price Updates'));
      console.log(chalk.white('   8. ❌ Close Position'));
      console.log(chalk.white('   9. 🚪 Exit Demo'));

      const choice = await this.askQuestion('\nSelect option (1-9): ');

      switch (choice) {
        case '1':
          await this.placeMarketOrder();
          break;
        case '2':
          await this.placeLimitOrder();
          break;
        case '3':
          await this.viewOrders();
          break;
        case '4':
          await this.depositMoreTokens();
          break;
        case '5':
          await this.createAnotherAccount();
          break;
        case '6':
          await this.viewPortfolioAnalytics();
          break;
        case '7':
          await this.simulatePriceUpdates();
          break;
        case '8':
          await this.closePosition();
          break;
        case '9':
          console.log(chalk.green('\n👋 Thanks for using QuantDesk Demo!'));
          process.exit(0);
        default:
          console.log(chalk.red('❌ Invalid option. Please try again.'));
          await this.delay(1000);
      }
    }
  }

  async placeMarketOrder() {
    console.log(chalk.yellow.bold('\n📈 Place Market Order'));
    
    const symbol = await this.askQuestion('Market (BTC-PERP, ETH-PERP, SOL-PERP): ');
    if (!MARKETS[symbol]) {
      console.log(chalk.red('❌ Invalid market'));
      await this.delay(1000);
      return;
    }

    const side = await this.askQuestion('Side (long/short): ');
    if (!['long', 'short'].includes(side)) {
      console.log(chalk.red('❌ Invalid side'));
      await this.delay(1000);
      return;
    }

    const size = parseFloat(await this.askQuestion('Size: '));
    const leverage = parseInt(await this.askQuestion('Leverage (1-100x): '));

    const spinner = ora('Executing market order...').start();

    try {
      // Calculate margin required
      const marginRequired = (size * MARKETS[symbol].price) / leverage;
      
      if (marginRequired > this.balances.USDC) {
        spinner.fail('❌ Insufficient margin');
        console.log(chalk.red(`Required: $${marginRequired.toFixed(2)} USDC`));
        await this.delay(2000);
        return;
      }

      // Simulate order execution
      await this.delay(1500);

      const order = {
        id: `order_${Date.now()}`,
        symbol,
        side,
        size,
        leverage,
        entryPrice: MARKETS[symbol].price,
        margin: marginRequired,
        status: 'filled',
        timestamp: new Date()
      };

      // Update position
      const existingPosition = this.positions.find(p => p.symbol === symbol);
      if (existingPosition) {
        if (existingPosition.side === side) {
          // Increase position
          existingPosition.size += size;
          existingPosition.entryPrice = (existingPosition.entryPrice * (existingPosition.size - size) + 
                                       order.entryPrice * size) / existingPosition.size;
        } else {
          // Close/reverse position
          if (size >= existingPosition.size) {
            this.positions = this.positions.filter(p => p.symbol !== symbol);
            if (size > existingPosition.size) {
              // Open new position in opposite direction
              this.positions.push({
                symbol,
                side,
                size: size - existingPosition.size,
                entryPrice: order.entryPrice,
                currentPrice: order.entryPrice,
                leverage,
                margin: marginRequired,
                unrealizedPnL: 0
              });
            }
          } else {
            existingPosition.size -= size;
          }
        }
      } else {
        // New position
        this.positions.push({
          symbol,
          side,
          size,
          entryPrice: order.entryPrice,
          currentPrice: order.entryPrice,
          leverage,
          margin: marginRequired,
          unrealizedPnL: 0
        });
      }

      // Update balances
      this.balances.USDC -= marginRequired;

      this.orders.push(order);

      spinner.succeed(chalk.green('✅ Market order executed!'));
      console.log(chalk.cyan(`   Order ID: ${order.id}`));
      console.log(chalk.cyan(`   ${side.toUpperCase()} ${size} ${symbol} @ $${order.entryPrice}`));
      console.log(chalk.cyan(`   Leverage: ${leverage}x | Margin: $${marginRequired.toFixed(2)}`));

      await this.delay(2000);

    } catch (error) {
      spinner.fail('❌ Order execution failed');
      console.log(chalk.red(error.message));
      await this.delay(2000);
    }
  }

  async placeLimitOrder() {
    console.log(chalk.yellow.bold('\n📊 Place Limit Order'));
    
    const symbol = await this.askQuestion('Market (BTC-PERP, ETH-PERP, SOL-PERP): ');
    if (!MARKETS[symbol]) {
      console.log(chalk.red('❌ Invalid market'));
      await this.delay(1000);
      return;
    }

    const side = await this.askQuestion('Side (long/short): ');
    const size = parseFloat(await this.askQuestion('Size: '));
    const price = parseFloat(await this.askQuestion('Limit Price: '));
    const leverage = parseInt(await this.askQuestion('Leverage (1-100x): '));

    const spinner = ora('Placing limit order...').start();

    try {
      const marginRequired = (size * price) / leverage;
      
      if (marginRequired > this.balances.USDC) {
        spinner.fail('❌ Insufficient margin');
        await this.delay(2000);
        return;
      }

      await this.delay(1000);

      const order = {
        id: `limit_${Date.now()}`,
        symbol,
        side,
        size,
        price,
        leverage,
        margin: marginRequired,
        status: 'pending',
        timestamp: new Date()
      };

      this.orders.push(order);
      this.balances.USDC -= marginRequired;

      spinner.succeed(chalk.green('✅ Limit order placed!'));
      console.log(chalk.cyan(`   Order ID: ${order.id}`));
      console.log(chalk.cyan(`   ${side.toUpperCase()} ${size} ${symbol} @ $${price} (limit)`));

      await this.delay(2000);

    } catch (error) {
      spinner.fail('❌ Limit order failed');
      await this.delay(2000);
    }
  }

  async viewOrders() {
    console.log(chalk.yellow.bold('\n📋 Order History'));
    
    if (this.orders.length === 0) {
      console.log(chalk.gray('No orders found\n'));
      await this.delay(2000);
      return;
    }

    this.orders.forEach((order, index) => {
      const statusColor = order.status === 'filled' ? chalk.green : chalk.yellow;
      console.log(chalk.white(`${index + 1}. ${order.symbol} ${order.side.toUpperCase()}`));
      console.log(chalk.gray(`   Size: ${order.size} | Price: $${order.price || order.entryPrice}`));
      console.log(chalk.gray(`   Status: ${statusColor(order.status)} | Time: ${order.timestamp.toLocaleTimeString()}`));
    });

    await this.askQuestion('\nPress Enter to continue...');
  }

  async depositMoreTokens() {
    const token = await this.askQuestion(`Token to deposit (${TOKENS.join(', ')}): `);
    const amount = parseFloat(await this.askQuestion('Amount: '));

    if (!TOKENS.includes(token)) {
      console.log(chalk.red('❌ Invalid token'));
      await this.delay(1000);
      return;
    }

    const spinner = ora(`Depositing ${amount} ${token}...`).start();
    await this.delay(1000);

    this.balances[token] = (this.balances[token] || 0) + amount;
    spinner.succeed(chalk.green(`✅ Deposited ${amount} ${token}!`));

    await this.delay(1000);
  }

  async createAnotherAccount() {
    const name = await this.askQuestion('Trading account name: ');

    const spinner = ora('Creating trading account...').start();
    await this.delay(1000);

    const newAccount = {
      id: `account_${Date.now()}`,
      name,
      accountIndex: this.tradingAccount.accountIndex + 1,
      isActive: true
    };

    spinner.succeed(chalk.green(`✅ Created "${name}"!`));
    console.log(chalk.cyan(`   Account Index: ${newAccount.accountIndex}`));

    await this.delay(1500);
  }

  async viewPortfolioAnalytics() {
    console.log(chalk.yellow.bold('\n📊 Portfolio Analytics'));
    
    const totalValue = Object.entries(this.balances).reduce((sum, [token, amount]) => {
      if (token === 'USDC' || token === 'USDT') return sum + amount;
      if (token === 'BTC') return sum + (amount * MARKETS['BTC-PERP'].price);
      if (token === 'ETH') return sum + (amount * MARKETS['ETH-PERP'].price);
      if (token === 'SOL') return sum + (amount * MARKETS['SOL-PERP'].price);
      return sum;
    }, 0);

    const totalPnL = this.positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);

    console.log(chalk.white(`Total Portfolio Value: $${totalValue.toFixed(2)}`));
    console.log(chalk.white(`Total Unrealized PnL: $${totalPnL.toFixed(2)}`));
    console.log(chalk.white(`Open Positions: ${this.positions.length}`));
    console.log(chalk.white(`Total Orders: ${this.orders.length}`));

    await this.askQuestion('\nPress Enter to continue...');
  }

  async simulatePriceUpdates() {
    console.log(chalk.yellow.bold('\n🔄 Simulating Price Updates...'));
    
    const spinner = ora('Updating market prices...').start();
    await this.delay(1500);

    // Simulate price changes
    Object.keys(MARKETS).forEach(symbol => {
      const change = (Math.random() - 0.5) * 0.1; // ±5% change
      MARKETS[symbol].price *= (1 + change);
      MARKETS[symbol].change = (change * 100).toFixed(2);
    });

    // Update position PnL
    this.positions.forEach(pos => {
      const currentPrice = MARKETS[pos.symbol].price;
      const priceChange = currentPrice - pos.entryPrice;
      pos.unrealizedPnL = pos.side === 'long' 
        ? priceChange * pos.size 
        : -priceChange * pos.size;
      pos.currentPrice = currentPrice;
    });

    spinner.succeed(chalk.green('✅ Prices updated!'));
    console.log(chalk.cyan('Market prices and position PnL have been updated.'));

    await this.delay(2000);
  }

  async closePosition() {
    if (this.positions.length === 0) {
      console.log(chalk.yellow('No positions to close'));
      await this.delay(1000);
      return;
    }

    console.log(chalk.yellow.bold('\n❌ Close Position'));
    this.displayPositions();
    
    const index = parseInt(await this.askQuestion('Position to close (number): ')) - 1;
    
    if (index < 0 || index >= this.positions.length) {
      console.log(chalk.red('❌ Invalid position'));
      await this.delay(1000);
      return;
    }

    const position = this.positions[index];
    const spinner = ora(`Closing ${position.symbol} position...`).start();
    
    await this.delay(1000);

    // Return margin and realize PnL
    this.balances.USDC += position.margin + position.unrealizedPnL;
    
    // Remove position
    this.positions.splice(index, 1);

    spinner.succeed(chalk.green('✅ Position closed!'));
    console.log(chalk.cyan(`Realized PnL: $${position.unrealizedPnL.toFixed(2)}`));

    await this.delay(2000);
  }

  async askQuestion(question) {
    return new Promise((resolve) => {
      this.rl.question(question, resolve);
    });
  }

  delay(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Start the demo
if (require.main === module) {
  const demo = new TradingDemo();
  demo.start().catch(console.error);
}

module.exports = TradingDemo;
