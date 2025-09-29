#!/usr/bin/env node

/**
 * 🚀 QuantDesk Standalone Trading Demo
 * 
 * This script runs a complete trading demo WITHOUT requiring the backend
 * It simulates all functionality locally for demonstration purposes
 */

const readline = require('readline');

// Simple color functions (no external dependencies)
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m',
  white: '\x1b[37m'
};

function colorize(text, color) {
  return `${colors[color]}${text}${colors.reset}`;
}

// Demo trading data
const MARKETS = {
  'BTC-PERP': { price: 45000, change: 2.5 },
  'ETH-PERP': { price: 3200, change: -1.2 },
  'SOL-PERP': { price: 95, change: 5.8 }
};

const TOKENS = ['SOL', 'USDC', 'USDT', 'BTC', 'ETH'];

class StandaloneTradingDemo {
  constructor() {
    this.rl = readline.createInterface({
      input: process.stdin,
      output: process.stdout
    });
    this.user = {
      id: 'demo_user_123',
      walletAddress: 'DemoWallet123456789012345678901234567890',
      username: 'DemoTrader'
    };
    this.tradingAccount = {
      id: 'trading_account_1',
      name: 'Main Trading Account',
      accountIndex: 1
    };
    this.positions = [];
    this.balances = {
      'SOL': 100,
      'USDC': 10000,
      'USDT': 5000,
      'BTC': 0.5,
      'ETH': 5
    };
    this.orders = [];
  }

  async start() {
    console.log(colorize('\n🚀 Welcome to QuantDesk Trading Demo!', 'blue'));
    console.log(colorize('Simulating a complete trading session...\n', 'white'));

    try {
      await this.connectWallet();
      await this.createAccount();
      await this.depositTokens();
      await this.createTradingAccount();
      await this.showMainMenu();
    } catch (error) {
      console.error(colorize('Demo failed:', 'red'), error.message);
    }
  }

  async connectWallet() {
    console.log(colorize('🔗 Connecting wallet...', 'yellow'));
    await this.delay(1000);
    
    console.log(colorize('✅ Wallet connected successfully!', 'green'));
    console.log(colorize(`   User ID: ${this.user.id}`, 'cyan'));
    console.log(colorize(`   Wallet: ${this.user.walletAddress}\n`, 'cyan'));
  }

  async createAccount() {
    console.log(colorize('👤 Creating user account...', 'yellow'));
    await this.delay(800);
    
    console.log(colorize('✅ Account created successfully!', 'green'));
    console.log(colorize(`   Account Type: Standard`, 'cyan'));
    console.log(colorize(`   KYC Status: Pending\n`, 'cyan'));
  }

  async depositTokens() {
    console.log(colorize('💰 Depositing demo tokens...', 'yellow'));
    
    const depositAmounts = {
      'SOL': 100,
      'USDC': 10000,
      'USDT': 5000,
      'BTC': 0.5,
      'ETH': 5
    };

    for (const [token, amount] of Object.entries(depositAmounts)) {
      await this.delay(300);
      console.log(colorize(`   ✅ Deposited ${amount} ${token}`, 'green'));
    }

    console.log(colorize('✅ All tokens deposited successfully!\n', 'green'));
    this.displayBalances();
  }

  async createTradingAccount() {
    console.log(colorize('🏦 Creating trading account...', 'yellow'));
    await this.delay(800);
    
    console.log(colorize('✅ Trading account created!', 'green'));
    console.log(colorize(`   Account: ${this.tradingAccount.name}`, 'cyan'));
    console.log(colorize(`   Index: ${this.tradingAccount.accountIndex}\n`, 'cyan'));
  }

  displayBalances() {
    console.log(colorize('\n💰 Current Balances:', 'yellow'));
    Object.entries(this.balances).forEach(([token, amount]) => {
      console.log(colorize(`   ${token}: ${amount}`, 'white'));
    });
    console.log();
  }

  displayMarkets() {
    console.log(colorize('\n📈 Available Markets:', 'yellow'));
    Object.entries(MARKETS).forEach(([symbol, data]) => {
      const changeColor = data.change >= 0 ? 'green' : 'red';
      const changeSymbol = data.change >= 0 ? '+' : '';
      console.log(colorize(`   ${symbol}: $${data.price} (${changeSymbol}${data.change}%)`, 'white'));
    });
    console.log();
  }

  displayPositions() {
    if (this.positions.length === 0) {
      console.log(colorize('📊 No open positions\n', 'yellow'));
      return;
    }

    console.log(colorize('\n📊 Open Positions:', 'yellow'));
    this.positions.forEach((pos, index) => {
      const pnlColor = pos.unrealizedPnL >= 0 ? 'green' : 'red';
      console.log(colorize(`   ${index + 1}. ${pos.symbol} ${pos.side.toUpperCase()}`, 'white'));
      console.log(colorize(`      Size: ${pos.size} | Entry: $${pos.entryPrice} | Current: $${pos.currentPrice}`, 'white'));
      console.log(colorize(`      Leverage: ${pos.leverage}x | PnL: $${pos.unrealizedPnL}`, 'white'));
    });
    console.log();
  }

  async showMainMenu() {
    while (true) {
      console.clear();
      console.log(colorize('🚀 QuantDesk Trading Demo', 'blue'));
      console.log(colorize('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━', 'white'));
      
      this.displayBalances();
      this.displayMarkets();
      this.displayPositions();

      console.log(colorize('📋 Trading Options:', 'cyan'));
      console.log(colorize('   1. 📈 Place Market Order', 'white'));
      console.log(colorize('   2. 📊 Place Limit Order', 'white'));
      console.log(colorize('   3. 📋 View Orders', 'white'));
      console.log(colorize('   4. 💰 Deposit More Tokens', 'white'));
      console.log(colorize('   5. 🏦 Create Another Trading Account', 'white'));
      console.log(colorize('   6. 📊 View Portfolio Analytics', 'white'));
      console.log(colorize('   7. 🔄 Simulate Price Updates', 'white'));
      console.log(colorize('   8. ❌ Close Position', 'white'));
      console.log(colorize('   9. 🚪 Exit Demo', 'white'));

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
          console.log(colorize('\n👋 Thanks for using QuantDesk Demo!', 'green'));
          process.exit(0);
        default:
          console.log(colorize('❌ Invalid option. Please try again.', 'red'));
          await this.delay(1000);
      }
    }
  }

  async placeMarketOrder() {
    console.log(colorize('\n📈 Place Market Order', 'yellow'));
    
    const symbol = await this.askQuestion('Market (BTC-PERP, ETH-PERP, SOL-PERP): ');
    if (!MARKETS[symbol]) {
      console.log(colorize('❌ Invalid market', 'red'));
      await this.delay(1000);
      return;
    }

    const side = await this.askQuestion('Side (long/short): ');
    if (!['long', 'short'].includes(side)) {
      console.log(colorize('❌ Invalid side', 'red'));
      await this.delay(1000);
      return;
    }

    const size = parseFloat(await this.askQuestion('Size: '));
    const leverage = parseInt(await this.askQuestion('Leverage (1-100x): '));

    console.log(colorize('⚡ Executing market order...', 'yellow'));
    await this.delay(1500);

    // Calculate margin required
    const marginRequired = (size * MARKETS[symbol].price) / leverage;
    
    if (marginRequired > this.balances.USDC) {
      console.log(colorize('❌ Insufficient margin', 'red'));
      console.log(colorize(`Required: $${marginRequired.toFixed(2)} USDC`, 'red'));
      await this.delay(2000);
      return;
    }

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

    console.log(colorize('✅ Market order executed!', 'green'));
    console.log(colorize(`   Order ID: ${order.id}`, 'cyan'));
    console.log(colorize(`   ${side.toUpperCase()} ${size} ${symbol} @ $${order.entryPrice}`, 'cyan'));
    console.log(colorize(`   Leverage: ${leverage}x | Margin: $${marginRequired.toFixed(2)}`, 'cyan'));

    await this.delay(2000);
  }

  async placeLimitOrder() {
    console.log(colorize('\n📊 Place Limit Order', 'yellow'));
    
    const symbol = await this.askQuestion('Market (BTC-PERP, ETH-PERP, SOL-PERP): ');
    if (!MARKETS[symbol]) {
      console.log(colorize('❌ Invalid market', 'red'));
      await this.delay(1000);
      return;
    }

    const side = await this.askQuestion('Side (long/short): ');
    const size = parseFloat(await this.askQuestion('Size: '));
    const price = parseFloat(await this.askQuestion('Limit Price: '));
    const leverage = parseInt(await this.askQuestion('Leverage (1-100x): '));

    console.log(colorize('📋 Placing limit order...', 'yellow'));
    await this.delay(1000);

    const marginRequired = (size * price) / leverage;
    
    if (marginRequired > this.balances.USDC) {
      console.log(colorize('❌ Insufficient margin', 'red'));
      await this.delay(2000);
      return;
    }

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

    console.log(colorize('✅ Limit order placed!', 'green'));
    console.log(colorize(`   Order ID: ${order.id}`, 'cyan'));
    console.log(colorize(`   ${side.toUpperCase()} ${size} ${symbol} @ $${price} (limit)`, 'cyan'));

    await this.delay(2000);
  }

  async viewOrders() {
    console.log(colorize('\n📋 Order History', 'yellow'));
    
    if (this.orders.length === 0) {
      console.log(colorize('No orders found\n', 'white'));
      await this.delay(2000);
      return;
    }

    this.orders.forEach((order, index) => {
      const statusColor = order.status === 'filled' ? 'green' : 'yellow';
      console.log(colorize(`${index + 1}. ${order.symbol} ${order.side.toUpperCase()}`, 'white'));
      console.log(colorize(`   Size: ${order.size} | Price: $${order.price || order.entryPrice}`, 'white'));
      console.log(colorize(`   Status: ${order.status} | Time: ${order.timestamp.toLocaleTimeString()}`, 'white'));
    });

    await this.askQuestion('\nPress Enter to continue...');
  }

  async depositMoreTokens() {
    const token = await this.askQuestion(`Token to deposit (${TOKENS.join(', ')}): `);
    const amount = parseFloat(await this.askQuestion('Amount: '));

    if (!TOKENS.includes(token)) {
      console.log(colorize('❌ Invalid token', 'red'));
      await this.delay(1000);
      return;
    }

    console.log(colorize(`💰 Depositing ${amount} ${token}...`, 'yellow'));
    await this.delay(1000);

    this.balances[token] = (this.balances[token] || 0) + amount;
    console.log(colorize(`✅ Deposited ${amount} ${token}!`, 'green'));

    await this.delay(1000);
  }

  async createAnotherAccount() {
    const name = await this.askQuestion('Trading account name: ');

    console.log(colorize('🏦 Creating trading account...', 'yellow'));
    await this.delay(1000);

    const newAccount = {
      id: `account_${Date.now()}`,
      name,
      accountIndex: this.tradingAccount.accountIndex + 1,
      isActive: true
    };

    console.log(colorize(`✅ Created "${name}"!`, 'green'));
    console.log(colorize(`   Account Index: ${newAccount.accountIndex}`, 'cyan'));

    await this.delay(1500);
  }

  async viewPortfolioAnalytics() {
    console.log(colorize('\n📊 Portfolio Analytics', 'yellow'));
    
    const totalValue = Object.entries(this.balances).reduce((sum, [token, amount]) => {
      if (token === 'USDC' || token === 'USDT') return sum + amount;
      if (token === 'BTC') return sum + (amount * MARKETS['BTC-PERP'].price);
      if (token === 'ETH') return sum + (amount * MARKETS['ETH-PERP'].price);
      if (token === 'SOL') return sum + (amount * MARKETS['SOL-PERP'].price);
      return sum;
    }, 0);

    const totalPnL = this.positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);

    console.log(colorize(`Total Portfolio Value: $${totalValue.toFixed(2)}`, 'white'));
    console.log(colorize(`Total Unrealized PnL: $${totalPnL.toFixed(2)}`, 'white'));
    console.log(colorize(`Open Positions: ${this.positions.length}`, 'white'));
    console.log(colorize(`Total Orders: ${this.orders.length}`, 'white'));

    await this.askQuestion('\nPress Enter to continue...');
  }

  async simulatePriceUpdates() {
    console.log(colorize('\n🔄 Simulating Price Updates...', 'yellow'));
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

    console.log(colorize('✅ Prices updated!', 'green'));
    console.log(colorize('Market prices and position PnL have been updated.', 'cyan'));

    await this.delay(2000);
  }

  async closePosition() {
    if (this.positions.length === 0) {
      console.log(colorize('No positions to close', 'yellow'));
      await this.delay(1000);
      return;
    }

    console.log(colorize('\n❌ Close Position', 'yellow'));
    this.displayPositions();
    
    const index = parseInt(await this.askQuestion('Position to close (number): ')) - 1;
    
    if (index < 0 || index >= this.positions.length) {
      console.log(colorize('❌ Invalid position', 'red'));
      await this.delay(1000);
      return;
    }

    const position = this.positions[index];
    console.log(colorize(`⚡ Closing ${position.symbol} position...`, 'yellow'));
    
    await this.delay(1000);

    // Return margin and realize PnL
    this.balances.USDC += position.margin + position.unrealizedPnL;
    
    // Remove position
    this.positions.splice(index, 1);

    console.log(colorize('✅ Position closed!', 'green'));
    console.log(colorize(`Realized PnL: $${position.unrealizedPnL.toFixed(2)}`, 'cyan'));

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
  const demo = new StandaloneTradingDemo();
  demo.start().catch(console.error);
}

module.exports = StandaloneTradingDemo;
