// Portfolio Service for QuantDesk
// Provides comprehensive portfolio management, P&L analytics, and risk metrics

export interface PortfolioSummary {
  totalBalance: number
  totalEquity: number
  totalMargin: number
  totalUnrealizedPnl: number
  totalRealizedPnl: number
  totalFees: number
  marginRatio: number
  liquidationPrice: number
  lastUpdate: number
}

export interface Position {
  id: string
  symbol: string
  side: 'long' | 'short'
  size: number
  entryPrice: number
  currentPrice: number
  leverage: number
  margin: number
  unrealizedPnl: number
  realizedPnl: number
  fees: number
  liquidationPrice: number
  marginRatio: number
  openTime: number
  lastUpdate: number
}

export interface Trade {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  size: number
  price: number
  fees: number
  pnl: number
  timestamp: number
  orderId: string
}

export interface Order {
  id: string
  symbol: string
  type: 'market' | 'limit' | 'stopLoss' | 'takeProfit' | 'trailingStop'
  side: 'buy' | 'sell'
  size: number
  price: number
  status: 'pending' | 'filled' | 'cancelled' | 'expired'
  filledSize: number
  fees: number
  createdAt: number
  updatedAt: number
}

export interface RiskMetrics {
  portfolioValue: number
  totalMargin: number
  marginRatio: number
  liquidationPrice: number
  maxDrawdown: number
  sharpeRatio: number
  winRate: number
  avgWin: number
  avgLoss: number
  profitFactor: number
  totalTrades: number
  winningTrades: number
  losingTrades: number
}

export interface PerformanceMetrics {
  totalReturn: number
  dailyReturn: number
  weeklyReturn: number
  monthlyReturn: number
  yearlyReturn: number
  volatility: number
  maxDrawdown: number
  sharpeRatio: number
  sortinoRatio: number
  calmarRatio: number
  winRate: number
  profitFactor: number
  avgTradeDuration: number
  totalTrades: number
}

class PortfolioService {
  private positions: Map<string, Position> = new Map()
  private trades: Trade[] = []
  private orders: Map<string, Order> = new Map()
  private portfolioSummary: PortfolioSummary
  private riskMetrics: RiskMetrics
  private performanceMetrics: PerformanceMetrics

  constructor() {
    this.portfolioSummary = this.initializePortfolioSummary()
    this.riskMetrics = this.initializeRiskMetrics()
    this.performanceMetrics = this.initializePerformanceMetrics()
    this.generateMockData()
  }

  private initializePortfolioSummary(): PortfolioSummary {
    return {
      totalBalance: 10000, // Starting balance
      totalEquity: 10000,
      totalMargin: 0,
      totalUnrealizedPnl: 0,
      totalRealizedPnl: 0,
      totalFees: 0,
      marginRatio: 0,
      liquidationPrice: 0,
      lastUpdate: Date.now(),
    }
  }

  private initializeRiskMetrics(): RiskMetrics {
    return {
      portfolioValue: 10000,
      totalMargin: 0,
      marginRatio: 0,
      liquidationPrice: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      winRate: 0,
      avgWin: 0,
      avgLoss: 0,
      profitFactor: 0,
      totalTrades: 0,
      winningTrades: 0,
      losingTrades: 0,
    }
  }

  private initializePerformanceMetrics(): PerformanceMetrics {
    return {
      totalReturn: 0,
      dailyReturn: 0,
      weeklyReturn: 0,
      monthlyReturn: 0,
      yearlyReturn: 0,
      volatility: 0,
      maxDrawdown: 0,
      sharpeRatio: 0,
      sortinoRatio: 0,
      calmarRatio: 0,
      winRate: 0,
      profitFactor: 0,
      avgTradeDuration: 0,
      totalTrades: 0,
    }
  }

  private generateMockData(): void {
    // Generate mock positions
    const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'ARB/USDT', 'OP/USDT', 'DOGE/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT']
    
    symbols.forEach((symbol, index) => {
      const side = index % 2 === 0 ? 'long' : 'short'
      const size = Math.random() * 10 + 0.1
      const entryPrice = this.getMockPrice(symbol)
      const currentPrice = entryPrice + (Math.random() - 0.5) * (entryPrice * 0.05)
      const leverage = Math.floor(Math.random() * 20) + 1
      const margin = (size * entryPrice) / leverage
      const unrealizedPnl = side === 'long' 
        ? (currentPrice - entryPrice) * size
        : (entryPrice - currentPrice) * size
      
      const position: Position = {
        id: `pos_${symbol}_${Date.now()}`,
        symbol,
        side,
        size: Number(size.toFixed(3)),
        entryPrice: Number(entryPrice.toFixed(2)),
        currentPrice: Number(currentPrice.toFixed(2)),
        leverage,
        margin: Number(margin.toFixed(2)),
        unrealizedPnl: Number(unrealizedPnl.toFixed(2)),
        realizedPnl: 0,
        fees: Number((margin * 0.001).toFixed(2)),
        liquidationPrice: this.calculateLiquidationPrice(entryPrice, side, leverage),
        marginRatio: this.calculateMarginRatio(margin, unrealizedPnl),
        openTime: Date.now() - Math.random() * 86400000, // Random time within last 24h
        lastUpdate: Date.now(),
      }

      this.positions.set(position.id, position)
    })

    // Generate mock trades
    this.generateMockTrades()

    // Generate mock orders
    this.generateMockOrders()

    // Update portfolio summary
    this.updatePortfolioSummary()
    this.updateRiskMetrics()
    this.updatePerformanceMetrics()
  }

  private getMockPrice(symbol: string): number {
    const prices: Record<string, number> = {
      'BTC/USDT': 43250,
      'ETH/USDT': 2650,
      'SOL/USDT': 220,
    }
    return prices[symbol] || 100
  }

  private calculateLiquidationPrice(entryPrice: number, side: string, leverage: number): number {
    const maintenanceMargin = 0.05 // 5% maintenance margin
    if (side === 'long') {
      return entryPrice * (1 - (1 - maintenanceMargin) / leverage)
    } else {
      return entryPrice * (1 + (1 - maintenanceMargin) / leverage)
    }
  }

  private calculateMarginRatio(margin: number, unrealizedPnl: number): number {
    const equity = margin + unrealizedPnl
    return equity > 0 ? (margin / equity) * 100 : 0
  }

  private generateMockTrades(): void {
    const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'ARB/USDT', 'OP/USDT', 'DOGE/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT']
    
    for (let i = 0; i < 50; i++) {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)]
      const side = Math.random() > 0.5 ? 'buy' : 'sell'
      const size = Math.random() * 5 + 0.1
      const price = this.getMockPrice(symbol) + (Math.random() - 0.5) * (this.getMockPrice(symbol) * 0.02)
      const fees = size * price * 0.001
      const pnl = (Math.random() - 0.5) * 100 // Random P&L

      const trade: Trade = {
        id: `trade_${i}`,
        symbol,
        side,
        size: Number(size.toFixed(3)),
        price: Number(price.toFixed(2)),
        fees: Number(fees.toFixed(2)),
        pnl: Number(pnl.toFixed(2)),
        timestamp: Date.now() - Math.random() * 604800000, // Random time within last week
        orderId: `order_${i}`,
      }

      this.trades.push(trade)
    }

    // Sort trades by timestamp (newest first)
    this.trades.sort((a, b) => b.timestamp - a.timestamp)
  }

  private generateMockOrders(): void {
    const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'AVAX/USDT', 'MATIC/USDT', 'ARB/USDT', 'OP/USDT', 'DOGE/USDT', 'ADA/USDT', 'DOT/USDT', 'LINK/USDT']
    const orderTypes = ['market', 'limit', 'stopLoss', 'takeProfit']
    const statuses = ['pending', 'filled', 'cancelled', 'expired']

    for (let i = 0; i < 20; i++) {
      const symbol = symbols[Math.floor(Math.random() * symbols.length)]
      const type = orderTypes[Math.floor(Math.random() * orderTypes.length)] as any
      const side = Math.random() > 0.5 ? 'buy' : 'sell'
      const size = Math.random() * 5 + 0.1
      const price = this.getMockPrice(symbol) + (Math.random() - 0.5) * (this.getMockPrice(symbol) * 0.02)
      const status = statuses[Math.floor(Math.random() * statuses.length)] as any
      const filledSize = status === 'filled' ? size : Math.random() * size
      const fees = filledSize * price * 0.001

      const order: Order = {
        id: `order_${i}`,
        symbol,
        type,
        side,
        size: Number(size.toFixed(3)),
        price: Number(price.toFixed(2)),
        status,
        filledSize: Number(filledSize.toFixed(3)),
        fees: Number(fees.toFixed(2)),
        createdAt: Date.now() - Math.random() * 86400000,
        updatedAt: Date.now() - Math.random() * 3600000,
      }

      this.orders.set(order.id, order)
    }
  }

  private updatePortfolioSummary(): void {
    let totalMargin = 0
    let totalUnrealizedPnl = 0
    let totalRealizedPnl = 0
    let totalFees = 0

    // Calculate from positions
    this.positions.forEach(position => {
      totalMargin += position.margin
      totalUnrealizedPnl += position.unrealizedPnl
      totalRealizedPnl += position.realizedPnl
      totalFees += position.fees
    })

    // Calculate from trades
    this.trades.forEach(trade => {
      totalRealizedPnl += trade.pnl
      totalFees += trade.fees
    })

    const totalEquity = this.portfolioSummary.totalBalance + totalUnrealizedPnl + totalRealizedPnl
    const marginRatio = totalEquity > 0 ? (totalMargin / totalEquity) * 100 : 0

    this.portfolioSummary = {
      totalBalance: this.portfolioSummary.totalBalance,
      totalEquity: Number(totalEquity.toFixed(2)),
      totalMargin: Number(totalMargin.toFixed(2)),
      totalUnrealizedPnl: Number(totalUnrealizedPnl.toFixed(2)),
      totalRealizedPnl: Number(totalRealizedPnl.toFixed(2)),
      totalFees: Number(totalFees.toFixed(2)),
      marginRatio: Number(marginRatio.toFixed(2)),
      liquidationPrice: this.calculatePortfolioLiquidationPrice(),
      lastUpdate: Date.now(),
    }
  }

  private calculatePortfolioLiquidationPrice(): number {
    // Simplified portfolio liquidation price calculation
    let totalValue = 0
    let totalMargin = 0

    this.positions.forEach(position => {
      totalValue += position.size * position.currentPrice
      totalMargin += position.margin
    })

    return totalValue > 0 ? (totalMargin / totalValue) * 100 : 0
  }

  private updateRiskMetrics(): void {
    const winningTrades = this.trades.filter(trade => trade.pnl > 0)
    const losingTrades = this.trades.filter(trade => trade.pnl < 0)
    
    const totalTrades = this.trades.length
    const winningCount = winningTrades.length
    const losingCount = losingTrades.length

    const avgWin = winningTrades.length > 0 
      ? winningTrades.reduce((sum, trade) => sum + trade.pnl, 0) / winningTrades.length 
      : 0

    const avgLoss = losingTrades.length > 0 
      ? Math.abs(losingTrades.reduce((sum, trade) => sum + trade.pnl, 0) / losingTrades.length)
      : 0

    const profitFactor = avgLoss > 0 ? avgWin / avgLoss : 0
    const winRate = totalTrades > 0 ? (winningCount / totalTrades) * 100 : 0

    this.riskMetrics = {
      portfolioValue: this.portfolioSummary.totalEquity,
      totalMargin: this.portfolioSummary.totalMargin,
      marginRatio: this.portfolioSummary.marginRatio,
      liquidationPrice: this.portfolioSummary.liquidationPrice,
      maxDrawdown: this.calculateMaxDrawdown(),
      sharpeRatio: this.calculateSharpeRatio(),
      winRate: Number(winRate.toFixed(2)),
      avgWin: Number(avgWin.toFixed(2)),
      avgLoss: Number(avgLoss.toFixed(2)),
      profitFactor: Number(profitFactor.toFixed(2)),
      totalTrades,
      winningTrades: winningCount,
      losingTrades: losingCount,
    }
  }

  private calculateMaxDrawdown(): number {
    // Simplified max drawdown calculation
    const returns = this.trades.map(trade => trade.pnl)
    let maxDrawdown = 0
    let peak = 0
    let runningTotal = 0

    returns.forEach(return_ => {
      runningTotal += return_
      if (runningTotal > peak) {
        peak = runningTotal
      }
      const drawdown = peak - runningTotal
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown
      }
    })

    return Number(maxDrawdown.toFixed(2))
  }

  private calculateSharpeRatio(): number {
    // Simplified Sharpe ratio calculation
    const returns = this.trades.map(trade => trade.pnl)
    if (returns.length === 0) return 0

    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length
    const stdDev = Math.sqrt(variance)

    return stdDev > 0 ? Number((avgReturn / stdDev).toFixed(2)) : 0
  }

  private updatePerformanceMetrics(): void {
    const totalReturn = this.portfolioSummary.totalRealizedPnl + this.portfolioSummary.totalUnrealizedPnl
    const initialBalance = this.portfolioSummary.totalBalance
    const returnPercentage = initialBalance > 0 ? (totalReturn / initialBalance) * 100 : 0

    this.performanceMetrics = {
      totalReturn: Number(returnPercentage.toFixed(2)),
      dailyReturn: Number((returnPercentage * 0.1).toFixed(2)), // Simplified
      weeklyReturn: Number((returnPercentage * 0.5).toFixed(2)),
      monthlyReturn: Number(returnPercentage.toFixed(2)),
      yearlyReturn: Number((returnPercentage * 12).toFixed(2)),
      volatility: this.calculateVolatility(),
      maxDrawdown: this.riskMetrics.maxDrawdown,
      sharpeRatio: this.riskMetrics.sharpeRatio,
      sortinoRatio: this.calculateSortinoRatio(),
      calmarRatio: this.calculateCalmarRatio(),
      winRate: this.riskMetrics.winRate,
      profitFactor: this.riskMetrics.profitFactor,
      avgTradeDuration: this.calculateAvgTradeDuration(),
      totalTrades: this.riskMetrics.totalTrades,
    }
  }

  private calculateVolatility(): number {
    const returns = this.trades.map(trade => trade.pnl)
    if (returns.length === 0) return 0

    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length
    const stdDev = Math.sqrt(variance)

    return Number(stdDev.toFixed(2))
  }

  private calculateSortinoRatio(): number {
    // Simplified Sortino ratio (similar to Sharpe but only considers downside deviation)
    return Number((this.performanceMetrics.sharpeRatio * 1.2).toFixed(2))
  }

  private calculateCalmarRatio(): number {
    // Calmar ratio = Annual Return / Max Drawdown
    const annualReturn = this.performanceMetrics.yearlyReturn
    const maxDrawdown = this.performanceMetrics.maxDrawdown
    return maxDrawdown > 0 ? Number((annualReturn / maxDrawdown).toFixed(2)) : 0
  }

  private calculateAvgTradeDuration(): number {
    // Simplified average trade duration calculation
    return Number((Math.random() * 24 + 1).toFixed(1)) // Random duration between 1-25 hours
  }

  // Public API methods
  public getPortfolioSummary(): PortfolioSummary {
    return this.portfolioSummary
  }

  public getPositions(): Position[] {
    return Array.from(this.positions.values())
  }

  public getPosition(id: string): Position | undefined {
    return this.positions.get(id)
  }

  public getTrades(limit?: number): Trade[] {
    const trades = this.trades.slice(0, limit || this.trades.length)
    return trades
  }

  public getOrders(): Order[] {
    return Array.from(this.orders.values())
  }

  public getOrder(id: string): Order | undefined {
    return this.orders.get(id)
  }

  public getRiskMetrics(): RiskMetrics {
    return this.riskMetrics
  }

  public getPerformanceMetrics(): PerformanceMetrics {
    return this.performanceMetrics
  }

  public updatePositionPrice(symbol: string, newPrice: number): void {
    this.positions.forEach(position => {
      if (position.symbol === symbol) {
        position.currentPrice = newPrice
        position.unrealizedPnl = position.side === 'long' 
          ? (newPrice - position.entryPrice) * position.size
          : (position.entryPrice - newPrice) * position.size
        position.marginRatio = this.calculateMarginRatio(position.margin, position.unrealizedPnl)
        position.lastUpdate = Date.now()
      }
    })
    this.updatePortfolioSummary()
    this.updateRiskMetrics()
  }

  public addTrade(trade: Trade): void {
    this.trades.unshift(trade) // Add to beginning
    if (this.trades.length > 1000) {
      this.trades.pop() // Keep only last 1000 trades
    }
    this.updatePortfolioSummary()
    this.updateRiskMetrics()
    this.updatePerformanceMetrics()
  }

  public addOrder(order: Order): void {
    this.orders.set(order.id, order)
  }

  public updateOrder(id: string, updates: Partial<Order>): void {
    const order = this.orders.get(id)
    if (order) {
      const updatedOrder = { ...order, ...updates, updatedAt: Date.now() }
      this.orders.set(id, updatedOrder)
    }
  }
}

// Export singleton instance
// Lazy initialization to avoid constructor running at module level
let _portfolioService: PortfolioService | null = null

export const portfolioService = {
  get instance() {
    if (!_portfolioService) {
      _portfolioService = new PortfolioService()
    }
    return _portfolioService
  }
}
