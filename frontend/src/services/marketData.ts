// Market Data Service for QuantDesk
// Provides real-time-like market data simulation

export interface MarketData {
  symbol: string
  price: number
  change24h: number
  volume24h: number
  high24h: number
  low24h: number
  open24h: number
  lastUpdate: number
}

export interface CandlestickData {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

export interface OrderBookData {
  bids: Array<{ price: number; size: number }>
  asks: Array<{ price: number; size: number }>
  spread: number
  lastUpdate: number
}

export interface TradeData {
  id: string
  price: number
  size: number
  side: 'buy' | 'sell'
  timestamp: number
}

class MarketDataService {
  private markets: Map<string, MarketData> = new Map()
  private candlestickData: Map<string, CandlestickData[]> = new Map()
  private orderBookData: Map<string, OrderBookData> = new Map()
  private tradeData: Map<string, TradeData[]> = new Map()
  private subscribers: Map<string, Set<(data: any) => void>> = new Map()

  constructor() {
    this.initializeMarkets()
    this.startDataUpdates()
  }

  private initializeMarkets() {
    const initialMarkets: MarketData[] = [
      {
        symbol: 'BTC/USDT',
        price: 43250.50,
        change24h: 2.45,
        volume24h: 1234567.89,
        high24h: 43500.00,
        low24h: 42800.00,
        open24h: 42200.00,
        lastUpdate: Date.now(),
      },
      {
        symbol: 'ETH/USDT',
        price: 2650.30,
        change24h: -1.20,
        volume24h: 987654.32,
        high24h: 2700.00,
        low24h: 2600.00,
        open24h: 2680.00,
        lastUpdate: Date.now(),
      },
      {
        symbol: 'SOL/USDT',
        price: 220.65,
        change24h: -7.05,
        volume24h: 456789.12,
        high24h: 240.00,
        low24h: 215.00,
        open24h: 237.50,
        lastUpdate: Date.now(),
      },
    ]

    initialMarkets.forEach(market => {
      this.markets.set(market.symbol, market)
      this.generateCandlestickData(market.symbol)
      this.generateOrderBookData(market.symbol)
      this.generateTradeData(market.symbol)
    })
  }

  private generateCandlestickData(symbol: string) {
    const market = this.markets.get(symbol)
    if (!market) return

    const data: CandlestickData[] = []
    const now = Date.now()
    const interval = 60 * 60 * 1000 // 1 hour intervals
    let basePrice = market.price

    for (let i = 100; i >= 0; i--) {
      const time = now - (i * interval)
      const open = basePrice + (Math.random() - 0.5) * (market.price * 0.02)
      const close = open + (Math.random() - 0.5) * (market.price * 0.03)
      const high = Math.max(open, close) + Math.random() * (market.price * 0.01)
      const low = Math.min(open, close) - Math.random() * (market.price * 0.01)
      const volume = Math.random() * 1000 + 100

      data.push({
        time,
        open: Number(open.toFixed(2)),
        high: Number(high.toFixed(2)),
        low: Number(low.toFixed(2)),
        close: Number(close.toFixed(2)),
        volume: Number(volume.toFixed(2)),
      })

      basePrice = close
    }

    this.candlestickData.set(symbol, data)
  }

  private generateOrderBookData(symbol: string) {
    const market = this.markets.get(symbol)
    if (!market) return

    const bids: Array<{ price: number; size: number }> = []
    const asks: Array<{ price: number; size: number }> = []
    
    // Generate bids (below current price)
    for (let i = 0; i < 10; i++) {
      const price = market.price - (i + 1) * (market.price * 0.001)
      const size = Math.random() * 10 + 0.1
      bids.push({ price: Number(price.toFixed(2)), size: Number(size.toFixed(3)) })
    }

    // Generate asks (above current price)
    for (let i = 0; i < 10; i++) {
      const price = market.price + (i + 1) * (market.price * 0.001)
      const size = Math.random() * 10 + 0.1
      asks.push({ price: Number(price.toFixed(2)), size: Number(size.toFixed(3)) })
    }

    const spread = asks[0].price - bids[0].price

    this.orderBookData.set(symbol, {
      bids: bids.sort((a, b) => b.price - a.price),
      asks: asks.sort((a, b) => a.price - b.price),
      spread: Number(spread.toFixed(2)),
      lastUpdate: Date.now(),
    })
  }

  private generateTradeData(symbol: string) {
    const market = this.markets.get(symbol)
    if (!market) return

    const trades: TradeData[] = []
    const now = Date.now()

    for (let i = 0; i < 20; i++) {
      const price = market.price + (Math.random() - 0.5) * (market.price * 0.002)
      const size = Math.random() * 5 + 0.1
      const side = Math.random() > 0.5 ? 'buy' : 'sell'
      const timestamp = now - (i * 30000) // 30 seconds apart

      trades.push({
        id: `trade_${i}`,
        price: Number(price.toFixed(2)),
        size: Number(size.toFixed(3)),
        side,
        timestamp,
      })
    }

    this.tradeData.set(symbol, trades.sort((a, b) => b.timestamp - a.timestamp))
  }

  private startDataUpdates() {
    setInterval(() => {
      this.updateMarketPrices()
      this.updateOrderBooks()
      this.addNewTrades()
    }, 1000) // Update every second
  }

  private updateMarketPrices() {
    this.markets.forEach((market, symbol) => {
      const change = (Math.random() - 0.5) * (market.price * 0.001) // 0.1% max change
      market.price = Number((market.price + change).toFixed(2))
      market.lastUpdate = Date.now()

      this.notifySubscribers(`market:${symbol}`, market)
    })
  }

  private updateOrderBooks() {
    this.orderBookData.forEach((orderBook, symbol) => {
      // Simulate order book updates
      const market = this.markets.get(symbol)
      if (!market) return

      // Update some bid/ask prices slightly
      orderBook.bids.forEach(bid => {
        bid.price += (Math.random() - 0.5) * 0.1
        bid.price = Number(bid.price.toFixed(2))
      })

      orderBook.asks.forEach(ask => {
        ask.price += (Math.random() - 0.5) * 0.1
        ask.price = Number(ask.price.toFixed(2))
      })

      orderBook.spread = orderBook.asks[0].price - orderBook.bids[0].price
      orderBook.lastUpdate = Date.now()

      this.notifySubscribers(`orderbook:${symbol}`, orderBook)
    })
  }

  private addNewTrades() {
    this.tradeData.forEach((trades, symbol) => {
      const market = this.markets.get(symbol)
      if (!market) return

      // Add new trade occasionally
      if (Math.random() > 0.7) {
        const price = market.price + (Math.random() - 0.5) * (market.price * 0.001)
        const size = Math.random() * 3 + 0.1
        const side = Math.random() > 0.5 ? 'buy' : 'sell'

        const newTrade: TradeData = {
          id: `trade_${Date.now()}`,
          price: Number(price.toFixed(2)),
          size: Number(size.toFixed(3)),
          side,
          timestamp: Date.now(),
        }

        trades.unshift(newTrade)
        if (trades.length > 50) {
          trades.pop() // Keep only last 50 trades
        }

        this.notifySubscribers(`trades:${symbol}`, trades.slice(0, 10))
      }
    })
  }

  private notifySubscribers(channel: string, data: any) {
    const subscribers = this.subscribers.get(channel)
    if (subscribers) {
      subscribers.forEach(callback => callback(data))
    }
  }

  // Public API methods
  public getMarketData(symbol: string): MarketData | undefined {
    return this.markets.get(symbol)
  }

  public getAllMarkets(): MarketData[] {
    return Array.from(this.markets.values())
  }

  public getCandlestickData(symbol: string): CandlestickData[] {
    return this.candlestickData.get(symbol) || []
  }

  public getOrderBookData(symbol: string): OrderBookData | undefined {
    return this.orderBookData.get(symbol)
  }

  public getTradeData(symbol: string): TradeData[] {
    return this.tradeData.get(symbol) || []
  }

  public subscribe(channel: string, callback: (data: any) => void): () => void {
    if (!this.subscribers.has(channel)) {
      this.subscribers.set(channel, new Set())
    }
    
    this.subscribers.get(channel)!.add(callback)

    // Return unsubscribe function
    return () => {
      const subscribers = this.subscribers.get(channel)
      if (subscribers) {
        subscribers.delete(callback)
        if (subscribers.size === 0) {
          this.subscribers.delete(channel)
        }
      }
    }
  }

  public unsubscribe(channel: string, callback: (data: any) => void) {
    const subscribers = this.subscribers.get(channel)
    if (subscribers) {
      subscribers.delete(callback)
      if (subscribers.size === 0) {
        this.subscribers.delete(channel)
      }
    }
  }
}

// Export singleton instance
export const marketDataService = new MarketDataService()
