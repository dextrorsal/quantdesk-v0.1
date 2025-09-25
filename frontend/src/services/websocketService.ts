// WebSocket Service for QuantDesk
// Provides real-time data streaming for market data, order book, trades, and positions

export interface WebSocketMessage {
  type: 'market_data' | 'order_book' | 'trade' | 'position_update' | 'order_update' | 'error' | 'ping' | 'pong' | 'subscribe' | 'unsubscribe'
  channel: string
  data: any
  timestamp: number
}

export interface MarketDataUpdate {
  symbol: string
  price: number
  change24h: number
  volume24h: number
  high24h: number
  low24h: number
  open24h: number
  lastUpdate: number
}

export interface OrderBookUpdate {
  symbol: string
  bids: Array<{ price: number; size: number }>
  asks: Array<{ price: number; size: number }>
  spread: number
  timestamp: number
}

export interface TradeUpdate {
  symbol: string
  id: string
  price: number
  size: number
  side: 'buy' | 'sell'
  timestamp: number
}

export interface PositionUpdate {
  symbol: string
  side: 'long' | 'short'
  size: number
  entryPrice: number
  currentPrice: number
  unrealizedPnl: number
  margin: number
  leverage: number
  liquidationPrice: number
  timestamp: number
}

export interface OrderUpdate {
  id: string
  symbol: string
  type: 'market' | 'limit' | 'stopLoss' | 'takeProfit' | 'trailingStop'
  side: 'buy' | 'sell'
  size: number
  price: number
  status: 'pending' | 'filled' | 'cancelled' | 'expired'
  filledSize: number
  timestamp: number
}

class WebSocketService {
  private ws: WebSocket | null = null
  private url: string
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private reconnectInterval = 1000
  private isConnected = false
  private subscribers: Map<string, Set<(data: any) => void>> = new Map()
  private heartbeatInterval: NodeJS.Timeout | null = null
  private subscriptions: Set<string> = new Set()

  constructor(url: string = 'ws://localhost:8080/ws') {
    this.url = url
  }

  public connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      try {
        this.ws = new WebSocket(this.url)
        
        this.ws.onopen = () => {
          console.log('WebSocket connected')
          this.isConnected = true
          this.reconnectAttempts = 0
          this.startHeartbeat()
          this.resubscribe()
          resolve()
        }

        this.ws.onmessage = (event) => {
          this.handleMessage(event.data)
        }

        this.ws.onclose = (event) => {
          console.log('WebSocket disconnected:', event.code, event.reason)
          this.isConnected = false
          this.stopHeartbeat()
          this.handleReconnect()
        }

        this.ws.onerror = (error) => {
          console.error('WebSocket error:', error)
          reject(error)
        }
      } catch (error) {
        reject(error)
      }
    })
  }

  public disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
    }
    this.isConnected = false
    this.stopHeartbeat()
  }

  private handleMessage(data: string): void {
    try {
      const message: WebSocketMessage = JSON.parse(data)
      
      switch (message.type) {
        case 'market_data':
          this.notifySubscribers(`market:${message.channel}`, message.data)
          break
        case 'order_book':
          this.notifySubscribers(`orderbook:${message.channel}`, message.data)
          break
        case 'trade':
          this.notifySubscribers(`trades:${message.channel}`, message.data)
          break
        case 'position_update':
          this.notifySubscribers(`positions:${message.channel}`, message.data)
          break
        case 'order_update':
          this.notifySubscribers(`orders:${message.channel}`, message.data)
          break
        case 'pong':
          // Heartbeat response
          break
        case 'error':
          console.error('WebSocket server error:', message.data)
          break
        default:
          console.warn('Unknown message type:', message.type)
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error)
    }
  }

  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = this.reconnectInterval * Math.pow(2, this.reconnectAttempts - 1)
      
      console.log(`Attempting to reconnect in ${delay}ms (attempt ${this.reconnectAttempts})`)
      
      setTimeout(() => {
        this.connect().catch(console.error)
      }, delay)
    } else {
      console.error('Max reconnection attempts reached')
    }
  }

  private startHeartbeat(): void {
    this.heartbeatInterval = setInterval(() => {
      if (this.isConnected && this.ws) {
        this.send({
          type: 'ping',
          channel: 'heartbeat',
          data: { timestamp: Date.now() },
          timestamp: Date.now()
        })
      }
    }, 30000) // Send ping every 30 seconds
  }

  private stopHeartbeat(): void {
    if (this.heartbeatInterval) {
      clearInterval(this.heartbeatInterval)
      this.heartbeatInterval = null
    }
  }

  private resubscribe(): void {
    this.subscriptions.forEach(channel => {
      this.send({
        type: 'subscribe',
        channel,
        data: {},
        timestamp: Date.now()
      })
    })
  }

  private send(message: WebSocketMessage): void {
    if (this.isConnected && this.ws) {
      this.ws.send(JSON.stringify(message))
    }
  }

  protected notifySubscribers(channel: string, data: any): void {
    const subscribers = this.subscribers.get(channel)
    if (subscribers) {
      subscribers.forEach(callback => callback(data))
    }
  }

  // Public API methods
  public subscribe(channel: string, callback: (data: any) => void): () => void {
    if (!this.subscribers.has(channel)) {
      this.subscribers.set(channel, new Set())
    }
    
    this.subscribers.get(channel)!.add(callback)

    // Subscribe to the channel if connected
    if (this.isConnected) {
      this.subscriptions.add(channel)
      this.send({
        type: 'subscribe',
        channel,
        data: {},
        timestamp: Date.now()
      })
    }

    // Return unsubscribe function
    return () => {
      const subscribers = this.subscribers.get(channel)
      if (subscribers) {
        subscribers.delete(callback)
        if (subscribers.size === 0) {
          this.subscribers.delete(channel)
          
          // Unsubscribe from the channel if connected
          if (this.isConnected) {
            this.subscriptions.delete(channel)
            this.send({
              type: 'unsubscribe',
              channel,
              data: {},
              timestamp: Date.now()
            })
          }
        }
      }
    }
  }

  public unsubscribe(channel: string, callback: (data: any) => void): void {
    const subscribers = this.subscribers.get(channel)
    if (subscribers) {
      subscribers.delete(callback)
      if (subscribers.size === 0) {
        this.subscribers.delete(channel)
        
        // Unsubscribe from the channel if connected
        if (this.isConnected) {
          this.subscriptions.delete(channel)
          this.send({
            type: 'unsubscribe',
            channel,
            data: {},
            timestamp: Date.now()
          })
        }
      }
    }
  }

  public getConnectionStatus(): boolean {
    return this.isConnected
  }

  public getReconnectAttempts(): number {
    return this.reconnectAttempts
  }
}

// Mock WebSocket Service for development
class MockWebSocketService extends WebSocketService {
  private mockDataInterval: NodeJS.Timeout | null = null
  private mockData: Map<string, any> = new Map()

  constructor() {
    super('mock://localhost')
    this.initializeMockData()
  }

  public connect(): Promise<void> {
    return new Promise((resolve) => {
      // Simulate connection delay
      setTimeout(() => {
        this.startMockDataUpdates()
        resolve()
      }, 100)
    })
  }

  public disconnect(): void {
    this.stopMockDataUpdates()
  }

  private initializeMockData(): void {
    // Initialize mock market data
    this.mockData.set('BTC/USDT', {
      symbol: 'BTC/USDT',
      price: 43250.50,
      change24h: 2.45,
      volume24h: 1234567.89,
      high24h: 43500.00,
      low24h: 42800.00,
      open24h: 42200.00,
      lastUpdate: Date.now(),
    })

    this.mockData.set('ETH/USDT', {
      symbol: 'ETH/USDT',
      price: 2650.30,
      change24h: -1.20,
      volume24h: 987654.32,
      high24h: 2700.00,
      low24h: 2600.00,
      open24h: 2680.00,
      lastUpdate: Date.now(),
    })

    this.mockData.set('SOL/USDT', {
      symbol: 'SOL/USDT',
      price: 220.65,
      change24h: -7.05,
      volume24h: 456789.12,
      high24h: 240.00,
      low24h: 215.00,
      open24h: 237.50,
      lastUpdate: Date.now(),
    })
  }

  private startMockDataUpdates(): void {
    this.mockDataInterval = setInterval(() => {
      this.updateMockMarketData()
      this.generateMockTrades()
      this.generateMockOrderBookUpdates()
    }, 1000) // Update every second
  }

  private stopMockDataUpdates(): void {
    if (this.mockDataInterval) {
      clearInterval(this.mockDataInterval)
      this.mockDataInterval = null
    }
  }

  private updateMockMarketData(): void {
    this.mockData.forEach((market, symbol) => {
      const change = (Math.random() - 0.5) * (market.price * 0.001) // 0.1% max change
      market.price = Number((market.price + change).toFixed(2))
      market.lastUpdate = Date.now()

      // Notify subscribers
      this.notifySubscribers(`market:${symbol}`, market)
    })
  }

  private generateMockTrades(): void {
    const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    symbols.forEach(symbol => {
      if (Math.random() > 0.7) { // 30% chance of new trade
        const market = this.mockData.get(symbol)
        if (market) {
          const trade: TradeUpdate = {
            symbol,
            id: `trade_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
            price: market.price + (Math.random() - 0.5) * (market.price * 0.001),
            size: Math.random() * 3 + 0.1,
            side: Math.random() > 0.5 ? 'buy' : 'sell',
            timestamp: Date.now(),
          }

          this.notifySubscribers(`trades:${symbol}`, trade)
        }
      }
    })
  }

  private generateMockOrderBookUpdates(): void {
    const symbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT']
    
    symbols.forEach(symbol => {
      const market = this.mockData.get(symbol)
      if (market) {
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

        const orderBookUpdate: OrderBookUpdate = {
          symbol,
          bids: bids.sort((a, b) => b.price - a.price),
          asks: asks.sort((a, b) => a.price - b.price),
          spread: Number(spread.toFixed(2)),
          timestamp: Date.now(),
        }

        this.notifySubscribers(`orderbook:${symbol}`, orderBookUpdate)
      }
    })
  }

  protected notifySubscribers(channel: string, data: any): void {
    // Access the parent class method
    const subscribers = (this as any).subscribers.get(channel)
    if (subscribers) {
      subscribers.forEach((callback: (data: any) => void) => callback(data))
    }
  }
}

// Export singleton instance (using mock for development)
export const websocketService = new MockWebSocketService()
