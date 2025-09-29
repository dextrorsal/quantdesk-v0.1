/**
 * Centralized Price Store - Industry Standard Implementation
 * Follows the same pattern as Drift Protocol, Hyperliquid, and other top DEX platforms
 */

export interface PriceData {
  symbol: string
  price: number
  change: number
  changePercent: number
  timestamp: number
  confidence?: number
  volume?: number
}

export interface PriceUpdate {
  symbol: string
  price: number
  change: number
  changePercent: number
  timestamp: number
}

type PriceSubscriber = (prices: Map<string, PriceData>) => void

class PriceStore {
  private prices = new Map<string, PriceData>()
  private subscribers = new Set<PriceSubscriber>()
  private isConnected = false
  private lastUpdate = 0
  private updateInterval: NodeJS.Timeout | null = null

  // Singleton pattern
  private static instance: PriceStore
  public static getInstance(): PriceStore {
    if (!PriceStore.instance) {
      PriceStore.instance = new PriceStore()
    }
    return PriceStore.instance
  }

  private constructor() {
    // Initialize with default prices to prevent UI flicker
    this.initializeDefaultPrices()
  }

  /**
   * Initialize with default prices to prevent UI flicker
   */
  private initializeDefaultPrices() {
    const defaultSymbols = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'ADA/USDT', 'DOT/USDT', 'AVAX/USDT', 'MATIC/USDT', 'DOGE/USDT', 'LINK/USDT']
    
    defaultSymbols.forEach(symbol => {
      this.prices.set(symbol, {
        symbol,
        price: 0,
        change: 0,
        changePercent: 0,
        timestamp: Date.now(),
        confidence: 0
      })
    })
  }

  /**
   * Subscribe to price updates
   */
  subscribe(callback: PriceSubscriber): () => void {
    this.subscribers.add(callback)
    
    // Immediately call with current prices
    callback(new Map(this.prices))
    
    // Return unsubscribe function
    return () => {
      this.subscribers.delete(callback)
    }
  }

  /**
   * Update prices from WebSocket or API
   */
  updatePrices(priceUpdates: PriceUpdate[]) {
    const now = Date.now()
    let hasUpdates = false

    priceUpdates.forEach(update => {
      const existing = this.prices.get(update.symbol)
      const previousPrice = existing?.price || 0
      
      // Calculate change if we have a previous price
      const change = previousPrice > 0 ? update.price - previousPrice : 0
      const changePercent = previousPrice > 0 ? (change / previousPrice) * 100 : 0

      this.prices.set(update.symbol, {
        symbol: update.symbol,
        price: update.price,
        change: update.change || change,
        changePercent: update.changePercent || changePercent,
        timestamp: update.timestamp || now,
        confidence: existing?.confidence || 0
      })
      
      hasUpdates = true
    })

    if (hasUpdates) {
      this.lastUpdate = now
      this.isConnected = true
      this.notifySubscribers()
    }
  }

  /**
   * Update connection status
   */
  setConnectionStatus(connected: boolean) {
    this.isConnected = connected
    this.notifySubscribers()
  }

  /**
   * Get current prices
   */
  getPrices(): Map<string, PriceData> {
    return new Map(this.prices)
  }

  /**
   * Get price for specific symbol
   */
  getPrice(symbol: string): PriceData | undefined {
    return this.prices.get(symbol)
  }

  /**
   * Get connection status
   */
  getConnectionStatus(): boolean {
    return this.isConnected
  }

  /**
   * Get last update timestamp
   */
  getLastUpdate(): number {
    return this.lastUpdate
  }

  /**
   * Notify all subscribers
   */
  private notifySubscribers() {
    const currentPrices = new Map(this.prices)
    this.subscribers.forEach(callback => {
      try {
        callback(currentPrices)
      } catch (error) {
        console.error('Error notifying price subscriber:', error)
      }
    })
  }

  /**
   * Start fallback polling if WebSocket fails
   */
  startFallbackPolling(apiUrl: string, intervalMs: number = 5000) {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
    }

    this.updateInterval = setInterval(async () => {
      try {
        const response = await fetch(apiUrl)
        const data = await response.json()
        
        if (data.success && data.data) {
          this.updatePrices(data.data)
        }
      } catch (error) {
        console.error('Fallback polling error:', error)
        this.setConnectionStatus(false)
      }
    }, intervalMs)
  }

  /**
   * Stop fallback polling
   */
  stopFallbackPolling() {
    if (this.updateInterval) {
      clearInterval(this.updateInterval)
      this.updateInterval = null
    }
  }

  /**
   * Clear all data
   */
  clear() {
    this.prices.clear()
    this.isConnected = false
    this.lastUpdate = 0
    this.notifySubscribers()
  }
}

export default PriceStore
