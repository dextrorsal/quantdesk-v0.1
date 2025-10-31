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
  // optional enriched fields from authoritative backend
  status?: 'live' | 'stale' | 'high_conf' | 'fallback' | 'unavailable'
  source?: string
  publishTime?: number
  feedId?: string
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
    // Try exact match first
    const direct = this.prices.get(symbol)
    if (direct) return direct
    // Be lenient: allow base asset lookups like 'BTC' by mapping to 'BTC/USDT'
    const upper = symbol.toUpperCase()
    const usdtSymbol = `${upper}/USDT`
    const usdt = this.prices.get(usdtSymbol)
    if (usdt) return usdt
    // Also try stripping common separators
    for (const key of this.prices.keys()) {
      if (key.toUpperCase().startsWith(upper + '/')) {
        return this.prices.get(key)
      }
    }
    return undefined
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
        const json = await response.json()
        // Support both legacy array shape and new authoritative map shape
        if (json && json.success) {
          const payload = json.data ?? json
          if (Array.isArray(payload)) {
            this.updatePrices(payload as unknown as PriceUpdate[])
          } else if (payload && typeof payload === 'object') {
            // Transform map { BTC: { price, updatedAt, ... }, ETH: {...} } → PriceUpdate[] with symbol 'ASSET/USDT'
            const updates: PriceUpdate[] = []
            for (const [assetKey, rec] of Object.entries(payload as Record<string, any>)) {
              if (!rec) continue
              const priceNum = Number((rec as any).price)
              if (!isFinite(priceNum) || priceNum <= 0) continue
              const symbol = `${assetKey.toUpperCase()}/USDT`
              updates.push({
                symbol,
                price: priceNum,
                change: Number((rec as any).change24h) || 0,
                changePercent: Number((rec as any).changePercent24h) || 0,
                timestamp: Number((rec as any).updatedAt) || Date.now(),
              })
              // Preserve enriched fields if desired
              const existing = this.prices.get(symbol)
              this.prices.set(symbol, {
                symbol,
                price: priceNum,
                change: Number((rec as any).change24h) || (existing?.change ?? 0),
                changePercent: Number((rec as any).changePercent24h) || (existing?.changePercent ?? 0),
                timestamp: Number((rec as any).updatedAt) || Date.now(),
                confidence: Number((rec as any).confidence) || existing?.confidence,
                volume: Number((rec as any).volume24h) || existing?.volume,
                status: (rec as any).status,
                source: (rec as any).source,
                publishTime: Number((rec as any).publishTime) || existing?.publishTime,
                feedId: (rec as any).feedId || existing?.feedId,
              })
            }
            // Notify subscribers if we enriched directly
            if (updates.length > 0) {
              this.lastUpdate = Date.now()
              this.isConnected = true
              this.notifySubscribers()
            }
          }
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
