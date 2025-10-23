import axios from 'axios'

export interface MarketData {
  symbol: string
  price: number
  change24h: number
  volume24h: number
  openInterest: number
  fundingRate: number
  timestamp: string
}

export interface PriceHistory {
  symbol: string
  hours: number
  data: Array<{
    price: number
    timestamp: string
  }>
}

class MarketDataService {
  private baseUrl: string

  constructor() {
    this.baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:3002'
  }

  /**
   * Get all markets with real-time data
   */
  async getMarkets(): Promise<MarketData[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/markets`)
      return response.data.markets || []
    } catch (error) {
      console.error('Error fetching markets:', error)
      return []
    }
  }

  /**
   * Get real-time price for a specific market
   */
  async getMarketPrice(symbol: string): Promise<number | null> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/markets/${symbol}/price`)
      return response.data.price || null
    } catch (error) {
      console.error(`Error fetching price for ${symbol}:`, error)
      return null
    }
  }

  /**
   * Get price history for charts
   */
  async getPriceHistory(symbol: string, hours: number = 24): Promise<PriceHistory | null> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/markets/${symbol}/price-history`, {
        params: { hours }
      })
      return response.data || null
    } catch (error) {
      console.error(`Error fetching price history for ${symbol}:`, error)
      return null
    }
  }

  /**
   * Get market details
   */
  async getMarketDetails(symbol: string): Promise<any> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/markets/${symbol}`)
      return response.data || null
    } catch (error) {
      console.error(`Error fetching market details for ${symbol}:`, error)
      return null
    }
  }

  /**
   * Get funding rates for a market
   */
  async getFundingRates(symbol: string, limit: number = 24): Promise<any[]> {
    try {
      const response = await axios.get(`${this.baseUrl}/api/markets/${symbol}/funding`, {
        params: { limit }
      })
      return response.data.fundingRates || []
    } catch (error) {
      console.error(`Error fetching funding rates for ${symbol}:`, error)
      return []
    }
  }

  /**
   * Convert price history to TradingView format
   */
  convertToTradingViewFormat(priceHistory: PriceHistory): Array<{
    time: number
    open: number
    high: number
    low: number
    close: number
    volume: number
  }> {
    if (!priceHistory || !priceHistory.data) return []

    // Convert price history to candlestick format
    const candles: Array<{
      time: number
      open: number
      high: number
      low: number
      close: number
      volume: number
    }> = []

    // Group prices by hour for candlestick data
    const hourlyData = new Map<string, number[]>()
    
    priceHistory.data.forEach(point => {
      const hour = new Date(point.timestamp).toISOString().substring(0, 13) + ':00:00'
      if (!hourlyData.has(hour)) {
        hourlyData.set(hour, [])
      }
      hourlyData.get(hour)!.push(point.price)
    })

    // Create candlesticks
    hourlyData.forEach((prices, hour) => {
      if (prices.length > 0) {
        const open = prices[0]
        const close = prices[prices.length - 1]
        const high = Math.max(...prices)
        const low = Math.min(...prices)
        const volume = prices.length * 1000 // Mock volume

        candles.push({
          time: Math.floor(new Date(hour).getTime() / 1000),
          open,
          high,
          low,
          close,
          volume
        })
      }
    })

    return candles.sort((a, b) => a.time - b.time)
  }

  /**
   * Get supported market symbols
   */
  getSupportedSymbols(): string[] {
    return ['BTC-PERP', 'ETH-PERP', 'SOL-PERP', 'AVAX-PERP', 'MATIC-PERP', 'ARB-PERP', 'OP-PERP', 'DOGE-PERP', 'ADA-PERP', 'DOT-PERP', 'LINK-PERP']
  }
}

// Lazy initialization to avoid constructor running at module level
let _marketDataService: MarketDataService | null = null

export const marketDataService = {
  get instance() {
    if (!_marketDataService) {
      _marketDataService = new MarketDataService()
    }
    return _marketDataService
  }
}
