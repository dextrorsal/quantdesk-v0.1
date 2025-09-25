import axios from 'axios'
import { DatabaseService } from './database'
import { Logger } from '../utils/logger'

export interface PythPriceFeed {
  id: string
  price: {
    price: string
    conf: string
    expo: number
    publish_time: number
  }
}

export interface PythHermesResponse {
  data: {
    [feedId: string]: PythPriceFeed
  }
}

class PythHermesService {
  private static instance: PythHermesService
  private db: DatabaseService
  private logger: Logger
  private readonly HERMES_API_URL = 'https://hermes.pyth.network/v2/updates/price/latest'
  
  // Pyth Network price feed IDs (from official documentation)
  private readonly PYTH_FEED_IDS = {
    BTC: 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J', // BTC/USD
    ETH: 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB', // ETH/USD
    SOL: 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG', // SOL/USD
    USDC: 'Gnt27xtC473ZT2Mw5u8wZ68Z3gULkSTb5DuxJy7eJotD', // USDC/USD
    USDT: '3vxLXJqLqF3JG5TCbYycbKWRBbCJQLxQmBGCkyqEEefL'  // USDT/USD
  }

  private readonly MARKET_SYMBOLS = {
    'BTC-PERP': 'BTC',
    'ETH-PERP': 'ETH', 
    'SOL-PERP': 'SOL'
  }

  private constructor() {
    this.db = DatabaseService.getInstance()
    this.logger = new Logger()
  }

  public static getInstance(): PythHermesService {
    if (!PythHermesService.instance) {
      PythHermesService.instance = new PythHermesService()
    }
    return PythHermesService.instance
  }

  /**
   * Fetch latest prices from Pyth Hermes API
   * Based on official Pyth Network documentation
   */
  public async fetchLatestPrices(): Promise<Map<string, PythPriceFeed>> {
    try {
      const feedIds = Object.values(this.PYTH_FEED_IDS)
      
      // Pyth Hermes API expects feed IDs as array
      const response = await axios.get<PythHermesResponse>(this.HERMES_API_URL, {
        params: {
          ids: feedIds
        },
        paramsSerializer: {
          indexes: null // This will serialize arrays as ids[]=value1&ids[]=value2
        },
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-DEX/1.0'
        }
      })

      const priceMap = new Map<string, PythPriceFeed>()
      
      // Map the response data to our symbols
      for (const [symbol, feedId] of Object.entries(this.PYTH_FEED_IDS)) {
        const priceData = response.data.data[feedId]
        if (priceData) {
          priceMap.set(symbol, priceData)
          this.logger.info(`Fetched ${symbol} price from Pyth: $${this.parsePrice(priceData.price.price, priceData.price.expo)}`)
        }
      }

      this.logger.info(`Successfully fetched ${priceMap.size} price feeds from Pyth Hermes API`)
      return priceMap

    } catch (error) {
      this.logger.error('Error fetching Pyth prices from Hermes API:', error)
      throw new Error('Failed to fetch price data from Pyth Network')
    }
  }

  /**
   * Parse Pyth price with exponent
   */
  private parsePrice(priceStr: string, exponent: number): number {
    const price = parseFloat(priceStr)
    return price * Math.pow(10, exponent)
  }

  /**
   * Store price data in Supabase oracle_prices table
   */
  public async storePriceData(priceMap: Map<string, PythPriceFeed>): Promise<void> {
    try {
      // Get all markets from database
      const markets = await this.db.getMarkets()
      
      for (const market of markets) {
        const symbol = this.MARKET_SYMBOLS[market.symbol as keyof typeof this.MARKET_SYMBOLS]
        if (!symbol) continue

        const priceData = priceMap.get(symbol)
        if (!priceData) continue

        // Parse price data
        const price = this.parsePrice(priceData.price.price, priceData.price.expo)
        const confidence = this.parsePrice(priceData.price.conf, priceData.price.expo)
        const exponent = priceData.price.expo
        const publishTime = priceData.price.publish_time

        // Store in oracle_prices table
        await this.db.query(
          `INSERT INTO oracle_prices (market_id, price, confidence, exponent, created_at)
           VALUES ($1, $2, $3, $4, NOW())`,
          [market.id, price, confidence, exponent]
        )

        this.logger.info(`Stored ${symbol} price: $${price.toFixed(2)} (confidence: ${confidence.toFixed(4)}, publish time: ${new Date(publishTime * 1000).toISOString()})`)
      }

    } catch (error) {
      this.logger.error('Error storing price data:', error)
      throw error
    }
  }

  /**
   * Get latest price for a specific market
   */
  public async getLatestPrice(marketSymbol: string): Promise<number | null> {
    try {
      const symbol = this.MARKET_SYMBOLS[marketSymbol as keyof typeof this.MARKET_SYMBOLS]
      if (!symbol) return null

      const priceData = await this.db.query(
        `SELECT price FROM oracle_prices 
         WHERE market_id = (
           SELECT id FROM markets WHERE symbol = $1
         ) 
         ORDER BY created_at DESC LIMIT 1`,
        [marketSymbol]
      )

      return priceData.rows[0]?.price || null

    } catch (error) {
      this.logger.error(`Error getting latest price for ${marketSymbol}:`, error)
      return null
    }
  }

  /**
   * Get price history for charts
   */
  public async getPriceHistory(marketSymbol: string, hours: number = 24): Promise<Array<{price: number, timestamp: string}>> {
    try {
      const priceHistory = await this.db.query(
        `SELECT price, created_at as timestamp FROM oracle_prices 
         WHERE market_id = (
           SELECT id FROM markets WHERE symbol = $1
         ) 
         AND created_at >= NOW() - INTERVAL '${hours} hours'
         ORDER BY created_at ASC`,
        [marketSymbol]
      )

      return priceHistory.rows.map(row => ({
        price: parseFloat(row.price),
        timestamp: row.timestamp
      }))

    } catch (error) {
      this.logger.error(`Error getting price history for ${marketSymbol}:`, error)
      return []
    }
  }

  /**
   * Start price feed service (runs every 30 seconds)
   */
  public startPriceFeed(): void {
    this.logger.info('🚀 Starting Pyth Hermes price feed service...')
    
    // Initial fetch
    this.updatePrices()
    
    // Set up interval
    setInterval(() => {
      this.updatePrices()
    }, 30000) // Update every 30 seconds
  }

  /**
   * Update prices and store in database
   */
  private async updatePrices(): Promise<void> {
    try {
      const priceMap = await this.fetchLatestPrices()
      await this.storePriceData(priceMap)
      
      this.logger.info('✅ Pyth price update completed successfully')
    } catch (error) {
      this.logger.error('❌ Pyth price update failed:', error)
    }
  }

  /**
   * Get all supported symbols
   */
  public getSupportedSymbols(): string[] {
    return Object.keys(this.MARKET_SYMBOLS)
  }

  /**
   * Test Pyth API connection
   */
  public async testConnection(): Promise<boolean> {
    try {
      const priceMap = await this.fetchLatestPrices()
      return priceMap.size > 0
    } catch (error) {
      this.logger.error('Pyth API connection test failed:', error)
      return false
    }
  }
}

export const pythHermesService = PythHermesService.getInstance()
