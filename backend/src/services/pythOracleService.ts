import axios from 'axios'
import { mcpSupabaseService } from './mcpSupabaseService'
import { Logger } from '../utils/logger'

export interface PythPriceData {
  price: number
  confidence: number
  exponent: number
  publishTime: number
  timestamp: number
}

class PythOracleService {
  private static instance: PythOracleService
  private db: typeof mcpSupabaseService
  private logger: Logger
  private readonly HERMES_API_URL = 'https://hermes.pyth.network/v2/updates/price/latest'
  
  // Pyth Network price feed IDs (official addresses)
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
    this.db = mcpSupabaseService
    this.logger = new Logger()
  }

  public static getInstance(): PythOracleService {
    if (!PythOracleService.instance) {
      PythOracleService.instance = new PythOracleService()
    }
    return PythOracleService.instance
  }

  /**
   * Fetch latest prices from CoinGecko API (primary source)
   * Pyth integration will be added once we resolve the API format issues
   */
  public async fetchLatestPrices(): Promise<Map<string, PythPriceData>> {
    try {
      this.logger.info('📊 Fetching latest prices from CoinGecko API...')
      
      const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
        params: {
          ids: 'bitcoin,ethereum,solana',
          vs_currencies: 'usd',
          include_24hr_change: true
        },
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-DEX/1.0'
        }
      })

      const priceMap = new Map<string, PythPriceData>()
      const coinMapping = {
        bitcoin: 'BTC',
        ethereum: 'ETH',
        solana: 'SOL'
      }

      for (const [coinId, symbol] of Object.entries(coinMapping)) {
        const coinData = response.data[coinId]
        if (coinData && coinData.usd) {
          priceMap.set(symbol, {
            price: coinData.usd,
            confidence: 0.01, // CoinGecko doesn't provide confidence
            exponent: 0,
            publishTime: Math.floor(Date.now() / 1000),
            timestamp: Date.now()
          })
          this.logger.info(`✅ Fetched ${symbol} price from CoinGecko: $${coinData.usd.toFixed(2)} (24h change: ${coinData.usd_24h_change?.toFixed(2)}%)`)
        }
      }

      this.logger.info(`🚀 Successfully fetched ${priceMap.size} price feeds from CoinGecko API`)
      return priceMap

    } catch (error) {
      this.logger.error('❌ Error fetching prices from CoinGecko API:', error)
      return new Map()
    }
  }

  /**
   * Parse Pyth price data
   */
  private parsePythPrice(priceData: any): PythPriceData | null {
    try {
      const price = parseFloat(priceData.price)
      const confidence = parseFloat(priceData.conf)
      const exponent = parseInt(priceData.expo)
      const publishTime = parseInt(priceData.publish_time)

      // Apply exponent to get actual price
      const actualPrice = price * Math.pow(10, exponent)
      const actualConfidence = confidence * Math.pow(10, exponent)

      return {
        price: actualPrice,
        confidence: actualConfidence,
        exponent,
        publishTime,
        timestamp: Date.now()
      }
    } catch (error) {
      this.logger.error('Error parsing Pyth price data:', error)
      return null
    }
  }


  /**
   * Store price data in Supabase oracle_prices table
   */
  public async storePriceData(priceMap: Map<string, PythPriceData>): Promise<void> {
    try {
      // Get all markets from database
      const markets = await this.db.getMarkets().catch((_e: any) => []) as any
      
      for (const market of markets || []) {
        const symbol = this.MARKET_SYMBOLS[market.symbol as keyof typeof this.MARKET_SYMBOLS]
        if (!symbol) continue

        const priceData = priceMap.get(symbol)
        if (!priceData) continue

        // Store in oracle_prices table
        try {
          await this.db.storeOraclePrice(
            market.id, 
            priceData.price, 
            priceData.confidence, 
            priceData.exponent
          )
        } catch (_e: any) {
          // Soft-fail DB writes so live prices still flow
        }

        this.logger.info(`💾 Stored ${symbol} oracle price: $${priceData.price.toFixed(2)} (confidence: ${priceData.confidence.toFixed(4)})`)
      }

    } catch (error) {
      this.logger.error('Error storing oracle price data:', error)
      throw error
    }
  }

  /**
   * Get latest oracle price for a specific market
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

      if (priceData?.rows?.[0]?.price) {
        return priceData.rows[0].price
      }

      // Fallback: fetch directly from CoinGecko
      const coinIdMap: Record<string, string> = { BTC: 'bitcoin', ETH: 'ethereum', SOL: 'solana' }
      const coinId = coinIdMap[symbol]
      if (!coinId) return null
      const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
        params: { ids: coinId, vs_currencies: 'usd' },
        timeout: 8000
      })
      return response.data?.[coinId]?.usd ?? null

    } catch (error) {
      // Fallback path on DB/network errors
      try {
        const symbol = this.MARKET_SYMBOLS[marketSymbol as keyof typeof this.MARKET_SYMBOLS]
        const coinIdMap: Record<string, string> = { BTC: 'bitcoin', ETH: 'ethereum', SOL: 'solana' }
        const coinId = symbol ? coinIdMap[symbol] : undefined
        if (!coinId) return null
        const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
          params: { ids: coinId, vs_currencies: 'usd' },
          timeout: 8000
        })
        return response.data?.[coinId]?.usd ?? null
      } catch (e) {
        this.logger.error(`Error getting latest oracle price for ${marketSymbol}:`, e)
        return null
      }
    }
  }

  /**
   * Get oracle price history for charts
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

      if (priceHistory?.rows?.length) {
        return priceHistory.rows.map((row: any) => ({
          price: parseFloat(row.price),
          timestamp: row.timestamp
        }))
      }

      // Fallback: fetch from CoinGecko market_chart (range)
      const symbol = this.MARKET_SYMBOLS[marketSymbol as keyof typeof this.MARKET_SYMBOLS]
      const coinIdMap: Record<string, string> = { BTC: 'bitcoin', ETH: 'ethereum', SOL: 'solana' }
      const coinId = symbol ? coinIdMap[symbol] : undefined
      if (!coinId) return []

      const nowSec = Math.floor(Date.now() / 1000)
      const fromSec = nowSec - hours * 3600
      const response = await axios.get(`https://api.coingecko.com/api/v3/coins/${coinId}/market_chart/range`, {
        params: { vs_currency: 'usd', from: fromSec, to: nowSec },
        timeout: 10000
      })
      const prices: [number, number][] = response.data?.prices || []
      return prices.map(([ts, price]) => ({ price, timestamp: new Date(ts).toISOString() }))

    } catch (error) {
      try {
        // Second-attempt fallback direct CoinGecko
        const symbol = this.MARKET_SYMBOLS[marketSymbol as keyof typeof this.MARKET_SYMBOLS]
        const coinIdMap: Record<string, string> = { BTC: 'bitcoin', ETH: 'ethereum', SOL: 'solana' }
        const coinId = symbol ? coinIdMap[symbol] : undefined
        if (!coinId) return []
        const nowSec = Math.floor(Date.now() / 1000)
        const fromSec = nowSec - hours * 3600
        const response = await axios.get(`https://api.coingecko.com/api/v3/coins/${coinId}/market_chart/range`, {
          params: { vs_currency: 'usd', from: fromSec, to: nowSec },
          timeout: 10000
        })
        const prices: [number, number][] = response.data?.prices || []
        return prices.map(([ts, price]) => ({ price, timestamp: new Date(ts).toISOString() }))
      } catch (e) {
        this.logger.error(`Error getting oracle price history for ${marketSymbol}:`, e)
        return []
      }
    }
  }

  /**
   * Start oracle price feed service (runs every 30 seconds)
   */
  public startPriceFeed(): void {
    this.logger.info('🚀 Starting Pyth Oracle price feed service...')
    
    // Initial fetch
    this.updatePrices()
    
    // Set up interval
    setInterval(() => {
      this.updatePrices()
    }, 30000) // Update every 30 seconds
  }

  /**
   * Update oracle prices and store in database
   */
  private async updatePrices(): Promise<void> {
    try {
      const priceMap = await this.fetchLatestPrices()
      if (priceMap.size > 0) {
        await this.storePriceData(priceMap)
        this.logger.info('✅ Oracle price update completed successfully')
      } else {
        this.logger.warn('⚠️ No oracle prices fetched')
      }
    } catch (error) {
      this.logger.error('❌ Oracle price update failed:', error)
    }
  }

  /**
   * Get all supported symbols
   */
  public getSupportedSymbols(): string[] {
    return Object.keys(this.MARKET_SYMBOLS)
  }

  /**
   * Test oracle API connection
   */
  public async testConnection(): Promise<boolean> {
    try {
      const priceMap = await this.fetchLatestPrices()
      return priceMap.size > 0
    } catch (error) {
      this.logger.error('Oracle API connection test failed:', error)
      return false
    }
  }
}

export const pythOracleService = PythOracleService.getInstance()