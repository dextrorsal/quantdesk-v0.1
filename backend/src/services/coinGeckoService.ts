import axios from 'axios'
import { SupabaseDatabaseService } from './supabaseDatabase'
import { Logger } from '../utils/logger'

export interface CoinGeckoPrice {
  price: number
  change24h: number
  timestamp: number
}

class CoinGeckoService {
  private static instance: CoinGeckoService
  private db: SupabaseDatabaseService
  private logger: Logger
  private readonly COINGECKO_API_URL = 'https://api.coingecko.com/api/v3/simple/price'
  
  // CoinGecko coin IDs
  private readonly COIN_IDS = {
    BTC: 'bitcoin',
    ETH: 'ethereum',
    SOL: 'solana',
    USDC: 'usd-coin',
    USDT: 'tether'
  }

  private readonly MARKET_SYMBOLS = {
    'BTC-PERP': 'BTC',
    'ETH-PERP': 'ETH', 
    'SOL-PERP': 'SOL'
  }

  private constructor() {
    this.db = SupabaseDatabaseService.getInstance()
    this.logger = new Logger()
  }

  public static getInstance(): CoinGeckoService {
    if (!CoinGeckoService.instance) {
      CoinGeckoService.instance = new CoinGeckoService()
    }
    return CoinGeckoService.instance
  }

  /**
   * Fetch latest prices from CoinGecko API
   */
  public async fetchLatestPrices(): Promise<Map<string, CoinGeckoPrice>> {
    try {
      const coinIds = Object.values(this.COIN_IDS).join(',')
      
      const response = await axios.get(this.COINGECKO_API_URL, {
        params: {
          ids: coinIds,
          vs_currencies: 'usd',
          include_24hr_change: true
        },
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-DEX/1.0'
        }
      })

      const priceMap = new Map<string, CoinGeckoPrice>()
      
      // Map the response data to our symbols
      for (const [symbol, coinId] of Object.entries(this.COIN_IDS)) {
        const coinData = response.data[coinId]
        if (coinData && coinData.usd) {
          priceMap.set(symbol, {
            price: coinData.usd,
            change24h: coinData.usd_24h_change || 0,
            timestamp: Date.now()
          })
          this.logger.info(`Fetched ${symbol} price from CoinGecko: $${coinData.usd.toFixed(2)} (24h: ${coinData.usd_24h_change?.toFixed(2) || 0}%)`)
        }
      }

      this.logger.info(`Successfully fetched ${priceMap.size} price feeds from CoinGecko`)
      return priceMap

    } catch (error) {
      this.logger.error('Error fetching CoinGecko prices:', error)
      throw new Error('Failed to fetch price data from CoinGecko')
    }
  }

  /**
   * Store price data in Supabase oracle_prices table
   */
  public async storePriceData(priceMap: Map<string, CoinGeckoPrice>): Promise<void> {
    try {
      // Get all markets from database
      const markets = await this.db.getMarkets()
      
      for (const market of markets) {
        const symbol = this.MARKET_SYMBOLS[market.symbol as keyof typeof this.MARKET_SYMBOLS]
        if (!symbol) continue

        const priceData = priceMap.get(symbol)
        if (!priceData) continue

        // Store in oracle_prices table
        await this.db.insert('oracle_prices', {
          market_id: market.id,
          price: priceData.price,
          confidence: 0.01,
          exponent: 0,
          created_at: new Date().toISOString()
        });

        this.logger.info(`Stored ${symbol} price: $${priceData.price.toFixed(2)} (24h: ${priceData.change24h.toFixed(2)}%)`)
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

      const priceData = await this.db.complexQuery(
        `SELECT price FROM oracle_prices 
         WHERE market_id = (
           SELECT id FROM markets WHERE symbol = $1
         ) 
         ORDER BY created_at DESC LIMIT 1`,
        [marketSymbol]
      );

      return priceData[0]?.price || null;

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
      const priceHistory = await this.db.complexQuery(
        `SELECT price, created_at as timestamp FROM oracle_prices 
         WHERE market_id = (
           SELECT id FROM markets WHERE symbol = $1
         ) 
         AND created_at >= NOW() - INTERVAL '${hours} hours'
         ORDER BY created_at ASC`,
        [marketSymbol]
      );

      return priceHistory.map(row => ({
        price: parseFloat(row.price),
        timestamp: row.timestamp
      }));

    } catch (error) {
      this.logger.error(`Error getting price history for ${marketSymbol}:`, error)
      return []
    }
  }

  /**
   * Start price feed service (runs every 30 seconds)
   */
  public startPriceFeed(): void {
    this.logger.info('🚀 Starting CoinGecko price feed service...')
    
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
      
      this.logger.info('✅ CoinGecko price update completed successfully')
    } catch (error) {
      this.logger.error('❌ CoinGecko price update failed:', error)
    }
  }

  /**
   * Get all supported symbols
   */
  public getSupportedSymbols(): string[] {
    return Object.keys(this.MARKET_SYMBOLS)
  }

  /**
   * Test API connection
   */
  public async testConnection(): Promise<boolean> {
    try {
      const priceMap = await this.fetchLatestPrices()
      return priceMap.size > 0
    } catch (error) {
      this.logger.error('CoinGecko API connection test failed:', error)
      return false
    }
  }
}

export const coinGeckoService = CoinGeckoService.getInstance()
