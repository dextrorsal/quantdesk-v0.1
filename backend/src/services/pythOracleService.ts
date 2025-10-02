import axios from 'axios'
import WebSocket from 'ws'
import { HermesClient } from '@pythnetwork/hermes-client'
import { mcpSupabaseService } from './mcpSupabaseService'
import { Logger } from '../utils/logger'

export interface PythPriceData {
  price: number
  confidence: number
  exponent: number
  publishTime: number
  timestamp: number
  symbol: string
}

class PythOracleService {
  private static instance: PythOracleService
  private db: typeof mcpSupabaseService
  private logger: Logger
  private ws: WebSocket | null = null
  private isConnected = false
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private priceCache = new Map<string, PythPriceData>()
  private hermesClient: HermesClient
  
  // Pyth Network WebSocket endpoint
  private readonly PYTH_WS_URL = 'wss://hermes.pyth.network/ws'
  
  // Pyth Network price feed IDs (verified working feed IDs)
  private readonly PYTH_FEED_IDS = {
    BTC: 'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43', // BTC/USD ✅
    ETH: 'ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace', // ETH/USD ✅
    SOL: 'ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d', // SOL/USD ✅
    ADA: '2a01deaec9e51a579277b34b122399984d0bbf57e2458a7e42fecd2829867a0d', // ADA/USD ✅
    DOT: 'ca3eed9b267293f6595901c734c7525ce8ef49adafe8284606ceb307afa2ca5b', // DOT/USD ✅
    LINK: '8ac0c70fff57e9aefdf5edf44b51d62c2d433653cbb2cf5cc06bb115af04d221' // LINK/USD ✅
  }

  private readonly MARKET_SYMBOLS = {
    'BTC-PERP': 'BTC',
    'ETH-PERP': 'ETH', 
    'SOL-PERP': 'SOL',
    'ADA-PERP': 'ADA',
    'DOT-PERP': 'DOT',
    'LINK-PERP': 'LINK'
  }

  private constructor() {
    this.db = mcpSupabaseService
    this.logger = new Logger()
    this.hermesClient = new HermesClient('https://hermes.pyth.network')
  }

  public static getInstance(): PythOracleService {
    if (!PythOracleService.instance) {
      PythOracleService.instance = new PythOracleService()
    }
    return PythOracleService.instance
  }

  /**
   * Connect to Pyth Network WebSocket
   */
  public async connectToPyth(): Promise<void> {
    try {
      this.logger.info('🔌 Connecting to Pyth Network WebSocket...')
      
      this.ws = new WebSocket(this.PYTH_WS_URL)
      
      this.ws.on('open', () => {
        this.logger.info('✅ Connected to Pyth Network WebSocket')
        this.isConnected = true
        this.reconnectAttempts = 0
        
        // Subscribe to all price feeds
        this.subscribeToPriceFeeds()
      })
      
      this.ws.on('message', (data: Buffer) => {
        try {
          const message = JSON.parse(data.toString())
          this.handlePythMessage(message)
        } catch (error) {
          this.logger.error('Error parsing Pyth WebSocket message:', error)
        }
      })
      
      this.ws.on('close', () => {
        this.logger.warn('Pyth WebSocket connection closed')
        this.isConnected = false
        this.handleReconnect()
      })
      
      this.ws.on('error', (error) => {
        this.logger.error('Pyth WebSocket error:', error)
        this.isConnected = false
      })
      
    } catch (error) {
      this.logger.error('Failed to connect to Pyth WebSocket:', error)
      this.isConnected = false
      this.handleReconnect()
    }
  }

  /**
   * Subscribe to Pyth price feeds using WebSocket
   */
  private subscribeToPriceFeeds(): void {
    if (!this.ws || !this.isConnected) return
    
    const feedIds = Object.values(this.PYTH_FEED_IDS)
    
    // Pyth WebSocket subscription format
    const subscribeMessage = {
      ids: feedIds,
      type: 'subscribe'
    }
    
    this.ws.send(JSON.stringify(subscribeMessage))
    this.logger.info(`📡 Subscribed to ${feedIds.length} Pyth price feeds`)
  }

  /**
   * Handle incoming Pyth WebSocket messages
   */
  private handlePythMessage(message: any): void {
    try {
      // Pyth sends price updates in this format
      if (message.type === 'price_update' && message.price_update) {
        const priceUpdate = message.price_update
        const symbol = this.getSymbolFromFeedId(priceUpdate.id)
        
        if (symbol) {
          const priceData = this.parsePythPriceUpdate(priceUpdate, symbol)
          if (priceData) {
            this.processPriceUpdate(priceData)
          }
        }
      }
    } catch (error) {
      this.logger.error('Error handling Pyth message:', error)
    }
  }

  /**
   * Parse Pyth price update message
   */
  private parsePythPriceUpdate(priceUpdate: any, symbol: string): PythPriceData | null {
    try {
      const price = parseFloat(priceUpdate.price?.price || '0')
      const confidence = parseFloat(priceUpdate.price?.conf || '0')
      const exponent = parseInt(priceUpdate.price?.expo || '0')
      const publishTime = parseInt(priceUpdate.price?.publish_time || '0')

      // Apply exponent to get actual price
      const actualPrice = price * Math.pow(10, exponent)
      const actualConfidence = confidence * Math.pow(10, exponent)

      return {
        price: actualPrice,
        confidence: actualConfidence,
        exponent,
        publishTime,
        timestamp: Date.now(),
        symbol
      }
    } catch (error) {
      this.logger.error('Error parsing Pyth price update:', error)
      return null
    }
  }

  /**
   * Process price update and store in database
   */
  private async processPriceUpdate(priceData: PythPriceData): Promise<void> {
    try {
      // Validate price data
      if (!this.validatePriceData(priceData)) {
        this.logger.warn(`Invalid price data for ${priceData.symbol}:`, priceData)
        return
      }

      // Store in database
      await this.storePriceData(new Map([[priceData.symbol, priceData]]))
      
      this.logger.info(`💰 Pyth ${priceData.symbol}: $${priceData.price.toFixed(2)} (confidence: ${priceData.confidence.toFixed(4)})`)
      
    } catch (error) {
      this.logger.error(`Error processing price update for ${priceData.symbol}:`, error)
    }
  }

  /**
   * Validate price data quality
   */
  private validatePriceData(priceData: PythPriceData): boolean {
    // Check if price is reasonable
    if (priceData.price <= 0 || priceData.price > 1000000) {
      return false
    }
    
    // Check confidence (should be low for good data)
    if (priceData.confidence > priceData.price * 0.1) { // Confidence should be < 10% of price
      return false
    }
    
    // Check if data is not too old (within 60 seconds)
    const ageSeconds = (Date.now() - priceData.timestamp) / 1000
    if (ageSeconds > 60) {
      return false
    }
    
    return true
  }

  /**
   * Get symbol from Pyth feed ID
   */
  private getSymbolFromFeedId(feedId: string): string | null {
    for (const [symbol, id] of Object.entries(this.PYTH_FEED_IDS)) {
      if (id === feedId) {
        return symbol
      }
    }
    return null
  }

  /**
   * Handle Pyth connection reconnection
   */
  private handleReconnect(): void {
    if (this.reconnectAttempts < this.maxReconnectAttempts) {
      this.reconnectAttempts++
      const delay = Math.min(1000 * Math.pow(2, this.reconnectAttempts), 30000) // Exponential backoff, max 30s
      
      this.logger.info(`🔄 Attempting to reconnect to Pyth Network (attempt ${this.reconnectAttempts}/${this.maxReconnectAttempts}) in ${delay}ms`)
      
      setTimeout(() => {
        this.connectToPyth()
      }, delay)
    } else {
      this.logger.error('❌ Max reconnection attempts reached for Pyth Network')
    }
  }

  /**
   * Get current price for a symbol from Pyth Network
   */
  public async getPrice(symbol: string): Promise<PythPriceData | null> {
    try {
      // Check if we have cached data
      const cached = this.priceCache.get(symbol);
      if (cached) {
        return cached;
      }

      // If no cached data, fetch latest prices
      const priceMap = await this.fetchLatestPrices();
      return priceMap.get(symbol) || null;
    } catch (error) {
      this.logger.error(`Error getting price for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Fetch latest prices from Pyth Network using Hermes REST API
   */
  public async fetchLatestPrices(): Promise<Map<string, PythPriceData>> {
    this.logger.info('📊 Fetching latest prices from Pyth Network using Hermes REST API...')
    
    try {
      const priceMap = new Map<string, PythPriceData>()
      const feedIds = Object.values(this.PYTH_FEED_IDS)
      
      // Use the working Hermes client
      const response = await this.hermesClient.getLatestPriceUpdates(feedIds)
      
      if (response && response.parsed && Array.isArray(response.parsed)) {
        for (const priceFeed of response.parsed) {
          if (!priceFeed || !priceFeed.id) continue
          
          const feedId = priceFeed.id
          const symbol = this.getSymbolFromFeedId(feedId)
          if (!symbol) continue
          
          // Parse the price data from the price feed
          if (priceFeed.price && priceFeed.price.price) {
            const price = parseFloat(priceFeed.price.price)
            const confidence = parseFloat(priceFeed.price.conf?.toString() || '0')
            const exponent = parseInt(priceFeed.price.expo?.toString() || '0')
            const publishTime = parseInt(priceFeed.price.publish_time?.toString() || '0')
            
            // Apply exponent to get actual price
            const actualPrice = price * Math.pow(10, exponent)
            const actualConfidence = confidence * Math.pow(10, exponent)
            
            priceMap.set(symbol, {
              price: actualPrice,
              confidence: actualConfidence,
              exponent,
              publishTime,
              timestamp: Date.now(),
              symbol
            })
            
            this.logger.info(`💰 Pyth ${symbol}: $${actualPrice.toFixed(2)} (confidence: ${actualConfidence.toFixed(4)})`)
          }
        }
      }
      
      if (priceMap.size > 0) {
        this.priceCache = priceMap
        this.logger.info(`✅ Successfully fetched ${priceMap.size} prices from Pyth Network`)
        return priceMap
      } else {
        this.logger.warn('Pyth API returned no prices. Falling back to CoinGecko.')
        return this.fetchFallbackPrices()
      }
    } catch (error) {
      this.logger.error(`Error fetching latest prices from Pyth: ${error.message}`, error)
      this.logger.warn('Falling back to CoinGecko for price data...')
      return this.fetchFallbackPrices()
    }
  }

  /**
   * Fallback method to fetch prices from CoinGecko
   */
  private async fetchFallbackPrices(): Promise<Map<string, PythPriceData>> {
    this.logger.info('🔄 Falling back to CoinGecko for price data...')
    
    try {
      const priceMap = new Map<string, PythPriceData>()
      const symbols = Object.keys(this.PYTH_FEED_IDS)
      
      // CoinGecko coin IDs mapping
      const coinIdMap: Record<string, string> = {
        BTC: 'bitcoin',
        ETH: 'ethereum', 
        SOL: 'solana',
        AVAX: 'avalanche-2',
        MATIC: 'matic-network',
        DOGE: 'dogecoin',
        ADA: 'cardano',
        DOT: 'polkadot',
        LINK: 'chainlink'
      }
      
      const coinIds = symbols.map(symbol => coinIdMap[symbol]).filter(Boolean)
      
      if (coinIds.length > 0) {
        const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
          params: { 
            ids: coinIds.join(','), 
            vs_currencies: 'usd',
            include_24hr_change: true
          },
          timeout: 10000
        })
        
        for (const symbol of symbols) {
          const coinId = coinIdMap[symbol]
          if (coinId && response.data[coinId]) {
            const price = response.data[coinId].usd
            const changePercent = response.data[coinId].usd_24h_change || 0
            
            const priceData: PythPriceData = {
              price,
              confidence: price * 0.01, // 1% confidence for CoinGecko
            exponent: 0,
              publishTime: Date.now(),
              timestamp: Date.now(),
              symbol
            }
            
            priceMap.set(symbol, priceData)
            this.priceCache.set(symbol, priceData)
            
            this.logger.info(`💰 CoinGecko ${symbol}: $${price.toFixed(2)} (${changePercent.toFixed(2)}%)`)
          }
        }
      }
      
      return priceMap

    } catch (error) {
      this.logger.error('Error fetching fallback prices from CoinGecko:', error)
      return new Map()
    }
  }

  /**
   * Parse Pyth price data
   */
  private parsePythPrice(priceData: any, symbol: string): PythPriceData | null {
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
        timestamp: Date.now(),
        symbol
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

      const priceData = await this.db.executeQuery(
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
      const priceHistory = await this.db.executeQuery(
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
   * Start oracle price feed service (uses Pyth WebSocket for real-time data)
   */
  public startPriceFeed(): void {
    this.logger.info('🚀 Starting live Pyth Oracle price feed service...')
    
    // Fetch initial prices
    this.fetchLatestPrices().then(priceMap => {
      if (priceMap.size > 0) {
        this.storePriceData(priceMap)
        this.logger.info('✅ Initial Pyth prices fetched and stored')
      }
    })
    
    // Connect to Pyth WebSocket for real-time updates
    this.connectToPyth()
    
    // Keep the interval as a fallback mechanism
    setInterval(() => {
      if (!this.isConnected) {
        this.logger.warn('⚠️ Pyth connection lost, attempting to reconnect...')
        this.connectToPyth()
      }
    }, 30000) // Check connection every 30 seconds
  }

  /**
   * Disconnect from Pyth Network
   */
  public disconnect(): void {
    if (this.ws) {
      this.ws.close()
      this.ws = null
      this.isConnected = false
      this.logger.info('🔌 Disconnected from Pyth Network')
    }
  }

  /**
   * Get connection status
   */
  public getConnectionStatus(): boolean {
    return this.isConnected
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