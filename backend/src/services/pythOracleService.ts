import axios from 'axios'
import WebSocket from 'ws'
import { HermesClient } from '@pythnetwork/hermes-client'
import { getSupabaseService } from './supabaseService'
import { Logger } from '../utils/logger'
import { performanceMonitoringService } from './performanceMonitoringService'
// Note: Feed IDs are sourced exclusively from environment variables

export interface PythPriceData {
  price: number
  confidence: number
  exponent: number
  publishTime: number
  timestamp: number
  symbol: string
  ema?: number | null
  feedId?: string | null
  change24h?: number | null
  volume24h?: number | null
  high24h?: number | null
  low24h?: number | null
}

class PythOracleService {
  private static instance: PythOracleService
  private db: ReturnType<typeof getSupabaseService>
  private logger: Logger
  private ws: WebSocket | null = null
  private isConnected = false
  private reconnectAttempts = 0
  private maxReconnectAttempts = 5
  private priceCache = new Map<string, PythPriceData>()
  private hermesClient: HermesClient
  private cacheTTL = 30000 // 30 seconds TTL for active feeds
  private staleThreshold = 60000 // 60 seconds staleness threshold
  private cacheStats = {
    hits: 0,
    misses: 0,
    stale: 0,
    errors: 0
  }
  
  // Enhanced caching with Redis-like functionality
  private redisCache = new Map<string, { data: PythPriceData; expiry: number }>()
  private batchRequestQueue = new Map<string, (value: number) => void>();
  private batchTimeout: NodeJS.Timeout | null = null
  private batchDelay = 100 // 100ms batch delay
  private maxBatchSize = 10
  private cacheTimeout = 30000 // 30 seconds cache timeout
  
  // Pyth Network WebSocket endpoint
  private readonly PYTH_WS_URL = 'wss://hermes.pyth.network/ws'
  
  // Pyth Network price feed IDs (verified working feed IDs)
  private PYTH_FEED_IDS: Record<string, string> = {
    BTC: 'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43', // BTC/USD ✅
    ETH: 'ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace', // ETH/USD ✅
    SOL: 'ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d', // SOL/USD ✅
    ADA: '2a01deaec9e51a579277b34b122399984d0bbf57e2458a7e42fecd2829867a0d', // ADA/USD ✅
    DOT: 'ca3eed9b267293f6595901c734c7525ce8ef49adafe8284606ceb307afa2ca5b', // DOT/USD ✅
    LINK: '8ac0c70fff57e9aefdf5edf44b51d62c2d433653cbb2cf5cc06bb115af04d221', // LINK/USD ✅
    // Solana Ecosystem Tokens - Using CoinGecko for now (add Pyth feeds when available)
    JUP: 'COINGECKO', // Jupiter Aggregator
    RAY: 'COINGECKO', // Raydium DEX
    JTO: 'COINGECKO', // Jito
    WIF: 'COINGECKO', // dogwifhat
    BONK: 'COINGECKO', // Bonk
    PYTH: 'COINGECKO', // Pyth Network Token
    ORCA: 'COINGECKO', // Orca DEX
    DRIFT: 'COINGECKO', // Drift Protocol
    FLOW: 'COINGECKO', // Flowfi
    MET: 'COINGECKO', // Met Token
    KMNO: 'COINGECKO' // Kamino
  }

  private readonly MARKET_SYMBOLS = {
    'BTC-PERP': 'BTC',
    'ETH-PERP': 'ETH', 
    'SOL-PERP': 'SOL',
    'ADA-PERP': 'ADA',
    'DOT-PERP': 'DOT',
    'LINK-PERP': 'LINK',
    // Solana Ecosystem
    'JUP-PERP': 'JUP',
    'RAY-PERP': 'RAY',
    'JTO-PERP': 'JTO',
    'WIF-PERP': 'WIF',
    'BONK-PERP': 'BONK',
    'PYTH-PERP': 'PYTH',
    'ORCA-PERP': 'ORCA',
    'DRIFT-PERP': 'DRIFT',
    'FLOW-PERP': 'FLOW',
    'MET-PERP': 'MET',
    'KMNO-PERP': 'KMNO'
  }

  private constructor() {
    this.db = getSupabaseService()
    this.logger = new Logger()
    this.hermesClient = new HermesClient('https://hermes.pyth.network')
    // Overlay feed IDs from environment if provided
    this.loadFeedIdsFromEnv()
  }

  public static getInstance(): PythOracleService {
    if (!PythOracleService.instance) {
      PythOracleService.instance = new PythOracleService()
    }
    return PythOracleService.instance
  }

  private loadFeedIdsFromEnv(): void {
    try {
      const env = process.env || {}
      // Accept vars: PYTH_PRICE_FEED_<SYMBOL>=<id>
      const prefix = 'PYTH_PRICE_FEED_'
      const hex64WithPrefix = /^0x[a-fA-F0-9]{64}$/
      const hex64WithoutPrefix = /^[a-fA-F0-9]{64}$/
      for (const [key, value] of Object.entries(env)) {
        if (!key.startsWith(prefix)) continue
        const symbol = key.substring(prefix.length).toUpperCase()
        const id = String(value || '').trim()
        if (!id) continue
        if (hex64WithPrefix.test(id) || hex64WithoutPrefix.test(id)) {
          // Strip 0x prefix for Hermes (expects hex without prefix)
          const cleanId = id.toLowerCase().startsWith('0x') ? id.toLowerCase().substring(2) : id.toLowerCase()
          this.PYTH_FEED_IDS[symbol] = cleanId
        } else {
          // Allow legacy Solana base58 for BTC/ETH/SOL; keep existing defaults if present
          // Log a soft warning so we can later add conversion if needed
          this.logger.warn(`PYTH feed for ${symbol} is not a hex feed id (got '${id}'). Using built-in/default until converted.`)
        }
      }
      this.logger.info(`Pyth feed catalog initialized for ${Object.keys(this.PYTH_FEED_IDS).length} symbols`)
    } catch (e: any) {
      this.logger.warn('Failed to load feed ids from env:', e?.message || e)
    }
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
    // Normalize feed ID (remove 0x prefix if present, lowercase)
    const normalized = (feedId || '').toLowerCase().replace(/^0x/, '')
    for (const [symbol, id] of Object.entries(this.PYTH_FEED_IDS)) {
      const normalizedId = (id || '').toLowerCase().replace(/^0x/, '')
      if (normalizedId === normalized) {
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
   * Get current price for a symbol from Pyth Network with cache-first architecture
   */
  public async getPrice(symbol: string): Promise<PythPriceData | null> {
    try {
      // Check cache first
      const cached = this.priceCache.get(symbol);
      if (cached && this.isCacheValid(cached)) {
        this.cacheStats.hits++;
        this.logger.debug(`💰 Cache hit for ${symbol}: $${cached.price.toFixed(2)}`);
        return cached;
      }

      // Cache miss or stale data
      if (cached) {
        this.cacheStats.stale++;
        this.logger.warn(`⚠️ Stale cache data for ${symbol}, age: ${this.getCacheAge(cached)}ms`);
      } else {
        this.cacheStats.misses++;
        this.logger.debug(`🔄 Cache miss for ${symbol}`);
      }

      // Fetch fresh data from Pyth
      const priceMap = await this.fetchLatestPrices();
      const freshData = priceMap.get(symbol);
      
      if (freshData) {
        // Update cache
        this.priceCache.set(symbol, freshData);
        this.logger.info(`✅ Fresh price for ${symbol}: $${freshData.price.toFixed(2)}`);
        return freshData;
      }

      // If Pyth fails, try database fallback
      this.logger.warn(`⚠️ Pyth failed for ${symbol}, trying database fallback`);
      return await this.getDatabaseFallback(symbol);
      
    } catch (error) {
      this.cacheStats.errors++;
      this.logger.error(`Error getting price for ${symbol}:`, error);
      
      // Try database fallback on error
      try {
        return await this.getDatabaseFallback(symbol);
      } catch (fallbackError) {
        this.logger.error(`Database fallback failed for ${symbol}:`, fallbackError);
        return null;
      }
    }
  }

  /**
   * Check if cached data is still valid
   */
  private isCacheValid(priceData: PythPriceData): boolean {
    const age = this.getCacheAge(priceData);
    return age < this.cacheTTL;
  }

  /**
   * Get cache age in milliseconds
   */
  private getCacheAge(priceData: PythPriceData): number {
    return Date.now() - priceData.timestamp;
  }

  /**
   * Get database fallback price
   */
  private async getDatabaseFallback(symbol: string): Promise<PythPriceData | null> {
    try {
      const marketSymbol = this.getMarketSymbolFromSymbol(symbol);
      if (!marketSymbol) return null;

      const market = await this.db.getClient()
        .from('markets')
        .select('id')
        .eq('symbol', marketSymbol)
        .single();

      if (!market.data) return null;

      const priceData = await this.db.getClient()
        .from('oracle_prices')
        .select('price, confidence, exponent, created_at')
        .eq('market_id', market.data.id)
        .order('created_at', { ascending: false })
        .limit(1)
        .single();

      if (priceData.data) {
        const fallbackData: PythPriceData = {
          price: priceData.data.price,
          confidence: priceData.data.confidence || priceData.data.price * 0.01,
          exponent: priceData.data.exponent || 0,
          publishTime: new Date(priceData.data.created_at).getTime(),
          timestamp: Date.now(),
          symbol
        };

        this.logger.info(`💾 Database fallback for ${symbol}: $${fallbackData.price.toFixed(2)}`);
        return fallbackData;
      }

      return null;
    } catch (error) {
      this.logger.error(`Database fallback error for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Get market symbol from asset symbol
   */
  private getMarketSymbolFromSymbol(symbol: string): string | null {
    for (const [marketSymbol, assetSymbol] of Object.entries(this.MARKET_SYMBOLS)) {
      if (assetSymbol === symbol) {
        return marketSymbol;
      }
    }
    return null;
  }

  /**
   * Fetch latest prices from Pyth Network using Hermes REST API
   */
  public async fetchLatestPrices(): Promise<Map<string, PythPriceData>> {
    this.logger.info('📊 Fetching latest prices from Pyth Network + CoinGecko...')
    
    try {
      const priceMap = new Map<string, PythPriceData>()
      
      // Step 1: Fetch Pyth prices for assets with Pyth feeds
      const pythFeedIds = Object.entries(this.PYTH_FEED_IDS)
        .filter(([_, feedId]) => feedId !== 'COINGECKO')
        .map(([_, feedId]) => feedId)
      
      if (pythFeedIds.length > 0) {
        try {
          const response = await this.hermesClient.getLatestPriceUpdates(pythFeedIds)
          
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
                
                // Check for staleness
                const ageSeconds = (Date.now() - publishTime * 1000) / 1000
                if (ageSeconds > this.staleThreshold / 1000) {
                  this.logger.warn(`⚠️ Stale price data for ${symbol}: ${ageSeconds.toFixed(1)}s old`)
                }
                
                priceMap.set(symbol, {
                  price: actualPrice,
                  confidence: actualConfidence,
                  exponent,
                  publishTime,
                  timestamp: Date.now(),
                  symbol,
                  ema: null,
                  feedId: feedId,
                  change24h: null,
                  volume24h: null,
                  high24h: null,
                  low24h: null
                })
                
                this.logger.info(`💰 Pyth ${symbol}: $${actualPrice.toFixed(2)} (confidence: ${actualConfidence.toFixed(4)}, age: ${ageSeconds.toFixed(1)}s)`)
              }
            }
          }
        } catch (pythError) {
          this.logger.warn('⚠️ Pyth fetch failed, will use CoinGecko:', pythError)
        }
      }
      
      // Step 2: Fetch CoinGecko prices for Solana ecosystem tokens
      const coingeckoSymbols = Object.entries(this.PYTH_FEED_IDS)
        .filter(([_, feedId]) => feedId === 'COINGECKO')
        .map(([symbol, _]) => symbol)
      
      if (coingeckoSymbols.length > 0) {
        this.logger.info(`📊 Fetching ${coingeckoSymbols.length} Solana ecosystem tokens from CoinGecko...`)
        const coinGeckoMap = await this.fetchFallbackPricesForSymbols(coingeckoSymbols)
        for (const [symbol, priceData] of coinGeckoMap) {
          priceMap.set(symbol, priceData)
        }
      }
      
      if (priceMap.size > 0) {
        // Update cache with fresh data
        for (const [symbol, priceData] of priceMap) {
          this.priceCache.set(symbol, priceData)
        }
        this.logger.info(`✅ Successfully fetched ${priceMap.size} prices from Pyth + CoinGecko`)
        return priceMap
      } else {
        this.logger.error('❌ No prices fetched from any source')
        throw new Error('No price data available')
      }
    } catch (error) {
      this.logger.error(`❌ Error fetching latest prices: ${error.message}`, error)
      throw error
    }
  }
  
  /**
   * Fetch CoinGecko prices for specific symbols
   */
  private async fetchFallbackPricesForSymbols(symbols: string[]): Promise<Map<string, PythPriceData>> {
    try {
      const priceMap = new Map<string, PythPriceData>()
      const coinIdMap: Record<string, string> = {
        JUP: 'jupiter-exchange-solana',
        RAY: 'raydium',
        JTO: 'jito-governance-token',
        WIF: 'dogwifcoin',
        BONK: 'bonk',
        PYTH: 'pyth-network',
        ORCA: 'orca',
        DRIFT: 'drift-protocol',
        FLOW: 'flowfi',
        MET: 'met-token',
        KMNO: 'kamino',
        FIDA: 'bonfida',
        FARTCOIN: 'fartcoin',
        POPCAT: 'popcat-wif-hat',
        MYRO: 'myro'
      }
      
      const coinIds = symbols
        .map(symbol => coinIdMap[symbol])
        .filter(Boolean)
        .join(',')
      
      if (coinIds.length === 0) return priceMap
      
      const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
        params: { 
          ids: coinIds, 
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
            confidence: price * 0.01,
            exponent: 0,
            publishTime: Date.now(),
            timestamp: Date.now(),
            symbol,
            ema: null,
            feedId: null,
            change24h: changePercent,
            volume24h: null,
            high24h: null,
            low24h: null
          }
          
          priceMap.set(symbol, priceData)
          this.logger.info(`💰 CoinGecko ${symbol}: $${price.toFixed(6)} (${changePercent.toFixed(2)}%)`)
        }
      }
      
      return priceMap
    } catch (error) {
      this.logger.error('Error fetching CoinGecko prices:', error)
      return new Map()
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
      
      // CoinGecko coin IDs mapping (expanded for Solana ecosystem)
      const coinIdMap: Record<string, string> = {
        BTC: 'bitcoin',
        ETH: 'ethereum', 
        SOL: 'solana',
        AVAX: 'avalanche-2',
        MATIC: 'matic-network',
        DOGE: 'dogecoin',
        ADA: 'cardano',
        DOT: 'polkadot',
        LINK: 'chainlink',
        // Solana Ecosystem
        JUP: 'jupiter-exchange-solana',
        RAY: 'raydium',
        JTO: 'jito-governance-token',
        WIF: 'dogwifcoin',
        BONK: 'bonk',
        MNGO: 'mango-markets',
        PYTH: 'pyth-network',
        ORCA: 'orca',
        SRM: 'serum',
        FIDA: 'bonfida',
        FARTCOIN: 'fartcoin',
        POPCAT: 'popcat-wif-hat',
        MYRO: 'myro',
        MEOW: 'meowcoin'
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

      // Notify portfolio service that prices have been updated
      this.notifyPortfolioServiceOfPriceUpdate();

    } catch (error) {
      this.logger.error('Error storing oracle price data:', error)
      throw error
    }
  }

  /**
   * Notify portfolio service that prices have been updated
   */
  private async notifyPortfolioServiceOfPriceUpdate(): Promise<void> {
    try {
      // Import portfolio service dynamically to avoid circular dependencies
      const { portfolioCalculationService } = await import('./portfolioCalculationService.js');
      
      // Invalidate all portfolio caches since prices have changed
      // This will force fresh calculations on next portfolio request
      await portfolioCalculationService.invalidateAllPortfolioCaches();
      
      this.logger.debug('📊 Notified portfolio service of price update');
    } catch (error) {
      this.logger.error('Error notifying portfolio service of price update:', error);
    }
  }

  // Redis-like cache methods
  private setRedisCache(symbol: string, priceData: PythPriceData): void {
    this.redisCache.set(symbol, {
      data: priceData,
      expiry: Date.now() + this.cacheTimeout
    });
  }

  private getRedisCache(symbol: string): PythPriceData | null {
    const cached = this.redisCache.get(symbol);
    if (!cached) return null;

    if (Date.now() > cached.expiry) {
      this.redisCache.delete(symbol);
      return null;
    }

    return cached.data;
  }

  private clearRedisCache(): void {
    this.redisCache.clear();
  }

  // Batch processing methods
  private async processBatchRequests(): Promise<void> {
    if (this.batchRequestQueue.size === 0) return;

    const symbols = Array.from(this.batchRequestQueue.keys());
    this.logger.info(`Processing batch request for ${symbols.length} symbols`);

    try {
      const prices = await this.getAllPrices();
      const batchResults = new Map<string, number>();

      for (const symbol of symbols) {
        const price = prices[symbol];
        if (price !== undefined) {
          batchResults.set(symbol, price);
        }
      }

      // Resolve all pending promises
      for (const [symbol, resolve] of this.batchRequestQueue) {
        const price = batchResults.get(symbol);
        if (price !== undefined) {
          resolve(price);
        } else {
          resolve(0); // Fallback to 0 for missing prices
        }
      }

      this.batchRequestQueue.clear();
    } catch (error) {
      this.logger.error('Batch processing failed:', error);
      // Reject all pending promises
      for (const [symbol, resolve] of this.batchRequestQueue) {
        resolve(0); // Fallback to 0 for errors
      }
      this.batchRequestQueue.clear();
    }
  }

  private scheduleBatchProcessing(): void {
    if (this.batchTimeout) {
      clearTimeout(this.batchTimeout);
    }

    this.batchTimeout = setTimeout(() => {
      this.processBatchRequests();
    }, this.batchDelay);
  }

  // Optimized batch price fetching
  async getBatchPrices(symbols: string[]): Promise<Record<string, number>> {
    const results: Record<string, number> = {};
    const uncachedSymbols: string[] = [];

    // Check cache first
    for (const symbol of symbols) {
      const cached = this.getRedisCache(symbol);
      if (cached) {
        results[symbol] = cached.price;
      } else {
        uncachedSymbols.push(symbol);
      }
    }

    // Fetch uncached symbols in batch
    if (uncachedSymbols.length > 0) {
      try {
        const allPrices = await this.getAllPrices();
        for (const symbol of uncachedSymbols) {
          const price = allPrices[symbol];
          if (price !== undefined) {
            results[symbol] = price;
            // Cache the result
            this.setRedisCache(symbol, {
              symbol,
              price,
              confidence: 0.95,
              exponent: -8,
              publishTime: Date.now(),
              timestamp: Date.now()
            });
          }
        }
      } catch (error) {
        this.logger.error('Batch price fetching failed:', error);
        // Fill missing prices with 0
        for (const symbol of uncachedSymbols) {
          if (!(symbol in results)) {
            results[symbol] = 0;
          }
        }
      }
    }

    return results;
  }

  /**
   * Get latest oracle price for a specific market with cache-first architecture
   */
  public async getLatestPrice(marketSymbol: string): Promise<number | null> {
    const startTime = Date.now();
    
    try {
      this.logger.debug(`🔍 Getting latest price for market: ${marketSymbol}`);
      const symbol = this.MARKET_SYMBOLS[marketSymbol as keyof typeof this.MARKET_SYMBOLS]
      this.logger.debug(`🔍 Mapped symbol: ${symbol}`);
      if (!symbol) {
        this.logger.warn(`❌ Invalid market symbol: ${marketSymbol}`);
        return null;
      }

      // Try Redis-like cache first
      const cached = this.getRedisCache(symbol);
      if (cached) {
        this.cacheStats.hits++;
        this.logger.debug(`💰 Cache hit for ${symbol}: $${cached.price.toFixed(2)}`);
        return cached.price;
      }

      // Try legacy cache
      const legacyCached = this.priceCache.get(symbol);
      if (legacyCached && this.isCacheValid(legacyCached)) {
        this.cacheStats.hits++;
        this.logger.debug(`💰 Legacy cache hit for ${symbol}: $${legacyCached.price.toFixed(2)}`);
        this.setRedisCache(symbol, legacyCached); // Migrate to Redis cache
        return legacyCached.price;
      }

      // Try fresh Pyth data
      try {
        const priceData = await this.getPrice(symbol);
        if (priceData) {
          this.logger.info(`✅ Fresh Pyth price for ${symbol}: $${priceData.price.toFixed(2)}`);
          this.setRedisCache(symbol, priceData); // Cache the fresh data
          return priceData.price;
        }
      } catch (pythError) {
        this.logger.warn(`⚠️ Pyth failed for ${symbol}, trying database fallback`);
      }

      // Database fallback
      const market = await this.db.getClient()
        .from('markets')
        .select('id')
        .eq('symbol', marketSymbol)
        .single()

      this.logger.debug(`🔍 Market lookup result:`, market);
      if (!market.data) return null

      const priceData = await this.db.getClient()
        .from('oracle_prices')
        .select('price')
        .eq('market_id', market.data.id)
        .order('created_at', { ascending: false })
        .limit(1)
        .single()

      this.logger.debug(`🔍 Price data result:`, priceData);
      if (priceData.data?.price) {
        this.logger.info(`💾 Database fallback price for ${symbol}: $${priceData.data.price}`);
        return priceData.data.price
      }

      this.logger.error(`❌ No price data available for ${symbol} from any source`);
      return null

    } catch (error) {
      this.cacheStats.errors++;
      this.logger.error(`Error getting latest oracle price for ${marketSymbol}:`, error)
      return null
    } finally {
      const duration = Date.now() - startTime;
      if (duration > 1000) { // Log slow oracle calls
        this.logger.warn(`🐌 Slow oracle call for ${marketSymbol}: ${duration}ms`);
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
    // Ensure env-based catalog is loaded after dotenv.config() in server startup
    this.loadFeedIdsFromEnv()
    
    // Warm cache with initial prices
    this.warmCache().then(() => {
      this.logger.info('✅ Cache warmed with initial prices')
    }).catch(error => {
      this.logger.error('❌ Cache warming failed:', error)
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
   * Warm cache with initial prices
   */
  private async warmCache(): Promise<void> {
    try {
      const priceMap = await this.fetchLatestPrices()
      if (priceMap.size > 0) {
        await this.storePriceData(priceMap)
        this.logger.info(`✅ Cache warmed with ${priceMap.size} prices`)
      }
    } catch (error) {
      this.logger.error('Cache warming failed:', error)
      throw error
    }
  }

  /**
   * Get cache statistics
   */
  public getCacheStats(): any {
    const total = this.cacheStats.hits + this.cacheStats.misses
    const hitRate = total > 0 ? (this.cacheStats.hits / total * 100).toFixed(2) : '0.00'
    
    return {
      ...this.cacheStats,
      hitRate: `${hitRate}%`,
      cacheSize: this.priceCache.size,
      staleThreshold: this.staleThreshold,
      cacheTTL: this.cacheTTL
    }
  }

  /**
   * Health check for oracle service
   */
  public async healthCheck(): Promise<{ status: string; details: any }> {
    try {
      const stats = this.getCacheStats()
      const connectionStatus = this.isConnected
      
      // Test Pyth connection
      const testPrices = await this.fetchLatestPrices()
      const pythWorking = testPrices.size > 0
      
      // Check for stale data
      let staleCount = 0
      for (const [symbol, priceData] of this.priceCache) {
        if (this.getCacheAge(priceData) > this.staleThreshold) {
          staleCount++
        }
      }
      
      const status = pythWorking && connectionStatus && staleCount === 0 ? 'healthy' : 'degraded'
      
      return {
        status,
        details: {
          pythConnection: connectionStatus,
          pythWorking,
          cacheStats: stats,
          staleCount,
          lastUpdate: this.priceCache.size > 0 ? Math.max(...Array.from(this.priceCache.values()).map(p => p.timestamp)) : null
        }
      }
    } catch (error) {
      return {
        status: 'unhealthy',
        details: {
          error: error.message,
          pythConnection: this.isConnected,
          cacheStats: this.getCacheStats()
        }
      }
    }
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

  /**
   * Get all prices (compatibility method for pythService)
   */
  /**
   * Get all current oracle prices from Pyth Network
   * @returns Promise<Record<string, number>> - Asset symbol to price mapping (e.g., {"BTC": 0.0012116391507573001})
   * @example
   * const prices = await pythOracleService.getAllPrices();
   * console.log(prices.BTC); // 0.0012116391507573001
   * console.log(prices.ETH); // 0.000043722750125000004
   */
  public async getAllPrices(): Promise<Record<string, number>> {
    try {
      const priceMap = await this.fetchLatestPrices()
      const result: Record<string, number> = {}
      
      for (const [symbol, priceData] of priceMap) {
        if (priceData && priceData.price) {
          // Price is already scaled with exponent in fetchLatestPrices() - don't apply it again!
          result[symbol] = parseFloat(priceData.price.toString())
        }
      }
      
      return result
    } catch (error) {
      this.logger.error('Error getting all prices:', error)
      return {}
    }
  }

  /**
   * Get asset price by asset name (compatibility method for pythService)
   */
  public async getAssetPrice(asset: string): Promise<number> {
    try {
      const symbol = asset.toUpperCase()
      const priceData = await this.getPrice(symbol)
      
      if (priceData && priceData.price) {
        // Price is already scaled with exponent - don't apply it again!
        return parseFloat(priceData.price.toString())
      }
      
      return 0
    } catch (error) {
      this.logger.error(`Error getting asset price for ${asset}:`, error)
      return 0
    }
  }

  /**
   * Get price confidence (compatibility method for pythService)
   */
  public async getPriceConfidence(asset: string): Promise<number> {
    try {
      const symbol = asset.toUpperCase()
      const priceData = await this.getPrice(symbol)
      
      if (priceData && priceData.price) {
        return parseFloat(priceData.confidence.toString()) * Math.pow(10, priceData.exponent || 0)
      }
      
      return 0
    } catch (error) {
      this.logger.error(`Error getting price confidence for ${asset}:`, error)
      return 0
    }
  }

  /**
   * Get latest prices in Pyth format (compatibility method for pythService)
   * @returns Promise<any> - Object with parsed array of price data
   * @example
   * const response = await pythOracleService.getLatestPrices();
   * console.log(response.parsed); // [{ id: 'BTC', price: 0.0012116391507573001, ema_price: 0.0012116391507573001 }]
   */
  public async getLatestPrices(): Promise<any> {
    try {
      const priceMap = await this.fetchLatestPrices()
      const parsed: any[] = []
      
      for (const [symbol, priceData] of priceMap) {
        if (priceData) {
          parsed.push({
            id: symbol,
            price: priceData.price,
            ema_price: priceData.price // Use same data for ema_price
          })
        }
      }
      
      return {
        parsed
      }
    } catch (error) {
      this.logger.error('Error getting latest prices:', error)
      return { parsed: [] }
    }
  }
}

export const pythOracleService = PythOracleService.getInstance()