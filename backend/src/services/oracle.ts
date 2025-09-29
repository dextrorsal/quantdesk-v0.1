import axios from 'axios';
import { Connection, PublicKey } from '@solana/web3.js';
import { DatabaseService } from './database';
import { Logger } from '../utils/logger';
import { config } from '../config/environment';

const logger = new Logger();

export interface OraclePrice {
  symbol: string;
  price: number;
  confidence: number;
  exponent: number;
  timestamp: number;
  source: 'pyth' | 'switchboard';
}

export interface PythPriceData {
  id: string;
  price: {
    price: string;
    conf: string;
    expo: number;
    publish_time: number;
  };
}

export class OracleService {
  private static instance: OracleService;
  private connection: Connection;
  private db: DatabaseService;
  private priceCache: Map<string, OraclePrice> = new Map();
  private lastUpdate: Map<string, number> = new Map();
  private readonly CACHE_TTL = 5000; // 5 seconds

  private constructor() {
    this.connection = new Connection(config.RPC_URL, 'confirmed');
    this.db = DatabaseService.getInstance();
  }

  public static getInstance(): OracleService {
    if (!OracleService.instance) {
      OracleService.instance = new OracleService();
    }
    return OracleService.instance;
  }

  /**
   * Get current price for a symbol from Pyth Network
   */
  public async getPrice(symbol: string): Promise<OraclePrice | null> {
    try {
      // Check cache first
      const cached = this.priceCache.get(symbol);
      const lastUpdateTime = this.lastUpdate.get(symbol) || 0;
      
      if (cached && Date.now() - lastUpdateTime < this.CACHE_TTL) {
        return cached;
      }

      // Fetch from Pyth Network
      const priceData = await this.fetchPythPrice(symbol);
      if (!priceData) {
        logger.warn(`No price data found for ${symbol}`);
        return null;
      }

      const oraclePrice: OraclePrice = {
        symbol,
        price: parseFloat(priceData.price.price) * Math.pow(10, priceData.price.expo),
        confidence: parseFloat(priceData.price.conf) * Math.pow(10, priceData.price.expo),
        exponent: priceData.price.expo,
        timestamp: priceData.price.publish_time,
        source: 'pyth'
      };

      // Update cache
      this.priceCache.set(symbol, oraclePrice);
      this.lastUpdate.set(symbol, Date.now());

      // Store in database
      await this.storePriceInDatabase(symbol, oraclePrice);

      return oraclePrice;
    } catch (error) {
      logger.error(`Error fetching price for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Get prices for multiple symbols
   */
  public async getPrices(symbols: string[]): Promise<Map<string, OraclePrice>> {
    const prices = new Map<string, OraclePrice>();
    
    const promises = symbols.map(async (symbol) => {
      const price = await this.getPrice(symbol);
      if (price) {
        prices.set(symbol, price);
      }
    });

    await Promise.all(promises);
    return prices;
  }

  /**
   * Fetch price data from Pyth Network API
   */
  private async fetchPythPrice(symbol: string): Promise<PythPriceData | null> {
    try {
      const response = await axios.get(config.PYTH_NETWORK_URL, {
        params: {
          ids: this.getPythPriceFeedId(symbol)
        },
        timeout: 5000
      });

      if (response.data && response.data.length > 0) {
        return response.data[0];
      }

      return null;
    } catch (error) {
      logger.error(`Pyth API error for ${symbol}:`, error);
      return null;
    }
  }

  /**
   * Get Pyth price feed ID for a symbol
   */
  private getPythPriceFeedId(symbol: string): string {
    const feedIds: Record<string, string> = {
      'BTC': config.PYTH_PRICE_FEEDS.BTC,
      'ETH': config.PYTH_PRICE_FEEDS.ETH,
      'SOL': config.PYTH_PRICE_FEEDS.SOL,
    };

    return feedIds[symbol.toUpperCase()] || '';
  }

  /**
   * Store price data in database
   */
  private async storePriceInDatabase(symbol: string, price: OraclePrice): Promise<void> {
    try {
      // Get market ID
      const market = await this.db.getMarketBySymbol(`${symbol}-PERP`);
      if (!market) {
        logger.warn(`Market not found for symbol ${symbol}`);
        return;
      }

      // Store oracle price
      await this.db.query(
        `INSERT INTO oracle_prices (market_id, price, confidence, exponent) 
         VALUES ($1, $2, $3, $4)`,
        [market.id, price.price, price.confidence, price.exponent]
      );

      // Update market with latest oracle price
      await this.db.query(
        `UPDATE markets SET updated_at = NOW() WHERE id = $1`,
        [market.id]
      );

    } catch (error) {
      logger.error(`Error storing price in database for ${symbol}:`, error);
    }
  }

  /**
   * Validate price data quality
   */
  public validatePrice(price: OraclePrice): boolean {
    // Check if price is positive
    if (price.price <= 0) {
      return false;
    }

    // Check confidence interval (should be less than 1% of price)
    const confidenceRatio = price.confidence / price.price;
    if (confidenceRatio > 0.01) {
      logger.warn(`High confidence interval for ${price.symbol}: ${confidenceRatio * 100}%`);
      return false;
    }

    // Check if price is stale (older than 30 seconds)
    const age = Date.now() - price.timestamp * 1000;
    if (age > 30000) {
      logger.warn(`Stale price for ${price.symbol}: ${age}ms old`);
      return false;
    }

    return true;
  }

  /**
   * Get historical prices for a symbol
   */
  public async getHistoricalPrices(symbol: string, hours: number = 24): Promise<OraclePrice[]> {
    try {
      const market = await this.db.getMarketBySymbol(`${symbol}-PERP`);
      if (!market) {
        return [];
      }

      const result = await this.db.query(
        `SELECT price, confidence, exponent, created_at 
         FROM oracle_prices 
         WHERE market_id = $1 
         AND created_at >= NOW() - INTERVAL '${hours} hours'
         ORDER BY created_at DESC`,
        [market.id]
      );

      return result.rows.map(row => ({
        symbol,
        price: parseFloat(row.price),
        confidence: parseFloat(row.confidence),
        exponent: row.exponent,
        timestamp: new Date(row.created_at).getTime() / 1000,
        source: 'pyth' as const
      }));
    } catch (error) {
      logger.error(`Error fetching historical prices for ${symbol}:`, error);
      return [];
    }
  }

  /**
   * Calculate funding rate based on premium index
   */
  public async calculateFundingRate(symbol: string): Promise<number> {
    try {
      const market = await this.db.getMarketBySymbol(`${symbol}-PERP`);
      if (!market) {
        return 0;
      }

      // Get current oracle price
      const oraclePrice = await this.getPrice(symbol);
      if (!oraclePrice) {
        return 0;
      }

      // Get current mark price (simplified - in production, use order book)
      const markPrice = oraclePrice.price; // This should be calculated from order book

      // Calculate premium index
      const premiumIndex = ((markPrice - oraclePrice.price) / oraclePrice.price) * 10000;

      // Calculate funding rate (simplified formula)
      const fundingRate = premiumIndex + 100; // Base rate of 1%

      // Clamp funding rate to reasonable bounds
      return Math.max(-1000, Math.min(1000, fundingRate)); // ±10%
    } catch (error) {
      logger.error(`Error calculating funding rate for ${symbol}:`, error);
      return 0;
    }
  }

  /**
   * Start price update service
   */
  public startPriceUpdateService(): void {
    const symbols = ['BTC', 'ETH', 'SOL'];
    
    setInterval(async () => {
      try {
        await this.getPrices(symbols);
        logger.info('Oracle prices updated successfully');
      } catch (error) {
        logger.error('Error updating oracle prices:', error);
      }
    }, 10000); // Update every 10 seconds

    logger.info('Oracle price update service started');
  }

  /**
   * Health check for oracle service
   */
  public async healthCheck(): Promise<boolean> {
    try {
      const btcPrice = await this.getPrice('BTC');
      return btcPrice !== null && this.validatePrice(btcPrice);
    } catch (error) {
      logger.error('Oracle health check failed:', error);
      return false;
    }
  }
}
