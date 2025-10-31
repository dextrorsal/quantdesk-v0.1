import ccxt, { Exchange } from 'ccxt';
import { config } from '@/config';
import { systemLogger, errorLogger } from '@/utils/logger';
import { PriceData } from '@/types';

/**
 * CCXT Service for Centralized Exchange Market Intelligence
 * Pure data collection and analysis - NO TRADING LOGIC
 */

export interface CEXMarketData {
  exchange: string;
  symbol: string;
  price: number;
  volume24h: number;
  change24h: number;
  high24h: number;
  low24h: number;
  timestamp: Date;
  bid: number;
  ask: number;
  spread: number;
}

export interface CEXLiquidationData {
  exchange: string;
  symbol: string;
  side: 'long' | 'short';
  amount: number;
  price: number;
  timestamp: Date;
  liquidationId: string;
}

export interface CEXOrderBookData {
  exchange: string;
  symbol: string;
  bids: Array<[number, number]>; // [price, amount]
  asks: Array<[number, number]>;
  timestamp: Date;
  spread: number;
  midPrice: number;
}

export interface CEXFundingRateData {
  exchange: string;
  symbol: string;
  fundingRate: number;
  nextFundingTime: Date;
  timestamp: Date;
}

export interface CEXOpenInterestData {
  exchange: string;
  symbol: string;
  openInterest: number;
  timestamp: Date;
}

export class CCXTService {
  private exchanges: Map<string, Exchange> = new Map();
  private supportedExchanges: string[] = [
    'binance',
    'kraken', 
    'kucoin',
    'deribit',
    'bybit',
    'okx',
    'coinbase',
    'bitget',
    'mexc',
    'gate'
  ];

  constructor() {
    this.initializeExchanges();
  }

  /**
   * Initialize supported exchanges
   */
  private initializeExchanges(): void {
    for (const exchangeId of this.supportedExchanges) {
      try {
        const ExchangeClass = ccxt[exchangeId as keyof typeof ccxt] as any;
        if (ExchangeClass) {
          const exchange = new ExchangeClass({
            timeout: 10000,
            enableRateLimit: true,
            sandbox: false, // Use live data for market intelligence
            options: {
              defaultType: 'spot' // Start with spot markets
            }
          });

          this.exchanges.set(exchangeId, exchange);
          systemLogger.externalApiCall(exchangeId, 'initialization', true, 0);
        }
      } catch (error) {
        errorLogger.externalApiError(error as Error, exchangeId, 'initialization');
      }
    }

    systemLogger.startup('1.0.0', config.dev.nodeEnv);
  }

  /**
   * Get price for a specific symbol
   */
  async getPrice(symbol: string): Promise<PriceData> {
    try {
      const marketData = await this.getMarketData(symbol);
      if (marketData.length === 0) {
        throw new Error(`No price data available for ${symbol}`);
      }
      
      // Return the first available price data
      const firstData = marketData[0];
      if (!firstData) {
        throw new Error(`No price data available for ${symbol}`);
      }
      
      return {
        symbol: firstData.symbol,
        price: firstData.price,
        change24h: 0, // We'll calculate this later
        volume24h: firstData.volume24h || 0,
        volume: firstData.volume24h || 0,
        timestamp: new Date(),
        source: 'aggregated' as const
      };
    } catch (error) {
      errorLogger.externalApiError(error as Error, 'price_fetch', symbol);
      throw error;
    }
  }

  /**
   * Get market data from multiple exchanges
   */
  async getMarketData(symbol: string = 'SOL/USDT'): Promise<CEXMarketData[]> {
    const marketData: CEXMarketData[] = [];
    const promises: Promise<void>[] = [];

    for (const [exchangeId, exchange] of this.exchanges) {
      promises.push(
        this.fetchMarketDataFromExchange(exchangeId, exchange, symbol)
          .then(data => {
            if (data) marketData.push(data);
          })
          .catch(error => {
            errorLogger.externalApiError(error, exchangeId, 'market_data');
          })
      );
    }

    await Promise.allSettled(promises);
    return marketData;
  }

  /**
   * Fetch market data from a specific exchange
   */
  private async fetchMarketDataFromExchange(
    exchangeId: string, 
    exchange: Exchange, 
    symbol: string
  ): Promise<CEXMarketData | null> {
    try {
      const startTime = Date.now();
      
      // Check if exchange supports the symbol
      await exchange.loadMarkets();
      if (!exchange.markets[symbol]) {
        return null;
      }

      const ticker = await exchange.fetchTicker(symbol);
      const duration = Date.now() - startTime;

      systemLogger.externalApiCall(exchangeId, 'fetchTicker', true, duration);

      return {
        exchange: exchangeId,
        symbol,
        price: ticker.last || ticker.close || 0,
        volume24h: ticker.quoteVolume || 0,
        change24h: ticker.change || 0,
        high24h: ticker.high || 0,
        low24h: ticker.low || 0,
        timestamp: new Date(ticker.timestamp || Date.now()),
        bid: ticker.bid || 0,
        ask: ticker.ask || 0,
        spread: (ticker.ask || 0) - (ticker.bid || 0)
      };
    } catch (error) {
      errorLogger.externalApiError(error as Error, exchangeId, 'market_data');
      return null;
    }
  }

  /**
   * Get liquidations data from exchanges that support it
   */
  async getLiquidationsData(symbol: string = 'SOL/USDT'): Promise<CEXLiquidationData[]> {
    const liquidations: CEXLiquidationData[] = [];
    const promises: Promise<void>[] = [];

    // Only some exchanges support liquidations API
    const liquidationExchanges = ['binance', 'bybit', 'okx', 'deribit'];

    for (const exchangeId of liquidationExchanges) {
      const exchange = this.exchanges.get(exchangeId);
      if (exchange) {
        promises.push(
          this.fetchLiquidationsFromExchange(exchangeId, exchange, symbol)
            .then(data => {
              liquidations.push(...data);
            })
            .catch(error => {
              errorLogger.externalApiError(error, exchangeId, 'liquidations');
            })
        );
      }
    }

    await Promise.allSettled(promises);
    return liquidations;
  }

  /**
   * Fetch liquidations from a specific exchange
   */
  private async fetchLiquidationsFromExchange(
    exchangeId: string,
    exchange: Exchange,
    symbol: string
  ): Promise<CEXLiquidationData[]> {
    try {
      const startTime = Date.now();
      
      // Check if exchange supports liquidations
      if (!exchange.has['fetchLiquidations']) {
        return [];
      }

      const liquidations = await exchange.fetchLiquidations(symbol, undefined, 100);
      const duration = Date.now() - startTime;

      systemLogger.externalApiCall(exchangeId, 'fetchLiquidations', true, duration);

      return liquidations.map((liq: any) => ({
        exchange: exchangeId,
        symbol,
        side: liq.side === 'buy' ? 'long' : 'short',
        amount: liq.amount || 0,
        price: liq.price || 0,
        timestamp: new Date(liq.timestamp || Date.now()),
        liquidationId: liq.id || `${exchangeId}_${Date.now()}`
      }));
    } catch (error) {
      errorLogger.externalApiError(error, exchangeId, 'liquidations');
      return [];
    }
  }

  /**
   * Get order book data from multiple exchanges
   */
  async getOrderBookData(symbol: string = 'SOL/USDT'): Promise<CEXOrderBookData[]> {
    const orderBooks: CEXOrderBookData[] = [];
    const promises: Promise<void>[] = [];

    for (const [exchangeId, exchange] of this.exchanges) {
      promises.push(
        this.fetchOrderBookFromExchange(exchangeId, exchange, symbol)
          .then(data => {
            if (data) orderBooks.push(data);
          })
          .catch(error => {
            errorLogger.externalApiError(error, exchangeId, 'order_book');
          })
      );
    }

    await Promise.allSettled(promises);
    return orderBooks;
  }

  /**
   * Fetch order book from a specific exchange
   */
  private async fetchOrderBookFromExchange(
    exchangeId: string,
    exchange: Exchange,
    symbol: string
  ): Promise<CEXOrderBookData | null> {
    try {
      const startTime = Date.now();
      
      const orderBook = await exchange.fetchOrderBook(symbol, 20); // Top 20 levels
      const duration = Date.now() - startTime;

      systemLogger.externalApiCall(exchangeId, 'fetchOrderBook', true, duration);

      const bestBid = orderBook.bids[0]?.[0] || 0;
      const bestAsk = orderBook.asks[0]?.[0] || 0;
      const spread = bestAsk - bestBid;
      const midPrice = (bestBid + bestAsk) / 2;

      return {
        exchange: exchangeId,
        symbol,
        bids: orderBook.bids.slice(0, 10), // Top 10 bids
        asks: orderBook.asks.slice(0, 10), // Top 10 asks
        timestamp: new Date(orderBook.timestamp || Date.now()),
        spread,
        midPrice
      };
    } catch (error) {
      errorLogger.externalApiError(error, exchangeId, 'order_book');
      return null;
    }
  }

  /**
   * Get funding rates from perpetual exchanges
   */
  async getFundingRates(symbol: string = 'SOL/USDT'): Promise<CEXFundingRateData[]> {
    const fundingRates: CEXFundingRateData[] = [];
    const promises: Promise<void>[] = [];

    // Exchanges that support funding rates
    const fundingExchanges = ['binance', 'bybit', 'okx', 'deribit', 'bitget'];

    for (const exchangeId of fundingExchanges) {
      const exchange = this.exchanges.get(exchangeId);
      if (exchange) {
        promises.push(
          this.fetchFundingRateFromExchange(exchangeId, exchange, symbol)
            .then(data => {
              if (data) fundingRates.push(data);
            })
            .catch(error => {
              errorLogger.externalApiError(error, exchangeId, 'funding_rate');
            })
        );
      }
    }

    await Promise.allSettled(promises);
    return fundingRates;
  }

  /**
   * Fetch funding rate from a specific exchange
   */
  private async fetchFundingRateFromExchange(
    exchangeId: string,
    exchange: Exchange,
    symbol: string
  ): Promise<CEXFundingRateData | null> {
    try {
      const startTime = Date.now();
      
      // Check if exchange supports funding rates
      if (!exchange.has['fetchFundingRate']) {
        return null;
      }

      const fundingRate = await exchange.fetchFundingRate(symbol);
      const duration = Date.now() - startTime;

      systemLogger.externalApiCall(exchangeId, 'fetchFundingRate', true, duration);

      return {
        exchange: exchangeId,
        symbol,
        fundingRate: fundingRate.fundingRate || 0,
        nextFundingTime: new Date(fundingRate.nextFundingDatetime || Date.now()),
        timestamp: new Date(fundingRate.timestamp || Date.now())
      };
    } catch (error) {
      errorLogger.externalApiError(error, exchangeId, 'funding_rate');
      return null;
    }
  }

  /**
   * Get open interest data
   */
  async getOpenInterest(symbol: string = 'SOL/USDT'): Promise<CEXOpenInterestData[]> {
    const openInterest: CEXOpenInterestData[] = [];
    const promises: Promise<void>[] = [];

    // Exchanges that support open interest
    const oiExchanges = ['binance', 'bybit', 'okx', 'deribit'];

    for (const exchangeId of oiExchanges) {
      const exchange = this.exchanges.get(exchangeId);
      if (exchange) {
        promises.push(
          this.fetchOpenInterestFromExchange(exchangeId, exchange, symbol)
            .then(data => {
              if (data) openInterest.push(data);
            })
            .catch(error => {
              errorLogger.externalApiError(error, exchangeId, 'open_interest');
            })
        );
      }
    }

    await Promise.allSettled(promises);
    return openInterest;
  }

  /**
   * Fetch open interest from a specific exchange
   */
  private async fetchOpenInterestFromExchange(
    exchangeId: string,
    exchange: Exchange,
    symbol: string
  ): Promise<CEXOpenInterestData | null> {
    try {
      const startTime = Date.now();
      
      // Check if exchange supports open interest
      if (!exchange.has['fetchOpenInterest']) {
        return null;
      }

      const oi = await exchange.fetchOpenInterest(symbol);
      const duration = Date.now() - startTime;

      systemLogger.externalApiCall(exchangeId, 'fetchOpenInterest', true, duration);

      return {
        exchange: exchangeId,
        symbol,
        openInterest: oi.openInterestAmount || 0,
        timestamp: new Date(oi.timestamp || Date.now())
      };
    } catch (error) {
      errorLogger.externalApiError(error, exchangeId, 'open_interest');
      return null;
    }
  }

  /**
   * Get comprehensive market intelligence
   */
  async getMarketIntelligence(symbol: string = 'SOL/USDT'): Promise<{
    marketData: CEXMarketData[];
    liquidations: CEXLiquidationData[];
    orderBooks: CEXOrderBookData[];
    fundingRates: CEXFundingRateData[];
    openInterest: CEXOpenInterestData[];
    arbitrageOpportunities: any[];
  }> {
    console.log(`🔍 Gathering market intelligence for ${symbol}...`);

    const [marketData, liquidations, orderBooks, fundingRates, openInterest] = await Promise.all([
      this.getMarketData(symbol),
      this.getLiquidationsData(symbol),
      this.getOrderBookData(symbol),
      this.getFundingRates(symbol),
      this.getOpenInterest(symbol)
    ]);

    // Find arbitrage opportunities
    const arbitrageOpportunities = this.findArbitrageOpportunities(marketData);

    return {
      marketData,
      liquidations,
      orderBooks,
      fundingRates,
      openInterest,
      arbitrageOpportunities
    };
  }

  /**
   * Find arbitrage opportunities between exchanges
   */
  private findArbitrageOpportunities(marketData: CEXMarketData[]): any[] {
    const opportunities: any[] = [];
    
    for (let i = 0; i < marketData.length; i++) {
      for (let j = i + 1; j < marketData.length; j++) {
        const exchange1 = marketData[i];
        const exchange2 = marketData[j];
        
        const priceDiff = Math.abs(exchange1.price - exchange2.price);
        const avgPrice = (exchange1.price + exchange2.price) / 2;
        const spreadPercent = (priceDiff / avgPrice) * 100;
        
        // Arbitrage opportunity if spread > 0.1%
        if (spreadPercent > 0.1) {
          opportunities.push({
            exchange1: exchange1.exchange,
            exchange2: exchange2.exchange,
            price1: exchange1.price,
            price2: exchange2.price,
            spread: priceDiff,
            spreadPercent,
            opportunity: exchange1.price > exchange2.price ? 
              `Buy on ${exchange2.exchange}, sell on ${exchange1.exchange}` :
              `Buy on ${exchange1.exchange}, sell on ${exchange2.exchange}`
          });
        }
      }
    }
    
    return opportunities.sort((a, b) => b.spreadPercent - a.spreadPercent);
  }

  /**
   * Get supported exchanges
   */
  getSupportedExchanges(): string[] {
    return Array.from(this.exchanges.keys());
  }

  /**
   * Get exchange capabilities
   */
  getExchangeCapabilities(exchangeId: string): {
    liquidations: boolean;
    fundingRates: boolean;
    openInterest: boolean;
    orderBook: boolean;
  } {
    const exchange = this.exchanges.get(exchangeId);
    if (!exchange) {
      return { liquidations: false, fundingRates: false, openInterest: false, orderBook: false };
    }

    return {
      liquidations: Boolean(exchange.has['fetchLiquidations']),
      fundingRates: Boolean(exchange.has['fetchFundingRate']),
      openInterest: Boolean(exchange.has['fetchOpenInterest']),
      orderBook: Boolean(exchange.has['fetchOrderBook'])
    };
  }

  /**
   * Cleanup resources
   */
  async cleanup(): Promise<void> {
    try {
      // Close any active connections
      for (const [exchangeId, exchange] of this.exchanges) {
        if (exchange.close) {
          await exchange.close();
        }
      }
      
      this.exchanges.clear();
      systemLogger.shutdown('CCXT service cleanup');
    } catch (error) {
      errorLogger.externalApiError(error as Error, 'cleanup', 'service');
    }
  }
}

// Export singleton instance
export const ccxtService = new CCXTService();
