import Redis from 'redis';
import { logger } from '../config';
import { 
  PriceData, 
  WalletData, 
  TransactionData, 
  SentimentData,
  LiquidationData,
  MarketAnalysis 
} from '../types';

/**
 * QuantDesk Data Bridge Service
 * Connects MIKEY-AI with QuantDesk data pipeline streams
 */
export class QuantDeskDataBridge {
  private redis: Redis.RedisClientType;
  private streams = {
    TICKS_RAW: 'ticks.raw',
    WHALES_RAW: 'whales.raw',
    NEWS_RAW: 'news.raw',
    TRENCH_RAW: 'trench.raw',
    DEFI_RAW: 'defi.raw',
    ANALYTICS_RAW: 'analytics.raw',
    MARKET_RAW: 'market.raw',
    PERPS_RAW: 'perps.raw',
    SIGNALS_RAW: 'signals.raw',
    ALERTS_RAW: 'alerts.raw',
    REPORTS_RAW: 'reports.raw'
  };

  constructor() {
    this.redis = Redis.createClient({
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379
    });
  }

  async connect(): Promise<void> {
    await this.redis.connect();
    logger.info('🔗 QuantDesk Data Bridge connected to Redis streams');
  }

  /**
   * Get real-time price data from QuantDesk pipeline
   */
  async getRealTimePrices(symbols: string[]): Promise<PriceData[]> {
    try {
      const messages = await this.redis.xRead(
        { key: this.streams.TICKS_RAW, id: '$' },
        { COUNT: 100, BLOCK: 1000 }
      );

      if (!messages || messages.length === 0) {
        return [];
      }

      const prices: PriceData[] = [];
      for (const message of messages[0].messages) {
        try {
          const data = JSON.parse(message.message);
          if (symbols.includes(data.symbol)) {
            prices.push({
              symbol: data.symbol,
              price: data.price,
              change24h: data.change_24h || 0,
              volume24h: data.volume_24h || 0,
              source: 'quantdesk',
              timestamp: new Date(data.timestamp),
              confidence: data.confidence || 1.0
            });
          }
        } catch (error) {
          logger.warn('Error parsing price message:', error);
        }
      }

      return prices;
    } catch (error) {
      logger.error('Error getting real-time prices:', error);
      return [];
    }
  }

  /**
   * Get whale movements from QuantDesk pipeline
   */
  async getWhaleMovements(): Promise<WalletData[]> {
    try {
      const messages = await this.redis.xRead(
        { key: this.streams.WHALES_RAW, id: '$' },
        { COUNT: 50, BLOCK: 1000 }
      );

      if (!messages || messages.length === 0) {
        return [];
      }

      const whales: WalletData[] = [];
      for (const message of messages[0].messages) {
        try {
          const data = JSON.parse(message.message);
          whales.push({
            address: data.wallet,
            label: data.label || `Whale ${data.wallet.slice(0, 8)}`,
            totalValue: data.amount,
            portfolio: {
              [data.token]: {
                amount: data.amount,
                value: data.amount * (data.price || 0),
                percentage: 100
              }
            },
            recentActivity: {
              transactions24h: 1,
              volume24h: data.amount,
              largestTransaction: data.amount
            }
          });
        } catch (error) {
          logger.warn('Error parsing whale message:', error);
        }
      }

      return whales;
    } catch (error) {
      logger.error('Error getting whale movements:', error);
      return [];
    }
  }

  /**
   * Get news sentiment from QuantDesk pipeline
   */
  async getNewsSentiment(symbol: string): Promise<SentimentData> {
    try {
      const messages = await this.redis.xRead(
        { key: this.streams.NEWS_RAW, id: '$' },
        { COUNT: 20, BLOCK: 1000 }
      );

      if (!messages || messages.length === 0) {
        return this.getDefaultSentiment(symbol);
      }

      let totalSentiment = 0;
      let count = 0;
      let socialVolume = 0;

      for (const message of messages[0].messages) {
        try {
          const data = JSON.parse(message.message);
          if (data.sentiment_score !== undefined) {
            totalSentiment += data.sentiment_score;
            count++;
          }
          if (data.social_volume) {
            socialVolume += data.social_volume;
          }
        } catch (error) {
          logger.warn('Error parsing sentiment message:', error);
        }
      }

      const avgSentiment = count > 0 ? totalSentiment / count : 0;
      
      return {
        symbol,
        sentimentScore: avgSentiment,
        sentimentLabel: this.getSentimentLabel(avgSentiment),
        socialVolume,
        newsSentiment: avgSentiment,
        twitterSentiment: avgSentiment,
        redditSentiment: avgSentiment,
        timestamp: new Date()
      };
    } catch (error) {
      logger.error('Error getting news sentiment:', error);
      return this.getDefaultSentiment(symbol);
    }
  }

  /**
   * Get liquidation data from QuantDesk pipeline
   */
  async getLiquidations(): Promise<LiquidationData[]> {
    try {
      const messages = await this.redis.xRead(
        { key: this.streams.PERPS_RAW, id: '$' },
        { COUNT: 50, BLOCK: 1000 }
      );

      if (!messages || messages.length === 0) {
        return [];
      }

      const liquidations: LiquidationData[] = [];
      for (const message of messages[0].messages) {
        try {
          const data = JSON.parse(message.message);
          if (data.type === 'liquidation') {
            liquidations.push({
              liquidationId: data.liquidation_id || `liq_${Date.now()}`,
              timestamp: new Date(data.timestamp),
              protocol: data.protocol || 'drift',
              walletAddress: data.wallet,
              positionType: data.side || 'long',
              size: data.amount || 0,
              collateral: data.collateral || 0,
              liquidationPrice: data.liquidation_price || 0,
              marketPrice: data.market_price || 0,
              pnl: data.pnl || 0
            });
          }
        } catch (error) {
          logger.warn('Error parsing liquidation message:', error);
        }
      }

      return liquidations;
    } catch (error) {
      logger.error('Error getting liquidations:', error);
      return [];
    }
  }

  /**
   * Get trading signals from QuantDesk Analytics Writer
   */
  async getTradingSignals(): Promise<any[]> {
    try {
      const messages = await this.redis.xRead(
        { key: this.streams.SIGNALS_RAW, id: '$' },
        { COUNT: 20, BLOCK: 1000 }
      );

      if (!messages || messages.length === 0) {
        return [];
      }

      const signals: any[] = [];
      for (const message of messages[0].messages) {
        try {
          const data = JSON.parse(message.message);
          signals.push(data);
        } catch (error) {
          logger.warn('Error parsing signal message:', error);
        }
      }

      return signals;
    } catch (error) {
      logger.error('Error getting trading signals:', error);
      return [];
    }
  }

  /**
   * Get DeFi protocol data from QuantDesk pipeline
   */
  async getDeFiData(): Promise<any[]> {
    try {
      const messages = await this.redis.xRead(
        { key: this.streams.DEFI_RAW, id: '$' },
        { COUNT: 50, BLOCK: 1000 }
      );

      if (!messages || messages.length === 0) {
        return [];
      }

      const defiData: any[] = [];
      for (const message of messages[0].messages) {
        try {
          const data = JSON.parse(message.message);
          defiData.push(data);
        } catch (error) {
          logger.warn('Error parsing DeFi message:', error);
        }
      }

      return defiData;
    } catch (error) {
      logger.error('Error getting DeFi data:', error);
      return [];
    }
  }

  /**
   * Get trench token data from QuantDesk pipeline
   */
  async getTrenchTokens(): Promise<any[]> {
    try {
      const messages = await this.redis.xRead(
        { key: this.streams.TRENCH_RAW, id: '$' },
        { COUNT: 20, BLOCK: 1000 }
      );

      if (!messages || messages.length === 0) {
        return [];
      }

      const trenchTokens: any[] = [];
      for (const message of messages[0].messages) {
        try {
          const data = JSON.parse(message.message);
          trenchTokens.push(data);
        } catch (error) {
          logger.warn('Error parsing trench token message:', error);
        }
      }

      return trenchTokens;
    } catch (error) {
      logger.error('Error getting trench tokens:', error);
      return [];
    }
  }

  private getSentimentLabel(score: number): 'bearish' | 'neutral' | 'bullish' {
    if (score > 0.2) return 'bullish';
    if (score < -0.2) return 'bearish';
    return 'neutral';
  }

  private getDefaultSentiment(symbol: string): SentimentData {
    return {
      symbol,
      sentimentScore: 0,
      sentimentLabel: 'neutral',
      socialVolume: 0,
      newsSentiment: 0,
      twitterSentiment: 0,
      redditSentiment: 0,
      timestamp: new Date()
    };
  }

  async disconnect(): Promise<void> {
    await this.redis.quit();
    logger.info('🔗 QuantDesk Data Bridge disconnected');
  }
}

export const quantDeskDataBridge = new QuantDeskDataBridge();
