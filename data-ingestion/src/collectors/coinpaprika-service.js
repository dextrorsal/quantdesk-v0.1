const axios = require('axios');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');
require('dotenv').config();

class CoinPaprikaService {
  constructor() {
    this.baseUrl = 'https://api.coinpaprika.com/v1';
    this.isRunning = false;
    this.coins = new Map(); // Cache for supported coins
    this.markets = new Map(); // Cache for market data
    this.scanInterval = parseInt(process.env.COINPAPRIKA_SCAN_INTERVAL_MS || '300000'); // 5 minutes
  }

  async start() {
    if (this.isRunning) {
      logger.warn('CoinPaprika service already running');
      return;
    }

    logger.info('📊 Starting CoinPaprika service for market data...');
    this.isRunning = true;

    // Load supported coins
    await this.loadSupportedCoins();

    // Start monitoring market data
    await this.startMarketMonitoring();

    logger.info('📊 CoinPaprika service started - monitoring market data');
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping CoinPaprika service...');
    this.isRunning = false;
    logger.info('CoinPaprika service stopped');
  }

  async loadSupportedCoins() {
    try {
      const response = await axios.get(`${this.baseUrl}/coins`, {
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-CoinPaprika/1.0'
        },
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        // Filter for active coins only
        const activeCoins = response.data.filter(coin => coin.is_active);
        
        activeCoins.forEach(coin => {
          this.coins.set(coin.id, {
            id: coin.id,
            name: coin.name,
            symbol: coin.symbol,
            rank: coin.rank,
            is_new: coin.is_new,
            is_active: coin.is_active,
            type: coin.type
          });
        });

        logger.info(`📊 Loaded ${this.coins.size} active coins from CoinPaprika`);
      }

    } catch (error) {
      logger.error('Error loading CoinPaprika coins:', error);
    }
  }

  async startMarketMonitoring() {
    try {
      // Monitor market data every 5 minutes
      setInterval(async () => {
        try {
          await this.monitorMarketData();
        } catch (error) {
          logger.error('Error monitoring market data:', error);
        }
      }, this.scanInterval);

      // Initial scan
      await this.monitorMarketData();

      logger.info('Started monitoring market data');
    } catch (error) {
      logger.error('Error starting market monitoring:', error);
    }
  }

  async monitorMarketData() {
    try {
      logger.info('📊 Scanning market data for price signals...');

      // Get global market data
      await this.getGlobalMarketData();

      // Get top coins by market cap
      await this.getTopCoinsData();

      // Get trending coins
      await this.getTrendingCoins();

      logger.info('📊 Completed market data scan');
    } catch (error) {
      logger.error('Error monitoring market data:', error);
    }
  }

  async getGlobalMarketData() {
    try {
      const response = await axios.get(`${this.baseUrl}/global`, {
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-CoinPaprika/1.0'
        },
        timeout: 10000
      });

      if (response.data) {
        const globalData = response.data;
        
        const event = {
          type: 'coinpaprika_global_data',
          data: {
            market_cap_usd: globalData.market_cap_usd,
            volume_24h_usd: globalData.volume_24h_usd,
            bitcoin_dominance_percentage: globalData.bitcoin_dominance_percentage,
            cryptocurrencies_number: globalData.cryptocurrencies_number,
            market_cap_ath_value: globalData.market_cap_ath_value,
            market_cap_ath_date: globalData.market_cap_ath_date,
            volume_24h_ath_value: globalData.volume_24h_ath_value,
            volume_24h_ath_date: globalData.volume_24h_ath_date,
            market_cap_change_24h: globalData.market_cap_change_24h,
            volume_change_24h: globalData.volume_change_24h
          },
          timestamp: new Date().toISOString()
        };

        await this.publishCoinPaprikaEvent(event);
        logger.info(`📊 Global market cap: $${(globalData.market_cap_usd / 1e12).toFixed(2)}T, BTC dominance: ${globalData.bitcoin_dominance_percentage.toFixed(1)}%`);
      }

    } catch (error) {
      logger.error('Error getting global market data:', error);
    }
  }

  async getTopCoinsData() {
    try {
      // Get top coins by market cap using tickers endpoint
      const response = await axios.get(`${this.baseUrl}/tickers`, {
        params: {
          quotes: 'USD',
          limit: 50
        },
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-CoinPaprika/1.0'
        },
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        for (const coin of response.data) {
          await this.analyzeCoinData(coin);
        }

        logger.info(`📊 Analyzed ${response.data.length} top coins`);
      }

    } catch (error) {
      logger.error('Error getting top coins data:', error);
    }
  }

  async getTrendingCoins() {
    try {
      // Get coins with highest 24h change using tickers endpoint
      const response = await axios.get(`${this.baseUrl}/tickers`, {
        params: {
          quotes: 'USD',
          limit: 100 // Get more coins to find trending ones
        },
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-CoinPaprika/1.0'
        },
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        // Sort by 24h change and get top movers
        const sortedCoins = response.data
          .filter(coin => coin.quotes && coin.quotes.USD && coin.quotes.USD.percent_change_24h)
          .sort((a, b) => Math.abs(b.quotes.USD.percent_change_24h) - Math.abs(a.quotes.USD.percent_change_24h))
          .slice(0, 20);

        for (const coin of sortedCoins) {
          if (Math.abs(coin.quotes.USD.percent_change_24h) > 10) { // Only significant changes
            await this.analyzeTrendingCoin(coin);
          }
        }

        logger.info(`📊 Analyzed ${sortedCoins.length} trending coins`);
      }

    } catch (error) {
      logger.error('Error getting trending coins:', error);
    }
  }

  async analyzeCoinData(coin) {
    try {
      const analysis = this.analyzePriceSignals(coin);

      if (analysis.isSignificant) {
        const usdData = coin.quotes?.USD || {};
        const event = {
          type: 'coinpaprika_price_signal',
          coin_id: coin.id,
          symbol: coin.symbol,
          name: coin.name,
          data: {
            // Price Data
            price_usd: usdData.price,
            market_cap_usd: usdData.market_cap,
            volume_24h_usd: usdData.volume_24h,
            volume_7d_usd: usdData.volume_7d,
            volume_30d_usd: usdData.volume_30d,
            
            // Price Changes
            percent_change_1h: usdData.percent_change_1h,
            percent_change_24h: usdData.percent_change_24h,
            percent_change_7d: usdData.percent_change_7d,
            percent_change_30d: usdData.percent_change_30d,
            
            // Market Data
            market_cap_rank: usdData.market_cap_rank,
            market_cap_dominance: usdData.market_cap_dominance,
            fully_diluted_market_cap: usdData.fully_diluted_market_cap,
            
            // Circulating Supply
            circulating_supply: coin.circulating_supply,
            total_supply: coin.total_supply,
            max_supply: coin.max_supply,
            
            // Additional Metrics
            ath_price: usdData.ath_price,
            ath_date: usdData.ath_date,
            atl_price: usdData.atl_price,
            atl_date: usdData.atl_date,
            
            // Coin Metadata
            is_new: coin.is_new,
            is_active: coin.is_active,
            type: coin.type,
            rank: coin.rank
          },
          analysis: analysis,
          timestamp: new Date().toISOString()
        };

        await this.publishCoinPaprikaEvent(event);
        logger.info(`📊 Price signal: ${coin.symbol} - ${analysis.signal_type} (${usdData.percent_change_24h?.toFixed(2)}%)`);
      }

    } catch (error) {
      logger.error(`Error analyzing coin ${coin.symbol}:`, error);
    }
  }

  async analyzeTrendingCoin(coin) {
    try {
      const usdData = coin.quotes?.USD || {};
      const event = {
        type: 'coinpaprika_trending_coin',
        coin_id: coin.id,
        symbol: coin.symbol,
        name: coin.name,
        data: {
          // Price Data
          price_usd: usdData.price,
          market_cap_usd: usdData.market_cap,
          volume_24h_usd: usdData.volume_24h,
          volume_7d_usd: usdData.volume_7d,
          volume_30d_usd: usdData.volume_30d,
          
          // Price Changes
          percent_change_1h: usdData.percent_change_1h,
          percent_change_24h: usdData.percent_change_24h,
          percent_change_7d: usdData.percent_change_7d,
          percent_change_30d: usdData.percent_change_30d,
          
          // Market Data
          market_cap_rank: usdData.market_cap_rank,
          market_cap_dominance: usdData.market_cap_dominance,
          fully_diluted_market_cap: usdData.fully_diluted_market_cap,
          
          // Circulating Supply
          circulating_supply: coin.circulating_supply,
          total_supply: coin.total_supply,
          max_supply: coin.max_supply,
          
          // Additional Metrics
          ath_price: usdData.ath_price,
          ath_date: usdData.ath_date,
          atl_price: usdData.atl_price,
          atl_date: usdData.atl_date,
          
          // Coin Metadata
          is_new: coin.is_new,
          is_active: coin.is_active,
          type: coin.type,
          rank: coin.rank
        },
        timestamp: new Date().toISOString()
      };

      await this.publishCoinPaprikaEvent(event);
      logger.info(`📊 Trending coin: ${coin.symbol} - ${usdData.percent_change_24h?.toFixed(2)}% change`);

    } catch (error) {
      logger.error(`Error analyzing trending coin ${coin.symbol}:`, error);
    }
  }

  analyzePriceSignals(coin) {
    try {
      const analysis = {
        isSignificant: false,
        signal_type: 'neutral',
        confidence: 0,
        reasons: []
      };

      // Analyze 24h price change
      if (coin.percent_change_24h) {
        const change24h = Math.abs(coin.percent_change_24h);
        if (change24h > 15) {
          analysis.isSignificant = true;
          analysis.signal_type = coin.percent_change_24h > 0 ? 'price_surge' : 'price_crash';
          analysis.confidence += 0.4;
          analysis.reasons.push(`24h change: ${coin.percent_change_24h.toFixed(2)}%`);
        }
      }

      // Analyze volume spike
      if (coin.volume_24h && coin.market_cap) {
        const volumeRatio = coin.volume_24h / coin.market_cap;
        if (volumeRatio > 0.5) { // Volume > 50% of market cap
          analysis.isSignificant = true;
          analysis.signal_type = 'volume_spike';
          analysis.confidence += 0.3;
          analysis.reasons.push(`Volume spike: ${(volumeRatio * 100).toFixed(1)}% of market cap`);
        }
      }

      // Analyze market cap rank changes (if available)
      if (coin.market_cap_rank && coin.market_cap_rank <= 100) {
        analysis.isSignificant = true;
        analysis.signal_type = 'top_coin';
        analysis.confidence += 0.2;
        analysis.reasons.push(`Top ${coin.market_cap_rank} coin`);
      }

      return analysis;

    } catch (error) {
      logger.error(`Error analyzing price signals for ${coin.symbol}:`, error);
      return { isSignificant: false, signal_type: 'error', confidence: 0, reasons: [] };
    }
  }

  async publishCoinPaprikaEvent(event) {
    try {
      await redis.xadd(
        STREAMS.MARKET_RAW || 'market.raw',
        'MAXLEN',
        '~',
        STREAM_CONFIG.maxLen,
        '*',
        'data', JSON.stringify(event),
        'event_type', event.type,
        'symbol', event.symbol || 'unknown',
        'timestamp', event.timestamp
      );

      logger.info(`📊 Published CoinPaprika event: ${event.type}`);
    } catch (error) {
      logger.error('Error publishing CoinPaprika event:', error);
      throw error;
    }
  }

  async getCoinPaprikaStats() {
    try {
      const stats = {
        coins_tracked: this.coins.size,
        markets_cached: this.markets.size,
        is_running: this.isRunning
      };

      return stats;
    } catch (error) {
      logger.error('Error getting CoinPaprika stats:', error);
      throw error;
    }
  }
}

// Start service if run directly
if (require.main === module) {
  const service = new CoinPaprikaService();
  
  service.start().catch(error => {
    logger.error('Failed to start CoinPaprika service:', error);
    process.exit(1);
  });

  // Graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Shutting down CoinPaprika service...');
    await service.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down CoinPaprika service...');
    await service.stop();
    process.exit(0);
  });
}

module.exports = CoinPaprikaService;
