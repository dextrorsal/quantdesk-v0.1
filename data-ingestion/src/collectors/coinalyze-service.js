const axios = require('axios');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');
require('dotenv').config();

class CoinalyzeService {
  constructor() {
    this.apiKey = process.env.COINALYZE_API_KEY;
    this.baseUrl = 'https://api.coinalyze.net/v1';
    this.isRunning = false;
    this.scanInterval = parseInt(process.env.COINALYZE_SCAN_INTERVAL_MS || '60000'); // 1 minute
    this.intervalId = null;
    this.monitoredSymbols = new Set([
      'BTCUSDT_PERP.A', // Binance BTC perp
      'ETHUSDT_PERP.A', // Binance ETH perp
      'SOLUSDT_PERP.A', // Binance SOL perp
      'BTCUSD_PERP.0',  // Bybit BTC perp
      'ETHUSD_PERP.0',  // Bybit ETH perp
      'SOLUSD_PERP.0'   // Bybit SOL perp
    ]);
    this.exchanges = new Map();
    this.futureMarkets = new Map();
    this.openInterestCache = new Map();
    this.fundingRateCache = new Map();

    if (!this.apiKey) {
      logger.warn('COINALYZE_API_KEY not found - service will run in limited mode');
    }
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Coinalyze service already running');
      return;
    }

    logger.info('📈 Starting Coinalyze service for perps analytics...');
    this.isRunning = true;

    await this.loadExchanges();
    await this.loadFutureMarkets();
    await this.monitorPerpsData();

    this.intervalId = setInterval(() => this.monitorPerpsData(), this.scanInterval);
    logger.info('📈 Coinalyze service started - monitoring perps markets');
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping Coinalyze service...');
    this.isRunning = false;
    if (this.intervalId) {
      clearInterval(this.intervalId);
    }
    logger.info('Coinalyze service stopped');
  }

  async loadExchanges() {
    try {
      const response = await axios.get(`${this.baseUrl}/exchanges`, {
        params: this.getParams(),
        headers: this.getHeaders(),
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        response.data.forEach(exchange => {
          this.exchanges.set(exchange.code, exchange);
        });
        logger.info(`📈 Loaded ${this.exchanges.size} exchanges from Coinalyze`);
      }
    } catch (error) {
      logger.error('Error loading exchanges:', error);
    }
  }

  async loadFutureMarkets() {
    try {
      const response = await axios.get(`${this.baseUrl}/future-markets`, {
        params: this.getParams(),
        headers: this.getHeaders(),
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        response.data.forEach(market => {
          this.futureMarkets.set(market.symbol, market);
        });
        logger.info(`📈 Loaded ${this.futureMarkets.size} future markets from Coinalyze`);
      }
    } catch (error) {
      logger.error('Error loading future markets:', error);
    }
  }

  async monitorPerpsData() {
    logger.info('📈 Scanning perps markets for whale activity...');
    
    await this.getCurrentOpenInterest();
    await this.getCurrentFundingRates();
    await this.getPredictedFundingRates();
    
    logger.info('📈 Completed perps market scan');
  }

  async getCurrentOpenInterest() {
    try {
      const symbols = Array.from(this.monitoredSymbols).join(',');
      const response = await axios.get(`${this.baseUrl}/open-interest`, {
        params: this.getParams({
          symbols: symbols,
          convert_to_usd: 'true'
        }),
        headers: this.getHeaders(),
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        for (const oiData of response.data) {
          const previousOI = this.openInterestCache.get(oiData.symbol);
          
          if (previousOI && oiData.value > 0) {
            const changePercent = ((oiData.value - previousOI) / previousOI) * 100;
            
            // Look for significant OI changes (>5%)
            if (Math.abs(changePercent) > 5) {
              await this.publishCoinalyzeEvent({
                type: 'open_interest_change',
                symbol: oiData.symbol,
                current_oi: oiData.value,
                previous_oi: previousOI,
                change_percent: changePercent,
                change_amount: oiData.value - previousOI,
                update_time: oiData.update,
                timestamp: new Date().toISOString(),
                source: 'coinalyze'
              });

              logger.info(`📈 OI Change: ${oiData.symbol} - ${changePercent.toFixed(1)}% (${changePercent > 0 ? '+' : ''}$${(oiData.value - previousOI).toLocaleString()})`);
            }
          }

          this.openInterestCache.set(oiData.symbol, oiData.value);
        }
      }
    } catch (error) {
      logger.error('Error getting open interest:', error);
    }
  }

  async getCurrentFundingRates() {
    try {
      const symbols = Array.from(this.monitoredSymbols).join(',');
      const response = await axios.get(`${this.baseUrl}/funding-rate`, {
        params: this.getParams({ symbols: symbols }),
        headers: this.getHeaders(),
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        for (const frData of response.data) {
          const previousFR = this.fundingRateCache.get(frData.symbol);
          
          if (previousFR !== undefined && frData.value !== undefined) {
            const changePercent = ((frData.value - previousFR) / Math.abs(previousFR)) * 100;
            
            // Look for significant funding rate changes (>10%)
            if (Math.abs(changePercent) > 10) {
              await this.publishCoinalyzeEvent({
                type: 'funding_rate_change',
                symbol: frData.symbol,
                current_funding_rate: frData.value,
                previous_funding_rate: previousFR,
                change_percent: changePercent,
                change_amount: frData.value - previousFR,
                update_time: frData.update,
                timestamp: new Date().toISOString(),
                source: 'coinalyze'
              });

              logger.info(`📈 Funding Rate Change: ${frData.symbol} - ${changePercent.toFixed(1)}% (${frData.value.toFixed(6)})`);
            }
          }

          this.fundingRateCache.set(frData.symbol, frData.value);
        }
      }
    } catch (error) {
      logger.error('Error getting funding rates:', error);
    }
  }

  async getPredictedFundingRates() {
    try {
      const symbols = Array.from(this.monitoredSymbols).join(',');
      const response = await axios.get(`${this.baseUrl}/predicted-funding-rate`, {
        params: this.getParams({ symbols: symbols }),
        headers: this.getHeaders(),
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        for (const pfrData of response.data) {
          await this.publishCoinalyzeEvent({
            type: 'predicted_funding_rate',
            symbol: pfrData.symbol,
            predicted_funding_rate: pfrData.value,
            update_time: pfrData.update,
            timestamp: new Date().toISOString(),
            source: 'coinalyze'
          });
        }
        logger.info(`📈 Published ${response.data.length} predicted funding rates`);
      }
    } catch (error) {
      logger.error('Error getting predicted funding rates:', error);
    }
  }

  async getLiquidationHistory(symbol, hours = 24) {
    try {
      const to = Math.floor(Date.now() / 1000);
      const from = to - (hours * 3600);
      
      const response = await axios.get(`${this.baseUrl}/liquidation-history`, {
        params: {
          symbols: symbol,
          interval: '1hour',
          from: from,
          to: to,
          convert_to_usd: 'true'
        },
        headers: this.getHeaders(),
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        for (const liqData of response.data) {
          if (liqData.history && Array.isArray(liqData.history)) {
            const totalLiquidations = liqData.history.reduce((sum, point) => sum + point.l + point.s, 0);
            
            if (totalLiquidations > 1000000) { // >$1M liquidations
              await this.publishCoinalyzeEvent({
                type: 'liquidation_spike',
                symbol: liqData.symbol,
                total_liquidations_usd: totalLiquidations,
                liquidation_data: liqData.history,
                time_range_hours: hours,
                timestamp: new Date().toISOString(),
                source: 'coinalyze'
              });

              logger.info(`📈 Liquidation Spike: ${liqData.symbol} - $${totalLiquidations.toLocaleString()} in ${hours}h`);
            }
          }
        }
      }
    } catch (error) {
      logger.error(`Error getting liquidation history for ${symbol}:`, error);
    }
  }

  async getLongShortRatio(symbol, hours = 24) {
    try {
      const to = Math.floor(Date.now() / 1000);
      const from = to - (hours * 3600);
      
      const response = await axios.get(`${this.baseUrl}/long-short-ratio-history`, {
        params: {
          symbols: symbol,
          interval: '1hour',
          from: from,
          to: to
        },
        headers: this.getHeaders(),
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        for (const ratioData of response.data) {
          if (ratioData.history && Array.isArray(ratioData.history)) {
            const latestRatio = ratioData.history[ratioData.history.length - 1];
            
            if (latestRatio) {
              await this.publishCoinalyzeEvent({
                type: 'long_short_ratio',
                symbol: ratioData.symbol,
                long_ratio: latestRatio.l,
                short_ratio: latestRatio.s,
                ratio: latestRatio.r,
                timestamp: new Date(latestRatio.t * 1000).toISOString(),
                source: 'coinalyze'
              });
            }
          }
        }
      }
    } catch (error) {
      logger.error(`Error getting long/short ratio for ${symbol}:`, error);
    }
  }

  getHeaders() {
    const headers = {
      'Accept': 'application/json'
    };
    
    return headers;
  }

  getParams(additionalParams = {}) {
    const params = {
      ...additionalParams
    };
    
    if (this.apiKey) {
      params['api_key'] = this.apiKey;
    }
    
    return params;
  }

  async publishCoinalyzeEvent(event) {
    try {
      await redis.xadd(
        STREAMS.PERPS_RAW || 'perps.raw',
        'MAXLEN',
        '~',
        STREAM_CONFIG.maxLen,
        '*',
        'data', JSON.stringify(event),
        'event_type', event.type,
        'symbol', event.symbol || 'unknown',
        'timestamp', event.timestamp
      );
      logger.info(`📈 Published Coinalyze event: ${event.type}`);
    } catch (error) {
      logger.error('Error publishing Coinalyze event:', error);
      throw error;
    }
  }

  async getCoinalyzeStats() {
    try {
      const stats = {
        exchanges_tracked: this.exchanges.size,
        future_markets_tracked: this.futureMarkets.size,
        symbols_monitored: this.monitoredSymbols.size,
        open_interest_cache_size: this.openInterestCache.size,
        funding_rate_cache_size: this.fundingRateCache.size
      };
      return stats;
    } catch (error) {
      logger.error('Error getting Coinalyze stats:', error);
      return {};
    }
  }
}

const coinalyzeService = new CoinalyzeService();

process.on('SIGINT', async () => {
  logger.info('SIGTERM received, shutting down Coinalyze service...');
  await coinalyzeService.stop();
  process.exit(0);
});

coinalyzeService.start();
