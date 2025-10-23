const axios = require('axios');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');

class TrenchWatcher {
  constructor() {
    this.isRunning = false;
    this.coinGeckoApiKey = process.env.COINGECKO_API_KEY || '';
    this.twelveDataApiKey = process.env.TWELVEDATA_API_KEY || '';
    this.newTokens = new Map(); // Cache for new tokens
    this.deFiProtocols = new Set(); // Track DeFi protocols
    this.smartMoneyWallets = new Set(); // Known smart money wallets
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Trench watcher already running');
      return;
    }

    logger.info('🏴‍☠️ Starting Trench Watcher for DeFi tokens...');
    this.isRunning = true;

    // Load known smart money wallets
    await this.loadSmartMoneyWallets();

    // Start monitoring new tokens
    await this.startNewTokenMonitoring();

    // Start DeFi protocol monitoring
    await this.startDeFiMonitoring();

    // Start technical analysis
    await this.startTechnicalAnalysis();

    logger.info('🏴‍☠️ Trench Watcher started - monitoring DeFi trenches');
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping Trench Watcher...');
    this.isRunning = false;
    logger.info('Trench Watcher stopped');
  }

  async loadSmartMoneyWallets() {
    try {
      // Load from Redis cache
      const cachedWallets = await redis.smembers('smart_money_wallets');
      if (cachedWallets.length > 0) {
        cachedWallets.forEach(wallet => this.smartMoneyWallets.add(wallet));
        logger.info(`Loaded ${cachedWallets.length} smart money wallets from cache`);
        return;
      }

      // Add some known smart money wallets (you can expand this)
      const knownSmartMoney = [
        '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM', // Example smart money
        'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  // USDC mint
        'So11111111111111111111111111111111111111112',     // SOL mint
        // Add more known smart money addresses
      ];

      knownSmartMoney.forEach(wallet => {
        this.smartMoneyWallets.add(wallet);
      });

      // Cache in Redis
      if (this.smartMoneyWallets.size > 0) {
        await redis.sadd('smart_money_wallets', ...this.smartMoneyWallets);
        await redis.expire('smart_money_wallets', 3600); // Cache for 1 hour
      }

      logger.info(`Loaded ${this.smartMoneyWallets.size} smart money wallets`);
    } catch (error) {
      logger.error('Error loading smart money wallets:', error);
    }
  }

  async startNewTokenMonitoring() {
    try {
      // Monitor for new tokens every 2 minutes
      setInterval(async () => {
        try {
          await this.scanNewTokens();
        } catch (error) {
          logger.error('Error scanning new tokens:', error);
        }
      }, 120000); // 2 minutes

      logger.info('Started monitoring for new tokens');
    } catch (error) {
      logger.error('Error starting new token monitoring:', error);
    }
  }

  async scanNewTokens() {
    try {
      // Get trending coins from CoinGecko
      const trendingResponse = await axios.get('https://api.coingecko.com/api/v3/search/trending', {
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-TrenchWatcher/1.0'
        }
      });

      if (trendingResponse.data && trendingResponse.data.coins) {
        for (const coin of trendingResponse.data.coins) {
          await this.analyzeNewToken(coin);
        }
      }

      // Also check for new listings
      await this.checkNewListings();
    } catch (error) {
      logger.error('Error scanning new tokens:', error);
    }
  }

  async analyzeNewToken(coin) {
    try {
      const tokenId = coin.item.id;
      const symbol = coin.item.symbol;
      const name = coin.item.name;
      const marketCapRank = coin.item.market_cap_rank;

      // Skip if we've already analyzed this token recently
      if (this.newTokens.has(tokenId)) {
        const lastAnalyzed = this.newTokens.get(tokenId);
        if (Date.now() - lastAnalyzed < 300000) { // 5 minutes
          return;
        }
      }

      this.newTokens.set(tokenId, Date.now());

      // Get detailed token data
      const tokenData = await this.getTokenData(tokenId);
      if (!tokenData) return;

      // Analyze for smart money activity
      const analysis = await this.analyzeSmartMoneyActivity(tokenData);

      if (analysis.isSignificant) {
        await this.publishTrenchEvent({
          type: 'new_token_detected',
          token_id: tokenId,
          symbol: symbol,
          name: name,
          market_cap_rank: marketCapRank,
          price_usd: tokenData.current_price,
          market_cap: tokenData.market_cap,
          volume_24h: tokenData.total_volume,
          price_change_24h: tokenData.price_change_percentage_24h,
          analysis: analysis,
          timestamp: new Date().toISOString()
        });

        logger.info(`🏴‍☠️ New token detected: ${symbol} (${name}) - Market Cap: $${tokenData.market_cap?.toLocaleString() || 'N/A'}`);
      }
    } catch (error) {
      logger.error('Error analyzing new token:', error);
    }
  }

  async getTokenData(tokenId) {
    try {
      const response = await axios.get(`https://api.coingecko.com/api/v3/coins/${tokenId}`, {
        params: {
          localization: false,
          tickers: false,
          market_data: true,
          community_data: false,
          developer_data: false,
          sparkline: false
        },
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-TrenchWatcher/1.0'
        }
      });

      return response.data.market_data;
    } catch (error) {
      logger.error(`Error getting token data for ${tokenId}:`, error);
      return null;
    }
  }

  async analyzeSmartMoneyActivity(tokenData) {
    try {
      const analysis = {
        isSignificant: false,
        signals: [],
        riskScore: 0,
        opportunityScore: 0
      };

      // Check for significant volume spike
      if (tokenData.total_volume && tokenData.market_cap) {
        const volumeToMarketCapRatio = tokenData.total_volume / tokenData.market_cap;
        if (volumeToMarketCapRatio > 0.5) { // Volume > 50% of market cap
          analysis.signals.push('high_volume_spike');
          analysis.opportunityScore += 30;
        }
      }

      // Check for price momentum
      if (tokenData.price_change_percentage_24h) {
        const priceChange = Math.abs(tokenData.price_change_percentage_24h);
        if (priceChange > 20) {
          analysis.signals.push('high_price_volatility');
          analysis.opportunityScore += 20;
        }
      }

      // Check for low market cap (potential for growth)
      if (tokenData.market_cap && tokenData.market_cap < 10000000) { // < $10M
        analysis.signals.push('low_market_cap');
        analysis.opportunityScore += 25;
      }

      // Check for recent listing (new token)
      if (tokenData.market_cap_rank && tokenData.market_cap_rank > 1000) {
        analysis.signals.push('new_listing');
        analysis.opportunityScore += 15;
      }

      // Calculate risk score
      analysis.riskScore = Math.min(100, analysis.opportunityScore * 0.8); // Higher opportunity = higher risk

      // Determine if significant
      analysis.isSignificant = analysis.opportunityScore > 50 || analysis.signals.length >= 2;

      return analysis;
    } catch (error) {
      logger.error('Error analyzing smart money activity:', error);
      return { isSignificant: false, signals: [], riskScore: 0, opportunityScore: 0 };
    }
  }

  async checkNewListings() {
    try {
      // Check for new listings on major exchanges
      const response = await axios.get('https://api.coingecko.com/api/v3/coins/list', {
        params: {
          include_platform: true
        },
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-TrenchWatcher/1.0'
        }
      });

      // This would be more sophisticated in production
      // For now, we'll focus on trending coins
      logger.info(`Checked ${response.data.length} total coins for new listings`);
    } catch (error) {
      logger.error('Error checking new listings:', error);
    }
  }

  async startDeFiMonitoring() {
    try {
      // Monitor DeFi protocols every 5 minutes
      setInterval(async () => {
        try {
          await this.monitorDeFiProtocols();
        } catch (error) {
          logger.error('Error monitoring DeFi protocols:', error);
        }
      }, 300000); // 5 minutes

      logger.info('Started monitoring DeFi protocols');
    } catch (error) {
      logger.error('Error starting DeFi monitoring:', error);
    }
  }

  async monitorDeFiProtocols() {
    try {
      // Get DeFi token data
      const response = await axios.get('https://api.coingecko.com/api/v3/coins/markets', {
        params: {
          vs_currency: 'usd',
          category: 'decentralized-finance-defi',
          order: 'market_cap_desc',
          per_page: 50,
          page: 1,
          sparkline: false,
          price_change_percentage: '24h'
        },
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-TrenchWatcher/1.0'
        }
      });

      for (const token of response.data) {
        await this.analyzeDeFiToken(token);
      }

      logger.info(`Monitored ${response.data.length} DeFi protocols`);
    } catch (error) {
      logger.error('Error monitoring DeFi protocols:', error);
    }
  }

  async analyzeDeFiToken(token) {
    try {
      // Look for significant changes in DeFi tokens
      const priceChange = Math.abs(token.price_change_percentage_24h || 0);
      const volumeRatio = token.total_volume / token.market_cap;

      if (priceChange > 15 || volumeRatio > 0.3) {
        await this.publishTrenchEvent({
          type: 'defi_protocol_activity',
          token_id: token.id,
          symbol: token.symbol,
          name: token.name,
          price_usd: token.current_price,
          market_cap: token.market_cap,
          volume_24h: token.total_volume,
          price_change_24h: token.price_change_percentage_24h,
          volume_ratio: volumeRatio,
          timestamp: new Date().toISOString()
        });

        logger.info(`🏴‍☠️ DeFi activity: ${token.symbol} - ${priceChange.toFixed(1)}% change, ${(volumeRatio * 100).toFixed(1)}% volume ratio`);
      }
    } catch (error) {
      logger.error('Error analyzing DeFi token:', error);
    }
  }

  async startTechnicalAnalysis() {
    try {
      // Start technical analysis every 10 minutes
      setInterval(async () => {
        try {
          await this.performTechnicalAnalysis();
        } catch (error) {
          logger.error('Error performing technical analysis:', error);
        }
      }, 600000); // 10 minutes

      logger.info('Started technical analysis');
    } catch (error) {
      logger.error('Error starting technical analysis:', error);
    }
  }

  async performTechnicalAnalysis() {
    try {
      // Get top tokens for technical analysis
      const response = await axios.get('https://api.coingecko.com/api/v3/coins/markets', {
        params: {
          vs_currency: 'usd',
          order: 'market_cap_desc',
          per_page: 20,
          page: 1,
          sparkline: true,
          price_change_percentage: '1h,24h,7d'
        },
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-TrenchWatcher/1.0'
        }
      });

      for (const token of response.data) {
        await this.analyzeTechnicalIndicators(token);
      }

      logger.info(`Performed technical analysis on ${response.data.length} tokens`);
    } catch (error) {
      logger.error('Error performing technical analysis:', error);
    }
  }

  async analyzeTechnicalIndicators(token) {
    try {
      // Simple technical analysis
      const sparkline = token.sparkline_in_7d?.price || [];
      if (sparkline.length < 7) return;

      // Calculate simple moving average
      const sma7 = sparkline.reduce((sum, price) => sum + price, 0) / sparkline.length;
      const currentPrice = token.current_price;

      // Look for breakout patterns
      if (currentPrice > sma7 * 1.1) { // Price > 10% above 7-day SMA
        await this.publishTrenchEvent({
          type: 'technical_breakout',
          token_id: token.id,
          symbol: token.symbol,
          name: token.name,
          current_price: currentPrice,
          sma_7d: sma7,
          breakout_percentage: ((currentPrice - sma7) / sma7) * 100,
          volume_24h: token.total_volume,
          timestamp: new Date().toISOString()
        });

        logger.info(`🏴‍☠️ Technical breakout: ${token.symbol} - ${((currentPrice - sma7) / sma7 * 100).toFixed(1)}% above 7-day SMA`);
      }
    } catch (error) {
      logger.error('Error analyzing technical indicators:', error);
    }
  }

  async publishTrenchEvent(event) {
    try {
      await redis.xadd(
        STREAMS.TRENCH_RAW || 'trench.raw',
        'MAXLEN',
        '~',
        STREAM_CONFIG.maxLen,
        '*',
        'data', JSON.stringify(event),
        'event_type', event.type,
        'token_symbol', event.symbol || 'unknown',
        'timestamp', event.timestamp
      );

      logger.info(`Published trench event: ${event.type}`);
    } catch (error) {
      logger.error('Error publishing trench event:', error);
      throw error;
    }
  }

  async getTrenchStats() {
    try {
      const stats = {
        new_tokens_tracked: this.newTokens.size,
        smart_money_wallets: this.smartMoneyWallets.size,
        defi_protocols: this.deFiProtocols.size,
        is_running: this.isRunning
      };

      return stats;
    } catch (error) {
      logger.error('Error getting trench stats:', error);
      throw error;
    }
  }
}

// Start watcher if run directly
if (require.main === module) {
  const watcher = new TrenchWatcher();
  
  watcher.start().catch(error => {
    logger.error('Failed to start Trench Watcher:', error);
    process.exit(1);
  });

  // Graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Shutting down Trench Watcher...');
    await watcher.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down Trench Watcher...');
    await watcher.stop();
    process.exit(0);
  });
}

module.exports = TrenchWatcher;
