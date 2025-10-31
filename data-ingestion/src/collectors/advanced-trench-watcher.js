const axios = require('axios');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');

class AdvancedTrenchWatcher {
  constructor() {
    this.isRunning = false;
    this.birdeyeApiKey = process.env.BIRDEYE_API_KEY || '';
    this.gmgnApiKey = process.env.GMGN_API_KEY || '';
    this.coinGeckoApiKey = process.env.COINGECKO_API_KEY || '';
    
    this.newTokens = new Map(); // Cache for new tokens
    this.smartMoneyWallets = new Set(); // Known smart money wallets
    this.trackedTokens = new Set(); // Currently tracked tokens
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Advanced Trench Watcher already running');
      return;
    }

    logger.info('🏴‍☠️ Starting Advanced Trench Watcher for Solana trench coins...');
    this.isRunning = true;

    // Load known smart money wallets
    await this.loadSmartMoneyWallets();

    // Start monitoring new Solana tokens via Birdeye
    await this.startBirdeyeMonitoring();

    // Start GMGN monitoring for early signals
    await this.startGMGNMonitoring();

    // Start CoinGecko verification
    await this.startCoinGeckoVerification();

    logger.info('🏴‍☠️ Advanced Trench Watcher started - monitoring Solana trenches');
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping Advanced Trench Watcher...');
    this.isRunning = false;
    logger.info('Advanced Trench Watcher stopped');
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

      // Add known Solana smart money wallets
      const knownSmartMoney = [
        '9WzDXwBbmkg8ZTbNMqUxvQRAyrZzDsGYdLVL9zYtAWWM', // Example smart money
        'EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v',  // USDC mint
        'So11111111111111111111111111111111111111112',     // SOL mint
        // Add more known Solana smart money addresses
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

  async startBirdeyeMonitoring() {
    try {
      // Monitor new tokens via Birdeye every 1 minute
      setInterval(async () => {
        try {
          await this.scanNewTokensBirdeye();
        } catch (error) {
          logger.error('Error scanning new tokens via Birdeye:', error);
        }
      }, 60000); // 1 minute

      logger.info('Started Birdeye monitoring for new Solana tokens');
    } catch (error) {
      logger.error('Error starting Birdeye monitoring:', error);
    }
  }

  async scanNewTokensBirdeye() {
    try {
      // Get trending tokens from Birdeye - try different endpoint
      const response = await axios.get('https://public-api.birdeye.so/public/v1/tokenlist', {
        params: {
          sort_by: 'v24hUSD',
          sort_type: 'desc',
          offset: 0,
          limit: 20
        },
        headers: {
          'X-API-KEY': this.birdeyeApiKey,
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-TrenchWatcher/1.0'
        },
        timeout: 10000
      });

      if (response.data && response.data.data && response.data.data.tokens) {
        for (const token of response.data.data.tokens) {
          await this.analyzeBirdeyeToken(token);
        }
      }

      logger.info(`Scanned ${response.data?.data?.tokens?.length || 0} tokens via Birdeye`);
    } catch (error) {
      logger.error('Error scanning new tokens via Birdeye:', error);
    }
  }

  async analyzeBirdeyeToken(token) {
    try {
      const tokenAddress = token.address;
      const symbol = token.symbol;
      const name = token.name;
      const price = token.price;
      const marketCap = token.mc;
      const volume24h = token.v24h;
      const priceChange24h = token.priceChange24h;

      // Skip if we've already analyzed this token recently
      if (this.newTokens.has(tokenAddress)) {
        const lastAnalyzed = this.newTokens.get(tokenAddress);
        if (Date.now() - lastAnalyzed < 300000) { // 5 minutes
          return;
        }
      }

      this.newTokens.set(tokenAddress, Date.now());

      // Analyze for trench potential
      const analysis = await this.analyzeTrenchPotential({
        address: tokenAddress,
        symbol: symbol,
        name: name,
        price: price,
        marketCap: marketCap,
        volume24h: volume24h,
        priceChange24h: priceChange24h
      });

      if (analysis.isSignificant) {
        await this.publishTrenchEvent({
          type: 'trench_token_detected',
          token_address: tokenAddress,
          symbol: symbol,
          name: name,
          price_usd: price,
          market_cap: marketCap,
          volume_24h: volume24h,
          price_change_24h: priceChange24h,
          analysis: analysis,
          source: 'birdeye',
          timestamp: new Date().toISOString()
        });

        logger.info(`🏴‍☠️ Trench token detected: ${symbol} (${name}) - MC: $${marketCap?.toLocaleString() || 'N/A'}`);
      }
    } catch (error) {
      logger.error('Error analyzing Birdeye token:', error);
    }
  }

  async checkNewTokensBirdeye() {
    try {
      // Get recently added tokens from Birdeye
      const response = await axios.get('https://public-api.birdeye.so/public/v1/tokenlist', {
        params: {
          sort_by: 'v24hUSD',
          sort_type: 'desc',
          offset: 0,
          limit: 50
        },
        headers: {
          'X-API-KEY': this.birdeyeApiKey,
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-TrenchWatcher/1.0'
        },
        timeout: 10000
      });

      if (response.data && response.data.data && response.data.data.tokens) {
        for (const token of response.data.data.tokens.slice(0, 10)) { // Check top 10
          await this.analyzeBirdeyeToken(token);
        }
      }

      logger.info(`Checked ${response.data?.data?.tokens?.length || 0} tokens for new listings`);
    } catch (error) {
      logger.error('Error checking new tokens via Birdeye:', error);
    }
  }

  async startGMGNMonitoring() {
    try {
      // Monitor GMGN for early signals every 2 minutes
      setInterval(async () => {
        try {
          await this.scanGMGN();
        } catch (error) {
          logger.error('Error scanning GMGN:', error);
        }
      }, 120000); // 2 minutes

      logger.info('Started GMGN monitoring for early signals');
    } catch (error) {
      logger.error('Error starting GMGN monitoring:', error);
    }
  }

  async scanGMGN() {
    try {
      // GMGN public API for new token detection (no API key needed)
      // Try different endpoint or headers
      const response = await axios.get('https://gmgn.ai/defi/quotation/v1/tokens/sol/newest', {
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
          'Referer': 'https://gmgn.ai/',
          'Origin': 'https://gmgn.ai'
        },
        timeout: 10000
      });

      if (response.data && response.data.data) {
        for (const token of response.data.data.slice(0, 20)) { // Check newest 20
          await this.analyzeGMGNToken(token);
        }
      }

      logger.info(`Scanned ${response.data?.data?.length || 0} newest tokens via GMGN (no API key needed)`);
    } catch (error) {
      logger.error('Error scanning GMGN:', error);
      // Try alternative approach - use CoinGecko for now
      await this.fallbackToCoinGecko();
    }
  }

  async fallbackToCoinGecko() {
    try {
      // Fallback to CoinGecko for new token detection
      const response = await axios.get('https://api.coingecko.com/api/v3/coins/markets', {
        params: {
          vs_currency: 'usd',
          order: 'market_cap_asc', // Get smallest market caps first
          per_page: 20,
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

      if (response.data) {
        for (const token of response.data) {
          await this.analyzeCoinGeckoToken(token);
        }
      }

      logger.info(`Fallback: Scanned ${response.data?.length || 0} tokens via CoinGecko`);
    } catch (error) {
      logger.error('Error in CoinGecko fallback:', error);
    }
  }

  async analyzeCoinGeckoToken(token) {
    try {
      const tokenId = token.id;
      const symbol = token.symbol;
      const name = token.name;
      const price = token.current_price;
      const marketCap = token.market_cap;
      const volume24h = token.total_volume;
      const priceChange24h = token.price_change_percentage_24h;

      // Skip if we've already analyzed this token recently
      if (this.newTokens.has(tokenId)) {
        return;
      }

      this.newTokens.set(tokenId, Date.now());

      // Analyze for trench potential
      const analysis = await this.analyzeTrenchPotential({
        address: tokenId,
        symbol: symbol,
        name: name,
        price: price,
        marketCap: marketCap,
        volume24h: volume24h,
        priceChange24h: priceChange24h
      });

      if (analysis.isSignificant) {
        await this.publishTrenchEvent({
          type: 'trench_token_detected',
          token_address: tokenId,
          symbol: symbol,
          name: name,
          price_usd: price,
          market_cap: marketCap,
          volume_24h: volume24h,
          price_change_24h: priceChange24h,
          analysis: analysis,
          source: 'coingecko_fallback',
          timestamp: new Date().toISOString()
        });

        logger.info(`🏴‍☠️ Trench token detected (CoinGecko): ${symbol} (${name}) - MC: $${marketCap?.toLocaleString() || 'N/A'}`);
      }
    } catch (error) {
      logger.error('Error analyzing CoinGecko token:', error);
    }
  }

  async analyzeGMGNToken(token) {
    try {
      const tokenAddress = token.address;
      const symbol = token.symbol;
      const name = token.name;
      const price = token.price;
      const marketCap = token.mc;
      const volume24h = token.v24h;
      const holderCount = token.holder_count;
      const liquidity = token.liquidity;

      // Skip if already analyzed recently
      if (this.newTokens.has(tokenAddress)) {
        return;
      }

      this.newTokens.set(tokenAddress, Date.now());

      // Analyze for early signals
      const analysis = await this.analyzeEarlySignals({
        address: tokenAddress,
        symbol: symbol,
        name: name,
        price: price,
        marketCap: marketCap,
        volume24h: volume24h,
        holderCount: holderCount,
        liquidity: liquidity
      });

      if (analysis.isSignificant) {
        await this.publishTrenchEvent({
          type: 'early_signal_detected',
          token_address: tokenAddress,
          symbol: symbol,
          name: name,
          price_usd: price,
          market_cap: marketCap,
          volume_24h: volume24h,
          holder_count: holderCount,
          liquidity: liquidity,
          analysis: analysis,
          source: 'gmgn',
          timestamp: new Date().toISOString()
        });

        logger.info(`🏴‍☠️ Early signal detected: ${symbol} (${name}) - Holders: ${holderCount}, Liquidity: $${liquidity?.toLocaleString() || 'N/A'}`);
      }
    } catch (error) {
      logger.error('Error analyzing GMGN token:', error);
    }
  }

  async startCoinGeckoVerification() {
    try {
      // Verify tokens via CoinGecko every 5 minutes
      setInterval(async () => {
        try {
          await this.verifyTokensCoinGecko();
        } catch (error) {
          logger.error('Error verifying tokens via CoinGecko:', error);
        }
      }, 300000); // 5 minutes

      logger.info('Started CoinGecko verification for tracked tokens');
    } catch (error) {
      logger.error('Error starting CoinGecko verification:', error);
    }
  }

  async verifyTokensCoinGecko() {
    try {
      // Get trending coins from CoinGecko for verification
      const response = await axios.get('https://api.coingecko.com/api/v3/search/trending', {
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-TrenchWatcher/1.0'
        }
      });

      if (response.data && response.data.coins) {
        for (const coin of response.data.coins) {
          await this.verifyTokenCoinGecko(coin);
        }
      }

      logger.info(`Verified ${response.data?.coins?.length || 0} tokens via CoinGecko`);
    } catch (error) {
      logger.error('Error verifying tokens via CoinGecko:', error);
    }
  }

  async verifyTokenCoinGecko(coin) {
    try {
      const tokenId = coin.item.id;
      const symbol = coin.item.symbol;
      const name = coin.item.name;
      const marketCapRank = coin.item.market_cap_rank;

      // Check if this token is already tracked
      if (this.trackedTokens.has(tokenId)) {
        return;
      }

      // Get detailed data
      const tokenData = await this.getTokenDataCoinGecko(tokenId);
      if (!tokenData) return;

      // Add to tracked tokens
      this.trackedTokens.add(tokenId);

      // Analyze for verification signals
      const analysis = await this.analyzeVerificationSignals(tokenData);

      if (analysis.isSignificant) {
        await this.publishTrenchEvent({
          type: 'token_verification',
          token_id: tokenId,
          symbol: symbol,
          name: name,
          market_cap_rank: marketCapRank,
          price_usd: tokenData.current_price,
          market_cap: tokenData.market_cap,
          volume_24h: tokenData.total_volume,
          price_change_24h: tokenData.price_change_percentage_24h,
          analysis: analysis,
          source: 'coingecko',
          timestamp: new Date().toISOString()
        });

        logger.info(`🏴‍☠️ Token verification: ${symbol} (${name}) - Rank: ${marketCapRank}`);
      }
    } catch (error) {
      logger.error('Error verifying token via CoinGecko:', error);
    }
  }

  async getTokenDataCoinGecko(tokenId) {
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

  async analyzeTrenchPotential(tokenData) {
    try {
      const analysis = {
        isSignificant: false,
        signals: [],
        riskScore: 0,
        opportunityScore: 0,
        trenchScore: 0
      };

      // Check for low market cap (trench potential)
      if (tokenData.marketCap && tokenData.marketCap < 1000000) { // < $1M
        analysis.signals.push('ultra_low_market_cap');
        analysis.trenchScore += 40;
        analysis.opportunityScore += 35;
      } else if (tokenData.marketCap && tokenData.marketCap < 10000000) { // < $10M
        analysis.signals.push('low_market_cap');
        analysis.trenchScore += 25;
        analysis.opportunityScore += 20;
      }

      // Check for high volume spike
      if (tokenData.volume24h && tokenData.marketCap) {
        const volumeToMarketCapRatio = tokenData.volume24h / tokenData.marketCap;
        if (volumeToMarketCapRatio > 1.0) { // Volume > 100% of market cap
          analysis.signals.push('extreme_volume_spike');
          analysis.trenchScore += 30;
          analysis.opportunityScore += 25;
        } else if (volumeToMarketCapRatio > 0.5) { // Volume > 50% of market cap
          analysis.signals.push('high_volume_spike');
          analysis.trenchScore += 20;
          analysis.opportunityScore += 15;
        }
      }

      // Check for price momentum
      if (tokenData.priceChange24h) {
        const priceChange = Math.abs(tokenData.priceChange24h);
        if (priceChange > 50) {
          analysis.signals.push('extreme_price_volatility');
          analysis.trenchScore += 25;
          analysis.opportunityScore += 20;
        } else if (priceChange > 20) {
          analysis.signals.push('high_price_volatility');
          analysis.trenchScore += 15;
          analysis.opportunityScore += 10;
        }
      }

      // Calculate risk score (higher trench score = higher risk)
      analysis.riskScore = Math.min(100, analysis.trenchScore * 0.9);

      // Determine if significant
      analysis.isSignificant = analysis.trenchScore > 50 || analysis.signals.length >= 2;

      return analysis;
    } catch (error) {
      logger.error('Error analyzing trench potential:', error);
      return { isSignificant: false, signals: [], riskScore: 0, opportunityScore: 0, trenchScore: 0 };
    }
  }

  async analyzeEarlySignals(tokenData) {
    try {
      const analysis = {
        isSignificant: false,
        signals: [],
        riskScore: 0,
        opportunityScore: 0,
        earlyScore: 0
      };

      // Check for low holder count (early stage)
      if (tokenData.holderCount && tokenData.holderCount < 100) {
        analysis.signals.push('ultra_early_stage');
        analysis.earlyScore += 35;
        analysis.opportunityScore += 30;
      } else if (tokenData.holderCount && tokenData.holderCount < 1000) {
        analysis.signals.push('early_stage');
        analysis.earlyScore += 25;
        analysis.opportunityScore += 20;
      }

      // Check for low liquidity (high risk/reward)
      if (tokenData.liquidity && tokenData.liquidity < 10000) { // < $10k liquidity
        analysis.signals.push('low_liquidity');
        analysis.earlyScore += 30;
        analysis.opportunityScore += 25;
      }

      // Check for recent creation (new token)
      if (tokenData.marketCap && tokenData.marketCap < 100000) { // < $100k market cap
        analysis.signals.push('brand_new_token');
        analysis.earlyScore += 25;
        analysis.opportunityScore += 20;
      }

      // Calculate risk score
      analysis.riskScore = Math.min(100, analysis.earlyScore * 0.8);

      // Determine if significant
      analysis.isSignificant = analysis.earlyScore > 60 || analysis.signals.length >= 2;

      return analysis;
    } catch (error) {
      logger.error('Error analyzing early signals:', error);
      return { isSignificant: false, signals: [], riskScore: 0, opportunityScore: 0, earlyScore: 0 };
    }
  }

  async analyzeVerificationSignals(tokenData) {
    try {
      const analysis = {
        isSignificant: false,
        signals: [],
        riskScore: 0,
        opportunityScore: 0,
        verificationScore: 0
      };

      // Check for trending status
      analysis.signals.push('trending_on_coingecko');
      analysis.verificationScore += 20;
      analysis.opportunityScore += 15;

      // Check for significant volume
      if (tokenData.total_volume && tokenData.market_cap) {
        const volumeRatio = tokenData.total_volume / tokenData.market_cap;
        if (volumeRatio > 0.3) {
          analysis.signals.push('high_volume_ratio');
          analysis.verificationScore += 25;
          analysis.opportunityScore += 20;
        }
      }

      // Check for price momentum
      if (tokenData.price_change_percentage_24h) {
        const priceChange = Math.abs(tokenData.price_change_percentage_24h);
        if (priceChange > 15) {
          analysis.signals.push('strong_price_momentum');
          analysis.verificationScore += 20;
          analysis.opportunityScore += 15;
        }
      }

      // Calculate risk score
      analysis.riskScore = Math.min(100, analysis.verificationScore * 0.7);

      // Determine if significant
      analysis.isSignificant = analysis.verificationScore > 40 || analysis.signals.length >= 2;

      return analysis;
    } catch (error) {
      logger.error('Error analyzing verification signals:', error);
      return { isSignificant: false, signals: [], riskScore: 0, opportunityScore: 0, verificationScore: 0 };
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
        'source', event.source || 'unknown',
        'timestamp', event.timestamp
      );

      logger.info(`Published trench event: ${event.type} from ${event.source}`);
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
        tracked_tokens: this.trackedTokens.size,
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
  const watcher = new AdvancedTrenchWatcher();
  
  watcher.start().catch(error => {
    logger.error('Failed to start Advanced Trench Watcher:', error);
    process.exit(1);
  });

  // Graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Shutting down Advanced Trench Watcher...');
    await watcher.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down Advanced Trench Watcher...');
    await watcher.stop();
    process.exit(0);
  });
}

module.exports = AdvancedTrenchWatcher;
