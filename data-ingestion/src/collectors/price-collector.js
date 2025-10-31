const { Connection, PublicKey } = require('@solana/web3.js');
const axios = require('axios');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');

class PriceCollector {
  constructor() {
    this.connection = new Connection(process.env.SOLANA_RPC_URL);
    this.pythBaseUrl = process.env.PYTH_BASE_URL;
    this.priceFeeds = {
      'BTC-PERP': process.env.PYTH_PRICE_FEED_BTC,
      'ETH-PERP': process.env.PYTH_PRICE_FEED_ETH,
      'SOL-PERP': process.env.PYTH_PRICE_FEED_SOL
    };
    this.isRunning = false;
    this.collectionInterval = 5000; // 5 seconds to avoid rate limits
    this.lastPrices = {};
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Price collector already running');
      return;
    }

    this.isRunning = true;
    logger.info('Starting price collector...');

    // Start collecting prices
    this.collectPrices();
  }

  async collectPrices() {
    if (!this.isRunning) return;

    try {
      const priceData = await this.fetchPythPrices();
      if (priceData.parsed && priceData.parsed.length > 0) {
        await this.publishPrices(priceData.parsed);
        logger.info(`Published ${priceData.parsed.length} price updates`);
      } else {
        logger.warn('No price data available');
      }
    } catch (error) {
      logger.error('Error collecting prices:', error);
    }

    // Schedule next collection
    setTimeout(() => this.collectPrices(), this.collectionInterval);
  }

  async fetchPythPrices() {
    try {
      // Try Pyth first, but fallback to CoinGecko if it fails
      const feedIds = Object.values(this.priceFeeds);
      
      // Try using the /v2/updates/price/latest endpoint with proper format
      const response = await axios.get(`${this.pythBaseUrl}/v2/updates/price/latest`, {
        params: {
          'ids[]': feedIds  // Use array format for multiple IDs
        },
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-DataIngestion/1.0'
        }
      });

      if (!response.data) {
        throw new Error('No price data received from Pyth');
      }

      logger.info('Pyth API response:', JSON.stringify(response.data, null, 2));

      // Process the response - Pyth returns data in parsed array
      const prices = [];
      
      if (response.data.parsed && Array.isArray(response.data.parsed)) {
        for (const priceFeed of response.data.parsed) {
          if (!priceFeed || !priceFeed.id) continue;
          
          const feedId = priceFeed.id;
          const symbol = this.getSymbolFromFeedId(feedId);
          if (!symbol) continue;
          
          // Parse the price data from the price feed
          if (priceFeed.price && priceFeed.price.price) {
            const price = parseFloat(priceFeed.price.price);
            const confidence = parseFloat(priceFeed.price.conf?.toString() || '0');
            const exponent = parseInt(priceFeed.price.expo?.toString() || '0');
            const publishTime = parseInt(priceFeed.price.publish_time?.toString() || '0');
            
            // Apply exponent to get actual price
            const actualPrice = price * Math.pow(10, exponent);
            const actualConfidence = confidence * Math.pow(10, exponent);

            prices.push({
              symbol,
              price: actualPrice,
              confidence: actualConfidence,
              timestamp: new Date(publishTime * 1000).toISOString(),
              source: 'pyth'
            });
            
            logger.info(`💰 Pyth ${symbol}: $${actualPrice.toFixed(2)} (confidence: ${actualConfidence.toFixed(4)})`);
          }
        }
      }

      return { parsed: prices };
    } catch (error) {
      logger.warn('Pyth API failed, falling back to CoinGecko:', error.response?.data || error.message);
      
      // Fallback to CoinGecko
      return await this.fetchCoinGeckoPrices();
    }
  }

  async fetchCoinGeckoPrices() {
    try {
      const coinGeckoIds = {
        'BTC-PERP': 'bitcoin',
        'ETH-PERP': 'ethereum',
        'SOL-PERP': 'solana'
      };

      const ids = Object.values(coinGeckoIds).join(',');
      const response = await axios.get(`https://api.coingecko.com/api/v3/simple/price`, {
        params: {
          ids: ids,
          vs_currencies: 'usd',
          include_24hr_change: true
        },
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-DataIngestion/1.0'
        }
      });

      if (!response.data) {
        throw new Error('No price data received from CoinGecko');
      }

      logger.info('CoinGecko API response:', JSON.stringify(response.data, null, 2));

      const prices = [];
      for (const [symbol, coinGeckoId] of Object.entries(coinGeckoIds)) {
        const priceData = response.data[coinGeckoId];
        
        if (priceData && priceData.usd) {
          prices.push({
            symbol,
            price: parseFloat(priceData.usd),
            confidence: null, // CoinGecko doesn't provide confidence
            timestamp: new Date().toISOString(),
            change_24h: priceData.usd_24h_change || null
          });
        } else {
          logger.warn(`No price data found for ${symbol} (coinGeckoId: ${coinGeckoId})`);
        }
      }

      return { parsed: prices };
    } catch (error) {
      logger.error('Error fetching CoinGecko prices:', error.response?.data || error.message);
      throw error;
    }
  }

  async publishPrices(prices) {
    try {
      for (const price of prices) {
        const message = {
          id: `price_${price.symbol}_${Date.now()}`,
          timestamp: price.timestamp,
          data: {
            symbol: price.symbol,
            price: price.price,
            confidence: price.confidence,
            source: 'pyth'
          }
        };

        await redis.xadd(
          STREAMS.PRICE_UPDATES,
          '*',
          'data', JSON.stringify(message),
          'type', 'price_update',
          'symbol', price.symbol,
          'price', price.price.toString(),
          'timestamp', price.timestamp
        );
      }
    } catch (error) {
      logger.error('Error publishing prices to Redis:', error);
      throw error;
    }
  }

  getSymbolFromFeedId(feedId) {
    // Map feed IDs to symbols
    for (const [symbol, id] of Object.entries(this.priceFeeds)) {
      if (id === feedId) {
        return symbol;
      }
    }
    return null;
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping price collector...');
    this.isRunning = false;
    
    if (this.ws) {
      this.ws.close();
    }
  }

  async getMarketId(symbol) {
    // Map symbols to market IDs
    const marketMap = {
      'BTC-PERP': 'btc-perp',
      'ETH-PERP': 'eth-perp',
      'SOL-PERP': 'sol-perp'
    };
    return marketMap[symbol] || symbol.toLowerCase();
  }
}

// Start the collector if this file is run directly
if (require.main === module) {
  const collector = new PriceCollector();
  
  // Handle graceful shutdown
  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down gracefully...');
    await collector.stop();
    process.exit(0);
  });

  process.on('SIGINT', async () => {
    logger.info('SIGINT received, shutting down gracefully...');
    await collector.stop();
    process.exit(0);
  });

  collector.start().catch(error => {
    logger.error('Failed to start price collector:', error);
    process.exit(1);
  });
}

module.exports = PriceCollector;