const express = require('express');
const axios = require('axios');
require('dotenv').config();

const app = express();
app.use(express.json());

// Enhanced Pyth API endpoint using environment variables
app.get('/api/oracle/prices', async (req, res) => {
  try {
    console.log('🔍 Fetching prices from Pyth Network...');
    const PYTH_API_URL = 'https://hermes.pyth.network/v2/updates/price/latest';
    
    // Use environment variables for feed IDs
    const FEED_IDS = {
      BTC: process.env.PYTH_PRICE_FEED_BTC || 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J',
      ETH: process.env.PYTH_PRICE_FEED_ETH || 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB',
      SOL: process.env.PYTH_PRICE_FEED_SOL || 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG',
    };
    
    console.log('📋 Using feed IDs:', Object.keys(FEED_IDS).map(asset => `${asset}: ${FEED_IDS[asset].substring(0, 8)}...`).join(', '));
    
    const prices = {};
    let pythSuccess = false;
    
    // Make a single request with all feed IDs
    try {
      console.log('🔍 Making single request with all feed IDs...');
      const feedIds = Object.values(FEED_IDS);
      const response = await axios.get(PYTH_API_URL, {
        params: { 'ids[]': feedIds },
        timeout: 10000
      });
      
      if (response.data.parsed && response.data.parsed.length > 0) {
        console.log(`📊 Received ${response.data.parsed.length} price feeds from Pyth`);
        
        // Process each price feed
        for (const priceData of response.data.parsed) {
          const price = parseFloat(priceData.price.price) * Math.pow(10, priceData.price.expo);
          
          // Find which asset this feed ID corresponds to
          for (const [asset, feedId] of Object.entries(FEED_IDS)) {
            if (priceData.id === feedId) {
              prices[asset] = price;
              pythSuccess = true;
              console.log(`✅ Pyth ${asset}: $${price.toFixed(2)}`);
              break;
            }
          }
        }
      } else {
        console.log('❌ No parsed data received');
      }
    } catch (error) {
      console.log('❌ Pyth API request failed:', error.response?.status || error.message);
    }
    
    if (pythSuccess) {
      console.log('✅ Pyth API success with', Object.keys(prices).length, 'assets');
      res.json({
        success: true,
        data: prices,
        timestamp: Date.now(),
        source: 'pyth-network',
        feeds_found: Object.keys(prices).length
      });
    } else {
      throw new Error('All Pyth feeds failed');
    }
    
  } catch (error) {
    console.log('❌ Pyth failed, trying CoinGecko fallback...', error.message);
    try {
      const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
        params: {
          ids: 'bitcoin,ethereum,solana,usd-coin',
          vs_currencies: 'usd'
        },
        timeout: 10000
      });
      
      console.log('✅ CoinGecko fallback success');
      res.json({
        success: true,
        data: {
          BTC: response.data.bitcoin.usd,
          ETH: response.data.ethereum.usd,
          SOL: response.data.solana.usd,
          USDC: response.data['usd-coin'].usd
        },
        timestamp: Date.now(),
        source: 'coingecko-fallback',
        feeds_found: 4
      });
    } catch (fallbackError) {
      console.log('❌ Both failed:', fallbackError.message);
      res.status(500).json({
        success: false,
        error: 'Both Pyth and CoinGecko failed',
        message: fallbackError.message
      });
    }
  }
});

// New endpoint to discover available Pyth feeds
app.get('/api/oracle/discover', async (req, res) => {
  try {
    console.log('🔍 Discovering Pyth feeds...');
    
    // Try common feed ID patterns
    const commonFeeds = [
      'BTC/USD', 'ETH/USD', 'SOL/USD', 'USDC/USD', 'USDT/USD',
      'AVAX/USD', 'MATIC/USD', 'DOT/USD', 'LINK/USD', 'UNI/USD'
    ];
    
    const discoveredFeeds = [];
    
    for (const feed of commonFeeds) {
      try {
        // Try to get feed by symbol (this might work with some APIs)
        const response = await axios.get('https://hermes.pyth.network/v2/updates/price/latest', {
          params: { symbol: feed },
          timeout: 3000
        });
        
        if (response.data && response.data.parsed) {
          discoveredFeeds.push({
            symbol: feed,
            available: true,
            data: response.data.parsed
          });
        }
      } catch (error) {
        // Feed not available, continue
      }
    }
    
    res.json({
      success: true,
      discovered_feeds: discoveredFeeds,
      note: 'Pyth Network requires specific feed IDs - use /api/oracle/prices for working feeds',
      timestamp: Date.now()
    });
    
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to discover feeds',
      message: error.message
    });
  }
});

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: Date.now() });
});

const PORT = 3002;
app.listen(PORT, () => {
  console.log('🚀 Enhanced Price API running on port', PORT);
  console.log('📊 Prices: http://localhost:' + PORT + '/api/oracle/prices');
  console.log('🔍 Discover: http://localhost:' + PORT + '/api/oracle/discover');
  console.log('❤️ Health: http://localhost:' + PORT + '/health');
});
