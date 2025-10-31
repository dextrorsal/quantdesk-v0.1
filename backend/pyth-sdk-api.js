const express = require('express');
const { HermesClient } = require('@pythnetwork/hermes-client');
require('dotenv').config();

const app = express();
app.use(express.json());

// Initialize Hermes client
const hermesClient = new HermesClient('https://hermes.pyth.network');

// Use environment variables for feed IDs
const FEED_IDS = {
  BTC: process.env.PYTH_PRICE_FEED_BTC || 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J',
  ETH: process.env.PYTH_PRICE_FEED_ETH || 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB',
  SOL: process.env.PYTH_PRICE_FEED_SOL || 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG',
};

console.log('📋 Using Pyth SDK with feed IDs:', Object.keys(FEED_IDS).map(asset => `${asset}: ${FEED_IDS[asset].substring(0, 8)}...`).join(', '));

// Pyth SDK API endpoint
app.get('/api/oracle/prices', async (req, res) => {
  try {
    console.log('🔍 Fetching prices using Pyth SDK...');
    
    const prices = {};
    let pythSuccess = false;
    
    try {
      // Get latest price updates using Hermes client
      const feedIds = Object.values(FEED_IDS);
      const priceUpdates = await hermesClient.getLatestPriceUpdates(feedIds);
      
      if (priceUpdates && priceUpdates.length > 0) {
        console.log(`📊 Received ${priceUpdates.length} price updates from Hermes`);
        
        // Process each price update
        for (const priceUpdate of priceUpdates) {
          if (priceUpdate.price && priceUpdate.price.price) {
            const priceValue = parseFloat(priceUpdate.price.price) * Math.pow(10, priceUpdate.price.expo);
            
            // Find which asset this feed ID corresponds to
            for (const [asset, feedId] of Object.entries(FEED_IDS)) {
              if (priceUpdate.id === feedId) {
                prices[asset] = priceValue;
                pythSuccess = true;
                console.log(`✅ Hermes ${asset}: $${priceValue.toFixed(2)}`);
                break;
              }
            }
          }
        }
      } else {
        console.log('❌ No price updates received from Hermes');
      }
    } catch (error) {
      console.log('❌ Hermes request failed:', error.message);
    }
    
    if (pythSuccess) {
      console.log('✅ Hermes success with', Object.keys(prices).length, 'assets');
      res.json({
        success: true,
        data: prices,
        timestamp: Date.now(),
        source: 'hermes-protocol',
        feeds_found: Object.keys(prices).length
      });
    } else {
      throw new Error('Hermes failed to fetch prices');
    }
    
  } catch (error) {
    console.log('❌ Hermes failed, trying CoinGecko fallback...', error.message);
    try {
      const axios = require('axios');
      const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
        params: {
          ids: 'bitcoin,ethereum,solana',
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
          SOL: response.data.solana.usd
        },
        timestamp: Date.now(),
        source: 'coingecko-fallback',
        feeds_found: 3
      });
    } catch (fallbackError) {
      console.log('❌ Both Hermes and CoinGecko failed:', fallbackError.message);
      res.status(500).json({
        success: false,
        error: 'Both Hermes and CoinGecko failed',
        message: fallbackError.message
      });
    }
  }
});

// Get available feeds endpoint
app.get('/api/oracle/feeds', async (req, res) => {
  try {
    console.log('🔍 Getting available Pyth feeds using Hermes...');
    
    // Try to get feed metadata
    const priceUpdates = await hermesClient.getLatestPriceUpdates(Object.values(FEED_IDS));
    
    res.json({
      success: true,
      message: 'Hermes Protocol is accessible',
      timestamp: Date.now(),
      available_feeds: Object.keys(FEED_IDS),
      feeds_found: priceUpdates ? priceUpdates.length : 0,
      feed_ids: FEED_IDS
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to access Hermes Protocol',
      message: error.message
    });
  }
});

// Health check endpoint
app.get('/health', (req, res) => {
  res.json({ 
    status: 'healthy', 
    timestamp: Date.now(),
    protocol: 'Hermes Protocol',
    feeds_configured: Object.keys(FEED_IDS).length
  });
});

const PORT = 3002;
app.listen(PORT, () => {
  console.log('🚀 Hermes Protocol API running on port', PORT);
  console.log('📊 Prices: http://localhost:' + PORT + '/api/oracle/prices');
  console.log('🔍 Feeds: http://localhost:' + PORT + '/api/oracle/feeds');
  console.log('❤️ Health: http://localhost:' + PORT + '/health');
  console.log('📦 Using Hermes Protocol:', '@pythnetwork/hermes-client');
});
