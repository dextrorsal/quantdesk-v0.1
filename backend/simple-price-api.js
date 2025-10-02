const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

// Enhanced Pyth API endpoint using proper SDK approach
app.get('/api/oracle/prices', async (req, res) => {
  try {
    console.log('🔍 Fetching prices from Pyth Network...');
    const PYTH_API_URL = 'https://hermes.pyth.network/v2/updates/price/latest';
    
    // Multiple price feed IDs to try (both working and common ones)
    const FEED_IDS = [
      // Working BTC ID
      'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43', // BTC/USD (working)
      // Common Pyth feed IDs (let's test these)
      'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J', // BTC/USD (alternative)
      'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB', // ETH/USD
      'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG', // SOL/USD
      'Gnt27xtC473ZT2Mw5u8wZ68Z3gULkSTb5DuxJy7eJotD', // USDC/USD
    ];
    
    const prices = {};
    let pythSuccess = false;
    
    // Try to fetch from Pyth for each feed ID
    for (const feedId of FEED_IDS) {
      try {
        const response = await axios.get(PYTH_API_URL, {
          params: { 'ids[]': feedId },
          timeout: 5000
        });
        
        if (response.data.parsed && response.data.parsed.length > 0) {
          const priceData = response.data.parsed[0];
          const price = parseFloat(priceData.price.price) * Math.pow(10, priceData.price.expo);
          
          // Determine asset type based on feed ID
          let asset = 'UNKNOWN';
          if (feedId.includes('e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43') || 
              feedId.includes('HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J')) {
            asset = 'BTC';
          } else if (feedId.includes('JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB')) {
            asset = 'ETH';
          } else if (feedId.includes('H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG')) {
            asset = 'SOL';
          } else if (feedId.includes('Gnt27xtC473ZT2Mw5u8wZ68Z3gULkSTb5DuxJy7eJotD')) {
            asset = 'USDC';
          }
          
          prices[asset] = price;
          pythSuccess = true;
          console.log(`✅ Pyth ${asset}: $${price.toFixed(2)}`);
        }
      } catch (error) {
        console.log(`❌ Pyth feed ${feedId.substring(0, 8)}... failed:`, error.message);
      }
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

// New endpoint to get available Pyth feeds dynamically
app.get('/api/oracle/feeds', async (req, res) => {
  try {
    console.log('🔍 Fetching available Pyth feeds...');
    
    // Try to get feed metadata
    const response = await axios.get('https://hermes.pyth.network/v2/updates/price/latest', {
      timeout: 10000
    });
    
    res.json({
      success: true,
      message: 'Pyth API is accessible',
      timestamp: Date.now(),
      note: 'Feed discovery requires specific feed IDs - see /api/oracle/prices for working feeds'
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to access Pyth API',
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
  console.log('🔍 Feeds: http://localhost:' + PORT + '/api/oracle/feeds');
  console.log('❤️ Health: http://localhost:' + PORT + '/health');
});
