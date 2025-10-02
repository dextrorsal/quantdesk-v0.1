const express = require('express');
const axios = require('axios');

const app = express();
app.use(express.json());

// Simple Pyth API endpoint
app.get('/api/oracle/prices', async (req, res) => {
  try {
    console.log('🔍 Fetching prices from Pyth...');
    const PYTH_API_URL = 'https://hermes.pyth.network/v2/updates/price/latest';
    const BTC_ID = 'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43';
    
    const response = await axios.get(PYTH_API_URL, {
      params: { 'ids[]': BTC_ID },
      timeout: 10000
    });
    
    if (response.data.parsed && response.data.parsed.length > 0) {
      const priceData = response.data.parsed[0];
      const price = parseFloat(priceData.price.price) * Math.pow(10, priceData.price.expo);
      
      console.log('✅ Pyth API success:', price);
      res.json({
        success: true,
        data: { BTC: price },
        timestamp: Date.now(),
        source: 'pyth-network'
      });
    } else {
      throw new Error('No parsed data');
    }
  } catch (error) {
    console.log('❌ Pyth failed, trying CoinGecko...', error.message);
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
        source: 'coingecko-fallback'
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

app.get('/health', (req, res) => {
  res.json({ status: 'healthy', timestamp: Date.now() });
});

const PORT = 3002;
app.listen(PORT, () => {
  console.log('🚀 Simple Price API running on port', PORT);
  console.log('📊 Endpoint: http://localhost:' + PORT + '/api/oracle/prices');
  console.log('❤️ Health: http://localhost:' + PORT + '/health');
});
