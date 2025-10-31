#!/usr/bin/env node

/**
 * Test Updated Pyth Oracle Service with Hermes Client
 */

const { HermesClient } = require('@pythnetwork/hermes-client');

console.log('🚀 Testing Updated Pyth Oracle Service with Hermes Client...\n');

async function testUpdatedService() {
  try {
    console.log('🔌 Creating Hermes client...');
    const hermesClient = new HermesClient('https://hermes.pyth.network');
    
    // Pyth Network price feed IDs (same as in our service)
    const PYTH_FEED_IDS = {
      BTC: 'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43', // BTC/USD
      ETH: 'ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace', // ETH/USD
      SOL: 'ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d', // SOL/USD
      AVAX: '93da3432f8d1d02ce1e6d3c4e62ca12276915010d9bcc1f2b0a7e37a6c71f4a7', // AVAX/USD
      MATIC: '5de33a9112c2b698b39765a0d7f49f14516033809528a4feca76f911096e0160', // MATIC/USD
      DOGE: 'dcef50dd0a4cd2dcc17e45df1676dcb336a11a61c69df7a0299b0150c672d25', // DOGE/USD
      ADA: '2a01deaec9e51a579277b34b122399984d0bbf57e2458a7e42fecd2829867a0d', // ADA/USD
      DOT: 'ca3eed9b267293f6595901c734c7525ce8ef49adafe8284606ceb307afa2ca5b', // DOT/USD
      LINK: '8ac0c70fff57e9aefdf5edf44b51d62c2d433653cbb2cf5cc06bb115af04d221' // LINK/USD
    };
    
    console.log('📊 Fetching latest price updates...');
    const feedIds = Object.values(PYTH_FEED_IDS);
    const response = await hermesClient.getLatestPriceUpdates(feedIds);
    
    console.log('✅ REST API call successful!');
    console.log('📈 Price updates received:', response.parsed.length);
    
    const priceMap = new Map();
    
    response.parsed.forEach((priceFeed, index) => {
      if (priceFeed && priceFeed.id && priceFeed.price) {
        const feedId = priceFeed.id;
        const symbol = Object.keys(PYTH_FEED_IDS).find(key => PYTH_FEED_IDS[key] === feedId);
        
        if (symbol) {
          const price = parseFloat(priceFeed.price.price);
          const confidence = parseFloat(priceFeed.price.conf || '0');
          const exponent = parseInt(priceFeed.price.expo || '0');
          const publishTime = parseInt(priceFeed.price.publish_time || '0');
          
          // Apply exponent to get actual price
          const actualPrice = price * Math.pow(10, exponent);
          const actualConfidence = confidence * Math.pow(10, exponent);
          
          priceMap.set(symbol, {
            price: actualPrice,
            confidence: actualConfidence,
            exponent,
            publishTime,
            timestamp: Date.now(),
            symbol
          });
          
          console.log(`💰 ${symbol}: $${actualPrice.toFixed(2)} ±$${actualConfidence.toFixed(2)}`);
        }
      }
    });
    
    console.log(`\n🎉 Successfully processed ${priceMap.size} prices from Pyth Network!`);
    console.log('📊 Price map keys:', Array.from(priceMap.keys()));
    
  } catch (error) {
    console.error('❌ Error testing updated service:', error.message);
    console.error('Stack:', error.stack);
    process.exit(1);
  }
}

// Start the test
testUpdatedService();
