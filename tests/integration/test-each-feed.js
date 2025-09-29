#!/usr/bin/env node

/**
 * Test Each Feed ID Individually
 */

const { HermesClient } = require('@pythnetwork/hermes-client');

console.log('🚀 Testing Each Feed ID Individually...\n');

async function testEachFeed() {
  try {
    const hermesClient = new HermesClient('https://hermes.pyth.network');
    
    const PYTH_FEED_IDS = {
      BTC: 'e62df6c8b4a85fe1a67db44dc12de5db330f7ac66b72dc658afedf0f4a415b43',
      ETH: 'ff61491a931112ddf1bd8147cd1b641375f79f5825126d665480874634fd0ace',
      SOL: 'ef0d8b6fda2ceba41da15d4095d1da392a0d2f8ed0c6c7bc0f4cfac8c280b56d',
      AVAX: '93da3432f8d1d02ce1e6d3c4e62ca12276915010d9bcc1f2b0a7e37a6c71f4a7',
      MATIC: '5de33a9112c2b698b39765a0d7f49f14516033809528a4feca76f911096e0160',
      DOGE: 'dcef50dd0a4cd2dcc17e45df1676dcb336a11a61c69df7a0299b0150c672d25',
      ADA: '2a01deaec9e51a579277b34b122399984d0bbf57e2458a7e42fecd2829867a0d',
      DOT: 'ca3eed9b267293f6595901c734c7525ce8ef49adafe8284606ceb307afa2ca5b',
      LINK: '8ac0c70fff57e9aefdf5edf44b51d62c2d433653cbb2cf5cc06bb115af04d221'
    };
    
    for (const [symbol, feedId] of Object.entries(PYTH_FEED_IDS)) {
      try {
        console.log(`Testing ${symbol} (${feedId})...`);
        const response = await hermesClient.getLatestPriceUpdates([feedId]);
        console.log(`✅ ${symbol}: Success - ${response.parsed.length} updates`);
        
        if (response.parsed.length > 0) {
          const priceData = response.parsed[0];
          const price = parseFloat(priceData.price.price) * Math.pow(10, parseInt(priceData.price.expo));
          console.log(`   Price: $${price.toFixed(2)}`);
        }
      } catch (error) {
        console.log(`❌ ${symbol}: Failed - ${error.message}`);
      }
    }
    
  } catch (error) {
    console.error('❌ Error:', error.message);
  }
}

testEachFeed();
