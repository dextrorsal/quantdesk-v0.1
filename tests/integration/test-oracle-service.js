const axios = require('axios');

// Test our oracle service directly
async function testOracleService() {
  console.log('🔍 Testing QuantDesk Oracle Service...\n');
  
  try {
    // Test CoinGecko API directly
    console.log('📊 Testing CoinGecko API...');
    const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
      params: {
        ids: 'bitcoin,ethereum,solana',
        vs_currencies: 'usd',
        include_24hr_change: true
      },
      timeout: 10000
    });
    
    console.log('✅ CoinGecko API Response:');
    console.log(JSON.stringify(response.data, null, 2));
    
    // Test our market API endpoints
    console.log('\n📡 Testing Market API Endpoints...');
    
    // Test BTC price
    try {
      const btcResponse = await axios.get('http://localhost:3001/api/markets/BTC-PERP/price');
      console.log('✅ BTC Price API:', btcResponse.data);
    } catch (error) {
      console.log('❌ BTC Price API failed:', error.message);
    }
    
    // Test ETH price
    try {
      const ethResponse = await axios.get('http://localhost:3001/api/markets/ETH-PERP/price');
      console.log('✅ ETH Price API:', ethResponse.data);
    } catch (error) {
      console.log('❌ ETH Price API failed:', error.message);
    }
    
    // Test SOL price
    try {
      const solResponse = await axios.get('http://localhost:3001/api/markets/SOL-PERP/price');
      console.log('✅ SOL Price API:', solResponse.data);
    } catch (error) {
      console.log('❌ SOL Price API failed:', error.message);
    }
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

testOracleService().catch(console.error);
