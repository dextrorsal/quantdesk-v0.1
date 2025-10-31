const axios = require('axios');

// Test Pyth API with different formats
async function testPythAPI() {
  console.log('🔍 Testing Pyth Hermes API formats...\n');
  
  const feedId = 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J'; // BTC/USD
  
  // Format 1: Single ID as string
  try {
    console.log('📡 Testing Format 1: Single ID as string');
    const response = await axios.get('https://hermes.pyth.network/v2/updates/price/latest', {
      params: { ids: feedId },
      timeout: 10000
    });
    console.log('✅ Success:', response.status);
    console.log('Data:', JSON.stringify(response.data, null, 2).substring(0, 200) + '...');
  } catch (error) {
    console.log('❌ Failed:', error.response?.data || error.message);
  }
  
  console.log('\n' + '='.repeat(50) + '\n');
  
  // Format 2: Array format
  try {
    console.log('📡 Testing Format 2: Array format');
    const response = await axios.get('https://hermes.pyth.network/v2/updates/price/latest', {
      params: { ids: [feedId] },
      paramsSerializer: { indexes: null },
      timeout: 10000
    });
    console.log('✅ Success:', response.status);
    console.log('Data:', JSON.stringify(response.data, null, 2).substring(0, 200) + '...');
  } catch (error) {
    console.log('❌ Failed:', error.response?.data || error.message);
  }
  
  console.log('\n' + '='.repeat(50) + '\n');
  
  // Format 3: Comma-separated
  try {
    console.log('📡 Testing Format 3: Comma-separated');
    const response = await axios.get('https://hermes.pyth.network/v2/updates/price/latest', {
      params: { ids: feedId },
      timeout: 10000
    });
    console.log('✅ Success:', response.status);
    console.log('Data:', JSON.stringify(response.data, null, 2).substring(0, 200) + '...');
  } catch (error) {
    console.log('❌ Failed:', error.response?.data || error.message);
  }
  
  console.log('\n' + '='.repeat(50) + '\n');
  
  // Format 4: Multiple IDs
  try {
    console.log('📡 Testing Format 4: Multiple IDs');
    const feedIds = [
      'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J', // BTC/USD
      'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB', // ETH/USD
      'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG'  // SOL/USD
    ];
    
    const response = await axios.get('https://hermes.pyth.network/v2/updates/price/latest', {
      params: { ids: feedIds },
      paramsSerializer: { indexes: null },
      timeout: 10000
    });
    console.log('✅ Success:', response.status);
    console.log('Data:', JSON.stringify(response.data, null, 2).substring(0, 200) + '...');
  } catch (error) {
    console.log('❌ Failed:', error.response?.data || error.message);
  }
  
  console.log('\n' + '='.repeat(50) + '\n');
  
  // Test CoinGecko fallback
  try {
    console.log('📡 Testing CoinGecko Fallback');
    const response = await axios.get('https://api.coingecko.com/api/v3/simple/price', {
      params: {
        ids: 'bitcoin,ethereum,solana',
        vs_currencies: 'usd',
        include_24hr_change: true
      },
      timeout: 10000
    });
    console.log('✅ Success:', response.status);
    console.log('Data:', JSON.stringify(response.data, null, 2));
  } catch (error) {
    console.log('❌ Failed:', error.response?.data || error.message);
  }
}

testPythAPI().catch(console.error);
