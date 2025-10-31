// test-quantdesk-api.js
require('dotenv').config({ path: './.env' });

async function testQuantDeskAPI() {
  console.log('🧪 Testing QuantDesk API Connectivity...\n');
  
  const baseURL = process.env.QUANTDESK_API_URL || 'http://localhost:3002';
  
  const tests = [
    {
      name: 'Health Check',
      url: `${baseURL}/health`,
      method: 'GET'
    },
    {
      name: 'Get Markets',
      url: `${baseURL}/api/markets`,
      method: 'GET'
    },
    {
      name: 'Get Account State',
      url: `${baseURL}/api/account/state`,
      method: 'GET'
    },
    {
      name: 'Get Price History',
      url: `${baseURL}/api/price-history?symbol=SOL&timeframe=1h`,
      method: 'GET'
    },
    {
      name: 'Get Funding Rates',
      url: `${baseURL}/api/funding-rates`,
      method: 'GET'
    }
  ];
  
  for (const test of tests) {
    console.log(`\n--- Testing: ${test.name} ---`);
    try {
      const response = await fetch(test.url, {
        method: test.method,
        headers: {
          'Content-Type': 'application/json',
          // Add auth headers if needed
          // 'Authorization': `Bearer ${process.env.QUANTDESK_TOKEN}`
        }
      });
      
      if (response.ok) {
        const data = await response.json();
        console.log(`✅ ${test.name}: Success`);
        console.log(`   Response: ${JSON.stringify(data).substring(0, 100)}...`);
      } else {
        console.log(`❌ ${test.name}: ${response.status} ${response.statusText}`);
      }
    } catch (error) {
      console.log(`❌ ${test.name}: ${error.message}`);
    }
  }
  
  console.log('\n🎯 QuantDesk API Test Complete!');
}

testQuantDeskAPI();
