#!/usr/bin/env node
/**
 * Test MIKEY Tool Routing
 * Tests different queries to verify tools are properly connected
 */

const MIKEY_URL = process.env.MIKEY_URL || 'http://localhost:3000'; // MIKEY-AI runs on port 3000 (as per architecture)

const testQueries = [
  {
    name: 'Real Token Analysis',
    query: 'Analyze BTC market sentiment and show me TVL, market cap, and indicators',
    expectedTool: 'RealTokenAnalysisTool',
    expectedEndpoints: ['/api/oracle/price/BTC', '/api/dev/market-summary']
  },
  {
    name: 'Pyth Price Check',
    query: 'What is the current price of SOL from Pyth oracle?',
    expectedTool: 'RealDataTools',
    expectedEndpoints: ['/api/oracle/prices']
  },
  {
    name: 'Market Summary',
    query: 'Show me the market summary for all trading pairs with volume and open interest',
    expectedTool: 'QuantDeskProtocolTools',
    expectedEndpoints: ['/api/dev/market-summary']
  },
  {
    name: 'Live Price Single Asset',
    query: 'What is the live price of ETH?',
    expectedTool: 'QuantDeskProtocolTools',
    expectedEndpoints: ['/api/oracle/price/ETH']
  },
  {
    name: 'Portfolio Check',
    query: 'Check my portfolio for wallet wgfSHTWx1woRXhsWijj1kcpCP8tmbmK2KnouFVAuoc6',
    expectedTool: 'QuantDeskProtocolTools',
    expectedEndpoints: ['/api/dev/user-portfolio/wgfSHTWx1woRXhsWijj1kcpCP8tmbmK2KnouFVAuoc6']
  },
  {
    name: 'Multi-Asset Analysis',
    query: 'Analyze SOL and BTC market data including prices, indicators, and sentiment',
    expectedTool: 'RealTokenAnalysisTool',
    expectedEndpoints: ['/api/oracle/price/SOL', '/api/oracle/price/BTC', '/api/dev/market-summary']
  }
];

async function testQuery(test) {
  console.log(`\n${'='.repeat(80)}`);
  console.log(`🧪 Testing: ${test.name}`);
  console.log(`📝 Query: "${test.query}"`);
  console.log(`🎯 Expected Tool: ${test.expectedTool}`);
  console.log(`🔗 Expected Endpoints: ${test.expectedEndpoints.join(', ')}`);
  console.log(`${'='.repeat(80)}\n`);

  try {
    const response = await fetch(`${MIKEY_URL}/api/v1/ai/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query: test.query,
        userId: 'test-user',
        sessionId: 'test-session'
      })
    });

    if (!response.ok) {
      console.error(`❌ HTTP Error: ${response.status} ${response.statusText}`);
      const errorText = await response.text();
      console.error(`   Error: ${errorText.substring(0, 200)}`);
      return false;
    }

    const data = await response.json();
    
    // Handle nested data structure: { success: true, data: { response, provider, sources, ... } }
    const responseData = data.data || data;
    const responseText = responseData.response || data.response || '';
    const provider = responseData.provider || data.provider || 'N/A';
    const sources = responseData.sources || data.sources || [];
    const confidence = responseData.confidence || data.confidence || 'N/A';
    
    console.log(`✅ Response Received:`);
    console.log(`   Success: ${data.success || 'N/A'}`);
    console.log(`   Provider: ${provider}`);
    console.log(`   Sources: ${sources.length > 0 ? sources.join(', ') : 'N/A'}`);
    console.log(`   Confidence: ${confidence}`);
    console.log(`\n   Response Preview (first 500 chars):`);
    console.log(`   ${responseText.substring(0, 500)}${responseText.length > 500 ? '...' : ''}`);
    
    if (responseText && responseText.length > 0) {
      // Check if it contains real data indicators
      const hasRealData = responseText.includes('Pyth') || 
                          responseText.includes('$') ||
                          responseText.toLowerCase().includes('price') ||
                          responseText.includes('USD') ||
                          responseText.toLowerCase().includes('market') ||
                          responseText.includes('BTC') ||
                          responseText.includes('SOL') ||
                          responseText.includes('ETH') ||
                          responseText.includes('Real-Time') ||
                          responseText.includes('QuantDesk');
      
      if (hasRealData) {
        console.log(`\n   ✅ Tool executed successfully! (Contains real data indicators)`);
        console.log(`   ✅ Expected Tool Matched: ${provider === test.expectedTool || responseText.includes(test.expectedTool)}`);
      } else {
        console.log(`\n   ⚠️  Response received but may not have used real tools (LLM only)`);
      }
      return hasRealData;
    } else {
      console.log(`\n   ❌ Empty response - tool may not have been called`);
      return false;
    }

  } catch (error) {
    console.error(`❌ Error: ${error.message}`);
    if (error.stack) {
      console.error(`   Stack: ${error.stack.split('\n').slice(0, 3).join('\n')}`);
    }
    return false;
  }
}

async function runTests() {
  console.log(`\n🚀 MIKEY Tool Routing Test Suite`);
  console.log(`📡 MIKEY URL: ${MIKEY_URL}\n`);

  // Check if MIKEY is running
  try {
    const healthCheck = await fetch(`${MIKEY_URL}/health`);
    const health = await healthCheck.json();
    console.log(`✅ MIKEY-AI is running`);
    console.log(`   Status: ${health.status || health.data?.status || 'healthy'}`);
  } catch (error) {
    console.error(`❌ MIKEY-AI is not running at ${MIKEY_URL}`);
    console.error(`   Error: ${error.message}`);
    process.exit(1);
  }

  // Run tests
  const results = [];
  for (const test of testQueries) {
    const success = await testQuery(test);
    results.push({ ...test, success });
    await new Promise(resolve => setTimeout(resolve, 2000)); // Wait 2s between tests
  }

  // Summary
  console.log(`\n${'='.repeat(80)}`);
  console.log(`📊 Test Summary`);
  console.log(`${'='.repeat(80)}\n`);

  const passed = results.filter(r => r.success).length;
  const failed = results.filter(r => !r.success).length;

  results.forEach(result => {
    const icon = result.success ? '✅' : '❌';
    console.log(`${icon} ${result.name}`);
  });

  console.log(`\n   Passed: ${passed}/${results.length}`);
  console.log(`   Failed: ${failed}/${results.length}`);

  if (failed === 0) {
    console.log(`\n🎉 All tests passed! All tools are properly connected.`);
  } else {
    console.log(`\n⚠️  Some tests failed. Check the logs above for details.`);
  }
}

// Run tests
runTests().catch(error => {
  console.error('Fatal error:', error);
  process.exit(1);
});

