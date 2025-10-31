// Simple test script for MIKEY-AI service
const http = require('http');

async function testService() {
  console.log('🧪 Testing MIKEY-AI Service...\n');

  // Test 1: Health Check
  try {
    const healthResponse = await fetch('http://localhost:3000/health');
    const healthData = await healthResponse.json();
    console.log('✅ Health Check:', healthData);
  } catch (error) {
    console.log('❌ Health Check failed:', error.message);
  }

  // Test 2: LLM Status
  try {
    const statusResponse = await fetch('http://localhost:3000/api/v1/llm/status');
    const statusData = await statusResponse.json();
    console.log('\n✅ LLM Status:', statusData);
  } catch (error) {
    console.log('❌ LLM Status failed:', error.message);
  }

  // Test 3: AI Query
  try {
    const queryResponse = await fetch('http://localhost:3000/api/v1/ai/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        query: 'Hello, this is a test'
      })
    });
    const queryData = await queryResponse.json();
    console.log('\n✅ AI Query:', queryData);
  } catch (error) {
    console.log('❌ AI Query failed:', error.message);
  }

  console.log('\n✅ Testing complete!');
}

// Test if fetch is available
if (typeof fetch === 'undefined') {
  console.log('⚠️ fetch not available, using http module');
  
  function makeRequest(url, options = {}) {
    return new Promise((resolve, reject) => {
      const urlObj = new URL(url);
      const httpOptions = {
        hostname: urlObj.hostname,
        port: urlObj.port,
        path: urlObj.pathname + urlObj.search,
        method: options.method || 'GET',
        headers: options.headers || {}
      };

      const req = http.request(httpOptions, (res) => {
        let data = '';
        res.on('data', (chunk) => { data += chunk; });
        res.on('end', () => {
          try {
            resolve({ json: () => Promise.resolve(JSON.parse(data)), status: res.statusCode });
          } catch (e) {
            resolve({ json: () => Promise.resolve({}), status: res.statusCode });
          }
        });
      });

      req.on('error', reject);
      if (options.body) req.write(options.body);
      req.end();
    });
  }

  global.fetch = makeRequest;
}

testService().catch(console.error);

