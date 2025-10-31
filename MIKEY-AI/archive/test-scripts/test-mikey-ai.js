// test-mikey-ai.js - Comprehensive test for running Mikey AI
require('dotenv').config({ path: './.env' });

async function testMikeyAI() {
  console.log('🤖 Testing Mikey AI - Hackathon Demo\n');
  
  const baseURL = 'http://localhost:3003';
  
  // Test 1: Health Check
  console.log('--- Health Check ---');
  try {
    const response = await fetch(`${baseURL}/health`);
    const data = await response.json();
    console.log('✅ Mikey AI Health:', data.status);
  } catch (error) {
    console.log('❌ Health Check Failed:', error.message);
    return;
  }
  
  // Test 2: Basic AI Query
  console.log('\n--- Basic AI Query ---');
  try {
    const response = await fetch(`${baseURL}/api/v1/ai/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: 'Hello! Can you help me with trading?',
        context: {}
      })
    });
    
    const data = await response.json();
    if (data.success) {
      console.log('✅ AI Response:', data.data.response.substring(0, 100) + '...');
      console.log('✅ Provider:', data.data.provider || 'Unknown');
    } else {
      console.log('❌ AI Query Failed:', data.error);
    }
  } catch (error) {
    console.log('❌ AI Query Error:', error.message);
  }
  
  // Test 3: Trading Analysis Query
  console.log('\n--- Trading Analysis Query ---');
  try {
    const response = await fetch(`${baseURL}/api/v1/ai/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: 'Get current SOL price and market data from QuantDesk',
        context: {}
      })
    });
    
    const data = await response.json();
    if (data.success) {
      console.log('✅ Trading Analysis:', data.data.response.substring(0, 150) + '...');
      console.log('✅ Provider:', data.data.provider || 'Unknown');
    } else {
      console.log('❌ Trading Analysis Failed:', data.error);
    }
  } catch (error) {
    console.log('❌ Trading Analysis Error:', error.message);
  }
  
  // Test 4: LLM Status
  console.log('\n--- LLM Provider Status ---');
  try {
    const response = await fetch(`${baseURL}/api/v1/llm/status`);
    const data = await response.json();
    console.log('✅ LLM Providers:', data.providers?.length || 0, 'available');
    console.log('✅ Status:', data.status);
  } catch (error) {
    console.log('❌ LLM Status Error:', error.message);
  }
  
  // Test 5: QuantDesk Integration
  console.log('\n--- QuantDesk Integration Test ---');
  try {
    const response = await fetch(`${baseURL}/api/v1/ai/query`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        query: 'Check QuantDesk API health and get available markets',
        context: {}
      })
    });
    
    const data = await response.json();
    if (data.success) {
      console.log('✅ QuantDesk Integration:', data.data.response.substring(0, 150) + '...');
    } else {
      console.log('❌ QuantDesk Integration Failed:', data.error);
    }
  } catch (error) {
    console.log('❌ QuantDesk Integration Error:', error.message);
  }
  
  console.log('\n🎯 Mikey AI Test Complete!');
  console.log('\n🚀 Your Mikey AI is ready for hackathon demo!');
  console.log('\nDemo Commands:');
  console.log('curl -X POST http://localhost:3003/api/v1/ai/query \\');
  console.log('  -H "Content-Type: application/json" \\');
  console.log('  -d \'{"query": "Analyze SOL price trends", "context": {}}\'');
}

testMikeyAI().catch(console.error);
