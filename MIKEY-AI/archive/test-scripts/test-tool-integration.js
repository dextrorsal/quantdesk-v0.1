#!/usr/bin/env node

/**
 * Quick Test for Mikey AI Tool Integration
 * Run this after restarting the server to verify tools are working
 */

require('dotenv').config({ path: './.env' });

const axios = require('axios');

const AI_URL = 'http://localhost:3003';

console.log('🧪 Testing Mikey AI Tool Integration');
console.log('📊 AI URL:', AI_URL);
console.log('');

async function testToolDetection() {
  console.log('🔍 Testing Tool Detection...');
  
  const testQueries = [
    {
      name: 'Pyth Oracle Test',
      query: 'What are the current Pyth oracle prices for BTC, ETH, and SOL?',
      expected: 'pyth'
    },
    {
      name: 'Whale Movement Test', 
      query: 'Show me recent whale movements and large transactions',
      expected: 'whale'
    },
    {
      name: 'News Sentiment Test',
      query: 'What is the latest crypto news and sentiment?',
      expected: 'news'
    }
  ];

  for (const test of testQueries) {
    try {
      console.log(`\n📡 Testing: ${test.name}`);
      console.log(`   Query: "${test.query}"`);
      
      const response = await axios.post(`${AI_URL}/api/v1/ai/query`, {
        query: test.query,
        context: {}
      }, {
        headers: { 'Content-Type': 'application/json' },
        timeout: 15000
      });

      if (response.data.success) {
        const data = response.data.data;
        console.log(`   ✅ Response received from ${data.provider || 'Unknown'}`);
        console.log(`   📊 Confidence: ${data.confidence}`);
        console.log(`   🔗 Sources: ${data.sources.join(', ')}`);
        
        const responseText = data.response;
        
        // Check if it used real data tools
        if (responseText.includes('Real Data Pipeline') || 
            responseText.includes('QuantDesk Data Pipeline') ||
            responseText.includes(test.expected)) {
          console.log(`   🎯 ✅ Correctly used real data tools!`);
        } else {
          console.log(`   ⚠️  ❌ Still giving generic advice`);
        }
        
        // Show first 150 chars of response
        const preview = responseText.substring(0, 150);
        console.log(`   📝 Response: "${preview}${responseText.length > 150 ? '...' : ''}"`);
        
      } else {
        console.log(`   ❌ Error: ${response.data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.log(`   ❌ Request failed: ${error.response?.status || error.message}`);
    }
  }
}

async function main() {
  try {
    await testToolDetection();
    
    console.log('\n🎉 Tool Integration Test Complete!');
    console.log('');
    console.log('📋 What to look for:');
    console.log('  • ✅ "Correctly used real data tools!" = Working');
    console.log('  • ❌ "Still giving generic advice" = Not working');
    console.log('');
    console.log('🔧 If tools are not working:');
    console.log('  1. Restart the Mikey AI server: PORT=3003 npm start');
    console.log('  2. Check server logs for debug output');
    console.log('  3. Verify QUANTDESK_URL is set correctly');
    console.log('  4. Test API endpoints directly with curl');
    
  } catch (error) {
    console.error('❌ Test failed:', error.message);
  }
}

main();
