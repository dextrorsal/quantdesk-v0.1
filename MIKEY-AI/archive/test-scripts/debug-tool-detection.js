#!/usr/bin/env node

/**
 * Debug Mikey AI Tool Detection
 * Test if the needsRealData method is working correctly
 */

require('dotenv').config({ path: './.env' });

// Simulate the needsRealData method
function needsRealData(queryText) {
  const lowerQuery = queryText.toLowerCase();
  return lowerQuery.includes('pyth') || 
         lowerQuery.includes('oracle') ||
         lowerQuery.includes('coingecko') ||
         lowerQuery.includes('whale') ||
         lowerQuery.includes('whales') ||
         lowerQuery.includes('large transaction') ||
         lowerQuery.includes('crypto news') ||
         lowerQuery.includes('news') ||
         lowerQuery.includes('sentiment') ||
         lowerQuery.includes('arbitrage') ||
         lowerQuery.includes('market analysis') ||
         lowerQuery.includes('real-time') ||
         lowerQuery.includes('live data');
}

console.log('🔍 Testing Mikey AI Tool Detection');
console.log('');

const testQueries = [
  'What are the current Pyth oracle prices for BTC, ETH, and SOL?',
  'Show me recent whale movements and large transactions',
  'What is the latest crypto news and sentiment?',
  'Find arbitrage opportunities across exchanges',
  'Show me market analysis for SOL',
  'What are the current prices?',
  'How do I trade crypto?'
];

testQueries.forEach((query, index) => {
  const detected = needsRealData(query);
  console.log(`${index + 1}. "${query}"`);
  console.log(`   Detected: ${detected ? '✅ YES' : '❌ NO'}`);
  console.log('');
});

console.log('🎯 Expected Results:');
console.log('  • Query 1 (Pyth): Should detect ✅');
console.log('  • Query 2 (Whale): Should detect ✅');
console.log('  • Query 3 (News): Should detect ✅');
console.log('  • Query 4 (Arbitrage): Should detect ✅');
console.log('  • Query 5 (Market): Should detect ✅');
console.log('  • Query 6 (Prices): Should detect ❌');
console.log('  • Query 7 (Trade): Should detect ❌');
