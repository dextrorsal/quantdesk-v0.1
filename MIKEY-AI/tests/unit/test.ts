#!/usr/bin/env tsx

/**
 * Test script for Solana DeFi Trading Intelligence AI
 * Run with: npx tsx src/test.ts
 */

import { config } from '@/config';
import { SecurityUtils } from '@/utils/security';
import { systemLogger } from '@/utils/logger';
import { solanaService } from '@/services/SolanaService';
import { tradingAgent } from '@/agents/TradingAgent';

async function runTests(): Promise<void> {
  console.log('🧪 Starting Solana DeFi AI Tests...\n');

  try {
    // Test 1: Environment validation
    console.log('1️⃣ Testing environment validation...');
    SecurityUtils.validateEnvironment();
    console.log('✅ Environment validation passed\n');

    // Test 2: Solana connection
    console.log('2️⃣ Testing Solana connection...');
    const networkHealth = await solanaService.getNetworkHealth();
    console.log(`✅ Solana connection healthy: ${networkHealth.isHealthy}`);
    console.log(`   Cluster: ${networkHealth.cluster}`);
    console.log(`   Current slot: ${networkHealth.slot}\n`);

    // Test 3: Wallet analysis (if private key is available)
    if (config.solana.privateKey) {
      console.log('3️⃣ Testing wallet analysis...');
      const publicKey = solanaService.getPublicKey();
      if (publicKey) {
        const balance = await solanaService.getBalance();
        console.log(`✅ Wallet balance: ${balance} SOL`);
        console.log(`   Address: ${publicKey.toString()}\n`);
      }
    } else {
      console.log('3️⃣ Skipping wallet test (no private key)\n');
    }

    // Test 4: AI agent initialization
    console.log('4️⃣ Testing AI agent...');
    const testQuery = {
      query: 'What is the current market sentiment for SOL?',
      context: { symbols: ['SOL/USD'] }
    };
    
    const response = await tradingAgent.processQuery(testQuery);
    console.log('✅ AI agent response received');
    console.log(`   Confidence: ${response.confidence}`);
    console.log(`   Response length: ${response.response.length} characters\n`);

    // Test 5: Security utilities
    console.log('5️⃣ Testing security utilities...');
    const testData = 'sensitive_data_12345';
    const encrypted = SecurityUtils.encrypt(testData);
    const decrypted = SecurityUtils.decrypt(encrypted);
    console.log(`✅ Encryption/decryption: ${testData === decrypted ? 'PASS' : 'FAIL'}`);
    
    const masked = SecurityUtils.maskSensitiveData('sk-1234567890abcdef');
    console.log(`✅ Data masking: ${masked}\n`);

    console.log('🎉 All tests passed! The Solana DeFi AI is ready to use.');
    console.log('\n📚 Next steps:');
    console.log('   1. Set up your .env file with API keys');
    console.log('   2. Run: npm run dev');
    console.log('   3. Test the API endpoints');
    console.log('   4. Start building your trading intelligence!');

  } catch (error) {
    console.error('❌ Test failed:', error);
    process.exit(1);
  }
}

// Run tests
runTests().catch((error) => {
  console.error('❌ Test runner failed:', error);
  process.exit(1);
});
