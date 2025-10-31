#!/usr/bin/env node

/**
 * MIKEY-AI Test Runner
 * Run all essential tests in the correct order
 */

const { exec } = require('child_process');
const path = require('path');

console.log('🧪 MIKEY-AI Test Runner');
console.log('======================');
console.log('');

const tests = [
  {
    name: '🔍 Tool Detection Test',
    file: 'debug-tool-detection.js',
    description: 'Verify tool detection logic'
  },
  {
    name: '📡 Endpoint Test',
    file: 'test-updated-endpoints.js',
    description: 'Test all API endpoints'
  },
  {
    name: '🤖 Mikey AI Integration',
    file: 'test-tool-integration.js',
    description: 'Test AI tool integration'
  },
  {
    name: '🏥 Service Health Check',
    file: 'check-running-services.js',
    description: 'Check running services'
  }
];

async function runTest(test) {
  return new Promise((resolve) => {
    console.log(`\n${test.name}`);
    console.log(`📋 ${test.description}`);
    console.log(`📁 Running: ${test.file}`);
    console.log('─'.repeat(50));
    
    exec(`node ${test.file}`, (error, stdout, stderr) => {
      if (error) {
        console.log(`❌ Test failed: ${error.message}`);
        resolve(false);
      } else {
        console.log(stdout);
        if (stderr) console.log(stderr);
        console.log(`✅ Test completed`);
        resolve(true);
      }
    });
  });
}

async function main() {
  console.log('🚀 Starting MIKEY-AI Test Suite...');
  console.log('');
  
  let passed = 0;
  let total = tests.length;
  
  for (const test of tests) {
    const success = await runTest(test);
    if (success) passed++;
  }
  
  console.log('\n' + '='.repeat(50));
  console.log('📊 Test Results Summary');
  console.log('='.repeat(50));
  console.log(`✅ Passed: ${passed}/${total}`);
  console.log(`❌ Failed: ${total - passed}/${total}`);
  
  if (passed === total) {
    console.log('\n🎉 All tests passed! Mikey AI is ready.');
  } else {
    console.log('\n⚠️  Some tests failed. Check the output above.');
    console.log('\n🔧 Next steps:');
    console.log('   1. Fix backend Pyth API issue');
    console.log('   2. Ensure all services are running');
    console.log('   3. Check environment variables');
  }
  
  console.log('\n📋 Individual Tests Available:');
  console.log('   • node test-openai-direct.js');
  console.log('   • node test-google-direct.js');
  console.log('   • node test-cohere-direct.js');
  console.log('   • node test-xai-direct.js');
  console.log('   • node test-mikey-ai.js');
  console.log('   • node test-quantdesk-endpoints.js');
}

main().catch(console.error);
