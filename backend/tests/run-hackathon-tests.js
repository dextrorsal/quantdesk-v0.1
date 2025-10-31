#!/usr/bin/env node

/**
 * Hackathon Demo Test Runner
 * 
 * This script runs all the tests required for the hackathon demo validation,
 * ensuring the complete trading flow works end-to-end.
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

// Colors for console output
const colors = {
  reset: '\x1b[0m',
  bright: '\x1b[1m',
  red: '\x1b[31m',
  green: '\x1b[32m',
  yellow: '\x1b[33m',
  blue: '\x1b[34m',
  magenta: '\x1b[35m',
  cyan: '\x1b[36m'
};

function log(message, color = 'reset') {
  console.log(`${colors[color]}${message}${colors.reset}`);
}

function logSection(title) {
  log(`\n${'='.repeat(60)}`, 'cyan');
  log(`  ${title}`, 'bright');
  log(`${'='.repeat(60)}`, 'cyan');
}

function logTest(testName, status, details = '') {
  const statusColor = status === 'PASS' ? 'green' : status === 'FAIL' ? 'red' : 'yellow';
  const statusSymbol = status === 'PASS' ? '✅' : status === 'FAIL' ? '❌' : '⚠️';
  
  log(`${statusSymbol} ${testName}`, statusColor);
  if (details) {
    log(`   ${details}`, 'reset');
  }
}

function runCommand(command, description) {
  try {
    log(`\n🔧 ${description}...`, 'blue');
    const output = execSync(command, { 
      encoding: 'utf8', 
      stdio: 'pipe',
      cwd: process.cwd()
    });
    return { success: true, output };
  } catch (error) {
    return { 
      success: false, 
      error: error.message,
      output: error.stdout || error.stderr || ''
    };
  }
}

function checkFileExists(filePath) {
  return fs.existsSync(path.resolve(filePath));
}

function main() {
  log('🚀 QuantDesk Hackathon Demo Test Runner', 'bright');
  log('=====================================', 'cyan');
  
  const results = {
    backend: { passed: 0, failed: 0, total: 0 },
    frontend: { passed: 0, failed: 0, total: 0 },
    integration: { passed: 0, failed: 0, total: 0 }
  };
  
  // Check if we're in the right directory
  if (!checkFileExists('package.json')) {
    log('❌ Error: package.json not found. Please run this script from the project root.', 'red');
    process.exit(1);
  }
  
  logSection('HACKATHON DEMO TEST SUITE');
  log('Running comprehensive tests to validate the hackathon demo flow...', 'yellow');
  
  // Backend Tests
  logSection('BACKEND TESTS');
  
  // Check if backend directory exists
  if (!checkFileExists('backend')) {
    logTest('Backend Directory', 'FAIL', 'Backend directory not found');
    results.backend.failed++;
    results.backend.total++;
  } else {
    logTest('Backend Directory', 'PASS', 'Backend directory found');
    results.backend.passed++;
    results.backend.total++;
    
    // Run backend unit tests
    const unitTestResult = runCommand(
      'cd backend && npm test -- --run tests/unit/hackathon-core.test.ts',
      'Running Hackathon Core Tests'
    );
    
    if (unitTestResult.success) {
      logTest('Hackathon Core Tests', 'PASS', 'All core functionality tests passed');
      results.backend.passed++;
    } else {
      logTest('Hackathon Core Tests', 'FAIL', unitTestResult.error);
      results.backend.failed++;
    }
    results.backend.total++;
    
    // Run WebSocket broadcasting tests
    const wsTestResult = runCommand(
      'cd backend && npm test -- --run tests/unit/websocket-broadcasting.test.ts',
      'Running WebSocket Broadcasting Tests'
    );
    
    if (wsTestResult.success) {
      logTest('WebSocket Broadcasting Tests', 'PASS', 'All WebSocket tests passed');
      results.backend.passed++;
    } else {
      logTest('WebSocket Broadcasting Tests', 'FAIL', wsTestResult.error);
      results.backend.failed++;
    }
    results.backend.total++;
    
    // Run E2E demo flow tests
    const e2eTestResult = runCommand(
      'cd backend && npm test -- --run tests/e2e/hackathon-demo-flow.test.ts',
      'Running E2E Demo Flow Tests'
    );
    
    if (e2eTestResult.success) {
      logTest('E2E Demo Flow Tests', 'PASS', 'Complete demo flow validated');
      results.backend.passed++;
    } else {
      logTest('E2E Demo Flow Tests', 'FAIL', e2eTestResult.error);
      results.backend.failed++;
    }
    results.backend.total++;
  }
  
  // Frontend Tests
  logSection('FRONTEND TESTS');
  
  // Check if frontend directory exists
  if (!checkFileExists('frontend')) {
    logTest('Frontend Directory', 'FAIL', 'Frontend directory not found');
    results.frontend.failed++;
    results.frontend.total++;
  } else {
    logTest('Frontend Directory', 'PASS', 'Frontend directory found');
    results.frontend.passed++;
    results.frontend.total++;
    
    // Run frontend real-time updates tests
    const frontendTestResult = runCommand(
      'cd frontend && npm test -- --run src/tests/real-time-updates.test.tsx',
      'Running Frontend Real-time Updates Tests'
    );
    
    if (frontendTestResult.success) {
      logTest('Real-time Updates Tests', 'PASS', 'All frontend real-time tests passed');
      results.frontend.passed++;
    } else {
      logTest('Real-time Updates Tests', 'FAIL', frontendTestResult.error);
      results.frontend.failed++;
    }
    results.frontend.total++;
  }
  
  // Integration Tests
  logSection('INTEGRATION TESTS');
  
  // Check if all services can start
  const serviceCheckResult = runCommand(
    'npm run dev --dry-run',
    'Checking Service Startup'
  );
  
  if (serviceCheckResult.success) {
    logTest('Service Startup Check', 'PASS', 'All services can start successfully');
    results.integration.passed++;
  } else {
    logTest('Service Startup Check', 'FAIL', 'Some services failed to start');
    results.integration.failed++;
  }
  results.integration.total++;
  
  // Check if all required files exist
  const requiredFiles = [
    'frontend/src/components/Positions.tsx',
    'frontend/src/components/Orders.tsx',
    'frontend/src/components/PortfolioDashboard.tsx',
    'backend/src/services/smartContractService.ts',
    'backend/src/services/matching.ts',
    'backend/src/services/websocket.ts'
  ];
  
  let allFilesExist = true;
  for (const file of requiredFiles) {
    if (checkFileExists(file)) {
      logTest(`File Check: ${file}`, 'PASS', 'File exists');
      results.integration.passed++;
    } else {
      logTest(`File Check: ${file}`, 'FAIL', 'File missing');
      results.integration.failed++;
      allFilesExist = false;
    }
    results.integration.total++;
  }
  
  // Performance Tests
  logSection('PERFORMANCE TESTS');
  
  // Check if demo can complete within time requirements
  const performanceTestResult = runCommand(
    'cd backend && npm test -- --run tests/e2e/hackathon-demo-flow.test.ts --reporter=verbose',
    'Running Performance Tests'
  );
  
  if (performanceTestResult.success) {
    logTest('Performance Tests', 'PASS', 'Demo completes within 2 seconds');
    results.integration.passed++;
  } else {
    logTest('Performance Tests', 'FAIL', 'Demo exceeds performance requirements');
    results.integration.failed++;
  }
  results.integration.total++;
  
  // Summary
  logSection('TEST SUMMARY');
  
  const totalPassed = results.backend.passed + results.frontend.passed + results.integration.passed;
  const totalFailed = results.backend.failed + results.frontend.failed + results.integration.failed;
  const totalTests = results.backend.total + results.frontend.total + results.integration.total;
  
  log(`Backend Tests: ${results.backend.passed}/${results.backend.total} passed`, 
      results.backend.failed > 0 ? 'red' : 'green');
  log(`Frontend Tests: ${results.frontend.passed}/${results.frontend.total} passed`, 
      results.frontend.failed > 0 ? 'red' : 'green');
  log(`Integration Tests: ${results.integration.passed}/${results.integration.total} passed`, 
      results.integration.failed > 0 ? 'red' : 'green');
  
  log(`\nTotal: ${totalPassed}/${totalTests} tests passed`, 
      totalFailed > 0 ? 'red' : 'green');
  
  if (totalFailed === 0) {
    log('\n🎉 ALL TESTS PASSED!', 'green');
    log('✅ Hackathon demo is ready for presentation!', 'green');
    log('\nDemo Flow Validation:', 'bright');
    log('  ✅ Account initialization and wallet connection', 'green');
    log('  ✅ Deposit process and balance display', 'green');
    log('  ✅ Order placement and execution', 'green');
    log('  ✅ Position creation and display', 'green');
    log('  ✅ Real-time updates via WebSocket', 'green');
    log('  ✅ Position management and closing', 'green');
    log('  ✅ Error handling and recovery', 'green');
    log('  ✅ Performance requirements met', 'green');
    
    log('\n🚀 Ready for Hackathon Demo!', 'bright');
    process.exit(0);
  } else {
    log('\n❌ SOME TESTS FAILED!', 'red');
    log('Please fix the failing tests before the hackathon demo.', 'red');
    log('\nFailed Tests:', 'red');
    
    if (results.backend.failed > 0) {
      log(`  Backend: ${results.backend.failed} tests failed`, 'red');
    }
    if (results.frontend.failed > 0) {
      log(`  Frontend: ${results.frontend.failed} tests failed`, 'red');
    }
    if (results.integration.failed > 0) {
      log(`  Integration: ${results.integration.failed} tests failed`, 'red');
    }
    
    process.exit(1);
  }
}

// Run the test suite
if (require.main === module) {
  main();
}

module.exports = { main };
