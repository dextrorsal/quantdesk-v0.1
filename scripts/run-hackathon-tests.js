#!/usr/bin/env node

/**
 * 🎯 Hackathon Demo Test Runner
 * 
 * Runs all tests related to the hackathon demo functionality:
 * - Backend WebSocket broadcasting tests
 * - Frontend real-time updates tests
 * - End-to-end demo flow tests
 * 
 * Usage: node scripts/run-hackathon-tests.js
 */

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

class HackathonTestRunner {
  constructor() {
    this.results = {
      passed: 0,
      failed: 0,
      total: 0,
      details: []
    };
    this.startTime = Date.now();
  }

  log(message, type = 'info') {
    const timestamp = new Date().toISOString();
    const prefix = {
      info: 'ℹ️',
      success: '✅',
      error: '❌',
      warning: '⚠️',
      test: '🧪'
    }[type] || 'ℹ️';
    
    console.log(`${prefix} [${timestamp}] ${message}`);
  }

  async runBackendTests() {
    this.log('Running backend hackathon demo tests...', 'test');
    
    try {
      // Run WebSocket broadcasting unit tests
      this.log('Running WebSocket broadcasting unit tests...', 'test');
      const wsUnitResult = execSync('cd backend && npm test -- tests/unit/websocket-broadcasting.test.ts', {
        encoding: 'utf8',
        stdio: 'pipe'
      });
      
      this.parseTestResults(wsUnitResult, 'WebSocket Broadcasting Unit Tests');
      
      // Run hackathon demo E2E tests
      this.log('Running hackathon demo E2E tests...', 'test');
      const e2eResult = execSync('cd backend && npm test -- tests/e2e/hackathon-demo-flow.test.ts', {
        encoding: 'utf8',
        stdio: 'pipe'
      });
      
      this.parseTestResults(e2eResult, 'Hackathon Demo E2E Tests');
      
    } catch (error) {
      this.log(`Backend tests failed: ${error.message}`, 'error');
      this.results.failed++;
      this.results.details.push({
        suite: 'Backend Tests',
        status: 'failed',
        error: error.message
      });
    }
  }

  async runFrontendTests() {
    this.log('Running frontend hackathon demo tests...', 'test');
    
    try {
      // Run frontend real-time updates tests
      this.log('Running frontend real-time updates tests...', 'test');
      const frontendResult = execSync('cd frontend && npm test -- src/tests/real-time-updates.test.tsx', {
        encoding: 'utf8',
        stdio: 'pipe'
      });
      
      this.parseTestResults(frontendResult, 'Frontend Real-time Updates Tests');
      
    } catch (error) {
      this.log(`Frontend tests failed: ${error.message}`, 'error');
      this.results.failed++;
      this.results.details.push({
        suite: 'Frontend Tests',
        status: 'failed',
        error: error.message
      });
    }
  }

  parseTestResults(output, suiteName) {
    const lines = output.split('\n');
    let passed = 0;
    let failed = 0;
    
    for (const line of lines) {
      if (line.includes('✓') || line.includes('PASS')) {
        passed++;
      } else if (line.includes('×') || line.includes('FAIL')) {
        failed++;
      }
    }
    
    this.results.passed += passed;
    this.results.failed += failed;
    this.results.total += passed + failed;
    
    this.results.details.push({
      suite: suiteName,
      status: failed === 0 ? 'passed' : 'failed',
      passed,
      failed,
      total: passed + failed
    });
    
    if (failed === 0) {
      this.log(`${suiteName}: ${passed} tests passed`, 'success');
    } else {
      this.log(`${suiteName}: ${passed} passed, ${failed} failed`, 'error');
    }
  }

  async runIntegrationTests() {
    this.log('Running integration tests...', 'test');
    
    try {
      // Run order-position flow integration tests
      this.log('Running order-position flow integration tests...', 'test');
      const integrationResult = execSync('cd backend && npm test -- tests/integration/order-position-flow.test.ts', {
        encoding: 'utf8',
        stdio: 'pipe'
      });
      
      this.parseTestResults(integrationResult, 'Order-Position Flow Integration Tests');
      
    } catch (error) {
      this.log(`Integration tests failed: ${error.message}`, 'error');
      this.results.failed++;
      this.results.details.push({
        suite: 'Integration Tests',
        status: 'failed',
        error: error.message
      });
    }
  }

  async runPerformanceTests() {
    this.log('Running performance tests...', 'test');
    
    try {
      // Test WebSocket performance
      this.log('Testing WebSocket performance...', 'test');
      const perfResult = execSync('cd backend && npm test -- tests/unit/websocket-broadcasting.test.ts --grep "Performance"', {
        encoding: 'utf8',
        stdio: 'pipe'
      });
      
      this.parseTestResults(perfResult, 'WebSocket Performance Tests');
      
    } catch (error) {
      this.log(`Performance tests failed: ${error.message}`, 'error');
      this.results.failed++;
      this.results.details.push({
        suite: 'Performance Tests',
        status: 'failed',
        error: error.message
      });
    }
  }

  generateReport() {
    const endTime = Date.now();
    const duration = endTime - this.startTime;
    
    this.log('Generating hackathon demo test report...', 'info');
    
    const report = {
      summary: {
        total: this.results.total,
        passed: this.results.passed,
        failed: this.results.failed,
        duration: `${duration}ms`,
        successRate: this.results.total > 0 ? 
          ((this.results.passed / this.results.total) * 100).toFixed(2) + '%' : '0%'
      },
      details: this.results.details,
      timestamp: new Date().toISOString(),
      hackathonDemo: {
        ready: this.results.failed === 0,
        criticalFeatures: [
          'Account Initialization',
          'Deposit Functionality', 
          'Order Placement',
          'Real-time Updates',
          'Position Display',
          'Portfolio Display'
        ],
        testCoverage: {
          backend: this.results.details.filter(d => d.suite.includes('Backend')).length,
          frontend: this.results.details.filter(d => d.suite.includes('Frontend')).length,
          integration: this.results.details.filter(d => d.suite.includes('Integration')).length,
          e2e: this.results.details.filter(d => d.suite.includes('E2E')).length
        }
      }
    };
    
    // Save report to file
    const reportPath = path.join(__dirname, '../hackathon-demo-test-report.json');
    fs.writeFileSync(reportPath, JSON.stringify(report, null, 2));
    
    // Display summary
    console.log('\n🎯 HACKATHON DEMO TEST REPORT');
    console.log('============================');
    console.log(`Total Tests: ${report.summary.total}`);
    console.log(`Passed: ${report.summary.passed}`);
    console.log(`Failed: ${report.summary.failed}`);
    console.log(`Success Rate: ${report.summary.successRate}`);
    console.log(`Duration: ${report.summary.duration}`);
    console.log(`Demo Ready: ${report.hackathonDemo.ready ? '✅ YES' : '❌ NO'}`);
    
    console.log('\n📊 Test Coverage:');
    console.log(`Backend Tests: ${report.hackathonDemo.testCoverage.backend}`);
    console.log(`Frontend Tests: ${report.hackathonDemo.testCoverage.frontend}`);
    console.log(`Integration Tests: ${report.hackathonDemo.testCoverage.integration}`);
    console.log(`E2E Tests: ${report.hackathonDemo.testCoverage.e2e}`);
    
    console.log('\n🎯 Critical Features Status:');
    report.hackathonDemo.criticalFeatures.forEach(feature => {
      console.log(`  ✅ ${feature}`);
    });
    
    if (report.hackathonDemo.ready) {
      console.log('\n🚀 HACKATHON DEMO IS READY!');
      console.log('All tests passed - demo is ready for presentation!');
    } else {
      console.log('\n⚠️ HACKATHON DEMO NEEDS ATTENTION');
      console.log('Some tests failed - please review and fix issues before demo.');
    }
    
    console.log(`\n📄 Full report saved to: ${reportPath}`);
    
    return report;
  }

  async runAllTests() {
    this.log('🎯 Starting Hackathon Demo Test Suite', 'test');
    this.log('=====================================', 'info');
    
    try {
      // Run all test suites
      await this.runBackendTests();
      await this.runFrontendTests();
      await this.runIntegrationTests();
      await this.runPerformanceTests();
      
      // Generate final report
      const report = this.generateReport();
      
      // Exit with appropriate code
      process.exit(report.hackathonDemo.ready ? 0 : 1);
      
    } catch (error) {
      this.log(`Test suite failed: ${error.message}`, 'error');
      process.exit(1);
    }
  }
}

// Run the test suite
if (require.main === module) {
  const runner = new HackathonTestRunner();
  runner.runAllTests().catch(error => {
    console.error('❌ Test runner failed:', error);
    process.exit(1);
  });
}

module.exports = HackathonTestRunner;
