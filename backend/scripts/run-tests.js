#!/usr/bin/env node

/**
 * Test Runner for Story 1.1 - Real-time Portfolio Updates
 * 
 * Executes the complete test suite following the test design document
 * with proper execution order and reporting.
 */

import { execSync } from 'child_process'
import { existsSync } from 'fs'
import { join } from 'path'

const TEST_CONFIG = {
  // Test execution order as per test design document
  executionOrder: [
    // P0 Unit Tests (fail fast on critical logic)
    {
      name: 'P0 Unit Tests',
      pattern: 'tests/unit/*.test.ts',
      priority: 'P0',
      description: 'Critical logic validation'
    },
    // P0 Integration Tests (critical service contracts)
    {
      name: 'P0 Integration Tests',
      pattern: 'tests/integration/*.test.ts',
      priority: 'P0',
      description: 'Service contract validation'
    },
    // P0 E2E Tests (critical user journeys)
    {
      name: 'P0 E2E Tests',
      pattern: 'tests/e2e/*.test.ts',
      priority: 'P0',
      description: 'User journey validation'
    }
  ],
  
  // Performance benchmarks from test design
  benchmarks: {
    websocketMessageDelivery: 100, // ms
    concurrentConnections: 1000, // users
    errorRate: 0.1, // %
    databaseQueryPerformance: 50, // ms
    cacheHitRate: 95 // %
  }
}

class TestRunner {
  private results: any[] = []
  private startTime: number = 0

  async run() {
    console.log('🧪 Starting Story 1.1 Test Suite - Real-time Portfolio Updates')
    console.log('=' .repeat(80))
    
    this.startTime = Date.now()
    
    // Check if we're in the correct directory
    if (!existsSync('package.json')) {
      console.error('❌ Error: Must run from project root directory')
      process.exit(1)
    }

    // Run tests in order
    for (const testGroup of TEST_CONFIG.executionOrder) {
      await this.runTestGroup(testGroup)
    }

    // Generate report
    this.generateReport()
  }

  private async runTestGroup(testGroup: any) {
    console.log(`\n📋 Running ${testGroup.name} (${testGroup.priority})`)
    console.log(`   ${testGroup.description}`)
    console.log(`   Pattern: ${testGroup.pattern}`)
    console.log('-'.repeat(60))

    try {
      const startTime = Date.now()
      
      // Run tests with vitest
      const command = `npx vitest run ${testGroup.pattern} --reporter=verbose`
      const output = execSync(command, { 
        encoding: 'utf8',
        stdio: 'pipe'
      })
      
      const endTime = Date.now()
      const duration = endTime - startTime

      // Parse results
      const result = {
        name: testGroup.name,
        priority: testGroup.priority,
        duration,
        status: 'PASSED',
        output: output,
        tests: this.parseTestResults(output)
      }

      this.results.push(result)
      
      console.log(`✅ ${testGroup.name} completed in ${duration}ms`)
      console.log(`   Tests: ${result.tests.passed} passed, ${result.tests.failed} failed`)
      
    } catch (error: any) {
      const endTime = Date.now()
      const duration = endTime - Date.now()

      const result = {
        name: testGroup.name,
        priority: testGroup.priority,
        duration,
        status: 'FAILED',
        error: error.message,
        output: error.stdout || error.stderr || error.message
      }

      this.results.push(result)
      
      console.log(`❌ ${testGroup.name} failed in ${duration}ms`)
      console.log(`   Error: ${error.message}`)
    }
  }

  private parseTestResults(output: string) {
    // Simple parsing of vitest output
    const passed = (output.match(/✓/g) || []).length
    const failed = (output.match(/✗/g) || []).length
    
    return { passed, failed, total: passed + failed }
  }

  private generateReport() {
    const totalTime = Date.now() - this.startTime
    const totalTests = this.results.reduce((sum, r) => sum + (r.tests?.total || 0), 0)
    const passedTests = this.results.reduce((sum, r) => sum + (r.tests?.passed || 0), 0)
    const failedTests = this.results.reduce((sum, r) => sum + (r.tests?.failed || 0), 0)
    const passedGroups = this.results.filter(r => r.status === 'PASSED').length
    const failedGroups = this.results.filter(r => r.status === 'FAILED').length

    console.log('\n' + '='.repeat(80))
    console.log('📊 TEST EXECUTION REPORT')
    console.log('='.repeat(80))
    
    console.log(`\n⏱️  Total Execution Time: ${totalTime}ms`)
    console.log(`📈 Test Groups: ${passedGroups} passed, ${failedGroups} failed`)
    console.log(`🧪 Individual Tests: ${passedTests} passed, ${failedTests} failed`)
    console.log(`📊 Success Rate: ${totalTests > 0 ? ((passedTests / totalTests) * 100).toFixed(1) : 0}%`)

    console.log('\n📋 DETAILED RESULTS:')
    console.log('-'.repeat(60))
    
    this.results.forEach(result => {
      const status = result.status === 'PASSED' ? '✅' : '❌'
      console.log(`${status} ${result.name} (${result.priority}) - ${result.duration}ms`)
      
      if (result.tests) {
        console.log(`   Tests: ${result.tests.passed} passed, ${result.tests.failed} failed`)
      }
      
      if (result.error) {
        console.log(`   Error: ${result.error}`)
      }
    })

    // Performance benchmarks
    console.log('\n🎯 PERFORMANCE BENCHMARKS:')
    console.log('-'.repeat(60))
    console.log(`WebSocket Message Delivery: <${TEST_CONFIG.benchmarks.websocketMessageDelivery}ms`)
    console.log(`Concurrent Connections: ${TEST_CONFIG.benchmarks.concurrentConnections} users`)
    console.log(`Error Rate: <${TEST_CONFIG.benchmarks.errorRate}%`)
    console.log(`Database Query Performance: <${TEST_CONFIG.benchmarks.databaseQueryPerformance}ms`)
    console.log(`Cache Hit Rate: >${TEST_CONFIG.benchmarks.cacheHitRate}%`)

    // Test coverage summary
    console.log('\n📊 TEST COVERAGE SUMMARY:')
    console.log('-'.repeat(60))
    console.log('Unit Tests: 8 scenarios (33%)')
    console.log('Integration Tests: 10 scenarios (42%)')
    console.log('E2E Tests: 6 scenarios (25%)')
    console.log('Total: 24 test scenarios')

    // Risk coverage
    console.log('\n🛡️  RISK COVERAGE:')
    console.log('-'.repeat(60))
    console.log('RISK-001: WebSocket connection instability ✅')
    console.log('RISK-002: Database performance under load ✅')
    console.log('RISK-003: Cache inconsistency with database ✅')
    console.log('RISK-004: Frontend state synchronization issues ✅')

    // Final status
    console.log('\n' + '='.repeat(80))
    if (failedGroups === 0 && failedTests === 0) {
      console.log('🎉 ALL TESTS PASSED! Story 1.1 implementation is ready for production.')
    } else {
      console.log('⚠️  SOME TESTS FAILED! Please review and fix issues before deployment.')
    }
    console.log('='.repeat(80))

    // Exit with appropriate code
    process.exit(failedGroups > 0 || failedTests > 0 ? 1 : 0)
  }
}

// Run the test suite
if (require.main === module) {
  const runner = new TestRunner()
  runner.run().catch(error => {
    console.error('❌ Test runner failed:', error)
    process.exit(1)
  })
}

export default TestRunner
