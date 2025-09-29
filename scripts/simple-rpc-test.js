#!/usr/bin/env node

/**
 * Simple RPC Performance Testing Script
 * Tests RPC load balancer via API endpoints (no Solana dependencies needed)
 */

class SimpleRPCTester {
  constructor() {
    this.baseUrl = 'http://localhost:3002';
  }

  /**
   * Test API endpoint performance
   */
  async testAPIEndpoints(requests = 20) {
    console.log(`\n🌐 Testing API Endpoints (${requests} requests)...`);
    
    const results = [];
    const startTime = Date.now();
    
    for (let i = 1; i <= requests; i++) {
      try {
        const requestStart = Date.now();
        const response = await fetch(`${this.baseUrl}/api/markets`);
        const responseTime = Date.now() - requestStart;
        
        if (response.ok) {
          results.push({ success: true, responseTime, request: i });
          process.stdout.write(`Request ${i}: ✅ Success (${responseTime}ms) `);
        } else {
          results.push({ success: false, responseTime, request: i, error: `HTTP ${response.status}` });
          process.stdout.write(`Request ${i}: ❌ Failed (${response.status}) `);
        }
      } catch (error) {
        const responseTime = Date.now() - requestStart;
        results.push({ success: false, responseTime, request: i, error: error.message });
        process.stdout.write(`Request ${i}: ❌ Error `);
      }
    }
    
    const totalTime = Date.now() - startTime;
    const successful = results.filter(r => r.success).length;
    const avgResponseTime = results.reduce((sum, r) => sum + r.responseTime, 0) / results.length;
    
    console.log(`\n📊 API Endpoint Test Results:`);
    console.log(`   Successful: ${successful}/${requests} (${(successful/requests*100).toFixed(1)}%)`);
    console.log(`   Average Response Time: ${avgResponseTime.toFixed(1)}ms`);
    console.log(`   Total Time: ${totalTime}ms`);
    
    return results;
  }

  /**
   * High-frequency rapid request test
   */
  async testHighFrequencyRequests(requests = 50) {
    console.log(`\n🔥 Testing High-Frequency Requests (${requests} rapid requests)...`);
    
    const results = [];
    const startTime = Date.now();
    
    for (let i = 1; i <= requests; i++) {
      try {
        const response = await fetch(`${this.baseUrl}/api/markets`);
        if (response.ok) {
          results.push({ success: true, request: i });
          process.stdout.write('.');
        } else {
          results.push({ success: false, request: i, error: `HTTP ${response.status}` });
          process.stdout.write('X');
        }
      } catch (error) {
        results.push({ success: false, request: i, error: error.message });
        process.stdout.write('E');
      }
    }
    
    const totalTime = Date.now() - startTime;
    const successful = results.filter(r => r.success).length;
    const actualRPS = requests / (totalTime / 1000);
    
    console.log(`\n📊 High-Frequency Test Results:`);
    console.log(`   Successful: ${successful}/${requests} (${(successful/requests*100).toFixed(1)}%)`);
    console.log(`   Duration: ${totalTime}ms`);
    console.log(`   Actual RPS: ${actualRPS.toFixed(2)}`);
    
    return { results, totalTime, successful, actualRPS };
  }

  /**
   * Ultimate stress test
   */
  async testUltimateStress(requests = 100, maxDuration = 10) {
    console.log(`\n⚡ Running Ultimate Stress Test (${requests} requests, max ${maxDuration}s)...`);
    
    const results = [];
    const startTime = Date.now();
    const endTime = startTime + (maxDuration * 1000);
    
    for (let i = 1; i <= requests && Date.now() < endTime; i++) {
      try {
        const response = await fetch(`${this.baseUrl}/api/markets`);
        if (response.ok) {
          results.push({ success: true, request: i });
          process.stdout.write('.');
        } else {
          results.push({ success: false, request: i, error: `HTTP ${response.status}` });
          process.stdout.write('X');
        }
      } catch (error) {
        results.push({ success: false, request: i, error: error.message });
        process.stdout.write('E');
      }
    }
    
    const actualDuration = Date.now() - startTime;
    const successful = results.filter(r => r.success).length;
    const actualRPS = results.length / (actualDuration / 1000);
    
    console.log(`\n🚀 Ultimate Stress Test Results:`);
    console.log(`   Requests Completed: ${results.length}/${requests}`);
    console.log(`   Successful: ${successful}/${results.length} (${(successful/results.length*100).toFixed(1)}%)`);
    console.log(`   Duration: ${actualDuration}ms`);
    console.log(`   Actual RPS: ${actualRPS.toFixed(2)}`);
    
    return { results, actualDuration, successful, actualRPS };
  }

  /**
   * Test rate limit detection and recovery
   */
  async testRateLimitRecovery() {
    console.log(`\n🚨 Testing Rate Limit Detection and Recovery...`);
    
    // First, try to hit the rate limit
    console.log(`   Attempting to trigger rate limit...`);
    const results = [];
    
    for (let i = 1; i <= 120; i++) { // Try to exceed 100 req/60s limit
      try {
        const response = await fetch(`${this.baseUrl}/api/markets`);
        if (response.ok) {
          results.push({ success: true, request: i });
          process.stdout.write('.');
        } else {
          results.push({ success: false, request: i, error: `HTTP ${response.status}` });
          process.stdout.write('X');
          
          // Check if we hit rate limit
          if (response.status === 429) {
            console.log(`\n   🎯 Rate limit hit at request ${i}!`);
            const retryAfter = response.headers.get('retry-after');
            console.log(`   Retry after: ${retryAfter} seconds`);
            break;
          }
        }
      } catch (error) {
        results.push({ success: false, request: i, error: error.message });
        process.stdout.write('E');
      }
    }
    
    const successful = results.filter(r => r.success).length;
    const rateLimited = results.filter(r => r.error && r.error.includes('429')).length;
    
    console.log(`\n📊 Rate Limit Test Results:`);
    console.log(`   Requests Sent: ${results.length}`);
    console.log(`   Successful: ${successful}`);
    console.log(`   Rate Limited: ${rateLimited}`);
    
    return { results, successful, rateLimited };
  }

  /**
   * Get RPC stats
   */
  async getRPCStats() {
    try {
      const response = await fetch(`${this.baseUrl}/api/rpc/stats`);
      if (response.ok) {
        return await response.json();
      } else {
        console.log(`❌ Failed to get RPC stats: HTTP ${response.status}`);
        return null;
      }
    } catch (error) {
      console.log(`❌ Error getting RPC stats: ${error.message}`);
      return null;
    }
  }

  /**
   * Get RPC health
   */
  async getRPCHealth() {
    try {
      const response = await fetch(`${this.baseUrl}/api/rpc/health`);
      if (response.ok) {
        return await response.json();
      } else {
        console.log(`❌ Failed to get RPC health: HTTP ${response.status}`);
        return null;
      }
    } catch (error) {
      console.log(`❌ Error getting RPC health: ${error.message}`);
      return null;
    }
  }

  /**
   * Generate comprehensive report
   */
  async generateReport() {
    console.log(`\n📋 COMPREHENSIVE RPC TEST REPORT`);
    console.log(`=====================================`);
    
    // Get current stats
    const stats = await this.getRPCStats();
    const health = await this.getRPCHealth();
    
    if (stats && stats.success) {
      console.log(`\n🏗️  Load Balancer Status:`);
      console.log(`   Total Providers: ${stats.stats.providers.length}`);
      console.log(`   Healthy Providers: ${stats.stats.healthyProviders}`);
      console.log(`   Total Requests: ${stats.stats.totalRequests}`);
      
      console.log(`\n📊 Provider Performance:`);
      stats.stats.providers.forEach(provider => {
        const successRate = provider.requestCount > 0 
          ? ((provider.requestCount - provider.errorCount) / provider.requestCount * 100).toFixed(1)
          : '0.0';
        
        console.log(`   ${provider.name}:`);
        console.log(`      Status: ${provider.isHealthy ? '✅ Healthy' : '❌ Unhealthy'}`);
        console.log(`      Requests: ${provider.requestCount}`);
        console.log(`      Errors: ${provider.errorCount}`);
        console.log(`      Success Rate: ${successRate}%`);
        console.log(`      Avg Response Time: ${provider.avgResponseTime.toFixed(1)}ms`);
        console.log(`      Last Used: ${provider.lastUsed ? new Date(provider.lastUsed).toLocaleTimeString() : 'Never'}`);
      });
    }
    
    if (health && health.success) {
      console.log(`\n💚 Health Status:`);
      console.log(`   Overall Health: ${health.health.isHealthy ? '✅ Healthy' : '❌ Unhealthy'}`);
      console.log(`   Healthy Providers: ${health.health.healthyProviders}/${health.health.totalProviders}`);
    }
    
    // Performance recommendations
    console.log(`\n💡 Performance Recommendations:`);
    
    if (stats && stats.success) {
      const healthyProviders = stats.stats.providers.filter(p => p.isHealthy);
      const avgResponseTime = healthyProviders.length > 0 
        ? healthyProviders.reduce((sum, p) => sum + p.avgResponseTime, 0) / healthyProviders.length 
        : 0;
      
      if (avgResponseTime < 100) {
        console.log(`   ✅ Excellent response times (${avgResponseTime.toFixed(1)}ms average)`);
      } else if (avgResponseTime < 200) {
        console.log(`   ⚠️  Good response times (${avgResponseTime.toFixed(1)}ms average)`);
      } else {
        console.log(`   ❌ Slow response times (${avgResponseTime.toFixed(1)}ms average) - consider upgrading providers`);
      }
      
      if (stats.stats.healthyProviders < stats.stats.providers.length) {
        console.log(`   ⚠️  ${stats.stats.providers.length - stats.stats.healthyProviders} providers are unhealthy`);
      } else {
        console.log(`   ✅ All providers are healthy`);
      }
      
      if (stats.stats.totalRequests > 1000) {
        console.log(`   📈 High request volume (${stats.stats.totalRequests}) - load balancer is working well`);
      }
    }
  }

  /**
   * Run all tests
   */
  async runAllTests() {
    console.log(`🚀 Starting RPC Load Balancer Tests`);
    console.log(`=====================================`);
    
    try {
      // Test 1: API endpoint performance
      const apiResults = await this.testAPIEndpoints(20);
      
      // Test 2: High-frequency requests
      const highFreqResults = await this.testHighFrequencyRequests(50);
      
      // Test 3: Ultimate stress test
      const stressResults = await this.testUltimateStress(100, 10);
      
      // Test 4: Rate limit recovery
      const recoveryResults = await this.testRateLimitRecovery();
      
      // Generate final report
      await this.generateReport();
      
      return {
        apiResults,
        highFreqResults,
        stressResults,
        recoveryResults
      };
      
    } catch (error) {
      console.error(`❌ Test failed:`, error);
      throw error;
    }
  }

  /**
   * Run specific test type
   */
  async runSpecificTest(testType, options = {}) {
    console.log(`🚀 Running ${testType} Test`);
    console.log(`========================`);
    
    try {
      switch (testType) {
        case 'api':
          return await this.testAPIEndpoints(options.requests || 20);
          
        case 'high-freq':
          return await this.testHighFrequencyRequests(options.requests || 50);
          
        case 'stress':
          return await this.testUltimateStress(options.requests || 100, options.duration || 10);
          
        case 'rate-limit':
          return await this.testRateLimitRecovery();
          
        default:
          throw new Error(`Unknown test type: ${testType}`);
      }
    } catch (error) {
      console.error(`❌ ${testType} test failed:`, error);
      throw error;
    }
  }
}

// Run tests if this script is executed directly
if (require.main === module) {
  const tester = new SimpleRPCTester();
  
  // Check command line arguments
  const args = process.argv.slice(2);
  
  if (args.includes('--stress')) {
    const duration = parseInt(args[args.indexOf('--duration') + 1]) || 10;
    const requests = parseInt(args[args.indexOf('--requests') + 1]) || 100;
    
    console.log(`Running stress test: ${requests} requests, max ${duration}s duration`);
    tester.runSpecificTest('stress', { requests, duration }).catch(console.error);
  } else if (args.includes('--api')) {
    const requests = parseInt(args[args.indexOf('--requests') + 1]) || 20;
    console.log(`Running API test: ${requests} requests`);
    tester.runSpecificTest('api', { requests }).catch(console.error);
  } else if (args.includes('--high-freq')) {
    const requests = parseInt(args[args.indexOf('--requests') + 1]) || 50;
    console.log(`Running high-frequency test: ${requests} requests`);
    tester.runSpecificTest('high-freq', { requests }).catch(console.error);
  } else if (args.includes('--rate-limit')) {
    console.log(`Running rate limit test`);
    tester.runSpecificTest('rate-limit').catch(console.error);
  } else {
    tester.runAllTests().catch(console.error);
  }
}

module.exports = { SimpleRPCTester };
