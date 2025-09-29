#!/usr/bin/env node

/**
 * RPC Load Balancer Testing Script
 * Tests RPC provider performance, rate limits, and load balancing
 */

const { Connection, PublicKey, LAMPORTS_PER_SOL } = require('@solana/web3.js');
const { RPCLoadBalancer } = require('../backend/src/services/rpcLoadBalancer');
const { SolanaService } = require('../backend/src/services/solana');

class RPCTester {
  constructor() {
    this.loadBalancer = RPCLoadBalancer.getInstance();
    this.solanaService = SolanaService.getInstance();
    this.results = [];
    this.providerStats = new Map();
  }

  /**
   * Test individual RPC provider
   */
  async testProvider(providerName, url) {
    const startTime = Date.now();
    
    try {
      const connection = new Connection(url, 'confirmed');
      
      // Test basic RPC calls
      await connection.getLatestBlockhash();
      await connection.getSlot();
      
      const responseTime = Date.now() - startTime;
      
      return {
        provider: providerName,
        success: true,
        responseTime,
        rateLimited: false
      };
    } catch (error) {
      const responseTime = Date.now() - startTime;
      const isRateLimited = this.isRateLimitError(error);
      
      return {
        provider: providerName,
        success: false,
        responseTime,
        error: error.message,
        rateLimited: isRateLimited
      };
    }
  }

  /**
   * Test load balancer with multiple concurrent requests
   */
  async testLoadBalancer(concurrentRequests = 10) {
    console.log(`\n🔄 Testing Load Balancer with ${concurrentRequests} concurrent requests...`);
    
    const promises = Array.from({ length: concurrentRequests }, async (_, index) => {
      const startTime = Date.now();
      
      try {
        // Test various RPC operations
        const connection = this.loadBalancer.getConnection();
        await connection.getLatestBlockhash();
        await connection.getSlot();
        
        const responseTime = Date.now() - startTime;
        
        return {
          provider: 'LoadBalancer',
          success: true,
          responseTime,
          rateLimited: false
        };
      } catch (error) {
        const responseTime = Date.now() - startTime;
        const isRateLimited = this.isRateLimitError(error);
        
        return {
          provider: 'LoadBalancer',
          success: false,
          responseTime,
          error: error.message,
          rateLimited: isRateLimited
        };
      }
    });

    return Promise.all(promises);
  }

  /**
   * Stress test with high frequency requests
   */
  async stressTest(duration = 30, requestsPerSecond = 10) {
    console.log(`\n⚡ Starting Stress Test: ${duration}s duration, ${requestsPerSecond} req/s`);
    
    const startTime = Date.now();
    const endTime = startTime + (duration * 1000);
    const requestInterval = 1000 / requestsPerSecond;
    
    let requestCount = 0;
    let successCount = 0;
    let rateLimitedCount = 0;
    
    while (Date.now() < endTime) {
      const requestStart = Date.now();
      
      try {
        await this.solanaService.getCurrentSlot();
        requestCount++;
        successCount++;
        
        // Log progress every 5 seconds
        if (requestCount % (requestsPerSecond * 5) === 0) {
          const elapsed = (Date.now() - startTime) / 1000;
          console.log(`📊 ${requestCount} requests completed in ${elapsed.toFixed(1)}s`);
        }
      } catch (error) {
        requestCount++;
        if (this.isRateLimitError(error)) {
          rateLimitedCount++;
        }
        console.error(`❌ Request ${requestCount} failed:`, error.message);
      }
      
      // Maintain request rate
      const requestDuration = Date.now() - requestStart;
      const sleepTime = Math.max(0, requestInterval - requestDuration);
      
      if (sleepTime > 0) {
        await new Promise(resolve => setTimeout(resolve, sleepTime));
      }
    }
    
    const totalTime = (Date.now() - startTime) / 1000;
    const actualRPS = requestCount / totalTime;
    const successRate = (successCount / requestCount * 100).toFixed(1);
    
    console.log(`\n📈 Stress Test Complete:`);
    console.log(`   Total Requests: ${requestCount}`);
    console.log(`   Successful: ${successCount} (${successRate}%)`);
    console.log(`   Rate Limited: ${rateLimitedCount}`);
    console.log(`   Duration: ${totalTime.toFixed(1)}s`);
    console.log(`   Actual RPS: ${actualRPS.toFixed(2)}`);
    
    return {
      totalRequests: requestCount,
      successfulRequests: successCount,
      rateLimitedRequests: rateLimitedCount,
      successRate: parseFloat(successRate),
      actualRPS: actualRPS,
      duration: totalTime
    };
  }

  /**
   * Test rate limit detection
   */
  async testRateLimits() {
    console.log(`\n🚨 Testing Rate Limit Detection...`);
    
    // Get current provider stats
    const stats = this.solanaService.getRPCStats();
    
    console.log(`\n📊 Current Provider Status:`);
    stats.providers.forEach(provider => {
      console.log(`   ${provider.name}: ${provider.isHealthy ? '✅' : '❌'} (${provider.requestCount} requests, ${provider.errorCount} errors)`);
    });
    
    // Test with rapid requests to trigger rate limits
    const rapidRequests = 50;
    console.log(`\n🔥 Sending ${rapidRequests} rapid requests to test rate limits...`);
    
    const promises = Array.from({ length: rapidRequests }, async (_, index) => {
      try {
        await this.solanaService.getCurrentSlot();
        return { success: true, rateLimited: false };
      } catch (error) {
        return { 
          success: false, 
          rateLimited: this.isRateLimitError(error),
          error: error.message 
        };
      }
    });
    
    const results = await Promise.all(promises);
    
    const successful = results.filter(r => r.success).length;
    const rateLimited = results.filter(r => r.rateLimited).length;
    const failed = results.filter(r => !r.success && !r.rateLimited).length;
    
    console.log(`\n📊 Rate Limit Test Results:`);
    console.log(`   Successful: ${successful}/${rapidRequests}`);
    console.log(`   Rate Limited: ${rateLimited}`);
    console.log(`   Other Failures: ${failed}`);
    
    if (rateLimited > 0) {
      console.log(`   ⚠️  Rate limits detected! Load balancer should switch providers.`);
    } else {
      console.log(`   ✅ No rate limits detected - load balancer working well!`);
    }
    
    return { successful, rateLimited, failed };
  }

  /**
   * Measure typical RPC speeds
   */
  async measureRPCSpeeds() {
    console.log(`\n⏱️  Measuring RPC Response Times...`);
    
    const testOperations = [
      { 
        name: 'getLatestBlockhash', 
        operation: () => this.solanaService.getLatestBlockhash() 
      },
      { 
        name: 'getCurrentSlot', 
        operation: () => this.solanaService.getCurrentSlot() 
      },
      { 
        name: 'getBalance', 
        operation: () => this.solanaService.getBalance(new PublicKey('11111111111111111111111111111112')) 
      },
    ];
    
    const results = {};
    
    for (const test of testOperations) {
      const times = [];
      const iterations = 10;
      
      console.log(`\n   Testing ${test.name}...`);
      
      for (let i = 0; i < iterations; i++) {
        try {
          const startTime = Date.now();
          await test.operation();
          const responseTime = Date.now() - startTime;
          times.push(responseTime);
          
          process.stdout.write(`   ${i + 1}/${iterations} (${responseTime}ms) `);
        } catch (error) {
          console.log(`   ❌ Failed: ${error.message}`);
        }
      }
      
      if (times.length > 0) {
        const avg = times.reduce((a, b) => a + b, 0) / times.length;
        const min = Math.min(...times);
        const max = Math.max(...times);
        
        console.log(`\n   📊 ${test.name} Results:`);
        console.log(`      Average: ${avg.toFixed(1)}ms`);
        console.log(`      Min: ${min}ms`);
        console.log(`      Max: ${max}ms`);
        console.log(`      Success Rate: ${times.length}/${iterations} (${(times.length/iterations*100).toFixed(1)}%)`);
        
        results[test.name] = {
          average: avg,
          min: min,
          max: max,
          successRate: times.length / iterations,
          times: times
        };
      }
    }
    
    return results;
  }

  /**
   * Check if error is rate limit related
   */
  isRateLimitError(error) {
    const message = error.message?.toLowerCase() || '';
    return message.includes('rate limit') || 
           message.includes('429') || 
           message.includes('too many requests') ||
           error.code === 429;
  }

  /**
   * Generate comprehensive report
   */
  generateReport() {
    console.log(`\n📋 COMPREHENSIVE RPC TEST REPORT`);
    console.log(`=====================================`);
    
    // Load balancer stats
    const stats = this.solanaService.getRPCStats();
    
    console.log(`\n🏗️  Load Balancer Status:`);
    console.log(`   Total Providers: ${stats.providers.length}`);
    console.log(`   Healthy Providers: ${stats.healthyProviders}`);
    console.log(`   Total Requests: ${stats.totalRequests}`);
    
    console.log(`\n📊 Provider Performance:`);
    stats.providers.forEach(provider => {
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
    
    // Performance recommendations
    console.log(`\n💡 Performance Recommendations:`);
    
    const healthyProviders = stats.providers.filter(p => p.isHealthy);
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
    
    if (stats.healthyProviders < stats.providers.length) {
      console.log(`   ⚠️  ${stats.providers.length - stats.healthyProviders} providers are unhealthy`);
    } else {
      console.log(`   ✅ All providers are healthy`);
    }
    
    if (stats.totalRequests > 1000) {
      console.log(`   📈 High request volume (${stats.totalRequests}) - load balancer is working well`);
    }
  }

  /**
   * Test API endpoint performance (like we did manually)
   */
  async testAPIEndpoints(requests = 20) {
    console.log(`\n🌐 Testing API Endpoints (${requests} requests)...`);
    
    const results = [];
    const startTime = Date.now();
    
    for (let i = 1; i <= requests; i++) {
      try {
        const requestStart = Date.now();
        const response = await fetch('http://localhost:3002/api/markets');
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
        const response = await fetch('http://localhost:3002/api/markets');
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
   * Ultimate stress test (like our manual test)
   */
  async testUltimateStress(requests = 100, maxDuration = 10) {
    console.log(`\n⚡ Running Ultimate Stress Test (${requests} requests, max ${maxDuration}s)...`);
    
    const results = [];
    const startTime = Date.now();
    const endTime = startTime + (maxDuration * 1000);
    
    for (let i = 1; i <= requests && Date.now() < endTime; i++) {
      try {
        const response = await fetch('http://localhost:3002/api/markets');
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
        const response = await fetch('http://localhost:3002/api/markets');
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
   * Run all tests
   */
  async runAllTests() {
    console.log(`🚀 Starting RPC Load Balancer Tests`);
    console.log(`=====================================`);
    
    try {
      // Test 1: Measure RPC speeds
      const speedResults = await this.measureRPCSpeeds();
      
      // Test 2: Test load balancer
      const loadBalancerResults = await this.testLoadBalancer(20);
      console.log(`\n📊 Load Balancer Test Results:`);
      console.log(`   Successful: ${loadBalancerResults.filter(r => r.success).length}/${loadBalancerResults.length}`);
      
      // Test 3: Rate limit detection
      const rateLimitResults = await this.testRateLimits();
      
      // Test 4: API endpoint performance (like our manual tests)
      const apiResults = await this.testAPIEndpoints(20);
      
      // Test 5: High-frequency requests
      const highFreqResults = await this.testHighFrequencyRequests(50);
      
      // Test 6: Ultimate stress test
      const stressResults = await this.testUltimateStress(100, 10);
      
      // Test 7: Rate limit recovery
      const recoveryResults = await this.testRateLimitRecovery();
      
      // Generate final report
      this.generateReport();
      
      return {
        speedResults,
        loadBalancerResults,
        rateLimitResults,
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
          
        case 'speeds':
          return await this.measureRPCSpeeds();
          
        case 'load-balancer':
          return await this.testLoadBalancer(options.requests || 20);
          
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
  const tester = new RPCTester();
  
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
  } else if (args.includes('--speeds')) {
    console.log(`Running RPC speed test`);
    tester.runSpecificTest('speeds').catch(console.error);
  } else if (args.includes('--load-balancer')) {
    const requests = parseInt(args[args.indexOf('--requests') + 1]) || 20;
    console.log(`Running load balancer test: ${requests} requests`);
    tester.runSpecificTest('load-balancer', { requests }).catch(console.error);
  } else {
    tester.runAllTests().catch(console.error);
  }
}

module.exports = { RPCTester };
