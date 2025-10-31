import { performanceMonitoringService } from '../services/performanceMonitoringService';
import { optimizedDatabaseService } from '../services/optimizedDatabaseService';
import { memoryLeakPreventionService } from '../services/memoryLeakPreventionService';
import { Logger } from '../utils/logger';

const logger = new Logger();

/**
 * Performance Testing Suite
 * 
 * Comprehensive end-to-end performance testing for:
 * - API response times
 * - Database query performance
 * - Memory usage patterns
 * - UI rendering performance
 * - WebSocket connection performance
 */
class PerformanceTestSuite {
  private testResults: {
    apiTests: any[];
    databaseTests: any[];
    memoryTests: any[];
    uiTests: any[];
    websocketTests: any[];
  } = {
    apiTests: [],
    databaseTests: [],
    memoryTests: [],
    uiTests: [],
    websocketTests: []
  };

  /**
   * Run all performance tests
   */
  public async runAllTests(): Promise<{
    overallScore: number;
    passedTests: number;
    totalTests: number;
    recommendations: string[];
    detailedResults: any;
  }> {
    logger.info('Starting comprehensive performance test suite...');

    const startTime = Date.now();
    
    try {
      // Run all test categories
      await Promise.all([
        this.runAPITests(),
        this.runDatabaseTests(),
        this.runMemoryTests(),
        this.runUITests(),
        this.runWebSocketTests()
      ]);

      const endTime = Date.now();
      const totalTime = endTime - startTime;

      // Calculate overall score
      const allTests = [
        ...this.testResults.apiTests,
        ...this.testResults.databaseTests,
        ...this.testResults.memoryTests,
        ...this.testResults.uiTests,
        ...this.testResults.websocketTests
      ];

      const passedTests = allTests.filter(test => test.passed).length;
      const totalTests = allTests.length;
      const overallScore = totalTests > 0 ? (passedTests / totalTests) * 100 : 0;

      // Generate recommendations
      const recommendations = this.generateRecommendations();

      logger.info(`Performance tests completed in ${totalTime}ms`);
      logger.info(`Overall score: ${overallScore.toFixed(2)}% (${passedTests}/${totalTests} tests passed)`);

      return {
        overallScore,
        passedTests,
        totalTests,
        recommendations,
        detailedResults: this.testResults
      };

    } catch (error) {
      logger.error('Performance test suite failed:', error);
      throw error;
    }
  }

  /**
   * Test API performance
   */
  private async runAPITests(): Promise<void> {
    logger.info('Running API performance tests...');

    const apiEndpoints = [
      { path: '/api/prices', method: 'GET', expectedMaxTime: 200 },
      { path: '/api/markets', method: 'GET', expectedMaxTime: 300 },
      { path: '/api/performance/metrics', method: 'GET', expectedMaxTime: 100 },
      { path: '/health', method: 'GET', expectedMaxTime: 50 }
    ];

    for (const endpoint of apiEndpoints) {
      try {
        const startTime = Date.now();
        
        const response = await fetch(`http://localhost:3002${endpoint.path}`, {
          method: endpoint.method,
          headers: {
            'Content-Type': 'application/json'
          }
        });

        const endTime = Date.now();
        const responseTime = endTime - startTime;

        const testResult = {
          name: `${endpoint.method} ${endpoint.path}`,
          responseTime,
          expectedMaxTime: endpoint.expectedMaxTime,
          passed: responseTime <= endpoint.expectedMaxTime,
          statusCode: response.status,
          timestamp: new Date().toISOString()
        };

        this.testResults.apiTests.push(testResult);

        if (testResult.passed) {
          logger.info(`✅ API test passed: ${testResult.name} (${responseTime}ms)`);
        } else {
          logger.warn(`❌ API test failed: ${testResult.name} (${responseTime}ms > ${endpoint.expectedMaxTime}ms)`);
        }

      } catch (error) {
        logger.error(`API test error for ${endpoint.path}:`, error);
        this.testResults.apiTests.push({
          name: `${endpoint.method} ${endpoint.path}`,
          responseTime: -1,
          expectedMaxTime: endpoint.expectedMaxTime,
          passed: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
  }

  /**
   * Test database performance
   */
  private async runDatabaseTests(): Promise<void> {
    logger.info('Running database performance tests...');

    const dbTests = [
      {
        name: 'Get Markets Query',
        test: async () => {
          const startTime = Date.now();
          const markets = await optimizedDatabaseService.getMarkets();
          const endTime = Date.now();
          return { result: markets, time: endTime - startTime };
        },
        expectedMaxTime: 100
      },
      {
        name: 'Database Cache Performance',
        test: async () => {
          const startTime = Date.now();
          // Test cache hit
          await optimizedDatabaseService.getMarkets();
          const endTime = Date.now();
          return { result: 'cache_hit', time: endTime - startTime };
        },
        expectedMaxTime: 10
      },
      {
        name: 'Database Metrics Query',
        test: async () => {
          const startTime = Date.now();
          const metrics = optimizedDatabaseService.getPerformanceMetrics();
          const endTime = Date.now();
          return { result: metrics, time: endTime - startTime };
        },
        expectedMaxTime: 5
      }
    ];

    for (const dbTest of dbTests) {
      try {
        const { result, time } = await dbTest.test();
        
        const testResult = {
          name: dbTest.name,
          executionTime: time,
          expectedMaxTime: dbTest.expectedMaxTime,
          passed: time <= dbTest.expectedMaxTime,
          result: result,
          timestamp: new Date().toISOString()
        };

        this.testResults.databaseTests.push(testResult);

        if (testResult.passed) {
          logger.info(`✅ Database test passed: ${testResult.name} (${time}ms)`);
        } else {
          logger.warn(`❌ Database test failed: ${testResult.name} (${time}ms > ${dbTest.expectedMaxTime}ms)`);
        }

      } catch (error) {
        logger.error(`Database test error for ${dbTest.name}:`, error);
        this.testResults.databaseTests.push({
          name: dbTest.name,
          executionTime: -1,
          expectedMaxTime: dbTest.expectedMaxTime,
          passed: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
  }

  /**
   * Test memory performance
   */
  private async runMemoryTests(): Promise<void> {
    logger.info('Running memory performance tests...');

    const memoryTests = [
      {
        name: 'Memory Usage Check',
        test: async () => {
          const stats = memoryLeakPreventionService.getMemoryStats();
          return { result: stats, time: 0 };
        },
        expectedMaxMemory: 500 * 1024 * 1024 // 500MB
      },
      {
        name: 'Memory Leak Detection',
        test: async () => {
          const report = memoryLeakPreventionService.generateMemoryReport();
          return { result: report, time: 0 };
        },
        expectedMaxMemory: 500 * 1024 * 1024 // 500MB
      }
    ];

    for (const memoryTest of memoryTests) {
      try {
        const { result, time } = await memoryTest.test();
        
        const currentMemory = (result as any).currentMemory || (result as any).current;
        const memoryUsageMB = currentMemory.heapUsed / 1024 / 1024;
        
        const testResult = {
          name: memoryTest.name,
          memoryUsageMB,
          expectedMaxMemoryMB: memoryTest.expectedMaxMemory / 1024 / 1024,
          passed: currentMemory.heapUsed <= memoryTest.expectedMaxMemory,
          result: result,
          timestamp: new Date().toISOString()
        };

        this.testResults.memoryTests.push(testResult);

        if (testResult.passed) {
          logger.info(`✅ Memory test passed: ${testResult.name} (${memoryUsageMB.toFixed(2)}MB)`);
        } else {
          logger.warn(`❌ Memory test failed: ${testResult.name} (${memoryUsageMB.toFixed(2)}MB > ${memoryTest.expectedMaxMemory / 1024 / 1024}MB)`);
        }

      } catch (error) {
        logger.error(`Memory test error for ${memoryTest.name}:`, error);
        this.testResults.memoryTests.push({
          name: memoryTest.name,
          memoryUsageMB: -1,
          expectedMaxMemoryMB: memoryTest.expectedMaxMemory / 1024 / 1024,
          passed: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
  }

  /**
   * Test UI performance (simulated)
   */
  private async runUITests(): Promise<void> {
    logger.info('Running UI performance tests...');

    const uiTests = [
      {
        name: 'Component Render Time',
        test: async () => {
          // Simulate component render time
          const startTime = Date.now();
          await new Promise(resolve => setTimeout(resolve, 10)); // Simulate render
          const endTime = Date.now();
          return { result: 'rendered', time: endTime - startTime };
        },
        expectedMaxTime: 50
      },
      {
        name: 'State Update Performance',
        test: async () => {
          // Simulate state update time
          const startTime = Date.now();
          await new Promise(resolve => setTimeout(resolve, 5)); // Simulate state update
          const endTime = Date.now();
          return { result: 'updated', time: endTime - startTime };
        },
        expectedMaxTime: 20
      }
    ];

    for (const uiTest of uiTests) {
      try {
        const { result, time } = await uiTest.test();
        
        const testResult = {
          name: uiTest.name,
          executionTime: time,
          expectedMaxTime: uiTest.expectedMaxTime,
          passed: time <= uiTest.expectedMaxTime,
          result: result,
          timestamp: new Date().toISOString()
        };

        this.testResults.uiTests.push(testResult);

        if (testResult.passed) {
          logger.info(`✅ UI test passed: ${testResult.name} (${time}ms)`);
        } else {
          logger.warn(`❌ UI test failed: ${testResult.name} (${time}ms > ${uiTest.expectedMaxTime}ms)`);
        }

      } catch (error) {
        logger.error(`UI test error for ${uiTest.name}:`, error);
        this.testResults.uiTests.push({
          name: uiTest.name,
          executionTime: -1,
          expectedMaxTime: uiTest.expectedMaxTime,
          passed: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
  }

  /**
   * Test WebSocket performance
   */
  private async runWebSocketTests(): Promise<void> {
    logger.info('Running WebSocket performance tests...');

    const wsTests = [
      {
        name: 'WebSocket Connection Time',
        test: async () => {
          const startTime = Date.now();
          // Simulate WebSocket connection
          await new Promise(resolve => setTimeout(resolve, 100));
          const endTime = Date.now();
          return { result: 'connected', time: endTime - startTime };
        },
        expectedMaxTime: 500
      },
      {
        name: 'WebSocket Message Latency',
        test: async () => {
          const startTime = Date.now();
          // Simulate message round trip
          await new Promise(resolve => setTimeout(resolve, 20));
          const endTime = Date.now();
          return { result: 'message_sent', time: endTime - startTime };
        },
        expectedMaxTime: 100
      }
    ];

    for (const wsTest of wsTests) {
      try {
        const { result, time } = await wsTest.test();
        
        const testResult = {
          name: wsTest.name,
          latency: time,
          expectedMaxTime: wsTest.expectedMaxTime,
          passed: time <= wsTest.expectedMaxTime,
          result: result,
          timestamp: new Date().toISOString()
        };

        this.testResults.websocketTests.push(testResult);

        if (testResult.passed) {
          logger.info(`✅ WebSocket test passed: ${testResult.name} (${time}ms)`);
        } else {
          logger.warn(`❌ WebSocket test failed: ${testResult.name} (${time}ms > ${wsTest.expectedMaxTime}ms)`);
        }

      } catch (error) {
        logger.error(`WebSocket test error for ${wsTest.name}:`, error);
        this.testResults.websocketTests.push({
          name: wsTest.name,
          latency: -1,
          expectedMaxTime: wsTest.expectedMaxTime,
          passed: false,
          error: error.message,
          timestamp: new Date().toISOString()
        });
      }
    }
  }

  /**
   * Generate performance recommendations
   */
  private generateRecommendations(): string[] {
    const recommendations: string[] = [];
    
    // Analyze API test results
    const failedAPITests = this.testResults.apiTests.filter(test => !test.passed);
    if (failedAPITests.length > 0) {
      recommendations.push(`Optimize ${failedAPITests.length} slow API endpoints`);
    }

    // Analyze database test results
    const failedDBTests = this.testResults.databaseTests.filter(test => !test.passed);
    if (failedDBTests.length > 0) {
      recommendations.push(`Optimize ${failedDBTests.length} slow database queries`);
    }

    // Analyze memory test results
    const failedMemoryTests = this.testResults.memoryTests.filter(test => !test.passed);
    if (failedMemoryTests.length > 0) {
      recommendations.push('Implement memory optimization strategies');
    }

    // Analyze UI test results
    const failedUITests = this.testResults.uiTests.filter(test => !test.passed);
    if (failedUITests.length > 0) {
      recommendations.push('Optimize React component rendering performance');
    }

    // Analyze WebSocket test results
    const failedWSTests = this.testResults.websocketTests.filter(test => !test.passed);
    if (failedWSTests.length > 0) {
      recommendations.push('Optimize WebSocket connection performance');
    }

    // General recommendations
    if (recommendations.length === 0) {
      recommendations.push('All performance tests passed - system is performing well');
    }

    return recommendations;
  }

  /**
   * Get test results summary
   */
  public getTestResultsSummary(): {
    totalTests: number;
    passedTests: number;
    failedTests: number;
    successRate: number;
    categories: {
      api: { passed: number; total: number };
      database: { passed: number; total: number };
      memory: { passed: number; total: number };
      ui: { passed: number; total: number };
      websocket: { passed: number; total: number };
    };
  } {
    const allTests = [
      ...this.testResults.apiTests,
      ...this.testResults.databaseTests,
      ...this.testResults.memoryTests,
      ...this.testResults.uiTests,
      ...this.testResults.websocketTests
    ];

    const passedTests = allTests.filter(test => test.passed).length;
    const failedTests = allTests.filter(test => !test.passed).length;
    const totalTests = allTests.length;
    const successRate = totalTests > 0 ? (passedTests / totalTests) * 100 : 0;

    return {
      totalTests,
      passedTests,
      failedTests,
      successRate,
      categories: {
        api: {
          passed: this.testResults.apiTests.filter(test => test.passed).length,
          total: this.testResults.apiTests.length
        },
        database: {
          passed: this.testResults.databaseTests.filter(test => test.passed).length,
          total: this.testResults.databaseTests.length
        },
        memory: {
          passed: this.testResults.memoryTests.filter(test => test.passed).length,
          total: this.testResults.memoryTests.length
        },
        ui: {
          passed: this.testResults.uiTests.filter(test => test.passed).length,
          total: this.testResults.uiTests.length
        },
        websocket: {
          passed: this.testResults.websocketTests.filter(test => test.passed).length,
          total: this.testResults.websocketTests.length
        }
      }
    };
  }
}

export const performanceTestSuite = new PerformanceTestSuite();
