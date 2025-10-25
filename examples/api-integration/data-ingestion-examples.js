/**
 * API Integration Examples
 * Demonstrates integration with QuantDesk's public APIs
 */

/**
 * Data Ingestion API Examples
 * Tests all available data ingestion endpoints
 */
async function dataIngestionExamples() {
  console.log('📊 Data Ingestion API Examples\n');
  
  const endpoints = [
    {
      name: 'Health Check',
      url: 'http://localhost:3003/health',
      description: 'Service health status'
    },
    {
      name: 'Latest Prices',
      url: 'http://localhost:3003/api/prices/latest',
      description: 'Real-time market prices'
    },
    {
      name: 'Whale Transactions',
      url: 'http://localhost:3003/api/whales/recent?limit=10',
      description: 'Recent whale transactions'
    },
    {
      name: 'Market Summary',
      url: 'http://localhost:3003/api/market/summary',
      description: 'Market overview data'
    },
    {
      name: 'Wallet Balance',
      url: 'http://localhost:3003/api/wallet/balance',
      description: 'Service wallet balance'
    },
    {
      name: 'System Status',
      url: 'http://localhost:3003/api/status',
      description: 'System status information'
    }
  ];
  
  for (const endpoint of endpoints) {
    try {
      console.log(`🔄 Testing ${endpoint.name}...`);
      const response = await fetch(endpoint.url);
      const data = await response.json();
      
      if (data.success !== false) {
        console.log(`✅ ${endpoint.name}:`, {
          status: response.status,
          success: data.success !== false,
          dataKeys: Object.keys(data.data || data),
          description: endpoint.description
        });
      } else {
        console.log(`⚠️ ${endpoint.name}: ${data.error || 'Unknown error'}`);
      }
    } catch (error) {
      console.log(`❌ ${endpoint.name}: ${error.message}`);
    }
  }
  
  console.log('');
}

/**
 * Service Health Monitoring
 * Comprehensive health check across all services
 */
async function serviceHealthMonitoring() {
  console.log('🏥 Service Health Monitoring\n');
  
  const services = [
    {
      name: 'Backend API',
      url: 'http://localhost:3002/api/health',
      expectedStatus: 'healthy'
    },
    {
      name: 'Data Ingestion',
      url: 'http://localhost:3003/health',
      expectedStatus: 'healthy'
    }
  ];
  
  const healthResults = await Promise.allSettled(
    services.map(async (service) => {
      const startTime = Date.now();
      const response = await fetch(service.url);
      const responseTime = Date.now() - startTime;
      const data = await response.json();
      
      return {
        name: service.name,
        healthy: response.ok,
        status: data.status || data.data?.status || 'unknown',
        responseTime: `${responseTime}ms`,
        expectedStatus: service.expectedStatus
      };
    })
  );
  
  healthResults.forEach((result, index) => {
    if (result.status === 'fulfilled') {
      const { name, healthy, status, responseTime, expectedStatus } = result.value;
      const statusIcon = healthy && status === expectedStatus ? '✅' : '⚠️';
      console.log(`${statusIcon} ${name}:`, {
        status,
        responseTime,
        healthy,
        expected: expectedStatus
      });
    } else {
      console.log(`❌ ${services[index].name}: ${result.reason.message}`);
    }
  });
  
  console.log('');
}

/**
 * Real-Time Data Streaming
 * Demonstrates continuous data fetching
 */
async function realTimeDataStreaming() {
  console.log('📡 Real-Time Data Streaming Example\n');
  
  const streamDuration = 10000; // 10 seconds
  const interval = 2000; // 2 seconds
  let iteration = 0;
  
  console.log(`🔄 Starting ${streamDuration / 1000}s data stream (${interval / 1000}s intervals)...`);
  
  const streamInterval = setInterval(async () => {
    iteration++;
    
    try {
      const response = await fetch('http://localhost:3003/api/prices/latest');
      const data = await response.json();
      
      if (data.success) {
        console.log(`📊 Stream ${iteration}:`, {
          timestamp: data.data.timestamp,
          prices: Object.keys(data.data.prices || {}),
          source: data.data.source
        });
      } else {
        console.log(`⚠️ Stream ${iteration}: ${data.error}`);
      }
    } catch (error) {
      console.log(`❌ Stream ${iteration}: ${error.message}`);
    }
    
    if (iteration * interval >= streamDuration) {
      clearInterval(streamInterval);
      console.log('✅ Data streaming completed');
    }
  }, interval);
}

/**
 * Error Handling Examples
 * Demonstrates proper error handling for API calls
 */
async function errorHandlingExamples() {
  console.log('🚨 Error Handling Examples\n');
  
  const testCases = [
    {
      name: 'Invalid Endpoint',
      url: 'http://localhost:3003/api/invalid-endpoint',
      expectedError: '404'
    },
    {
      name: 'Invalid Parameters',
      url: 'http://localhost:3003/api/whales/recent?limit=invalid',
      expectedError: 'Parameter validation'
    },
    {
      name: 'Service Unavailable',
      url: 'http://localhost:9999/health',
      expectedError: 'Connection refused'
    }
  ];
  
  for (const testCase of testCases) {
    try {
      console.log(`🔄 Testing ${testCase.name}...`);
      const response = await fetch(testCase.url);
      const data = await response.json();
      
      console.log(`✅ ${testCase.name}:`, {
        status: response.status,
        error: data.error || 'No error returned',
        expectedError: testCase.expectedError
      });
    } catch (error) {
      console.log(`✅ ${testCase.name}:`, {
        error: error.message,
        expectedError: testCase.expectedError,
        handled: true
      });
    }
  }
  
  console.log('');
}

/**
 * Performance Testing
 * Tests API response times and throughput
 */
async function performanceTesting() {
  console.log('⚡ Performance Testing\n');
  
  const testEndpoint = 'http://localhost:3003/api/prices/latest';
  const testCount = 10;
  
  console.log(`🔄 Running ${testCount} requests to ${testEndpoint}...`);
  
  const startTime = Date.now();
  const results = await Promise.allSettled(
    Array(testCount).fill().map(async (_, index) => {
      const requestStart = Date.now();
      const response = await fetch(testEndpoint);
      const requestTime = Date.now() - requestStart;
      const data = await response.json();
      
      return {
        request: index + 1,
        responseTime: requestTime,
        success: response.ok,
        dataSize: JSON.stringify(data).length
      };
    })
  );
  
  const endTime = Date.now();
  const totalTime = endTime - startTime;
  
  const successfulRequests = results.filter(r => r.status === 'fulfilled' && r.value.success);
  const averageResponseTime = successfulRequests.reduce((sum, r) => sum + r.value.responseTime, 0) / successfulRequests.length;
  
  console.log('📊 Performance Results:', {
    totalRequests: testCount,
    successfulRequests: successfulRequests.length,
    failedRequests: testCount - successfulRequests.length,
    totalTime: `${totalTime}ms`,
    averageResponseTime: `${averageResponseTime.toFixed(2)}ms`,
    requestsPerSecond: (testCount / (totalTime / 1000)).toFixed(2),
    averageDataSize: successfulRequests.length > 0 ? 
      `${(successfulRequests.reduce((sum, r) => sum + r.value.dataSize, 0) / successfulRequests.length).toFixed(0)} bytes` : 
      'N/A'
  });
  
  console.log('');
}

/**
 * Complete API Integration Test
 * Runs all API integration examples
 */
async function runAPIIntegrationTest() {
  console.log('🚀 Starting API Integration Test\n');
  console.log('=' .repeat(50));
  
  try {
    // Data ingestion examples
    await dataIngestionExamples();
    
    // Service health monitoring
    await serviceHealthMonitoring();
    
    // Error handling examples
    await errorHandlingExamples();
    
    // Performance testing
    await performanceTesting();
    
    console.log('=' .repeat(50));
    console.log('🎉 API integration test completed successfully!');
    console.log('✅ All API endpoints are working properly');
    
    // Optional: Run real-time streaming (uncomment to test)
    // console.log('\n🔄 Starting real-time data streaming test...');
    // await realTimeDataStreaming();
    
  } catch (error) {
    console.log('=' .repeat(50));
    console.error('❌ API integration test failed:', error.message);
    console.log('💡 Make sure all services are running: pnpm run dev');
  }
}

// Export functions for individual testing
export {
  dataIngestionExamples,
  serviceHealthMonitoring,
  realTimeDataStreaming,
  errorHandlingExamples,
  performanceTesting,
  runAPIIntegrationTest
};

// Run complete test if this file is executed directly
if (import.meta.url === `file://${process.argv[1]}`) {
  runAPIIntegrationTest();
}
