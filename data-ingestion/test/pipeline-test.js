const { redis, dbPool, logger, STREAMS } = require('../src/config');

class PipelineTester {
  constructor() {
    this.testResults = {
      redis: false,
      database: false,
      streams: false,
      consumers: false,
      dataFlow: false
    };
  }

  async runAllTests() {
    logger.info('🧪 Starting pipeline tests...');

    try {
      await this.testRedisConnection();
      await this.testDatabaseConnection();
      await this.testStreamsCreation();
      await this.testConsumerGroups();
      await this.testDataFlow();
      
      this.printResults();
    } catch (error) {
      logger.error('Test failed:', error);
    }
  }

  async testRedisConnection() {
    try {
      await redis.ping();
      logger.info('✅ Redis connection test passed');
      this.testResults.redis = true;
    } catch (error) {
      logger.error('❌ Redis connection test failed:', error.message);
    }
  }

  async testDatabaseConnection() {
    try {
      await dbPool.query('SELECT 1');
      logger.info('✅ Database connection test passed');
      this.testResults.database = true;
    } catch (error) {
      logger.error('❌ Database connection test failed:', error.message);
    }
  }

  async testStreamsCreation() {
    try {
      // Test creating streams
      await redis.xadd(STREAMS.TICKS_RAW, '*', 'test', 'data');
      await redis.xadd(STREAMS.WHALES_RAW, '*', 'test', 'data');
      await redis.xadd(STREAMS.NEWS_RAW, '*', 'test', 'data');
      await redis.xadd(STREAMS.SYSTEM_EVENTS, '*', 'test', 'data');

      // Verify streams exist
      const streams = await redis.xinfo('STREAM', STREAMS.TICKS_RAW);
      if (streams && streams.length > 0) {
        logger.info('✅ Streams creation test passed');
        this.testResults.streams = true;
      }
    } catch (error) {
      logger.error('❌ Streams creation test failed:', error.message);
    }
  }

  async testConsumerGroups() {
    try {
      const consumerGroup = 'test-group';
      
      // Create consumer group
      try {
        await redis.xgroup('CREATE', STREAMS.TICKS_RAW, consumerGroup, '$', 'MKSTREAM');
      } catch (error) {
        if (!error.message.includes('BUSYGROUP')) {
          throw error;
        }
      }

      // Test reading from consumer group
      const messages = await redis.xreadgroup(
        'GROUP', consumerGroup, 'test-consumer',
        'COUNT', 1,
        'STREAMS', STREAMS.TICKS_RAW, '>'
      );

      logger.info('✅ Consumer groups test passed');
      this.testResults.consumers = true;
    } catch (error) {
      logger.error('❌ Consumer groups test failed:', error.message);
    }
  }

  async testDataFlow() {
    try {
      // Send test price data
      const testPrice = {
        symbol: 'BTC-PERP',
        market_id: 1,
        price: 45000,
        confidence: 0.01,
        exponent: -8,
        timestamp: new Date().toISOString()
      };

      await redis.xadd(
        STREAMS.TICKS_RAW,
        '*',
        'data', JSON.stringify(testPrice),
        'timestamp', testPrice.timestamp,
        'symbol', testPrice.symbol
      );

      // Read the message back
      const messages = await redis.xrevrange(STREAMS.TICKS_RAW, '+', '-', 'COUNT', 1);
      
      if (messages && messages.length > 0) {
        const [messageId, fields] = messages[0];
        const data = JSON.parse(fields[1]);
        
        if (data.symbol === testPrice.symbol && data.price === testPrice.price) {
          logger.info('✅ Data flow test passed');
          this.testResults.dataFlow = true;
        }
      }
    } catch (error) {
      logger.error('❌ Data flow test failed:', error.message);
    }
  }

  printResults() {
    console.log('\n📊 Test Results:');
    console.log('================');
    
    for (const [test, passed] of Object.entries(this.testResults)) {
      const status = passed ? '✅ PASS' : '❌ FAIL';
      console.log(`${test.toUpperCase()}: ${status}`);
    }

    const allPassed = Object.values(this.testResults).every(result => result);
    
    if (allPassed) {
      console.log('\n🎉 All tests passed! Pipeline is ready to start.');
    } else {
      console.log('\n⚠️  Some tests failed. Please check the configuration.');
    }

    console.log('\n📋 Next Steps:');
    console.log('1. Copy env.example to .env and configure your settings');
    console.log('2. Start Redis: redis-server');
    console.log('3. Start the pipeline: ./start-pipeline.sh');
    console.log('4. Monitor at: http://localhost:3003');
  }

  async cleanup() {
    try {
      // Clean up test data
      await redis.del(STREAMS.TICKS_RAW);
      await redis.del(STREAMS.WHALES_RAW);
      await redis.del(STREAMS.NEWS_RAW);
      await redis.del(STREAMS.SYSTEM_EVENTS);
      
      logger.info('🧹 Test cleanup completed');
    } catch (error) {
      logger.error('Error during cleanup:', error);
    }
  }
}

// Run tests if called directly
if (require.main === module) {
  const tester = new PipelineTester();
  
  tester.runAllTests()
    .then(() => tester.cleanup())
    .then(() => process.exit(0))
    .catch(error => {
      logger.error('Test suite failed:', error);
      process.exit(1);
    });
}

module.exports = PipelineTester;
