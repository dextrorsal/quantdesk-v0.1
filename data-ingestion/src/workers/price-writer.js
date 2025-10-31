const { redis, dbPool, logger, STREAMS, BATCH_CONFIG } = require('../config');

class PriceWriter {
  constructor() {
    this.isRunning = false;
    this.batchBuffer = [];
    this.lastFlushTime = Date.now();
    this.consumerGroup = 'price-writers';
    this.consumerName = `price-writer-${process.pid}`;
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Price writer already running');
      return;
    }

    logger.info('Starting price writer...');
    this.isRunning = true;

    // Create consumer group
    await this.createConsumerGroup();

    // Start processing messages
    this.processMessages();

    // Start batch flush timer
    this.startBatchTimer();

    logger.info('Price writer started');
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping price writer...');
    this.isRunning = false;

    // Flush remaining batch
    await this.flushBatch();

    logger.info('Price writer stopped');
  }

  async createConsumerGroup() {
    try {
      await redis.xgroup('CREATE', STREAMS.TICKS_RAW, this.consumerGroup, '$', 'MKSTREAM');
      logger.info(`Created consumer group: ${this.consumerGroup}`);
    } catch (error) {
      if (error.message.includes('BUSYGROUP')) {
        logger.info(`Consumer group ${this.consumerGroup} already exists`);
      } else {
        logger.error('Error creating consumer group:', error);
        throw error;
      }
    }
  }

  async processMessages() {
    while (this.isRunning) {
      try {
        const messages = await redis.xreadgroup(
          'GROUP', this.consumerGroup, this.consumerName,
          'COUNT', BATCH_CONFIG.size,
          'BLOCK', 1000,
          'STREAMS', STREAMS.TICKS_RAW, '>'
        );

        if (messages && messages.length > 0) {
          const streamMessages = messages[0][1];
          
          for (const [messageId, fields] of streamMessages) {
            try {
              await this.processMessage(messageId, fields);
            } catch (error) {
              logger.error(`Error processing message ${messageId}:`, error);
              // Acknowledge message even if processing failed to prevent infinite retry
              await redis.xack(STREAMS.TICKS_RAW, this.consumerGroup, messageId);
            }
          }
        }
      } catch (error) {
        logger.error('Error reading messages:', error);
        await this.sleep(1000); // Wait before retrying
      }
    }
  }

  async processMessage(messageId, fields) {
    try {
      const data = JSON.parse(fields[1]); // fields[0] is 'data', fields[1] is the JSON
      
      // Add to batch buffer
      this.batchBuffer.push({
        market_id: data.market_id,
        price: parseFloat(data.price),
        confidence: parseFloat(data.confidence),
        exponent: parseInt(data.exponent),
        created_at: new Date(data.timestamp)
      });

      // Acknowledge message
      await redis.xack(STREAMS.TICKS_RAW, this.consumerGroup, messageId);

      // Flush if batch is full
      if (this.batchBuffer.length >= BATCH_CONFIG.size) {
        await this.flushBatch();
      }
    } catch (error) {
      logger.error('Error processing price message:', error);
      throw error;
    }
  }

  async flushBatch() {
    if (this.batchBuffer.length === 0) {
      return;
    }

    try {
      const startTime = Date.now();
      
      // Prepare batch insert query
      const values = this.batchBuffer.map((price, index) => {
        const offset = index * 5;
        return `($${offset + 1}, $${offset + 2}, $${offset + 3}, $${offset + 4}, $${offset + 5})`;
      }).join(', ');

      const query = `
        INSERT INTO oracle_prices (market_id, price, confidence, exponent, created_at)
        VALUES ${values}
        ON CONFLICT (market_id, created_at) DO UPDATE SET
          price = EXCLUDED.price,
          confidence = EXCLUDED.confidence,
          exponent = EXCLUDED.exponent
      `;

      // Flatten parameters
      const params = this.batchBuffer.flatMap(price => [
        price.market_id,
        price.price,
        price.confidence,
        price.exponent,
        price.created_at
      ]);

      // Execute batch insert
      await dbPool.query(query, params);

      const duration = Date.now() - startTime;
      logger.info(`Flushed ${this.batchBuffer.length} price records in ${duration}ms`);

      // Clear buffer
      this.batchBuffer = [];
      this.lastFlushTime = Date.now();

    } catch (error) {
      logger.error('Error flushing price batch:', error);
      throw error;
    }
  }

  startBatchTimer() {
    setInterval(async () => {
      if (this.batchBuffer.length > 0) {
        const timeSinceLastFlush = Date.now() - this.lastFlushTime;
        if (timeSinceLastFlush >= BATCH_CONFIG.intervalMs) {
          await this.flushBatch();
        }
      }
    }, BATCH_CONFIG.intervalMs);
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Start writer if run directly
if (require.main === module) {
  const writer = new PriceWriter();
  
  writer.start().catch(error => {
    logger.error('Failed to start price writer:', error);
    process.exit(1);
  });
}

module.exports = PriceWriter;
