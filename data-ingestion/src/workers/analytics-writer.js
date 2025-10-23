const { redis, dbPool, logger, STREAMS, BATCH_CONFIG } = require('../config');

class AnalyticsWriter {
  constructor() {
    this.isRunning = false;
    this.batchBuffer = [];
    this.lastFlushTime = Date.now();
    this.consumerGroup = 'analytics-writers';
    this.consumerName = `analytics-writer-${process.pid}`;
    this.marketStatsCache = new Map();
    this.userStatsCache = new Map();
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Analytics writer already running');
      return;
    }

    logger.info('Starting analytics writer...');
    this.isRunning = true;

    // Create consumer groups for all streams
    await this.createConsumerGroups();

    // Start processing messages from all streams
    this.processWhaleMessages();
    this.processNewsMessages();
    this.processSystemMessages();

    // Start batch flush timer
    this.startBatchTimer();

    logger.info('Analytics writer started');
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping analytics writer...');
    this.isRunning = false;

    // Flush remaining batches
    await this.flushAllBatches();

    logger.info('Analytics writer stopped');
  }

  async createConsumerGroups() {
    const streams = [STREAMS.WHALES_RAW, STREAMS.NEWS_RAW, STREAMS.SYSTEM_EVENTS];
    
    for (const stream of streams) {
      try {
        await redis.xgroup('CREATE', stream, this.consumerGroup, '$', 'MKSTREAM');
        logger.info(`Created consumer group for ${stream}`);
      } catch (error) {
        if (error.message.includes('BUSYGROUP')) {
          logger.info(`Consumer group for ${stream} already exists`);
        } else {
          logger.error(`Error creating consumer group for ${stream}:`, error);
        }
      }
    }
  }

  async processWhaleMessages() {
    while (this.isRunning) {
      try {
        const messages = await redis.xreadgroup(
          'GROUP', this.consumerGroup, this.consumerName,
          'COUNT', 10,
          'BLOCK', 1000,
          'STREAMS', STREAMS.WHALES_RAW, '>'
        );

        if (messages && messages.length > 0) {
          const streamMessages = messages[0][1];
          
          for (const [messageId, fields] of streamMessages) {
            try {
              await this.processWhaleMessage(messageId, fields);
            } catch (error) {
              logger.error(`Error processing whale message ${messageId}:`, error);
              await redis.xack(STREAMS.WHALES_RAW, this.consumerGroup, messageId);
            }
          }
        }
      } catch (error) {
        logger.error('Error reading whale messages:', error);
        await this.sleep(1000);
      }
    }
  }

  async processNewsMessages() {
    while (this.isRunning) {
      try {
        const messages = await redis.xreadgroup(
          'GROUP', this.consumerGroup, this.consumerName,
          'COUNT', 10,
          'BLOCK', 1000,
          'STREAMS', STREAMS.NEWS_RAW, '>'
        );

        if (messages && messages.length > 0) {
          const streamMessages = messages[0][1];
          
          for (const [messageId, fields] of streamMessages) {
            try {
              await this.processNewsMessage(messageId, fields);
            } catch (error) {
              logger.error(`Error processing news message ${messageId}:`, error);
              await redis.xack(STREAMS.NEWS_RAW, this.consumerGroup, messageId);
            }
          }
        }
      } catch (error) {
        logger.error('Error reading news messages:', error);
        await this.sleep(1000);
      }
    }
  }

  async processSystemMessages() {
    while (this.isRunning) {
      try {
        const messages = await redis.xreadgroup(
          'GROUP', this.consumerGroup, this.consumerName,
          'COUNT', 10,
          'BLOCK', 1000,
          'STREAMS', STREAMS.SYSTEM_EVENTS, '>'
        );

        if (messages && messages.length > 0) {
          const streamMessages = messages[0][1];
          
          for (const [messageId, fields] of streamMessages) {
            try {
              await this.processSystemMessage(messageId, fields);
            } catch (error) {
              logger.error(`Error processing system message ${messageId}:`, error);
              await redis.xack(STREAMS.SYSTEM_EVENTS, this.consumerGroup, messageId);
            }
          }
        }
      } catch (error) {
        logger.error('Error reading system messages:', error);
        await this.sleep(1000);
      }
    }
  }

  async processWhaleMessage(messageId, fields) {
    try {
      const data = JSON.parse(fields[1]);
      
      // Add to system events
      await this.addToBatch('system_events', {
        type: 'whale_activity',
        data: data,
        created_at: new Date(data.timestamp)
      });

      // Update user stats if it's a user wallet
      if (data.wallet_address) {
        await this.updateUserStats(data.wallet_address, {
          last_whale_activity: new Date(data.timestamp),
          whale_activity_count: 1
        });
      }

      // Acknowledge message
      await redis.xack(STREAMS.WHALES_RAW, this.consumerGroup, messageId);

    } catch (error) {
      logger.error('Error processing whale message:', error);
      throw error;
    }
  }

  async processNewsMessage(messageId, fields) {
    try {
      const data = JSON.parse(fields[1]);
      
      // Add to system events
      await this.addToBatch('system_events', {
        type: 'news_event',
        data: data,
        created_at: new Date(data.scraped_at)
      });

      // Update market sentiment if article mentions specific markets
      const mentionedMarkets = await this.extractMentionedMarkets(data);
      for (const marketId of mentionedMarkets) {
        await this.updateMarketStats(marketId, {
          news_sentiment_score: data.sentiment_score,
          news_sentiment_confidence: data.sentiment_confidence,
          last_news_update: new Date(data.scraped_at)
        });
      }

      // Acknowledge message
      await redis.xack(STREAMS.NEWS_RAW, this.consumerGroup, messageId);

    } catch (error) {
      logger.error('Error processing news message:', error);
      throw error;
    }
  }

  async processSystemMessage(messageId, fields) {
    try {
      const data = JSON.parse(fields[1]);
      
      // Add to system events
      await this.addToBatch('system_events', {
        type: data.type,
        data: data,
        created_at: new Date(data.timestamp)
      });

      // Acknowledge message
      await redis.xack(STREAMS.SYSTEM_EVENTS, this.consumerGroup, messageId);

    } catch (error) {
      logger.error('Error processing system message:', error);
      throw error;
    }
  }

  async addToBatch(table, record) {
    if (!this.batchBuffer[table]) {
      this.batchBuffer[table] = [];
    }
    
    this.batchBuffer[table].push(record);
  }

  async updateUserStats(walletAddress, stats) {
    try {
      const query = `
        INSERT INTO user_stats (wallet_address, ${Object.keys(stats).join(', ')})
        VALUES ($1, ${Object.keys(stats).map((_, i) => `$${i + 2}`).join(', ')})
        ON CONFLICT (wallet_address) DO UPDATE SET
          ${Object.keys(stats).map(key => `${key} = EXCLUDED.${key}`).join(', ')},
          updated_at = NOW()
      `;
      
      const params = [walletAddress, ...Object.values(stats)];
      await dbPool.query(query, params);
      
    } catch (error) {
      logger.error('Error updating user stats:', error);
    }
  }

  async updateMarketStats(marketId, stats) {
    try {
      const query = `
        INSERT INTO market_stats (market_id, ${Object.keys(stats).join(', ')})
        VALUES ($1, ${Object.keys(stats).map((_, i) => `$${i + 2}`).join(', ')})
        ON CONFLICT (market_id) DO UPDATE SET
          ${Object.keys(stats).map(key => `${key} = EXCLUDED.${key}`).join(', ')},
          updated_at = NOW()
      `;
      
      const params = [marketId, ...Object.values(stats)];
      await dbPool.query(query, params);
      
    } catch (error) {
      logger.error('Error updating market stats:', error);
    }
  }

  async extractMentionedMarkets(newsData) {
    try {
      // Simple keyword matching - in production, use NLP
      const text = `${newsData.title} ${newsData.content}`.toLowerCase();
      const mentionedMarkets = [];
      
      const marketKeywords = {
        'BTC-PERP': ['bitcoin', 'btc'],
        'ETH-PERP': ['ethereum', 'eth'],
        'SOL-PERP': ['solana', 'sol']
      };
      
      for (const [symbol, keywords] of Object.entries(marketKeywords)) {
        if (keywords.some(keyword => text.includes(keyword))) {
          // Get market ID
          const marketId = await redis.hget('market_ids', symbol);
          if (marketId) {
            mentionedMarkets.push(marketId);
          }
        }
      }
      
      return mentionedMarkets;
    } catch (error) {
      logger.error('Error extracting mentioned markets:', error);
      return [];
    }
  }

  async flushAllBatches() {
    try {
      for (const [table, records] of Object.entries(this.batchBuffer)) {
        if (records.length > 0) {
          await this.flushBatch(table, records);
        }
      }
      
      this.batchBuffer = {};
      this.lastFlushTime = Date.now();
      
    } catch (error) {
      logger.error('Error flushing all batches:', error);
      throw error;
    }
  }

  async flushBatch(table, records) {
    if (records.length === 0) {
      return;
    }

    try {
      const startTime = Date.now();
      
      if (table === 'system_events') {
        await this.flushSystemEvents(records);
      }
      
      const duration = Date.now() - startTime;
      logger.info(`Flushed ${records.length} records to ${table} in ${duration}ms`);

    } catch (error) {
      logger.error(`Error flushing batch to ${table}:`, error);
      throw error;
    }
  }

  async flushSystemEvents(records) {
    try {
      const values = records.map((event, index) => {
        const offset = index * 3;
        return `($${offset + 1}, $${offset + 2}, $${offset + 3})`;
      }).join(', ');

      const query = `
        INSERT INTO system_events (type, data, created_at)
        VALUES ${values}
      `;

      const params = records.flatMap(event => [
        event.type,
        JSON.stringify(event.data),
        event.created_at
      ]);

      await dbPool.query(query, params);
      
    } catch (error) {
      logger.error('Error flushing system events:', error);
      throw error;
    }
  }

  startBatchTimer() {
    setInterval(async () => {
      const timeSinceLastFlush = Date.now() - this.lastFlushTime;
      if (timeSinceLastFlush >= BATCH_CONFIG.intervalMs) {
        await this.flushAllBatches();
      }
    }, BATCH_CONFIG.intervalMs);
  }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Start writer if run directly
if (require.main === module) {
  const writer = new AnalyticsWriter();
  
  writer.start().catch(error => {
    logger.error('Failed to start analytics writer:', error);
    process.exit(1);
  });
}

module.exports = AnalyticsWriter;
