const axios = require('axios');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');
require('dotenv').config();

class DuneAnalyticsService {
  constructor() {
    this.apiKey = process.env.DUNE_API_KEY;
    this.baseUrl = 'https://api.dune.com/api/v1';
    this.isRunning = false;
    this.queryCache = new Map(); // Cache for query results
    this.queryInterval = parseInt(process.env.DUNE_QUERY_INTERVAL_MS || '300000'); // 5 minutes
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Dune Analytics service already running');
      return;
    }

    if (!this.apiKey) {
      logger.error('DUNE_API_KEY not found in environment variables');
      return;
    }

    logger.info('🔍 Starting Dune Analytics service for blockchain data...');
    this.isRunning = true;

    // Start monitoring predefined queries
    await this.startQueryMonitoring();

    logger.info('🔍 Dune Analytics service started - monitoring blockchain queries');
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping Dune Analytics service...');
    this.isRunning = false;
    logger.info('Dune Analytics service stopped');
  }

  async startQueryMonitoring() {
    try {
      // Monitor queries every 5 minutes
      setInterval(async () => {
        try {
          await this.executePredefinedQueries();
        } catch (error) {
          logger.error('Error executing Dune queries:', error);
        }
      }, this.queryInterval);

      logger.info('Started monitoring Dune queries');
    } catch (error) {
      logger.error('Error starting query monitoring:', error);
    }
  }

  async executePredefinedQueries() {
    try {
      // Define some useful queries for whale tracking and DeFi analytics
      const queries = [
        {
          name: 'whale_transfers',
          description: 'Large ETH transfers (>1000 ETH)',
          query: 'SELECT * FROM ethereum.transactions WHERE value > 1000000000000000000000 ORDER BY block_time DESC LIMIT 100'
        },
        {
          name: 'defi_tvl_changes',
          description: 'DeFi protocol TVL changes',
          query: 'SELECT protocol, tvl_change FROM defi.protocols WHERE tvl_change > 1000000 ORDER BY block_time DESC LIMIT 50'
        },
        {
          name: 'exchange_flows',
          description: 'Exchange inflow/outflow patterns',
          query: 'SELECT exchange, flow_type, amount FROM ethereum.exchange_flows WHERE amount > 100000000000000000000 ORDER BY block_time DESC LIMIT 100'
        }
      ];

      for (const queryDef of queries) {
        try {
          await this.executeQuery(queryDef);
        } catch (error) {
          logger.error(`Error executing query ${queryDef.name}:`, error);
        }
      }

      logger.info(`Executed ${queries.length} predefined Dune queries`);
    } catch (error) {
      logger.error('Error executing predefined queries:', error);
    }
  }

  async executeQuery(queryDef) {
    try {
      // Create a new query
      const createResponse = await axios.post(`${this.baseUrl}/query`, {
        query_sql: queryDef.query,
        name: queryDef.name,
        description: queryDef.description
      }, {
        headers: {
          'X-Dune-API-Key': this.apiKey,
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });

      const queryId = createResponse.data.query_id;
      logger.info(`🔍 Created Dune query: ${queryDef.name} (ID: ${queryId})`);

      // Execute the query
      const executeResponse = await axios.post(`${this.baseUrl}/query/${queryId}/execute`, {}, {
        headers: {
          'X-Dune-API-Key': this.apiKey,
          'Content-Type': 'application/json'
        },
        timeout: 30000
      });

      const executionId = executeResponse.data.execution_id;
      logger.info(`🔍 Executing Dune query: ${queryDef.name} (Execution ID: ${executionId})`);

      // Wait for execution to complete and get results
      await this.waitForExecution(queryId, executionId, queryDef);

    } catch (error) {
      logger.error(`Error executing Dune query ${queryDef.name}:`, error);
    }
  }

  async waitForExecution(queryId, executionId, queryDef) {
    try {
      let attempts = 0;
      const maxAttempts = 30; // 5 minutes max wait time

      while (attempts < maxAttempts) {
        const statusResponse = await axios.get(`${this.baseUrl}/execution/${executionId}/status`, {
          headers: {
            'X-Dune-API-Key': this.apiKey
          },
          timeout: 10000
        });

        const status = statusResponse.data.state;

        if (status === 'QUERY_STATE_SUCCESS') {
          // Get the results
          const resultsResponse = await axios.get(`${this.baseUrl}/execution/${executionId}/results`, {
            headers: {
              'X-Dune-API-Key': this.apiKey
            },
            timeout: 10000
          });

          await this.processQueryResults(queryDef, resultsResponse.data);
          break;

        } else if (status === 'QUERY_STATE_FAILED') {
          logger.error(`Dune query ${queryDef.name} failed`);
          break;

        } else {
          // Still running, wait and retry
          await new Promise(resolve => setTimeout(resolve, 10000)); // Wait 10 seconds
          attempts++;
        }
      }

      if (attempts >= maxAttempts) {
        logger.warn(`Dune query ${queryDef.name} timed out after ${maxAttempts} attempts`);
      }

    } catch (error) {
      logger.error(`Error waiting for Dune execution ${executionId}:`, error);
    }
  }

  async processQueryResults(queryDef, results) {
    try {
      if (!results.rows || results.rows.length === 0) {
        logger.info(`No results for Dune query: ${queryDef.name}`);
        return;
      }

      // Process each row of results
      for (const row of results.rows) {
        const event = {
          type: 'dune_query_result',
          query_name: queryDef.name,
          query_description: queryDef.description,
          data: row,
          timestamp: new Date().toISOString()
        };

        await this.publishDuneEvent(event);
      }

      logger.info(`🔍 Processed ${results.rows.length} results from Dune query: ${queryDef.name}`);

    } catch (error) {
      logger.error(`Error processing Dune query results for ${queryDef.name}:`, error);
    }
  }

  async executeCustomQuery(querySql, queryName, description) {
    try {
      const queryDef = {
        name: queryName,
        description: description,
        query: querySql
      };

      await this.executeQuery(queryDef);
      logger.info(`🔍 Executed custom Dune query: ${queryName}`);

    } catch (error) {
      logger.error(`Error executing custom Dune query ${queryName}:`, error);
      throw error;
    }
  }

  async getQueryHistory() {
    try {
      const response = await axios.get(`${this.baseUrl}/user/queries`, {
        headers: {
          'X-Dune-API-Key': this.apiKey
        },
        timeout: 10000
      });

      return response.data;
    } catch (error) {
      logger.error('Error getting Dune query history:', error);
      return null;
    }
  }

  async publishDuneEvent(event) {
    try {
      await redis.xadd(
        STREAMS.ANALYTICS_RAW || 'analytics.raw',
        'MAXLEN',
        '~',
        STREAM_CONFIG.maxLen,
        '*',
        'data', JSON.stringify(event),
        'event_type', event.type,
        'query_name', event.query_name || 'unknown',
        'query_id', event.query_id?.toString() || 'unknown',
        'results_count', event.results?.length?.toString() || '0',
        'execution_time_ms', event.execution_time_ms?.toString() || '0',
        'timestamp', event.timestamp
      );

      logger.info(`🔍 Published Dune Analytics event: ${event.query_name} (${event.results?.length || 0} results)`);
    } catch (error) {
      logger.error('Error publishing Dune event:', error);
      throw error;
    }
  }

  async getDuneStats() {
    try {
      const stats = {
        queries_cached: this.queryCache.size,
        is_running: this.isRunning,
        api_key_configured: !!this.apiKey
      };

      return stats;
    } catch (error) {
      logger.error('Error getting Dune stats:', error);
      throw error;
    }
  }
}

// Start service if run directly
if (require.main === module) {
  const service = new DuneAnalyticsService();
  
  service.start().catch(error => {
    logger.error('Failed to start Dune Analytics service:', error);
    process.exit(1);
  });

  // Graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Shutting down Dune Analytics service...');
    await service.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down Dune Analytics service...');
    await service.stop();
    process.exit(0);
  });
}

module.exports = DuneAnalyticsService;
