const axios = require('axios');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');

class DeFiLlamaService {
  constructor() {
    this.baseUrl = 'https://api.llama.fi';
    this.isRunning = false;
    this.protocols = new Map(); // Cache for protocols
    this.tvlHistory = new Map(); // Cache for TVL history
  }

  async start() {
    if (this.isRunning) {
      logger.warn('DeFiLlama service already running');
      return;
    }

    logger.info('🦙 Starting DeFiLlama service for TVL data...');
    this.isRunning = true;

    // Start monitoring TVL changes
    await this.startTVLMonitoring();

    // Start protocol monitoring
    await this.startProtocolMonitoring();

    logger.info('🦙 DeFiLlama service started - monitoring TVL and protocols');
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping DeFiLlama service...');
    this.isRunning = false;
    logger.info('DeFiLlama service stopped');
  }

  async startTVLMonitoring() {
    try {
      // Monitor TVL changes every 5 minutes
      setInterval(async () => {
        try {
          await this.scanTVLChanges();
        } catch (error) {
          logger.error('Error scanning TVL changes:', error);
        }
      }, 300000); // 5 minutes

      logger.info('Started monitoring TVL changes');
    } catch (error) {
      logger.error('Error starting TVL monitoring:', error);
    }
  }

  async scanTVLChanges() {
    try {
      // Get current TVL data
      const response = await axios.get(`${this.baseUrl}/tvl`, {
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-DeFiLlama/1.0'
        }
      });

      if (response.data) {
        await this.analyzeTVLData(response.data);
      }

      logger.info('Scanned TVL data from DeFiLlama');
    } catch (error) {
      logger.error('Error scanning TVL changes:', error);
    }
  }

  async analyzeTVLData(tvlData) {
    try {
      // Analyze TVL changes for significant movements
      for (const [protocol, tvl] of Object.entries(tvlData)) {
        const previousTvl = this.tvlHistory.get(protocol);
        
        if (previousTvl && tvl > 0) {
          const changePercent = ((tvl - previousTvl) / previousTvl) * 100;
          
          // Look for significant TVL changes (>10%)
          if (Math.abs(changePercent) > 10) {
            await this.publishTVLEvent({
              type: 'tvl_change',
              protocol: protocol,
              current_tvl: tvl,
              previous_tvl: previousTvl,
              change_percent: changePercent,
              change_amount: tvl - previousTvl,
              timestamp: new Date().toISOString()
            });

            logger.info(`🦙 TVL Change: ${protocol} - ${changePercent.toFixed(1)}% (${changePercent > 0 ? '+' : ''}$${(tvl - previousTvl).toLocaleString()})`);
          }
        }

        // Update cache
        this.tvlHistory.set(protocol, tvl);
      }
    } catch (error) {
      logger.error('Error analyzing TVL data:', error);
    }
  }

  async startProtocolMonitoring() {
    try {
      // Monitor protocols every 10 minutes
      setInterval(async () => {
        try {
          await this.scanProtocols();
        } catch (error) {
          logger.error('Error scanning protocols:', error);
        }
      }, 600000); // 10 minutes

      logger.info('Started monitoring protocols');
    } catch (error) {
      logger.error('Error starting protocol monitoring:', error);
    }
  }

  async scanProtocols() {
    try {
      // Get protocol data
      const response = await axios.get(`${this.baseUrl}/protocols`, {
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-DeFiLlama/1.0'
        }
      });

      if (response.data) {
        for (const protocol of response.data.slice(0, 50)) { // Check top 50
          await this.analyzeProtocol(protocol);
        }
      }

      logger.info(`Scanned ${response.data?.length || 0} protocols from DeFiLlama`);
    } catch (error) {
      logger.error('Error scanning protocols:', error);
    }
  }

  async analyzeProtocol(protocol) {
    try {
      const name = protocol.name;
      const tvl = protocol.tvl;
      const category = protocol.category;
      const chains = protocol.chains || [];

      // Look for new or significant protocols
      if (tvl > 1000000) { // > $1M TVL
        const previousData = this.protocols.get(name);
        
        if (!previousData) {
          // New protocol detected
          await this.publishTVLEvent({
            type: 'new_protocol',
            protocol: name,
            tvl: tvl,
            category: category,
            chains: chains,
            timestamp: new Date().toISOString()
          });

          logger.info(`🦙 New Protocol: ${name} - $${tvl.toLocaleString()} TVL (${category})`);
        } else if (Math.abs(tvl - previousData.tvl) > previousData.tvl * 0.2) {
          // Significant TVL change
          const changePercent = ((tvl - previousData.tvl) / previousData.tvl) * 100;
          
          await this.publishTVLEvent({
            type: 'protocol_tvl_change',
            protocol: name,
            current_tvl: tvl,
            previous_tvl: previousData.tvl,
            change_percent: changePercent,
            category: category,
            chains: chains,
            timestamp: new Date().toISOString()
          });

          logger.info(`🦙 Protocol TVL Change: ${name} - ${changePercent.toFixed(1)}% ($${tvl.toLocaleString()})`);
        }
      }

      // Update cache
      this.protocols.set(name, { tvl, category, chains });
    } catch (error) {
      logger.error('Error analyzing protocol:', error);
    }
  }

  async getProtocolDetails(protocolName) {
    try {
      const response = await axios.get(`${this.baseUrl}/protocol/${protocolName}`, {
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-DeFiLlama/1.0'
        }
      });

      return response.data;
    } catch (error) {
      logger.error(`Error getting protocol details for ${protocolName}:`, error);
      return null;
    }
  }

  async getChainTVL() {
    try {
      const response = await axios.get(`${this.baseUrl}/chains`, {
        timeout: 10000,
        headers: {
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-DeFiLlama/1.0'
        }
      });

      return response.data;
    } catch (error) {
      logger.error('Error getting chain TVL:', error);
      return null;
    }
  }

  async publishTVLEvent(event) {
    try {
      await redis.xadd(
        STREAMS.DEFI_RAW || 'defi.raw',
        'MAXLEN',
        '~',
        STREAM_CONFIG.maxLen,
        '*',
        'data', JSON.stringify(event),
        'event_type', event.type,
        'protocol', event.protocol || 'unknown',
        'timestamp', event.timestamp
      );

      logger.info(`🦙 Published DeFi event: ${event.type}`);
    } catch (error) {
      logger.error('Error publishing DeFi event:', error);
      throw error;
    }
  }

  async getDeFiLlamaStats() {
    try {
      const stats = {
        protocols_tracked: this.protocols.size,
        tvl_history_size: this.tvlHistory.size,
        is_running: this.isRunning
      };

      return stats;
    } catch (error) {
      logger.error('Error getting DeFiLlama stats:', error);
      throw error;
    }
  }
}

// Start service if run directly
if (require.main === module) {
  const service = new DeFiLlamaService();
  
  service.start().catch(error => {
    logger.error('Failed to start DeFiLlama service:', error);
    process.exit(1);
  });

  // Graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Shutting down DeFiLlama service...');
    await service.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down DeFiLlama service...');
    await service.stop();
    process.exit(0);
  });
}

module.exports = DeFiLlamaService;
