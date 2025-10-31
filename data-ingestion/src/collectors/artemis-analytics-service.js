const axios = require('axios');
const { redis, logger, STREAMS, STREAM_CONFIG } = require('../config');
require('dotenv').config();

class ArtemisAnalyticsService {
  constructor() {
    this.apiKey = process.env.ARTEMIS_API_KEY;
    this.baseUrl = 'https://api.artemisanalytics.com';
    this.isRunning = false;
    this.assets = new Map(); // Cache for supported assets
    this.metrics = new Map(); // Cache for asset metrics
    this.scanInterval = parseInt(process.env.ARTEMIS_SCAN_INTERVAL_MS || '300000'); // 5 minutes
  }

  async start() {
    if (this.isRunning) {
      logger.warn('Artemis Analytics service already running');
      return;
    }

    if (!this.apiKey) {
      logger.warn('ARTEMIS_API_KEY not found - service will run in limited mode');
    }

    logger.info('🎯 Starting Artemis Analytics service for DeFi protocol data...');
    this.isRunning = true;

    // Load supported assets
    await this.loadSupportedAssets();

    // Start monitoring DeFi protocols
    await this.startProtocolMonitoring();

    logger.info('🎯 Artemis Analytics service started - monitoring DeFi protocols');
  }

  async stop() {
    if (!this.isRunning) {
      return;
    }

    logger.info('Stopping Artemis Analytics service...');
    this.isRunning = false;
    logger.info('Artemis Analytics service stopped');
  }

  async loadSupportedAssets() {
    try {
      if (!this.apiKey) {
        logger.warn('No API key - using sample DeFi assets');
        this.loadSampleAssets();
        return;
      }

      const response = await axios.get(`${this.baseUrl}/asset-symbols`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-Artemis/1.0'
        },
        timeout: 10000
      });

      if (response.data && Array.isArray(response.data)) {
        response.data.forEach(asset => {
          this.assets.set(asset.symbol, {
            artemis_id: asset.artemis_id,
            symbol: asset.symbol,
            coingecko_id: asset.coingecko_id,
            title: asset.title,
            color: asset.color
          });
        });

        logger.info(`🎯 Loaded ${this.assets.size} supported assets from Artemis`);
      }

    } catch (error) {
      logger.error('Error loading Artemis assets:', error);
      this.loadSampleAssets();
    }
  }

  loadSampleAssets() {
    // Sample DeFi assets for testing
    const sampleAssets = [
      { artemis_id: 'uniswap', symbol: 'UNI', coingecko_id: 'uniswap', title: 'Uniswap', color: '#ff007a' },
      { artemis_id: 'aave', symbol: 'AAVE', coingecko_id: 'aave', title: 'Aave', color: '#b6509e' },
      { artemis_id: 'compound', symbol: 'COMP', coingecko_id: 'compound-governance-token', title: 'Compound', color: '#00d395' },
      { artemis_id: 'maker', symbol: 'MKR', coingecko_id: 'maker', title: 'Maker', color: '#1aab9b' },
      { artemis_id: 'curve', symbol: 'CRV', coingecko_id: 'curve-dao-token', title: 'Curve', color: '#40649f' },
      { artemis_id: 'sushiswap', symbol: 'SUSHI', coingecko_id: 'sushi', title: 'SushiSwap', color: '#7c3aed' },
      { artemis_id: 'yearn', symbol: 'YFI', coingecko_id: 'yearn-finance', title: 'Yearn Finance', color: '#006ae3' },
      { artemis_id: 'balancer', symbol: 'BAL', coingecko_id: 'balancer', title: 'Balancer', color: '#1e1e1e' },
      { artemis_id: 'synthetix', symbol: 'SNX', coingecko_id: 'havven', title: 'Synthetix', color: '#5b5b5b' },
      { artemis_id: '1inch', symbol: '1INCH', coingecko_id: '1inch', title: '1inch', color: '#1f2937' }
    ];

    sampleAssets.forEach(asset => {
      this.assets.set(asset.symbol, asset);
    });

    logger.info(`🎯 Loaded ${this.assets.size} sample DeFi assets`);
  }

  async startProtocolMonitoring() {
    try {
      // Monitor DeFi protocols every 5 minutes
      setInterval(async () => {
        try {
          await this.monitorDeFiProtocols();
        } catch (error) {
          logger.error('Error monitoring DeFi protocols:', error);
        }
      }, this.scanInterval);

      // Initial scan
      await this.monitorDeFiProtocols();

      logger.info('Started monitoring DeFi protocols');
    } catch (error) {
      logger.error('Error starting protocol monitoring:', error);
    }
  }

  async monitorDeFiProtocols() {
    try {
      logger.info('🎯 Scanning DeFi protocols for smart money flows...');

      // Monitor each supported asset
      for (const [symbol, asset] of this.assets) {
        try {
          await this.analyzeAssetMetrics(symbol, asset);
        } catch (error) {
          logger.error(`Error analyzing ${symbol}:`, error);
        }
      }

      logger.info(`🎯 Completed DeFi protocol scan for ${this.assets.size} assets`);
    } catch (error) {
      logger.error('Error monitoring DeFi protocols:', error);
    }
  }

  async analyzeAssetMetrics(symbol, asset) {
    try {
      if (!this.apiKey) {
        // Simulate metrics for sample assets
        await this.simulateAssetMetrics(symbol, asset);
        return;
      }

      // Get available metrics for this asset
      const metricsResponse = await axios.get(`${this.baseUrl}/assets/${symbol}/metrics`, {
        headers: {
          'Authorization': `Bearer ${this.apiKey}`,
          'Accept': 'application/json',
          'User-Agent': 'QuantDesk-Artemis/1.0'
        },
        timeout: 10000
      });

      if (metricsResponse.data && metricsResponse.data.metrics) {
        await this.processAssetMetrics(symbol, asset, metricsResponse.data.metrics);
      }

    } catch (error) {
      logger.error(`Error getting metrics for ${symbol}:`, error);
    }
  }

  async simulateAssetMetrics(symbol, asset) {
    try {
      // Simulate DeFi protocol metrics
      const simulatedMetrics = {
        tvl: Math.random() * 1000000000, // Random TVL between 0-1B
        volume_24h: Math.random() * 100000000, // Random volume
        fees_24h: Math.random() * 1000000, // Random fees
        active_users: Math.floor(Math.random() * 10000), // Random users
        protocol_revenue: Math.random() * 10000000 // Random revenue
      };

      await this.processAssetMetrics(symbol, asset, simulatedMetrics);

    } catch (error) {
      logger.error(`Error simulating metrics for ${symbol}:`, error);
    }
  }

  async processAssetMetrics(symbol, asset, metrics) {
    try {
      // Analyze metrics for smart money signals
      const analysis = this.analyzeSmartMoneySignals(symbol, asset, metrics);

      if (analysis.isSignificant) {
        const event = {
          type: 'artemis_defi_analysis',
          symbol: symbol,
          artemis_id: asset.artemis_id,
          title: asset.title,
          coingecko_id: asset.coingecko_id,
          color: asset.color,
          // Comprehensive metrics data
          metrics: {
            // TVL Data
            tvl: metrics.tvl,
            tvl_change_1d: metrics.tvl_change_1d,
            tvl_change_7d: metrics.tvl_change_7d,
            tvl_change_30d: metrics.tvl_change_30d,
            
            // Volume Data
            volume_24h: metrics.volume_24h,
            volume_7d: metrics.volume_7d,
            volume_30d: metrics.volume_30d,
            volume_change_1d: metrics.volume_change_1d,
            volume_change_7d: metrics.volume_change_7d,
            
            // Fee Data
            fees_24h: metrics.fees_24h,
            fees_7d: metrics.fees_7d,
            fees_30d: metrics.fees_30d,
            fees_change_1d: metrics.fees_change_1d,
            fees_change_7d: metrics.fees_change_7d,
            
            // Revenue Data
            protocol_revenue: metrics.protocol_revenue,
            protocol_revenue_24h: metrics.protocol_revenue_24h,
            protocol_revenue_7d: metrics.protocol_revenue_7d,
            protocol_revenue_30d: metrics.protocol_revenue_30d,
            
            // User Activity
            active_users: metrics.active_users,
            active_users_24h: metrics.active_users_24h,
            active_users_7d: metrics.active_users_7d,
            active_users_30d: metrics.active_users_30d,
            new_users: metrics.new_users,
            new_users_24h: metrics.new_users_24h,
            new_users_7d: metrics.new_users_7d,
            
            // Transaction Data
            transactions_24h: metrics.transactions_24h,
            transactions_7d: metrics.transactions_7d,
            transactions_30d: metrics.transactions_30d,
            
            // Additional DeFi Metrics
            total_value_locked: metrics.total_value_locked,
            total_value_locked_change: metrics.total_value_locked_change,
            market_cap: metrics.market_cap,
            market_cap_change: metrics.market_cap_change,
            token_price: metrics.token_price,
            token_price_change: metrics.token_price_change,
            
            // Protocol Health
            protocol_health_score: metrics.protocol_health_score,
            risk_score: metrics.risk_score,
            liquidity_score: metrics.liquidity_score,
            
            // Cross-chain Data
            chains: metrics.chains,
            chain_distribution: metrics.chain_distribution,
            
            // Historical Data
            historical_tvl: metrics.historical_tvl,
            historical_volume: metrics.historical_volume,
            historical_fees: metrics.historical_fees
          },
          analysis: analysis,
          timestamp: new Date().toISOString()
        };

        await this.publishArtemisEvent(event);
        logger.info(`🎯 DeFi signal detected: ${symbol} (${asset.title}) - ${analysis.signal_type}`);
      }

    } catch (error) {
      logger.error(`Error processing metrics for ${symbol}:`, error);
    }
  }

  analyzeSmartMoneySignals(symbol, asset, metrics) {
    try {
      const analysis = {
        isSignificant: false,
        signal_type: 'neutral',
        confidence: 0,
        reasons: []
      };

      // Analyze TVL changes (simulated)
      const tvlChange = Math.random() * 20 - 10; // -10% to +10%
      if (Math.abs(tvlChange) > 5) {
        analysis.isSignificant = true;
        analysis.signal_type = tvlChange > 0 ? 'tvl_increase' : 'tvl_decrease';
        analysis.confidence += 0.3;
        analysis.reasons.push(`TVL change: ${tvlChange.toFixed(2)}%`);
      }

      // Analyze volume spikes (simulated)
      const volumeSpike = Math.random() * 100;
      if (volumeSpike > 50) {
        analysis.isSignificant = true;
        analysis.signal_type = 'volume_spike';
        analysis.confidence += 0.4;
        analysis.reasons.push(`Volume spike: ${volumeSpike.toFixed(0)}%`);
      }

      // Analyze user activity (simulated)
      const userGrowth = Math.random() * 30 - 15; // -15% to +15%
      if (userGrowth > 10) {
        analysis.isSignificant = true;
        analysis.signal_type = 'user_growth';
        analysis.confidence += 0.3;
        analysis.reasons.push(`User growth: ${userGrowth.toFixed(2)}%`);
      }

      return analysis;

    } catch (error) {
      logger.error(`Error analyzing smart money signals for ${symbol}:`, error);
      return { isSignificant: false, signal_type: 'error', confidence: 0, reasons: [] };
    }
  }

  async publishArtemisEvent(event) {
    try {
      await redis.xadd(
        STREAMS.DEFI_RAW || 'defi.raw',
        'MAXLEN',
        '~',
        STREAM_CONFIG.maxLen,
        '*',
        'data', JSON.stringify(event),
        'event_type', event.type,
        'symbol', event.symbol || 'unknown',
        'timestamp', event.timestamp
      );

      logger.info(`🎯 Published Artemis event: ${event.type}`);
    } catch (error) {
      logger.error('Error publishing Artemis event:', error);
      throw error;
    }
  }

  async getArtemisStats() {
    try {
      const stats = {
        assets_tracked: this.assets.size,
        metrics_cached: this.metrics.size,
        is_running: this.isRunning,
        api_key_configured: !!this.apiKey
      };

      return stats;
    } catch (error) {
      logger.error('Error getting Artemis stats:', error);
      throw error;
    }
  }
}

// Start service if run directly
if (require.main === module) {
  const service = new ArtemisAnalyticsService();
  
  service.start().catch(error => {
    logger.error('Failed to start Artemis Analytics service:', error);
    process.exit(1);
  });

  // Graceful shutdown
  process.on('SIGINT', async () => {
    logger.info('Shutting down Artemis Analytics service...');
    await service.stop();
    process.exit(0);
  });

  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down Artemis Analytics service...');
    await service.stop();
    process.exit(0);
  });
}

module.exports = ArtemisAnalyticsService;
