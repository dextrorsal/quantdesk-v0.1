const Redis = require('redis');
const { logger } = require('../config');

class AnalyticsWriter {
  constructor() {
    this.redis = Redis.createClient({
      host: process.env.REDIS_HOST || 'localhost',
      port: process.env.REDIS_PORT || 6379
    });
    
    this.streams = {
      TICKS_RAW: 'ticks.raw',
      WHALES_RAW: 'whales.raw',
      NEWS_RAW: 'news.raw',
      TRENCH_RAW: 'trench.raw',
      DEFI_RAW: 'defi.raw',
      ANALYTICS_RAW: 'analytics.raw',
      MARKET_RAW: 'market.raw',
      PERPS_RAW: 'perps.raw'
    };
    
    this.outputStreams = {
      SIGNALS: 'signals.raw',
      ALERTS: 'alerts.raw',
      REPORTS: 'reports.raw'
    };
    
    this.isRunning = false;
    this.processors = new Map();
  }

  async start() {
    try {
      await this.redis.connect();
      logger.info('🧠 Analytics Writer started - processing data streams...');
      
      this.isRunning = true;
      
      // Start the main processing loop
      await this.processDataStreams();
      
    } catch (error) {
      logger.error('Error starting Analytics Writer:', error);
      throw error;
    }
  }

  async stop() {
    this.isRunning = false;
    await this.redis.quit();
    logger.info('🧠 Analytics Writer stopped');
  }

  async processDataStreams() {
    while (this.isRunning) {
      try {
        // Read from all input streams
        const streamData = await this.readAllStreams();
        
        // Process each data type
        await this.processWhaleMovements(streamData.whales);
        await this.processPriceCorrelations(streamData.prices);
        await this.processSentimentAnalysis(streamData.news);
        await this.processTrenchTokens(streamData.trench);
        await this.processDeFiProtocols(streamData.defi);
        await this.processPerpsMarkets(streamData.perps);
        
        // Generate cross-correlations
        await this.generateCrossCorrelations(streamData);
        
        // Wait before next cycle
        await this.sleep(5000); // 5 seconds
        
      } catch (error) {
        logger.error('Error processing data streams:', error);
        await this.sleep(10000); // Wait longer on error
      }
    }
  }

  async readAllStreams() {
    const streamData = {};
    
    for (const [name, stream] of Object.entries(this.streams)) {
      try {
        const messages = await this.redis.xRead(
          { key: stream, id: '$' },
          { COUNT: 100, BLOCK: 1000 }
        );
        
        if (messages && messages.length > 0) {
          streamData[name.toLowerCase().replace('_raw', '')] = messages[0].messages;
        }
      } catch (error) {
        logger.warn(`No messages in stream ${stream}:`, error.message);
      }
    }
    
    return streamData;
  }

  async processWhaleMovements(whaleData) {
    if (!whaleData || whaleData.length === 0) return;
    
    for (const message of whaleData) {
      try {
        const whale = JSON.parse(message.message);
        
        // Analyze whale transaction
        const analysis = {
          wallet: whale.wallet,
          amount: whale.amount,
          token: whale.token,
          direction: whale.direction,
          timestamp: whale.timestamp,
          price_impact: this.calculatePriceImpact(whale),
          smart_money_score: this.getSmartMoneyScore(whale.wallet),
          historical_performance: this.getHistoricalPerformance(whale.wallet),
          risk_level: this.assessRiskLevel(whale)
        };
        
        // Generate whale signal if significant
        if (analysis.smart_money_score > 0.7 && analysis.amount > 100000) {
          await this.generateWhaleSignal(analysis);
        }
        
      } catch (error) {
        logger.error('Error processing whale movement:', error);
      }
    }
  }

  async processPriceCorrelations(priceData) {
    if (!priceData || priceData.length === 0) return;
    
    // Track price movements vs whale activity
    const correlations = {
      whale_price_correlation: this.calculateWhalePriceCorrelation(),
      volume_price_correlation: this.calculateVolumePriceCorrelation(),
      funding_rate_correlation: this.calculateFundingRateCorrelation(),
      sentiment_price_correlation: this.calculateSentimentPriceCorrelation()
    };
    
    // Generate price signal if correlation is strong
    if (correlations.whale_price_correlation > 0.8) {
      await this.generatePriceSignal(correlations);
    }
  }

  async processSentimentAnalysis(newsData) {
    if (!newsData || newsData.length === 0) return;
    
    for (const message of newsData) {
      try {
        const news = JSON.parse(message.message);
        
        const sentiment = {
          headline: news.headline,
          sentiment_score: this.analyzeSentiment(news.headline),
          market_impact: this.assessMarketImpact(news),
          confidence: this.calculateConfidence(news),
          related_tokens: this.identifyRelatedTokens(news)
        };
        
        // Generate sentiment signal if strong
        if (sentiment.sentiment_score > 0.8 && sentiment.confidence > 0.7) {
          await this.generateSentimentSignal(sentiment);
        }
        
      } catch (error) {
        logger.error('Error processing sentiment:', error);
      }
    }
  }

  async processTrenchTokens(trenchData) {
    if (!trenchData || trenchData.length === 0) return;
    
    for (const message of trenchData) {
      try {
        const token = JSON.parse(message.message);
        
        const analysis = {
          token_address: token.address,
          launch_time: token.launch_time,
          volume: token.volume,
          holders: token.holders,
          liquidity: token.liquidity,
          dev_wallet: token.dev_wallet,
          similarity_score: this.calculateSimilarityToSuccessfulTokens(token),
          risk_score: this.assessTokenRisk(token),
          opportunity_score: this.calculateOpportunityScore(token)
        };
        
        // Generate trench signal if opportunity is high
        if (analysis.opportunity_score > 0.8 && analysis.risk_score < 0.3) {
          await this.generateTrenchSignal(analysis);
        }
        
      } catch (error) {
        logger.error('Error processing trench token:', error);
      }
    }
  }

  async processDeFiProtocols(defiData) {
    if (!defiData || defiData.length === 0) return;
    
    for (const message of defiData) {
      try {
        const protocol = JSON.parse(message.message);
        
        const analysis = {
          protocol_name: protocol.name,
          tvl: protocol.tvl,
          tvl_change: protocol.tvl_change,
          volume: protocol.volume,
          fees: protocol.fees,
          users: protocol.users,
          transactions: protocol.transactions,
          health_score: this.calculateProtocolHealth(protocol),
          risk_score: this.assessProtocolRisk(protocol),
          opportunity_score: this.calculateProtocolOpportunity(protocol)
        };
        
        // Generate DeFi signal if healthy and opportunistic
        if (analysis.health_score > 0.8 && analysis.opportunity_score > 0.7) {
          await this.generateDeFiSignal(analysis);
        }
        
      } catch (error) {
        logger.error('Error processing DeFi protocol:', error);
      }
    }
  }

  async processPerpsMarkets(perpsData) {
    if (!perpsData || perpsData.length === 0) return;
    
    for (const message of perpsData) {
      try {
        const market = JSON.parse(message.message);
        
        const analysis = {
          symbol: market.symbol,
          funding_rate: market.funding_rate,
          open_interest: market.open_interest,
          long_short_ratio: market.long_short_ratio,
          liquidations: market.liquidations,
          squeeze_potential: this.calculateSqueezePotential(market),
          risk_level: this.assessPerpsRisk(market),
          opportunity_score: this.calculatePerpsOpportunity(market)
        };
        
        // Generate perps signal if squeeze potential or opportunity is high
        if (analysis.squeeze_potential > 0.8 || analysis.opportunity_score > 0.7) {
          await this.generatePerpsSignal(analysis);
        }
        
      } catch (error) {
        logger.error('Error processing perps market:', error);
      }
    }
  }

  async generateCrossCorrelations(streamData) {
    // Cross-correlation analysis between different data streams
    const correlations = {
      whale_defi_correlation: this.calculateWhaleDeFiCorrelation(streamData),
      price_sentiment_correlation: this.calculatePriceSentimentCorrelation(streamData),
      perps_trench_correlation: this.calculatePerpsTrenchCorrelation(streamData)
    };
    
    // Generate cross-correlation signals
    if (correlations.whale_defi_correlation > 0.8) {
      await this.generateCrossCorrelationSignal(correlations);
    }
  }

  // Signal Generation Methods
  async generateWhaleSignal(analysis) {
    const signal = {
      type: 'WHALE_MOVEMENT',
      priority: 'HIGH',
      action: analysis.direction === 'buy' ? 'LONG' : 'SHORT',
      token: analysis.token,
      amount: analysis.amount,
      confidence: analysis.smart_money_score,
      reasoning: `Smart money wallet (${(analysis.smart_money_score * 100).toFixed(1)}% win rate) ${analysis.direction} ${analysis.amount} ${analysis.token}`,
      target_price: this.calculateTargetPrice(analysis),
      stop_loss: this.calculateStopLoss(analysis),
      risk_reward: this.calculateRiskReward(analysis),
      timestamp: Date.now()
    };
    
    await this.publishSignal(signal);
  }

  async generateTrenchSignal(analysis) {
    const signal = {
      type: 'TRENCH_OPPORTUNITY',
      priority: 'MEDIUM',
      action: 'MICRO_POSITION',
      token: analysis.token_address,
      amount: '100-500', // Micro position
      confidence: analysis.opportunity_score,
      reasoning: `Early-stage token with ${(analysis.opportunity_score * 100).toFixed(1)}% opportunity score, similar to successful launches`,
      target_price: this.calculateTargetPrice(analysis),
      stop_loss: this.calculateStopLoss(analysis),
      risk_reward: '10:1',
      timestamp: Date.now()
    };
    
    await this.publishSignal(signal);
  }

  async generatePerpsSignal(analysis) {
    const signal = {
      type: 'PERPS_SQUEEZE',
      priority: 'HIGH',
      action: analysis.squeeze_potential > 0.8 ? 'SHORT' : 'LONG',
      symbol: analysis.symbol,
      confidence: analysis.squeeze_potential,
      reasoning: `Funding rate ${analysis.funding_rate}%, OI ${analysis.open_interest}, L/S ratio ${analysis.long_short_ratio}`,
      target_price: this.calculateTargetPrice(analysis),
      stop_loss: this.calculateStopLoss(analysis),
      risk_reward: this.calculateRiskReward(analysis),
      timestamp: Date.now()
    };
    
    await this.publishSignal(signal);
  }

  async generateCrossCorrelationSignal(correlations) {
    const signal = {
      type: 'CROSS_CORRELATION',
      priority: 'MEDIUM',
      action: 'MONITOR',
      confidence: Math.max(...Object.values(correlations)),
      reasoning: `Strong cross-correlations detected: ${Object.entries(correlations).map(([key, value]) => `${key}: ${(value * 100).toFixed(1)}%`).join(', ')}`,
      correlations: correlations,
      timestamp: Date.now()
    };
    
    await this.publishSignal(signal);
  }

  async publishSignal(signal) {
    try {
      await this.redis.xAdd(
        this.outputStreams.SIGNALS,
        '*',
        { signal: JSON.stringify(signal) }
      );
      
      logger.info(`🚨 Generated ${signal.type} signal: ${signal.reasoning}`);
    } catch (error) {
      logger.error('Error publishing signal:', error);
    }
  }

  // Analysis Helper Methods (Placeholder implementations)
  calculatePriceImpact(whale) { return Math.random() * 0.1; }
  getSmartMoneyScore(wallet) { return Math.random() * 0.9 + 0.1; }
  getHistoricalPerformance(wallet) { return Math.random() * 0.4 + 0.6; }
  assessRiskLevel(whale) { return Math.random() * 0.5 + 0.2; }
  
  calculateWhalePriceCorrelation() { return Math.random() * 0.4 + 0.6; }
  calculateVolumePriceCorrelation() { return Math.random() * 0.3 + 0.7; }
  calculateFundingRateCorrelation() { return Math.random() * 0.2 + 0.8; }
  calculateSentimentPriceCorrelation() { return Math.random() * 0.3 + 0.7; }
  
  analyzeSentiment(headline) { return Math.random() * 0.4 + 0.6; }
  assessMarketImpact(news) { return Math.random() * 0.3 + 0.7; }
  calculateConfidence(news) { return Math.random() * 0.2 + 0.8; }
  identifyRelatedTokens(news) { return ['BTC', 'ETH', 'SOL']; }
  
  calculateSimilarityToSuccessfulTokens(token) { return Math.random() * 0.3 + 0.7; }
  assessTokenRisk(token) { return Math.random() * 0.4 + 0.3; }
  calculateOpportunityScore(token) { return Math.random() * 0.2 + 0.8; }
  
  calculateProtocolHealth(protocol) { return Math.random() * 0.2 + 0.8; }
  assessProtocolRisk(protocol) { return Math.random() * 0.3 + 0.2; }
  calculateProtocolOpportunity(protocol) { return Math.random() * 0.3 + 0.7; }
  
  calculateSqueezePotential(market) { return Math.random() * 0.3 + 0.7; }
  assessPerpsRisk(market) { return Math.random() * 0.4 + 0.3; }
  calculatePerpsOpportunity(market) { return Math.random() * 0.2 + 0.8; }
  
  calculateWhaleDeFiCorrelation(streamData) { return Math.random() * 0.2 + 0.8; }
  calculatePriceSentimentCorrelation(streamData) { return Math.random() * 0.3 + 0.7; }
  calculatePerpsTrenchCorrelation(streamData) { return Math.random() * 0.2 + 0.8; }
  
  calculateTargetPrice(analysis) { return Math.random() * 100 + 100; }
  calculateStopLoss(analysis) { return Math.random() * 50 + 50; }
  calculateRiskReward(analysis) { return `${Math.floor(Math.random() * 5) + 1}:1`; }

  async sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
  }
}

// Main execution
if (require.main === module) {
  const analyticsWriter = new AnalyticsWriter();
  
  // Graceful shutdown
  process.on('SIGTERM', async () => {
    logger.info('SIGTERM received, shutting down gracefully...');
    await analyticsWriter.stop();
    process.exit(0);
  });
  
  process.on('SIGINT', async () => {
    logger.info('SIGINT received, shutting down gracefully...');
    await analyticsWriter.stop();
    process.exit(0);
  });
  
  // Start the analytics writer
  analyticsWriter.start().catch(error => {
    logger.error('Failed to start Analytics Writer:', error);
    process.exit(1);
  });
}

module.exports = AnalyticsWriter;
