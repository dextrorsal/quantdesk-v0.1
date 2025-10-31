/**
 * MIKEY-AI to QuantDesk Integration Bridge
 * 
 * This service bridges MIKEY-AI's intelligence capabilities with QuantDesk's ML trading system
 * 
 * Features:
 * - Real-time market data from 100+ exchanges (MIKEY-AI)
 * - ML model predictions (QuantDesk)
 * - Cross-platform arbitrage detection
 * - Unified trading intelligence API
 */

import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import { config } from './config';
import { QuantDeskClient } from './clients/QuantDeskClient';
import { MikeyAIClient } from './clients/MikeyAIClient';
import { TradingIntelligenceService } from './services/TradingIntelligenceService';
import { ArbitrageDetectionService } from './services/ArbitrageDetectionService';
import { logger } from './utils/logger';

class MikeyQuantBridge {
  private app: express.Application;
  private quantDeskClient: QuantDeskClient;
  private mikeyAIClient: MikeyAIClient;
  private tradingIntelligenceService: TradingIntelligenceService;
  private arbitrageDetectionService: ArbitrageDetectionService;

  constructor() {
    this.app = express();
    this.quantDeskClient = new QuantDeskClient();
    this.mikeyAIClient = new MikeyAIClient();
    this.tradingIntelligenceService = new TradingIntelligenceService(
      this.quantDeskClient,
      this.mikeyAIClient
    );
    this.arbitrageDetectionService = new ArbitrageDetectionService(
      this.mikeyAIClient
    );
    
    this.setupMiddleware();
    this.setupRoutes();
  }

  private setupMiddleware(): void {
    this.app.use(helmet());
    this.app.use(cors());
    this.app.use(express.json());
    this.app.use(express.urlencoded({ extended: true }));
  }

  private setupRoutes(): void {
    // Health check
    this.app.get('/health', (req, res) => {
      res.json({ 
        status: 'healthy', 
        timestamp: new Date().toISOString(),
        services: {
          quantDesk: this.quantDeskClient.isConnected(),
          mikeyAI: this.mikeyAIClient.isConnected()
        }
      });
    });

    // Unified market analysis
    this.app.post('/api/analyze', async (req, res) => {
      try {
        const { symbol, analysisType = 'comprehensive' } = req.body;
        
        const analysis = await this.tradingIntelligenceService.analyzeMarket({
          symbol,
          analysisType,
          includeMLPredictions: true,
          includeArbitrage: true,
          includeWhaleTracking: true
        });

        res.json(analysis);
      } catch (error) {
        logger.error('Market analysis failed:', error);
        res.status(500).json({ error: 'Analysis failed' });
      }
    });

    // ML model predictions with market context
    this.app.post('/api/predict', async (req, res) => {
      try {
        const { symbol, timeframe = '15m', includeMarketData = true } = req.body;
        
        const prediction = await this.tradingIntelligenceService.getMLPrediction({
          symbol,
          timeframe,
          includeMarketData,
          includeSentiment: true,
          includeWhaleActivity: true
        });

        res.json(prediction);
      } catch (error) {
        logger.error('ML prediction failed:', error);
        res.status(500).json({ error: 'Prediction failed' });
      }
    });

    // Arbitrage opportunities
    this.app.get('/api/arbitrage/:symbol', async (req, res) => {
      try {
        const { symbol } = req.params;
        const { minSpreadPercent = 0.1 } = req.query;
        
        const opportunities = await this.arbitrageDetectionService.findOpportunities({
          symbol,
          minSpreadPercent: parseFloat(minSpreadPercent as string)
        });

        res.json(opportunities);
      } catch (error) {
        logger.error('Arbitrage detection failed:', error);
        res.status(500).json({ error: 'Arbitrage detection failed' });
      }
    });

    // Whale tracking with ML insights
    this.app.get('/api/whales/:symbol', async (req, res) => {
      try {
        const { symbol } = req.params;
        const { threshold = 100000 } = req.query;
        
        const whaleAnalysis = await this.tradingIntelligenceService.trackWhales({
          symbol,
          threshold: parseFloat(threshold as string),
          includeMLImpact: true,
          includeArbitrageImpact: true
        });

        res.json(whaleAnalysis);
      } catch (error) {
        logger.error('Whale tracking failed:', error);
        res.status(500).json({ error: 'Whale tracking failed' });
      }
    });

    // Natural language queries
    this.app.post('/api/query', async (req, res) => {
      try {
        const { query, context = {} } = req.body;
        
        const response = await this.tradingIntelligenceService.processNaturalLanguageQuery({
          query,
          context,
          includeMLData: true,
          includeMarketData: true,
          includeArbitrageData: true
        });

        res.json(response);
      } catch (error) {
        logger.error('Natural language query failed:', error);
        res.status(500).json({ error: 'Query processing failed' });
      }
    });

    // Real-time data stream
    this.app.get('/api/stream/:symbol', (req, res) => {
      const { symbol } = req.params;
      
      res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
        'Access-Control-Allow-Origin': '*'
      });

      const stream = this.tradingIntelligenceService.createDataStream(symbol);
      
      stream.on('data', (data) => {
        res.write(`data: ${JSON.stringify(data)}\n\n`);
      });

      stream.on('error', (error) => {
        logger.error('Stream error:', error);
        res.write(`data: ${JSON.stringify({ error: 'Stream error' })}\n\n`);
      });

      req.on('close', () => {
        stream.destroy();
      });
    });
  }

  public async start(): Promise<void> {
    try {
      // Initialize connections
      await this.quantDeskClient.connect();
      await this.mikeyAIClient.connect();
      
      // Start server
      this.app.listen(config.port, () => {
        logger.info(`MIKEY-QuantDesk Bridge running on port ${config.port}`);
        logger.info('Services connected:', {
          quantDesk: this.quantDeskClient.isConnected(),
          mikeyAI: this.mikeyAIClient.isConnected()
        });
      });
    } catch (error) {
      logger.error('Failed to start bridge service:', error);
      process.exit(1);
    }
  }
}

// Start the bridge service
const bridge = new MikeyQuantBridge();
bridge.start().catch(console.error);

export default MikeyQuantBridge;
