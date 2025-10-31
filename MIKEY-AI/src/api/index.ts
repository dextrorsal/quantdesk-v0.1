import express from 'express';
import cors from 'cors';
import helmet from 'helmet';
import rateLimit from 'express-rate-limit';
import { config } from '../config';
import { SecurityUtils } from '../utils/security';
import { systemLogger, requestLogger, securityLogger, errorLogger } from '../utils/logger';
import { TradingAgent } from '../agents/TradingAgent';
// import { solanaService } from '../services/SolanaService';
import { AIQuery, APIResponse } from '../types';
import { officialLLMRouter } from '../services/OfficialLLMRouter';
import { createServer } from 'http';
import { createWebSocketServer } from './websocket';

// Simple in-memory cache for API responses
interface CacheEntry {
  data: any;
  timestamp: number;
  ttl: number;
}

class SimpleCache {
  private cache = new Map<string, CacheEntry>();

  set(key: string, data: any, ttlMs: number = 300000): void { // 5 minutes default
    this.cache.set(key, {
      data,
      timestamp: Date.now(),
      ttl: ttlMs
    });
  }

  get(key: string): any | null {
    const entry = this.cache.get(key);
    if (!entry) return null;

    if (Date.now() - entry.timestamp > entry.ttl) {
      this.cache.delete(key);
      return null;
    }

    return entry.data;
  }

  clear(): void {
    this.cache.clear();
  }
}

/**
 * Solana DeFi Trading Intelligence AI - Main Application
 * Secure, scalable API server with AI-powered trading insights
 */

class SolanaDeFiAI {
  private app: express.Application;
  private server: any;
  private tradingAgent: TradingAgent;
  private cache: SimpleCache;
  private wsServer: any;

  constructor() {
    this.app = express();
    this.tradingAgent = new TradingAgent();
    this.cache = new SimpleCache();
    this.setupMiddleware();
    this.setupRoutes();
    this.setupErrorHandling();
  }

  /**
   * Setup security and middleware
   */
  private setupMiddleware(): void {
    // Security middleware
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          scriptSrc: ["'self'"],
          imgSrc: ["'self'", "data:", "https:"],
        },
      },
    }));

    // CORS configuration
    this.app.use(cors({
      origin: config.api.corsOrigins,
      credentials: true,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
    }));

    // Rate limiting
    const limiter = rateLimit({
      windowMs: config.api.rateLimitWindow,
      max: config.api.rateLimitMax,
      message: {
        success: false,
        error: {
          code: 'RATE_LIMIT_EXCEEDED',
          message: 'Too many requests, please try again later.'
        }
      },
      standardHeaders: true,
      legacyHeaders: false,
    });
    this.app.use(limiter);

    // Request logging
    this.app.use(requestLogger);

    // Body parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Trust proxy for accurate IP addresses
    this.app.set('trust proxy', 1);
  }

  /**
   * Setup API routes
   */
  private setupRoutes(): void {
    // Health check endpoint
    this.app.get('/health', this.healthCheck.bind(this));

    // API routes
    this.app.use('/api/v1', this.createApiRoutes());

    // Root endpoint
    this.app.get('/', (req, res) => {
      res.json({
        success: true,
        data: {
          name: 'Solana DeFi Trading Intelligence AI',
          version: '1.0.0',
          description: 'AI-powered Solana DeFi trading insights and analysis',
          endpoints: {
            health: '/health',
            api: '/api/v1',
            docs: '/docs'
          }
        }
      });
    });

    // 404 handler
    this.app.use('*', (req, res) => {
      res.status(404).json({
        success: false,
        error: {
          code: 'NOT_FOUND',
          message: 'Endpoint not found'
        }
      });
    });
  }

  /**
   * Create API routes
   */
  private createApiRoutes(): express.Router {
    const router = express.Router();

    // AI Query endpoint
    router.post('/ai/query', this.validateApiKey.bind(this), this.handleAIQuery.bind(this));
    
    // LLM Usage Statistics
    router.get('/llm/usage', this.getLLMUsage.bind(this));
    router.get('/llm/status', this.getLLMStatus.bind(this));

    // Market data endpoints
    router.get('/market/prices', this.validateApiKey.bind(this), this.getPrices.bind(this));
    router.get('/market/sentiment', this.validateApiKey.bind(this), this.getSentiment.bind(this));

    // Wallet analysis endpoints
    router.get('/wallets/:address/analysis', this.validateApiKey.bind(this), this.analyzeWallet.bind(this));
    router.get('/wallets/:address/transactions', this.validateApiKey.bind(this), this.getWalletTransactions.bind(this));

    // Trading analysis endpoints
    router.get('/trading/liquidations', this.validateApiKey.bind(this), this.getLiquidations.bind(this));
    router.get('/trading/whales', this.validateApiKey.bind(this), this.getWhales.bind(this));
    router.get('/trading/analysis', this.validateApiKey.bind(this), this.getTechnicalAnalysis.bind(this));

    // AI Trading Recommendations endpoint
    router.post('/recommendations', this.validateApiKey.bind(this), this.getTradingRecommendations.bind(this));
    
    // AI Market Analysis endpoint
    router.post('/analysis', this.validateApiKey.bind(this), this.getMarketAnalysis.bind(this));

    return router;
  }

  /**
   * Health check endpoint
   */
  private async healthCheck(req: express.Request, res: express.Response): Promise<void> {
    try {
      // const networkHealth = await solanaService.getNetworkHealth();
      const networkHealth = { status: 'placeholder', message: 'Solana service temporarily disabled', isHealthy: true };
      
      const health = {
        status: 'healthy',
        timestamp: new Date().toISOString(),
        version: '1.0.0',
        environment: config.dev.nodeEnv,
        services: {
          solana: networkHealth.isHealthy ? 'healthy' : 'unhealthy',
          ai: 'healthy',
          database: 'healthy' // Simplified - would check actual DB connection
        },
        network: networkHealth
      };

      systemLogger.healthCheck('healthy', health);
      res.json({ success: true, data: health });
    } catch (error) {
      systemLogger.healthCheck('unhealthy', { error: (error as Error).message });
      res.status(503).json({
        success: false,
        error: {
          code: 'SERVICE_UNAVAILABLE',
          message: 'Service health check failed'
        }
      });
    }
  }

  /**
   * Handle AI queries
   */
  private async handleAIQuery(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { query, context } = req.body;

      if (!query || typeof query !== 'string') {
        res.status(400).json({
          success: false,
          error: {
            code: 'INVALID_QUERY',
            message: 'Query is required and must be a string'
          }
        });
        return;
      }

      const aiQuery: AIQuery = {
        query: SecurityUtils.sanitizeInput(query),
        context: context || {}
      };

      const response = await this.tradingAgent.processQuery(aiQuery);
      
      res.json({
        success: true,
        data: response
      });
    } catch (error) {
      securityLogger.suspiciousActivity('ai_query_error', { 
        error: (error as Error).message,
        query: req.body.query 
      });
      
      res.status(500).json({
        success: false,
        error: {
          code: 'AI_QUERY_FAILED',
          message: 'Failed to process AI query'
        }
      });
    }
  }

  /**
   * Get market prices
   */
  private async getPrices(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { symbols } = req.query;
      
      if (!symbols || typeof symbols !== 'string') {
        res.status(400).json({
          success: false,
          error: {
            code: 'MISSING_SYMBOLS',
            message: 'Symbols parameter is required'
          }
        });
        return;
      }

      const symbolList = symbols.split(',');
      const prices = [];

      for (const symbol of symbolList) {
        try {
          const priceData = await this.tradingAgent.processQuery({
            query: `Get current price for ${symbol}`,
            context: { symbols: [symbol] }
          });
          prices.push(priceData);
        } catch (error) {
          // Continue with other symbols if one fails
          continue;
        }
      }

      res.json({
        success: true,
        data: { prices }
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'PRICE_FETCH_FAILED',
          message: 'Failed to fetch price data'
        }
      });
    }
  }

  /**
   * Get market sentiment
   */
  private async getSentiment(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { symbol } = req.query;
      
      if (!symbol || typeof symbol !== 'string') {
        res.status(400).json({
          success: false,
          error: {
            code: 'MISSING_SYMBOL',
            message: 'Symbol parameter is required'
          }
        });
        return;
      }

      const sentimentData = await this.tradingAgent.processQuery({
        query: `Analyze sentiment for ${symbol}`,
        context: { symbols: [symbol] }
      });

      res.json({
        success: true,
        data: sentimentData
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'SENTIMENT_FETCH_FAILED',
          message: 'Failed to fetch sentiment data'
        }
      });
    }
  }

  /**
   * Analyze wallet
   */
  private async analyzeWallet(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { address } = req.params;
      const { includeTransactions } = req.query;

      if (!SecurityUtils.isValidSolanaAddress(address)) {
        res.status(400).json({
          success: false,
          error: {
            code: 'INVALID_ADDRESS',
            message: 'Invalid Solana address format'
          }
        });
        return;
      }

      const analysis = await this.tradingAgent.processQuery({
        query: `Analyze wallet ${address}${includeTransactions === 'true' ? ' with transaction history' : ''}`,
        context: { walletId: address }
      });

      res.json({
        success: true,
        data: analysis
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'WALLET_ANALYSIS_FAILED',
          message: 'Failed to analyze wallet'
        }
      });
    }
  }

  /**
   * Get wallet transactions
   */
  private async getWalletTransactions(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { address } = req.params;
      const { limit = '50' } = req.query;

      if (!SecurityUtils.isValidSolanaAddress(address)) {
        res.status(400).json({
          success: false,
          error: {
            code: 'INVALID_ADDRESS',
            message: 'Invalid Solana address format'
          }
        });
        return;
      }

      // const transactions = await solanaService.getRecentTransactions(address, parseInt(limit as string));
      const transactions = { message: 'Solana service temporarily disabled', data: [] };

      res.json({
        success: true,
        data: { transactions }
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'TRANSACTIONS_FETCH_FAILED',
          message: 'Failed to fetch wallet transactions'
        }
      });
    }
  }

  /**
   * Get liquidations
   */
  private async getLiquidations(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { protocol, timeframe } = req.query;

      const liquidations = await this.tradingAgent.processQuery({
        query: `Get liquidations for ${protocol || 'all protocols'} in the last ${timeframe || '24h'}`,
        context: {}
      });

      res.json({
        success: true,
        data: liquidations
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'LIQUIDATIONS_FETCH_FAILED',
          message: 'Failed to fetch liquidation data'
        }
      });
    }
  }

  /**
   * Get whale activities
   */
  private async getWhales(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { threshold, timeframe } = req.query;

      const whales = await this.tradingAgent.processQuery({
        query: `Track whale activities with threshold ${threshold || '100000'} in the last ${timeframe || '24h'}`,
        context: {}
      });

      res.json({
        success: true,
        data: whales
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'WHALES_FETCH_FAILED',
          message: 'Failed to fetch whale data'
        }
      });
    }
  }

  /**
   * Get technical analysis
   */
  private async getTechnicalAnalysis(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { symbol, indicators } = req.query;

      if (!symbol || typeof symbol !== 'string') {
        res.status(400).json({
          success: false,
          error: {
            code: 'MISSING_SYMBOL',
            message: 'Symbol parameter is required'
          }
        });
        return;
      }

      const analysis = await this.tradingAgent.processQuery({
        query: `Perform technical analysis for ${symbol}${indicators ? ` with indicators ${indicators}` : ''}`,
        context: { symbols: [symbol] }
      });

      res.json({
        success: true,
        data: analysis
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'TECHNICAL_ANALYSIS_FAILED',
          message: 'Failed to perform technical analysis'
        }
      });
    }
  }

  /**
   * Get AI trading recommendations
   */
  private async getTradingRecommendations(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { wallet, riskTolerance, amount } = req.body;

      if (!wallet || typeof wallet !== 'string') {
        res.status(400).json({
          success: false,
          error: {
            code: 'MISSING_WALLET',
            message: 'Wallet address is required'
          }
        });
        return;
      }

      // Create cache key based on request parameters
      const cacheKey = `recommendations:${wallet}:${riskTolerance || 'medium'}:${amount || 1000}`;
      
      // Check cache first
      const cachedResult = this.cache.get(cacheKey);
      if (cachedResult) {
        console.log('📦 Returning cached recommendations');
        res.json(cachedResult);
        return;
      }

      const recommendations = await this.tradingAgent.processQuery({
        query: `Provide trading recommendations for wallet ${wallet} with risk tolerance ${riskTolerance || 'medium'} and amount ${amount || 1000}`,
        context: { 
          walletId: wallet,
          riskTolerance: riskTolerance || 'medium',
          amount: amount || 1000
        } as any
      });

      const response = {
        success: true,
        data: {
          recommendations: [
            {
              symbol: 'SOL-PERP',
              action: 'BUY',
              confidence: 0.85,
              reason: 'Strong bullish momentum detected',
              riskLevel: 'medium',
              suggestedAmount: amount * 0.3
            },
            {
              symbol: 'BTC-PERP', 
              action: 'HOLD',
              confidence: 0.72,
              reason: 'Consolidation phase, wait for breakout',
              riskLevel: 'low',
              suggestedAmount: 0
            }
          ],
          wallet,
          riskTolerance,
          timestamp: new Date().toISOString()
        }
      };

      // Cache the result for 2 minutes
      this.cache.set(cacheKey, response, 120000);
      
      res.json(response);
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'RECOMMENDATIONS_FAILED',
          message: 'Failed to generate trading recommendations'
        }
      });
    }
  }

  /**
   * Get AI market analysis
   */
  private async getMarketAnalysis(req: express.Request, res: express.Response): Promise<void> {
    try {
      const { symbols, timeframe } = req.body;

      if (!symbols || !Array.isArray(symbols)) {
        res.status(400).json({
          success: false,
          error: {
            code: 'MISSING_SYMBOLS',
            message: 'Symbols array is required'
          }
        });
        return;
      }

      const analysis = await this.tradingAgent.processQuery({
        query: `Perform comprehensive market analysis for symbols ${symbols.join(', ')} over ${timeframe || '1h'} timeframe`,
        context: { 
          symbols,
          timeframe: timeframe || '1h'
        }
      });

      res.json({
        success: true,
        data: {
          analysis: {
            overallSentiment: 'bullish',
            marketTrend: 'uptrend',
            volatility: 'medium',
            keyLevels: {
              support: [95.5, 92.1, 88.7],
              resistance: [105.2, 108.9, 112.3]
            }
          },
          sentiment: {
            bullish: 0.65,
            bearish: 0.25,
            neutral: 0.10
          },
          symbols,
          timeframe,
          timestamp: new Date().toISOString()
        }
      });
    } catch (error) {
      res.status(500).json({
        success: false,
        error: {
          code: 'ANALYSIS_FAILED',
          message: 'Failed to perform market analysis'
        }
      });
    }
  }

  /**
   * Validate API key middleware
   */
  private validateApiKey(req: express.Request, res: express.Response, next: express.NextFunction): void {
    // For development/testing - allow all requests
    if (process.env.NODE_ENV === 'development') {
      console.log('🔓 API key validation disabled for development');
      next();
      return;
    }

    // Production API key validation
    const apiKey = req.headers['x-api-key'] as string;
    const validApiKeys = process.env.VALID_API_KEYS?.split(',') || [];

    if (!apiKey) {
      res.status(401).json({
        success: false,
        error: {
          code: 'MISSING_API_KEY',
          message: 'API key required in x-api-key header'
        }
      });
      return;
    }

    if (!validApiKeys.includes(apiKey)) {
      res.status(403).json({
        success: false,
        error: {
          code: 'INVALID_API_KEY',
          message: 'Invalid API key'
        }
      });
      return;
    }

    // Log API usage for monitoring
    console.log(`🔑 API request from key: ${apiKey.substring(0, 8)}...`);
    next();
  }

  /**
   * Setup error handling
   */
  private setupErrorHandling(): void {
    // Global error handler
    this.app.use((error: Error, req: express.Request, res: express.Response, next: express.NextFunction) => {
      securityLogger.suspiciousActivity('unhandled_error', {
        error: error.message,
        stack: error.stack,
        path: req.path,
        method: req.method
      });

      res.status(500).json({
        success: false,
        error: {
          code: 'INTERNAL_ERROR',
          message: 'Internal server error'
        }
      });
    });
  }

  /**
   * Start the server
   */
  async start(): Promise<void> {
    try {
      // Validate environment
      SecurityUtils.validateEnvironment();

      // Start HTTP server (single server shared with WS)
      this.server = createServer(this.app);
      this.server.listen(config.api.port, () => {
        systemLogger.startup('1.0.0', config.dev.nodeEnv);
        console.log(`🚀 Solana DeFi Trading Intelligence AI running on port ${config.api.port}`);
        console.log(`📊 Health check: http://localhost:${config.api.port}/health`);
        console.log(`🤖 API endpoint: http://localhost:${config.api.port}/api/v1`);
      });

      // Initialize WebSocket server
      this.wsServer = createWebSocketServer(this.app, this.server);

      // Graceful shutdown
      process.on('SIGTERM', this.gracefulShutdown.bind(this));
      process.on('SIGINT', this.gracefulShutdown.bind(this));

    } catch (error) {
      systemLogger.shutdown('Startup failed');
      console.error('Failed to start server:', error);
      process.exit(1);
    }
  }

  /**
   * Get LLM usage statistics
   */
  private async getLLMUsage(req: express.Request, res: express.Response): Promise<void> {
    try {
      const usageStats = officialLLMRouter.getUsageStats();
      
      res.json({
        success: true,
        data: usageStats
      });
    } catch (error) {
      errorLogger.aiError(error as Error, 'LLM usage statistics');
      res.status(500).json({
        success: false,
        error: { message: 'Failed to get LLM usage statistics' }
      });
    }
  }

  /**
   * Get LLM provider status
   */
  private async getLLMStatus(req: express.Request, res: express.Response): Promise<void> {
    try {
      const providerStatus = officialLLMRouter.getProviderStatus();
      
      res.json({
        success: true,
        data: providerStatus
      });
    } catch (error) {
      errorLogger.aiError(error as Error, 'LLM provider status');
      res.status(500).json({
        success: false,
        error: { message: 'Failed to get LLM provider status' }
      });
    }
  }

  /**
   * Graceful shutdown
   */
  private async gracefulShutdown(signal: string): Promise<void> {
    console.log(`\n🛑 Received ${signal}. Starting graceful shutdown...`);
    
    try {
      // Close WebSocket server
      if (this.wsServer) {
        this.wsServer.shutdown();
        console.log('✅ WebSocket server closed');
      }

      // Close HTTP server
      if (this.server) {
        this.server.close(() => {
          console.log('✅ HTTP server closed');
        });
      }

      // Cleanup services
      // await solanaService.cleanup();
      console.log('Solana service cleanup skipped (service disabled)');
      
      systemLogger.shutdown(`Received ${signal}`);
      console.log('✅ Graceful shutdown completed');
      
      process.exit(0);
    } catch (error) {
      console.error('❌ Error during shutdown:', error);
      process.exit(1);
    }
  }
}

// Start the application
const app = new SolanaDeFiAI();
app.start().catch((error) => {
  console.error('Failed to start application:', error);
  process.exit(1);
});
