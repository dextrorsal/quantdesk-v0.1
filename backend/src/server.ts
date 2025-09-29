import express from 'express';
import dns from 'dns';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import dotenv from 'dotenv';
import { createServer } from 'http';
import { createServer as createHttpsServer } from 'https';
import { readFileSync } from 'fs';
import { Server as SocketIOServer } from 'socket.io';

// Import middleware
import { errorHandler, notFoundHandler, requestIdMiddleware, responseTimeMiddleware } from './middleware/errorHandling';
import { authMiddleware } from './middleware/auth';
import { rateLimiters, createTieredRateLimit } from './middleware/rateLimiting';

// Import routes
import authRoutes from './routes/auth';
import marketRoutes from './routes/markets';
import simpleMarketRoutes from './routes/simpleMarkets';
import realSupabaseMarketRoutes from './routes/realSupabaseMarkets';
import positionRoutes from './routes/positions';
import orderRoutes from './routes/orders';
import tradeRoutes from './routes/trades';
import userRoutes from './routes/users';
import adminRoutes from './routes/admin';
import liquidityRoutes from './routes/liquidity';
import oracleRoutes from './routes/oracle';
import supabaseOracleRoutes from './routes/supabaseOracle';
import metricsRoutes from './routes/metrics';
import grafanaIntegrationRoutes from './routes/grafanaIntegration';
import advancedOrdersRoutes from './routes/advancedOrders';
import crossCollateralRoutes from './routes/crossCollateral';
import portfolioAnalyticsRoutes from './routes/portfolioAnalytics';
import advancedRiskManagementRoutes from './routes/advancedRiskManagement';
import jitLiquidityRoutes from './routes/jitLiquidity';
import depositsRoutes from './routes/deposits';
import accountsRoutes from './routes/accounts';
import rpcStatsRoutes from './routes/rpcStats';

// Import API improvements
import { createWebhookRoutes } from './services/webhookService';
import { createAPIDocRoutes } from './services/apiDocumentation';

// Import services
import { WebSocketService } from './services/websocket';
import { pythOracleService } from './services/pythOracleService';
import { fundingService } from './services/funding';
import { LiquidationBot } from './services/liquidationBot';
import { metricsCollector } from './services/metricsCollector';
import { orderScheduler } from './services/orderScheduler';
import { Logger } from './utils/logger';

// Load environment variables
dotenv.config();

// Prefer IPv4 to avoid ENETUNREACH issues with some providers (e.g., Supabase)
try {
  // Node >=18
  dns.setDefaultResultOrder('ipv4first');
} catch (_e) {}

const app = express();

// Create HTTPS server in production, HTTP in development
let server;
if (process.env['NODE_ENV'] === 'production') {
  try {
    const options = {
      key: process.env['SSL_KEY_PATH'] ? readFileSync(process.env['SSL_KEY_PATH']) : undefined,
      cert: process.env['SSL_CERT_PATH'] ? readFileSync(process.env['SSL_CERT_PATH']) : undefined,
    };
    
    if (options.key && options.cert) {
      server = createHttpsServer(options, app);
      console.log('🔒 HTTPS server created with SSL certificates');
    } else {
      console.warn('⚠️  SSL certificate paths not provided, falling back to HTTP');
      server = createServer(app);
    }
  } catch (error) {
    console.error('❌ Failed to create HTTPS server:', error);
    console.log('🔄 Falling back to HTTP server');
    server = createServer(app);
  }
} else {
  server = createServer(app);
  console.log('🔓 HTTP server created for development');
}
const io = new SocketIOServer(server, {
  cors: {
    origin: [
      process.env['FRONTEND_URL'] || "http://localhost:3000",
      "http://localhost:3001", // Vite dev server
      "http://localhost:3003"  // Alternative frontend port
    ],
    methods: ["GET", "POST"]
  }
});

const PORT = process.env['PORT'] || 3002;
const NODE_ENV = process.env['NODE_ENV'] || 'development';

// Initialize logger
const logger = new Logger();

// Security middleware
app.use(requestIdMiddleware);
app.use(responseTimeMiddleware);
app.use(helmet({
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
app.use(cors({
  origin: [
    process.env['FRONTEND_URL'] || "http://localhost:3000",
    "http://localhost:3001", // Vite dev server
    "http://localhost:3003"  // Alternative frontend port
  ],
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With', 'X-Request-ID', 'X-API-Key']
}));

// Compression
app.use(compression());

// Request logging
app.use(morgan('combined', {
  stream: {
    write: (message: string) => logger.info(message.trim())
  }
}));

// Body parsing
app.use(express.json({ limit: '10mb' }));
app.use(express.urlencoded({ extended: true, limit: '10mb' }));

// Health check endpoint
app.get('/health', (_req, res) => {
  res.json({
    status: 'healthy',
    timestamp: new Date().toISOString(),
    uptime: process.uptime(),
    environment: NODE_ENV,
    version: process.env['npm_package_version'] || '1.0.0'
  });
});

// Simple API endpoint for latest prices from Pyth Network (no rate limiting)
app.get('/api/prices', async (req, res) => {
  try {
    const prices = await pythOracleService.fetchLatestPrices();
    const priceArray = Array.from(prices.entries()).map(([symbol, data]) => ({
      symbol: `${symbol}/USDT`,
      price: data.price,
      change: 0, // We'll calculate this later
      changePercent: 0, // We'll calculate this later
      timestamp: data.timestamp
    }));
    
    res.json({
      success: true,
      data: priceArray,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error fetching prices:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch prices'
    });
  }
});

// Apply rate limiting (order matters: specific before generic)
app.use('/api/admin/', rateLimiters.admin);
app.use('/api/auth/', rateLimiters.auth);
app.use('/api/trading/', rateLimiters.trading);
app.use('/api/webhooks/', rateLimiters.webhook);
app.use('/api/', rateLimiters.public);

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/markets', realSupabaseMarketRoutes);
app.use('/api/positions', authMiddleware, positionRoutes);
app.use('/api/orders', authMiddleware, orderRoutes);
app.use('/api/trades', authMiddleware, tradeRoutes);
app.use('/api/users', authMiddleware, userRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/liquidity', liquidityRoutes);
app.use('/api/oracle', oracleRoutes);
app.use('/api/supabase-oracle', supabaseOracleRoutes);
app.use('/api/metrics', metricsRoutes);
app.use('/api/grafana', grafanaIntegrationRoutes);
app.use('/api/advanced-orders', authMiddleware, advancedOrdersRoutes);
app.use('/api/cross-collateral', authMiddleware, crossCollateralRoutes);
app.use('/api/portfolio', authMiddleware, portfolioAnalyticsRoutes);
app.use('/api/risk', authMiddleware, advancedRiskManagementRoutes);
app.use('/api/jit-liquidity', authMiddleware, jitLiquidityRoutes);
app.use('/api/deposits', authMiddleware, depositsRoutes);
app.use('/api/accounts', authMiddleware, accountsRoutes);
app.use('/api/rpc', rpcStatsRoutes); // Public for testing

// API improvements routes
app.use('/api/webhooks', authMiddleware, createWebhookRoutes());
app.use('/api/docs', createAPIDocRoutes());

// WebSocket service
const wsService = WebSocketService.getInstance(io);
wsService.initialize();

// 404 handler (must be before error handler)
app.use('*', notFoundHandler);

// Error handling middleware (must be last)
app.use(errorHandler);

// Graceful shutdown
process.on('SIGTERM', () => {
  logger.info('SIGTERM received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

process.on('SIGINT', () => {
  logger.info('SIGINT received, shutting down gracefully');
  server.close(() => {
    logger.info('Process terminated');
    process.exit(0);
  });
});

// Start server
server.listen(PORT, () => {
  logger.info(`🚀 QuantDesk Backend API running on port ${PORT}`);
  logger.info(`📊 Environment: ${NODE_ENV}`);
  logger.info(`🔗 Frontend URL: ${process.env['FRONTEND_URL'] || 'http://localhost:3000'}`);
  logger.info(`📡 WebSocket enabled: ${io ? 'Yes' : 'No'}`);
  
  // Start Pyth Oracle price feed service
  logger.info(`💰 Starting Pyth Oracle price feed service...`);
  pythOracleService.startPriceFeed();
  logger.info(`✅ Pyth Oracle price feed service started`);

  // Start funding scheduler (placeholder)
  fundingService.start();

  // Start liquidation monitoring (scaffold)
  LiquidationBot.getInstance().start();

  // Start metrics collection for Grafana
  logger.info(`📊 Starting metrics collection service...`);
  metricsCollector.start();
  logger.info(`✅ Metrics collection service started`);

  // Start order scheduler for advanced orders
  logger.info(`⏰ Starting order scheduler service...`);
  orderScheduler.start();
  logger.info(`✅ Order scheduler service started`);
});

export default app;
