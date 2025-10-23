import express from 'express';
import dns from 'dns';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import dotenv from 'dotenv';
import { createServer } from 'http';
import { createServer as createHttpsServer } from 'https';
import * as http from 'http';
import { readFileSync } from 'fs';
import { Server as SocketIOServer } from 'socket.io';
import { WebSocket, WebSocketServer } from 'ws';
import jwt from 'jsonwebtoken';
import cookieParser from 'cookie-parser';
import { parse as parseCookie } from 'cookie';
import passport from 'passport';
import session from 'express-session';

// Import middleware
import { errorHandler, notFoundHandler, requestIdMiddleware, responseTimeMiddleware } from './middleware/errorHandling';
import { authMiddleware } from './middleware/auth';
import { rateLimiters } from './middleware/rateLimiting';
import './middleware/adminAuth'; // Initialize Passport strategies

// Import routes
import authRoutes from './routes/auth';
import siwsRoutes from './routes/siws';
import referralRoutes from './routes/referrals';
import chatRoutes from './routes/chat';
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
import protocolStatsRoutes from './routes/protocol-stats';
import aiAgentRoutes from './routes/aiAgent';
import aiRoutes from './routes/ai';
import apiDocsRoutes from './routes/apiDocs';
import portfolioRoutes from './routes/portfolio';
import marketManagementRoutes from './routes/marketManagement';
import accountStateRoutes from './routes/accountState';
import rpcStatsRoutes from './routes/rpcStats';

// Import API improvements
import { createWebhookRoutes } from './services/webhookService';
import { createAPIDocRoutes } from './services/apiDocumentation';
// Skip Redis imports in development mode
console.log('⚠️  Skipping Redis imports in development mode');
const connectRedis = async () => Promise.resolve();
const getRedisSubscriber = () => ({ 
  subscribe: async (channel: string, callback: (message: string, channel: string) => void) => Promise.resolve(),
  unsubscribe: async (channel: string) => Promise.resolve()
});
const pingRedis = async () => Promise.resolve({ ok: true, error: null });

// Import services
import { WebSocketService } from './services/websocket';
import { pythOracleService } from './services/pythOracleService';
import { fundingService } from './services/funding';
import { LiquidationBot } from './services/liquidationBot';
import { metricsCollector } from './services/metricsCollector';
import { orderScheduler } from './services/orderScheduler';
import { Logger } from './utils/logger';
import { getSystemMonitor } from './services/systemMonitor';
import { config, validateConfig } from './config/environment';

// Load environment variables
dotenv.config();

// Validate environment configuration
validateConfig();

// Prefer IPv4 to avoid ENETUNREACH issues with some providers (e.g., Supabase)
try {
  // Node >=18
  dns.setDefaultResultOrder('ipv4first');
} catch (error) {
  console.warn('Failed to set DNS order:', error);
}

const app = express();

// Create server - Railway handles SSL termination at load balancer level
let server;
if (process.env['NODE_ENV'] === 'production' && process.env['RAILWAY_ENVIRONMENT']) {
  // Railway deployment - use HTTP (SSL handled by Railway)
  server = createServer(app);
  console.log('🚀 Railway deployment detected - using HTTP server (SSL handled by Railway)');
} else if (process.env['NODE_ENV'] === 'production') {
  // Other production deployments - try HTTPS with custom certificates
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
  // Development - use HTTP
  server = createServer(app);
  console.log('🔓 HTTP server created for development');
}
const io = new SocketIOServer(server, {
  cors: {
    origin: [
      process.env['FRONTEND_URL'] || "http://localhost:3001",
      "http://localhost:3001", // Main frontend
      "http://localhost:5173"  // Admin dashboard
    ],
    methods: ["GET", "POST"]
  }
});

const PORT = process.env['PORT'] || 3002;
const NODE_ENV = process.env['NODE_ENV'] || 'development';

// Initialize logger
const logger = new Logger();

// Initialize system monitoring
const systemMonitor = getSystemMonitor(logger);
systemMonitor.startMonitoring(5000); // Update every 5 seconds

// Add monitoring middleware
app.use((req, res, next) => {
  const startTime = Date.now();
  
  res.on('finish', () => {
    const latency = Date.now() - startTime;
    const isError = res.statusCode >= 400;
    systemMonitor.recordRequest(latency, isError);
  });
  
  next();
});

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
      connectSrc: ["'self'", "wss:", "https:"],
      fontSrc: ["'self'", "https:", "data:"],
      objectSrc: ["'none'"],
      mediaSrc: ["'self'"],
      frameSrc: ["'none'"],
    },
  },
  hsts: {
    maxAge: 31536000,
    includeSubDomains: true,
    preload: true,
  },
  noSniff: true,
  xssFilter: true,
  referrerPolicy: { policy: "strict-origin-when-cross-origin" },
  frameguard: { action: "deny" },
  hidePoweredBy: true,
}));

// CORS configuration
app.use(cors({
  origin: [
    process.env['FRONTEND_URL'] || "http://localhost:3001",
    "http://localhost:3001", // Main frontend
    "http://localhost:5173"  // Admin dashboard
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
app.use(cookieParser());

// Session middleware for OAuth
app.use(session({
  secret: process.env.SESSION_SECRET || 'quantdesk-admin-session-secret',
  resave: false,
  saveUninitialized: false,
  cookie: {
    secure: process.env.NODE_ENV === 'production',
    httpOnly: true,
    maxAge: 24 * 60 * 60 * 1000 // 24 hours
  }
}));

// Initialize Passport
app.use(passport.initialize());
app.use(passport.session());

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

// Redis health endpoint
app.get('/health/redis', async (_req, res) => {
  const result = await pingRedis();
  if (result.ok) {
    res.json({ status: 'ok' });
  } else {
    res.status(503).json({ status: 'degraded', error: result.error });
  }
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

// Debug endpoint for JWT testing
app.get('/api/debug/jwt', (req, res) => {
  console.log('🚀 Debug endpoint called');
  const jwt = require('jsonwebtoken');
  const token = jwt.sign({ wallet_pubkey: 'test-wallet-address', iat: Math.floor(Date.now() / 1000), exp: Math.floor(Date.now() / 1000) + (60 * 60) }, 'test-jwt-secret');
  
  res.json({
    jwtSecret: process.env.JWT_SECRET,
    testToken: token,
    configJwtSecret: config.JWT_SECRET
  });
});

// Test endpoint to see if middleware is called
app.get('/api/debug/test', (req, res) => {
  console.log('🚀 Test endpoint called');
  res.json({ message: 'Test endpoint working' });
});

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/siws', siwsRoutes);
app.use('/api/referrals', referralRoutes);
app.use('/api/chat', chatRoutes);
app.use('/api/markets', realSupabaseMarketRoutes);
app.use('/api/market-management', marketManagementRoutes);
app.use('/api/positions', authMiddleware, positionRoutes);
app.use('/api/orders', authMiddleware, orderRoutes);
app.use('/api/trades', authMiddleware, tradeRoutes);
app.use('/api/users', authMiddleware, userRoutes);
app.use('/api/admin', adminRoutes);
app.use('/api/ai', aiRoutes);
app.use('/api/liquidity', liquidityRoutes);
app.use('/api/oracle', oracleRoutes);
app.use('/api/supabase-oracle', supabaseOracleRoutes);
app.use('/api/metrics', metricsRoutes);
app.use('/api/grafana', grafanaIntegrationRoutes);
app.use('/api/advanced-orders', authMiddleware, advancedOrdersRoutes);
app.use('/api/cross-collateral', authMiddleware, crossCollateralRoutes);
app.use('/api/portfolio', authMiddleware, portfolioAnalyticsRoutes);
app.use('/api/portfolio-data', authMiddleware, portfolioRoutes);
app.use('/api/risk', authMiddleware, advancedRiskManagementRoutes);
app.use('/api/jit-liquidity', authMiddleware, jitLiquidityRoutes);
app.use('/api/deposits', authMiddleware, depositsRoutes);
app.use('/api/accounts', authMiddleware, accountsRoutes);
app.use('/api/protocol', protocolStatsRoutes); // Protocol monitoring (public stats)
app.use('/api/dev', aiAgentRoutes);
app.use('/api/docs', apiDocsRoutes);
app.use('/api/account-state', authMiddleware, accountStateRoutes);
app.use('/api/rpc', rpcStatsRoutes); // Public for testing

// API improvements routes
app.use('/api/webhooks', authMiddleware, createWebhookRoutes());

// WebSocket service
const wsService = WebSocketService.getInstance(io);
wsService.initialize();

// WebSocket Server for Chat
const wss = new WebSocketServer({ server });

interface AuthenticatedWebSocket extends WebSocket {
  isAlive: boolean;
  walletPubkey?: string;
}

const clients: Map<string, AuthenticatedWebSocket[]> = new Map(); // channel -> list of client sockets

wss.on('connection', async (ws: AuthenticatedWebSocket, req: http.IncomingMessage) => {
  ws.isAlive = true;
  ws.on('pong', () => { ws.isAlive = true; });

    const cookieHeader = req.headers.cookie;
    const sessionCookie = cookieHeader ? parseCookie(cookieHeader).qd_session : null;
    const channelId = new URL(req.url || '/', `http://${req.headers.host}`).searchParams.get('channelId') || 'global';

    if (!sessionCookie) {
      ws.send(JSON.stringify({ type: 'error', message: 'Authentication required' }));
      ws.close();
      return;
    }

    try {
      const decoded = jwt.verify(sessionCookie, config.JWT_SECRET as jwt.Secret) as { wallet_pubkey: string };
      ws.walletPubkey = decoded.wallet_pubkey;
      logger.info(`WebSocket connected for ${ws.walletPubkey} to channel ${channelId}`);

      // Add client to the specific channel
      if (!clients.has(channelId)) {
        clients.set(channelId, []);
      }
      clients.get(channelId)?.push(ws);

      // Subscribe to Redis Pub/Sub for this channel
      const subscriber = getRedisSubscriber();
      const namespacedChannel = `qd:${process.env.ENV_NAME || process.env.NODE_ENV || 'dev'}:pubsub:${channelId}`;
      await subscriber.subscribe(namespacedChannel, (message, channelName) => {
        // Broadcast message to all clients in this channel
        clients.get(channelName)?.forEach(client => {
          if (client.readyState === WebSocket.OPEN && client !== ws) {
            client.send(message);
          }
        });
      });

    } catch (error) {
      logger.error('WebSocket authentication error:', error);
      ws.send(JSON.stringify({ type: 'error', message: 'Authentication failed' }));
      ws.close();
      return;
    }

    ws.on('message', (message: string) => {
      // For now, WebSocket messages are for client-side chat, not direct server processing here
      // Instead, messages are sent via REST endpoint and then broadcast via Redis Pub/Sub
      logger.info(`Received WebSocket message from ${ws.walletPubkey} in channel ${channelId}: ${message}`);
    });

    ws.on('close', () => {
      logger.info(`WebSocket disconnected for ${ws.walletPubkey} from channel ${channelId}`);
      // Remove from the specific channel
      const channelClients = clients.get(channelId);
      if (channelClients) {
        clients.set(channelId, channelClients.filter(client => client !== ws));
      }
      // Unsubscribe from Redis Pub/Sub
      const namespacedChannel = `qd:${process.env.ENV_NAME || process.env.NODE_ENV || 'dev'}:pubsub:${channelId}`;
      getRedisSubscriber().unsubscribe(namespacedChannel);
    });

    ws.on('error', (error) => {
      logger.error(`WebSocket error for ${ws.walletPubkey} in channel ${channelId}:`, error);
    });
});

// Ping-pong for WebSocket keep-alive
setInterval(() => {
  wss.clients.forEach((ws: WebSocket) => {
    const client = ws as AuthenticatedWebSocket;
    if (!client.isAlive) return client.terminate();

    client.isAlive = false;
    client.ping();
  });
}, 30000); // Every 30 seconds

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
  // Connect Redis for sessions/presence (non-blocking)
  if (process.env.NODE_ENV === 'development' && !process.env.REDIS_URL) {
    logger.info('⚠️  Redis not configured, skipping Redis connection for development');
  } else {
    connectRedis().then(() => logger.info('✅ Redis connected')).catch((e) => {
      logger.warn('Redis connection failed, continuing without Redis:', e.message);
    });
  }
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

