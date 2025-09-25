import express from 'express';
import dns from 'dns';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import morgan from 'morgan';
import rateLimit from 'express-rate-limit';
import slowDown from 'express-slow-down';
import dotenv from 'dotenv';
import { createServer } from 'http';
import { createServer as createHttpsServer } from 'https';
import { readFileSync } from 'fs';
import { Server as SocketIOServer } from 'socket.io';

// Import middleware
import { errorHandler } from './middleware/errorHandler';
import { authMiddleware } from './middleware/auth';
import { rateLimitMiddleware } from './middleware/rateLimit';

// Import routes
import authRoutes from './routes/auth';
import marketRoutes from './routes/markets';
import positionRoutes from './routes/positions';
import orderRoutes from './routes/orders';
import tradeRoutes from './routes/trades';
import userRoutes from './routes/users';
import adminRoutes from './routes/admin';
import liquidityRoutes from './routes/liquidity';

// Import services
import { WebSocketService } from './services/websocket';
import { pythOracleService } from './services/pythOracleService';
import { fundingService } from './services/funding';
import { LiquidationBot } from './services/liquidationBot';
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
    origin: process.env['FRONTEND_URL'] || "http://localhost:3000",
    methods: ["GET", "POST"]
  }
});

const PORT = process.env['PORT'] || 3001;
const NODE_ENV = process.env['NODE_ENV'] || 'development';

// Initialize logger
const logger = new Logger();

// Security middleware
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
  origin: process.env['FRONTEND_URL'] || "http://localhost:3000",
  credentials: true,
  methods: ['GET', 'POST', 'PUT', 'DELETE', 'OPTIONS'],
  allowedHeaders: ['Content-Type', 'Authorization', 'X-Requested-With']
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

// Rate limiting
const limiter = rateLimit({
  windowMs: 15 * 60 * 1000, // 15 minutes
  max: 1000, // limit each IP to 1000 requests per windowMs
  message: {
    error: 'Too many requests from this IP, please try again later.',
    retryAfter: '15 minutes'
  },
  standardHeaders: true,
  legacyHeaders: false,
});

const speedLimiter = slowDown({
  windowMs: 15 * 60 * 1000, // 15 minutes
  delayAfter: 100, // allow 100 requests per 15 minutes, then...
  // express-slow-down v2 expects delayMs to be a function for the old behavior
  delayMs: () => 500,
  validate: { delayMs: false }
});

app.use('/api/', limiter);
app.use('/api/', speedLimiter);

// Custom rate limiting for trading endpoints
app.use('/api/trading/', rateLimitMiddleware({
  windowMs: 60 * 1000, // 1 minute
  max: 60, // 60 requests per minute for trading
  skipSuccessfulRequests: true
}));

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

// API routes
app.use('/api/auth', authRoutes);
app.use('/api/markets', marketRoutes);
app.use('/api/positions', authMiddleware, positionRoutes);
app.use('/api/orders', authMiddleware, orderRoutes);
app.use('/api/trades', authMiddleware, tradeRoutes);
app.use('/api/users', authMiddleware, userRoutes);
app.use('/api/admin', authMiddleware, adminRoutes);
app.use('/api/liquidity', liquidityRoutes);

// WebSocket service
const wsService = WebSocketService.getInstance(io);
wsService.initialize();

// 404 handler
app.use('*', (req, res) => {
  res.status(404).json({
    error: 'Endpoint not found',
    path: req.originalUrl,
    method: req.method
  });
});

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
});

export default app;
