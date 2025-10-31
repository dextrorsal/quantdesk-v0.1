const Redis = require('ioredis');
const { Pool } = require('pg');
const winston = require('winston');
require('dotenv').config();

// Logger configuration
const logger = winston.createLogger({
  level: process.env.LOG_LEVEL || 'info',
  format: winston.format.combine(
    winston.format.timestamp(),
    winston.format.errors({ stack: true }),
    winston.format.json()
  ),
  transports: [
    new winston.transports.Console({
      format: winston.format.combine(
        winston.format.colorize(),
        winston.format.simple()
      )
    }),
    new winston.transports.File({ filename: 'logs/ingestion.log' })
  ]
});

// Redis connection with graceful error handling
const redis = new Redis({
  host: process.env.REDIS_HOST || 'localhost',
  port: process.env.REDIS_PORT || 6379,
  password: process.env.REDIS_PASSWORD,
  retryDelayOnFailover: 100,
  maxRetriesPerRequest: 3,
  lazyConnect: true,
  enableOfflineQueue: false, // Don't queue commands when offline
  connectTimeout: 5000,
  commandTimeout: 5000
});

// Handle Redis connection errors gracefully
redis.on('error', (error) => {
  logger.warn('Redis connection error (continuing without Redis):', error.message);
});

redis.on('connect', () => {
  logger.info('Redis connected successfully');
});

redis.on('ready', () => {
  logger.info('Redis ready for commands');
});

redis.on('close', () => {
  logger.warn('Redis connection closed');
});

redis.on('reconnecting', () => {
  logger.info('Redis reconnecting...');
});

// Database connection
const dbPool = new Pool({
  connectionString: process.env.DATABASE_URL,
  ssl: false, // Disable SSL for development
  max: 20,
  idleTimeoutMillis: 30000,
  connectionTimeoutMillis: 5000,
  statement_timeout: 15000,
  query_timeout: 15000,
  keepAlive: true,
  keepAliveInitialDelayMillis: 10000,
  application_name: 'quantdesk-ingestion'
});

// Redis Streams configuration
const STREAMS = {
  TICKS_RAW: 'ticks.raw',
  WHALES_RAW: 'whales.raw',
  NEWS_RAW: 'news.raw',
  TRENCH_RAW: 'trench.raw',
  DEFI_RAW: 'defi.raw',
  ANALYTICS_RAW: 'analytics.raw',
  MARKET_RAW: 'market.raw',
  PERPS_RAW: 'perps.raw',
  USER_EVENTS: 'user.events',
  SYSTEM_EVENTS: 'system.events',
  // Analytics Writer Output Streams
  SIGNALS_RAW: 'signals.raw',
  ALERTS_RAW: 'alerts.raw',
  REPORTS_RAW: 'reports.raw'
};

// Stream configuration
const STREAM_CONFIG = {
  maxLen: parseInt(process.env.REDIS_STREAM_MAX_LEN) || 10000,
  approximate: true
};

// Database batch configuration
const BATCH_CONFIG = {
  size: parseInt(process.env.BATCH_SIZE) || 100,
  intervalMs: parseInt(process.env.BATCH_INTERVAL_MS) || 100
};

// Error handling
process.on('uncaughtException', (error) => {
  logger.error('Uncaught Exception:', error);
  process.exit(1);
});

process.on('unhandledRejection', (reason, promise) => {
  logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

// Graceful shutdown
process.on('SIGINT', async () => {
  logger.info('Shutting down gracefully...');
  try {
    await redis.quit();
  } catch (error) {
    logger.warn('Error closing Redis connection:', error.message);
  }
  await dbPool.end();
  process.exit(0);
});

process.on('SIGTERM', async () => {
  logger.info('SIGTERM received, shutting down gracefully...');
  try {
    await redis.quit();
  } catch (error) {
    logger.warn('Error closing Redis connection:', error.message);
  }
  await dbPool.end();
  process.exit(0);
});

module.exports = {
  redis,
  dbPool,
  logger,
  STREAMS,
  STREAM_CONFIG,
  BATCH_CONFIG
};
