import dotenv from 'dotenv';
import { SolanaConfig, AIConfig, DatabaseConfig, APIConfig } from '../types';

// Load environment variables
dotenv.config();

// Set defaults for testing if not provided
if (!process.env['SOLANA_RPC_URL']) {
  process.env['SOLANA_RPC_URL'] = 'https://api.devnet.solana.com';
}
if (!process.env['SOLANA_PRIVATE_KEY']) {
  process.env['SOLANA_PRIVATE_KEY'] = 'test_key_placeholder';
}
if (!process.env['OPENAI_API_KEY']) {
  process.env['OPENAI_API_KEY'] = 'test_openai_key';
}
if (!process.env['POSTGRES_URL']) {
  process.env['POSTGRES_URL'] = 'postgresql://test:test@localhost:5432/test';
}
if (!process.env['REDIS_URL']) {
  process.env['REDIS_URL'] = 'redis://localhost:6379';
}

// Solana Configuration
export const solanaConfig: SolanaConfig = {
  rpcUrl: process.env['SOLANA_RPC_URL']!,
  wsUrl: process.env['SOLANA_WS_URL'] || process.env['SOLANA_RPC_URL']!.replace('http', 'ws'),
  privateKey: process.env['SOLANA_PRIVATE_KEY']!,
  publicKey: process.env['SOLANA_PUBLIC_KEY'] || '',
  cluster: (process.env['SOLANA_CLUSTER'] as 'mainnet-beta' | 'testnet' | 'devnet') || 'mainnet-beta'
};

// AI Configuration
export const aiConfig: AIConfig = {
  openaiApiKey: process.env['OPENAI_API_KEY']!,
  anthropicApiKey: process.env['ANTHROPIC_API_KEY'],
  modelName: process.env['AI_MODEL_NAME'] || 'gpt-4-turbo-preview',
  temperature: parseFloat(process.env['AI_TEMPERATURE'] || '0.7'),
  maxTokens: parseInt(process.env['AI_MAX_TOKENS'] || '4000')
};

// Database Configuration
export const databaseConfig: DatabaseConfig = {
  postgresUrl: process.env['POSTGRES_URL']!,
  redisUrl: process.env['REDIS_URL']!,
  influxdbUrl: process.env['INFLUXDB_URL'] || 'http://localhost:8086',
  influxdbToken: process.env['INFLUXDB_TOKEN'] || '',
  influxdbOrg: process.env['INFLUXDB_ORG'] || 'solana-ai',
  influxdbBucket: process.env['INFLUXDB_BUCKET'] || 'solana_data',
  elasticsearchUrl: process.env['ELASTICSEARCH_URL'] || 'http://localhost:9200'
};

// API Configuration
export const apiConfig: APIConfig = {
  port: parseInt(process.env['PORT'] || '3000'),
  rateLimitWindow: parseInt(process.env['RATE_LIMIT_WINDOW'] || '900000'), // 15 minutes
  rateLimitMax: parseInt(process.env['RATE_LIMIT_MAX'] || '1000'),
  corsOrigins: process.env['CORS_ORIGINS']?.split(',') || ['http://localhost:3100'],
  jwtSecret: process.env['JWT_SECRET'] || 'your-secret-key',
  encryptionKey: process.env['ENCRYPTION_KEY'] || 'your-32-character-encryption-key',
  quantdeskUrl: process.env['QUANTDESK_URL'] || 'http://localhost:3002'
};

// External API Keys
export const apiKeys = {
  helius: process.env['HELIUS_API_KEY'],
  quicknode: process.env['QUICKNODE_API_KEY'],
  alchemy: process.env['ALCHEMY_API_KEY'],
  pyth: process.env['PYTH_API_KEY'],
  switchboard: process.env['SWITCHBOARD_API_KEY'],
  coingecko: process.env['COINGECKO_API_KEY'],
  coinmarketcap: process.env['COINMARKETCAP_API_KEY'],
  jupiter: process.env['JUPITER_API_KEY'],
  drift: process.env['DRIFT_API_KEY'],
  raydium: process.env['RAYDIUM_API_KEY'],
  orca: process.env['ORCA_API_KEY'],
  mango: process.env['MANGO_API_KEY'],
  twitter: {
    apiKey: process.env['TWITTER_API_KEY'],
    apiSecret: process.env['TWITTER_API_SECRET'],
    accessToken: process.env['TWITTER_ACCESS_TOKEN'],
    accessSecret: process.env['TWITTER_ACCESS_SECRET'],
    bearerToken: process.env['TWITTER_BEARER_TOKEN']
  },
  discord: process.env['DISCORD_BOT_TOKEN'],
  telegram: process.env['TELEGRAM_BOT_TOKEN']
};

// Trading Configuration
export const tradingConfig = {
  minTransactionValue: parseInt(process.env['MIN_TRANSACTION_VALUE'] || '1000'),
  whaleThreshold: parseInt(process.env['WHALE_THRESHOLD'] || '100000'),
  liquidationThreshold: parseInt(process.env['LIQUIDATION_THRESHOLD'] || '50000'),
  sentimentUpdateInterval: parseInt(process.env['SENTIMENT_UPDATE_INTERVAL'] || '300000'), // 5 minutes
  priceUpdateInterval: parseInt(process.env['PRICE_UPDATE_INTERVAL'] || '10000'), // 10 seconds
  walletUpdateInterval: parseInt(process.env['WALLET_UPDATE_INTERVAL'] || '60000') // 1 minute
};

// WebSocket Configuration
export const wsConfig = {
  port: parseInt(process.env['WS_PORT'] || '8080'),
  heartbeatInterval: parseInt(process.env['WS_HEARTBEAT_INTERVAL'] || '30000'), // 30 seconds
  maxConnections: parseInt(process.env['WS_MAX_CONNECTIONS'] || '1000')
};

// Monitoring Configuration
export const monitoringConfig = {
  sentryDsn: process.env['SENTRY_DSN'],
  prometheusPort: parseInt(process.env['PROMETHEUS_PORT'] || '9090'),
  grafanaPort: parseInt(process.env['GRAFANA_PORT'] || '3000'),
  logLevel: process.env['LOG_LEVEL'] || 'info',
  debug: process.env['DEBUG'] || 'solana-ai:*'
};

// Development Configuration
export const devConfig = {
  nodeEnv: process.env['NODE_ENV'] || 'development',
  isDevelopment: process.env['NODE_ENV'] === 'development',
  isProduction: process.env['NODE_ENV'] === 'production',
  isTest: process.env['NODE_ENV'] === 'test'
};

// Export all configurations
export const config = {
  solana: solanaConfig,
  ai: aiConfig,
  database: databaseConfig,
  api: apiConfig,
  apiKeys,
  trading: tradingConfig,
  websocket: wsConfig,
  monitoring: monitoringConfig,
  dev: devConfig
};

export default config;
