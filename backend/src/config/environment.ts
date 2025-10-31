import dotenv from 'dotenv';
import { getStandardizedConfig, validateStandardizedConfig, logConfigurationStatus } from './standardizedConfig';

// Load ONLY backend/.env to avoid root .env interfering in dev
// ts-node runs with CWD=backend, so this loads backend/.env
dotenv.config();

// Clean configuration object using only standardized names
export const config = {
  // Server Configuration
  NODE_ENV: process.env['NODE_ENV'] || 'development',
  PORT: parseInt(process.env['PORT'] || '3002'),
  
  // Supabase Configuration
  SUPABASE_URL: process.env['SUPABASE_URL'] || '',
  SUPABASE_ANON_KEY: process.env['SUPABASE_ANON_KEY'] || '',
  SUPABASE_ACCESS_TOKEN: process.env['SUPABASE_ACCESS_TOKEN'] || '',
  SUPABASE_PROJECT_ID: process.env['SUPABASE_PROJECT_ID'] || '',
  DATABASE_URL: process.env['DATABASE_URL'] || '',
  
  // Solana Configuration - Only standardized names
  SOLANA_NETWORK: process.env['SOLANA_NETWORK'] || 'devnet',
  SOLANA_RPC_URL: process.env['SOLANA_RPC_URL'] || 'https://api.devnet.solana.com',
  SOLANA_WS_URL: process.env['SOLANA_WS_URL'] || 'wss://api.devnet.solana.com',
  QUANTDESK_PROGRAM_ID: process.env['QUANTDESK_PROGRAM_ID'] || 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
  
  // JWT Configuration
  JWT_SECRET: process.env['JWT_SECRET'] || '',
  JWT_EXPIRES_IN: process.env['JWT_EXPIRES_IN'] || '7d',
  
  // Frontend Configuration
  FRONTEND_URL: process.env['FRONTEND_URL'] || 'http://localhost:3000',
  
  // Rate Limiting
  RATE_LIMIT_WINDOW: parseInt(process.env['RATE_LIMIT_WINDOW'] || '900000'),
  RATE_LIMIT_MAX: parseInt(process.env['RATE_LIMIT_MAX'] || '1000'),
  
  // Redis Configuration
  REDIS_URL: process.env['REDIS_URL'] || 'redis://localhost:6379',
  
  // Oracle Configuration - Only standardized names
  PYTH_NETWORK_URL: process.env['PYTH_NETWORK_URL'] || 'https://hermes.pyth.network/v2/updates/price/latest',
  PYTH_PRICE_FEEDS: {
    BTC: process.env['PYTH_PRICE_FEED_BTC'] || 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J',
    ETH: process.env['PYTH_PRICE_FEED_ETH'] || 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB',
    SOL: process.env['PYTH_PRICE_FEED_SOL'] || 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG',
  },
  
  // Twitter/X API Configuration
  TWITTER_BEARER_TOKEN: process.env['TWITTER_BEARER_TOKEN'] || '',
  
  // News API Configuration
  NEWSDATA_API_KEY: process.env['NEWSDATA_API_KEY'] || '',
  CRYPTOPANIC_API_KEY: process.env['CRYPTOPANIC_API_KEY'] || '',
  
  // Security
  CORS_ORIGIN: process.env['CORS_ORIGIN'] || 'http://localhost:3000',
  HELMET_CSP_ENABLED: process.env['HELMET_CSP_ENABLED'] === 'true',
  
  // Logging
  LOG_LEVEL: process.env['LOG_LEVEL'] || 'info',
  LOG_FILE: process.env['LOG_FILE'] || 'logs/quantdesk.log',
  
  // Trading Configuration
  MAX_LEVERAGE: parseInt(process.env['MAX_LEVERAGE'] || '100'),
  MIN_ORDER_SIZE: parseFloat(process.env['MIN_ORDER_SIZE'] || '0.001'),
  MAX_ORDER_SIZE: parseFloat(process.env['MAX_ORDER_SIZE'] || '1000000'),
  DEFAULT_FUNDING_INTERVAL: parseInt(process.env['DEFAULT_FUNDING_INTERVAL'] || '3600'),
  
  // Liquidation Configuration
  LIQUIDATION_THRESHOLD: parseFloat(process.env['LIQUIDATION_THRESHOLD'] || '0.95'),
  LIQUIDATION_PENALTY: parseFloat(process.env['LIQUIDATION_PENALTY'] || '0.05'),
  
  // Monitoring
  ENABLE_METRICS: process.env['ENABLE_METRICS'] === 'true',
  METRICS_PORT: parseInt(process.env['METRICS_PORT'] || '9090'),
};

// Validate required environment variables using standardized approach
export const validateConfig = (): void => {
  try {
    const standardizedConfig = getStandardizedConfig();
    validateStandardizedConfig(standardizedConfig);
    logConfigurationStatus(standardizedConfig);
  } catch (error) {
    console.error('❌ Configuration validation failed:', error);
    throw error;
  }
};

export default config;
