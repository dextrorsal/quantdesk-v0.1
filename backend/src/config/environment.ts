import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

export const config = {
  // Server Configuration
  NODE_ENV: process.env['NODE_ENV'] || 'development',
  PORT: parseInt(process.env['PORT'] || '3001'),
  
  // Supabase Configuration
  SUPABASE_URL: process.env['SUPABASE_URL'] || '',
  SUPABASE_ANON_KEY: process.env['SUPABASE_ANON_KEY'] || '',
  SUPABASE_ACCESS_TOKEN: process.env['SUPABASE_ACCESS_TOKEN'] || '',
  SUPABASE_PROJECT_ID: process.env['SUPABASE_PROJECT_ID'] || '',
  DATABASE_URL: process.env['DATABASE_URL'] || '',
  
  // Solana Configuration
  SOLANA_NETWORK: process.env['SOLANA_NETWORK'] || 'devnet',
  RPC_URL: process.env['RPC_URL'] || 'https://api.devnet.solana.com',
  WS_URL: process.env['WS_URL'] || 'wss://api.devnet.solana.com',
  PROGRAM_ID: process.env['PROGRAM_ID'] || 'G7isTpCkw8TWhPhozSuZMbUjTEF8Jf8xxAguZyL39L8J',
  
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
  
  // Oracle Configuration
  PYTH_NETWORK_URL: process.env['PYTH_NETWORK_URL'] || 'https://hermes.pyth.network/v2/updates/price/latest',
  PYTH_PRICE_FEEDS: {
    BTC: process.env['PYTH_PRICE_FEED_BTC'] || 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J',
    ETH: process.env['PYTH_PRICE_FEED_ETH'] || 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB',
    SOL: process.env['PYTH_PRICE_FEED_SOL'] || 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG',
  },
  
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

// Validate required environment variables
export const validateConfig = (): void => {
  const required = [
    'SUPABASE_URL',
    'SUPABASE_ANON_KEY',
    'JWT_SECRET',
  ];
  
  const missing = required.filter(key => !process.env[key as keyof typeof process.env]);
  
  if (missing.length > 0) {
    throw new Error(`Missing required environment variables: ${missing.join(', ')}`);
  }
};

export default config;
