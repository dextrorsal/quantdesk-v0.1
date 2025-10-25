//! Updated Backend Environment Configuration
//! This file provides secure environment variable management for QuantDesk backend

import dotenv from 'dotenv';

// Load environment variables
dotenv.config();

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
  
  // Solana Configuration (Standardized)
  SOLANA_NETWORK: process.env['SOLANA_NETWORK'] || 'devnet',
  SOLANA_RPC_URL: process.env['SOLANA_RPC_URL'] || 'https://api.devnet.solana.com',
  SOLANA_WS_URL: process.env['SOLANA_WS_URL'] || 'wss://api.devnet.solana.com',
  SOLANA_COMMITMENT: process.env['SOLANA_COMMITMENT'] || 'confirmed',
  
  // Program Configuration (Consolidated)
  QUANTDESK_PROGRAM_ID: process.env['QUANTDESK_PROGRAM_ID'] || 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
  
  // Wallet Configuration (Secure)
  SOLANA_PRIVATE_KEY: process.env['SOLANA_PRIVATE_KEY'] || '',
  KEEPER_PRIVATE_KEY: process.env['KEEPER_PRIVATE_KEY'] || '',
  ADMIN_PRIVATE_KEY: process.env['ADMIN_PRIVATE_KEY'] || '',
  
  // JWT Configuration
  JWT_SECRET: process.env['JWT_SECRET'] || '',
  JWT_EXPIRES_IN: process.env['JWT_EXPIRES_IN'] || '7d',
  
  // Frontend Configuration
  FRONTEND_URL: process.env['FRONTEND_URL'] || 'http://localhost:3000',
  CORS_ORIGIN: process.env['CORS_ORIGIN'] || 'http://localhost:3000',
  
  // Rate Limiting
  RATE_LIMIT_WINDOW: parseInt(process.env['RATE_LIMIT_WINDOW'] || '900000'),
  RATE_LIMIT_MAX: parseInt(process.env['RATE_LIMIT_MAX'] || '1000'),
  
  // Redis Configuration
  REDIS_URL: process.env['REDIS_URL'] || 'redis://localhost:6379',
  
  // Oracle Configuration (Enhanced)
  PYTH_NETWORK_URL: process.env['PYTH_NETWORK_URL'] || 'https://hermes.pyth.network/v2/updates/price/latest',
  PYTH_PRICE_FEEDS: {
    BTC: process.env['PYTH_PRICE_FEED_BTC'] || 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J',
    ETH: process.env['PYTH_PRICE_FEED_ETH'] || 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB',
    SOL: process.env['PYTH_PRICE_FEED_SOL'] || 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG',
    USDC: process.env['PYTH_PRICE_FEED_USDC'] || 'Gnt27xtC473ZT2Mw5u8wZ68Z3gULkSTb5DuxJy7eJotD',
  },
  
  // Security Configuration (New)
  CIRCUIT_BREAKER_ENABLED: process.env['CIRCUIT_BREAKER_ENABLED'] === 'true',
  KEEPER_AUTHORIZATION_REQUIRED: process.env['KEEPER_AUTHORIZATION_REQUIRED'] === 'true',
  ORACLE_STALENESS_THRESHOLD: parseInt(process.env['ORACLE_STALENESS_THRESHOLD'] || '60'),
  MAX_PRICE_CHANGE_PERCENT: parseInt(process.env['MAX_PRICE_CHANGE_PERCENT'] || '1000'),
  MAX_VOLUME_SPIKE_PERCENT: parseInt(process.env['MAX_VOLUME_SPIKE_PERCENT'] || '5000'),
  
  // Testing Configuration (New)
  TEST_MODE: process.env['TEST_MODE'] === 'true',
  MOCK_ORACLES: process.env['MOCK_ORACLES'] === 'true',
  ENABLE_LIQUIDATION: process.env['ENABLE_LIQUIDATION'] === 'true',
  ENABLE_FUNDING: process.env['ENABLE_FUNDING'] === 'true',
};

/**
 * Validate required environment variables
 */
export function validateEnvironment(): void {
  const requiredVars = [
    'SUPABASE_URL',
    'SUPABASE_ANON_KEY',
    'JWT_SECRET',
    'SOLANA_PRIVATE_KEY',
  ];

  const missingVars = requiredVars.filter(varName => !process.env[varName]);
  
  if (missingVars.length > 0) {
    throw new Error(`Missing required environment variables: ${missingVars.join(', ')}`);
  }

  // Validate private key format
  if (process.env.SOLANA_PRIVATE_KEY && process.env.SOLANA_PRIVATE_KEY === 'your_base58_private_key_here') {
    throw new Error('SOLANA_PRIVATE_KEY must be set to a valid Base58 private key');
  }
}

/**
 * Get environment-specific configuration
 */
export function getEnvironmentConfig() {
  const env = config.NODE_ENV;
  
  switch (env) {
    case 'production':
      return {
        ...config,
        SOLANA_NETWORK: 'mainnet',
        SOLANA_RPC_URL: process.env['SOLANA_RPC_URL'] || 'https://api.mainnet-beta.solana.com',
        TEST_MODE: false,
        MOCK_ORACLES: false,
        ENABLE_LIQUIDATION: true,
        ENABLE_FUNDING: true,
      };
    
    case 'staging':
      return {
        ...config,
        SOLANA_NETWORK: 'testnet',
        SOLANA_RPC_URL: process.env['SOLANA_RPC_URL'] || 'https://api.testnet.solana.com',
        TEST_MODE: false,
        MOCK_ORACLES: false,
        ENABLE_LIQUIDATION: true,
        ENABLE_FUNDING: true,
      };
    
    case 'development':
    default:
      return {
        ...config,
        SOLANA_NETWORK: 'devnet',
        SOLANA_RPC_URL: process.env['SOLANA_RPC_URL'] || 'https://api.devnet.solana.com',
        TEST_MODE: true,
        MOCK_ORACLES: false,
        ENABLE_LIQUIDATION: false,
        ENABLE_FUNDING: false,
      };
  }
}

export default config;
