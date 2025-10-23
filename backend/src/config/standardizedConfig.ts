/**
 * Phase 4: Clean Environment Configuration
 * 
 * This module provides centralized configuration loading with:
 * - Only standardized variable names (no backward compatibility)
 * - Environment-specific configurations
 * - Type safety and validation
 * - Clean, maintainable code
 */

export interface StandardizedConfig {
  // Solana Configuration
  solanaNetwork: string;
  solanaRpcUrl: string;
  solanaWsUrl: string;
  solanaWalletPath: string;
  
  // Program Configuration
  quantdeskProgramId: string;
  
  // Database Configuration
  supabaseUrl: string;
  supabaseAnonKey: string;
  
  // Oracle Configuration
  pythNetworkUrl: string;
  pythPriceFeedSol: string;
  pythPriceFeedBtc: string;
  pythPriceFeedEth: string;
  
  // JWT Configuration
  jwtSecret: string;
  
  // Environment
  nodeEnv: string;
  port: number;
}

/**
 * Get clean, standardized configuration
 * 
 * Uses only standardized variable names - no backward compatibility.
 * This provides clean, maintainable code with clear variable naming.
 */
export const getStandardizedConfig = (): StandardizedConfig => {
  return {
    // Solana Configuration - Only standardized names
    solanaNetwork: process.env.SOLANA_NETWORK || 'devnet',
    solanaRpcUrl: process.env.SOLANA_RPC_URL || 'https://api.devnet.solana.com',
    solanaWsUrl: process.env.SOLANA_WS_URL || 'wss://api.devnet.solana.com',
    solanaWalletPath: process.env.SOLANA_WALLET || '',
    
    // Program Configuration - Only standardized names
    quantdeskProgramId: process.env.QUANTDESK_PROGRAM_ID || 'C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw',
    
    // Database Configuration - Already standardized
    supabaseUrl: process.env.SUPABASE_URL || '',
    supabaseAnonKey: process.env.SUPABASE_ANON_KEY || '',
    
    // Oracle Configuration - Only standardized names
    pythNetworkUrl: process.env.PYTH_NETWORK_URL || 'https://hermes.pyth.network/v2/updates/price/latest',
    pythPriceFeedSol: process.env.PYTH_PRICE_FEED_SOL || 'H6ARHf6YXhGYeQfUzQNGk6rDNnLBQKrenN712K4AQJEG',
    pythPriceFeedBtc: process.env.PYTH_PRICE_FEED_BTC || 'HovQMDrbAgAYPCmHVSrezcSmkMtXSSUsLDFANExrZh2J',
    pythPriceFeedEth: process.env.PYTH_PRICE_FEED_ETH || 'JBu1AL4obBcCMqKBBxhpWCNUt136ijcuMZLFvTP7iWdB',
    
    // JWT Configuration - Already standardized
    jwtSecret: process.env.JWT_SECRET || '',
    
    // Environment Configuration
    nodeEnv: process.env.NODE_ENV || 'development',
    port: parseInt(process.env.PORT || '3002', 10)
  };
};

/**
 * Validate clean configuration
 * 
 * Ensures all required configuration values are present
 * and provides clear error messages for missing values.
 */
export const validateStandardizedConfig = (config: StandardizedConfig): void => {
  const requiredFields: Array<keyof StandardizedConfig> = [
    'solanaWalletPath',
    'supabaseUrl',
    'supabaseAnonKey',
    'jwtSecret'
  ];
  
  const missing = requiredFields.filter(field => !config[field]);
  
  if (missing.length > 0) {
    throw new Error(`Missing required configuration: ${missing.join(', ')}`);
  }
  
  // Validate wallet file path
  if (config.solanaWalletPath === 'your_wallet_path_here' || 
      config.solanaWalletPath === '') {
    throw new Error('SOLANA_WALLET must be set to a valid wallet file path');
  }
  
  // Validate program ID length
  if (config.quantdeskProgramId.length < 32) {
    throw new Error('QUANTDESK_PROGRAM_ID must be a valid Solana program ID');
  }
  
  // Validate RPC URL format
  if (!config.solanaRpcUrl.startsWith('http')) {
    throw new Error('SOLANA_RPC_URL must be a valid HTTP/HTTPS URL');
  }
  
  if (!config.supabaseUrl.startsWith('http')) {
    throw new Error('SUPABASE_URL must be a valid HTTP/HTTPS URL');
  }
};

/**
 * Get environment-specific configuration
 * 
 * Returns configuration optimized for the current environment
 * (development, staging, production)
 */
export const getEnvironmentSpecificConfig = (): Partial<StandardizedConfig> => {
  const baseConfig = getStandardizedConfig();
  
  switch (baseConfig.nodeEnv) {
    case 'production':
      return {
        ...baseConfig,
        // Production-specific overrides
        solanaNetwork: 'mainnet-beta'
      };
      
    case 'staging':
      return {
        ...baseConfig,
        // Staging-specific overrides
        solanaNetwork: 'testnet'
      };
      
    case 'development':
    default:
      return {
        ...baseConfig,
        // Development-specific overrides
        solanaNetwork: 'devnet'
      };
  }
};

/**
 * Log configuration status for debugging
 * 
 * Shows current configuration values and environment status.
 */
export const logConfigurationStatus = (config: StandardizedConfig): void => {
  console.log('🔧 Configuration Status:');
  console.log(`  Environment: ${config.nodeEnv}`);
  console.log(`  Solana Network: ${config.solanaNetwork}`);
  console.log(`  RPC URL: ${config.solanaRpcUrl}`);
  console.log(`  Program ID: ${config.quantdeskProgramId}`);
  console.log('  ✅ Using standardized variable names');
};

export default getStandardizedConfig;
