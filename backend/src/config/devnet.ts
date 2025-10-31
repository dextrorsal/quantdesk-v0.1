export const DEVNET_CONFIG = {
  network: 'devnet',
  rpcEndpoint: process.env.SOLANA_RPC_URL || 'http://127.0.0.1:8899',
  wsEndpoint: process.env.SOLANA_WS_URL || 'ws://127.0.0.1:8900',
  commitment: 'confirmed' as const,
  
  // Program IDs (will be updated after deployment)
  programs: {
    quantdeskPerp: process.env.QUANTDESK_PROGRAM_ID || '',
  },
  
  // Markets configuration for devnet MVP
  markets: [
    { symbol: 'SOL-PERP', baseAsset: 'SOL', quoteAsset: 'USDC', maxLeverage: 5 },
    { symbol: 'BTC-PERP', baseAsset: 'BTC', quoteAsset: 'USDC', maxLeverage: 10 },
  ],
  
  // Oracle configuration (backend-only pattern)
  oracle: {
    updateInterval: 1000, // 1 second
    priceValidityWindow: 60000, // 60 seconds
  },
  
  // Devnet-specific settings
  settings: {
    enableLiquidation: false, // Disable liquidation for devnet testing
    enableFunding: false, // Disable funding for devnet testing
    mockPrices: true, // Use mock prices for testing
  }
};
