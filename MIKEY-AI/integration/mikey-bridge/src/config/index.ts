/**
 * Configuration for MIKEY-AI to QuantDesk Bridge
 */

export const config = {
  port: process.env.PORT || 3001,
  
  quantDesk: {
    baseUrl: process.env.QUANTDESK_BASE_URL || 'http://localhost:3000',
    apiKey: process.env.QUANTDESK_API_KEY || 'your-quantdesk-api-key',
    timeout: parseInt(process.env.QUANTDESK_TIMEOUT || '10000')
  },
  
  mikeyAI: {
    baseUrl: process.env.MIKEY_AI_BASE_URL || 'http://localhost:3002',
    apiKey: process.env.MIKEY_AI_API_KEY || 'your-mikey-ai-api-key',
    timeout: parseInt(process.env.MIKEY_AI_TIMEOUT || '10000')
  },
  
  redis: {
    host: process.env.REDIS_HOST || 'localhost',
    port: parseInt(process.env.REDIS_PORT || '6379'),
    password: process.env.REDIS_PASSWORD || undefined
  },
  
  logging: {
    level: process.env.LOG_LEVEL || 'info',
    format: process.env.LOG_FORMAT || 'json'
  },
  
  features: {
    enableRealTimeStreaming: process.env.ENABLE_REALTIME_STREAMING === 'true',
    enableArbitrageDetection: process.env.ENABLE_ARBITRAGE_DETECTION === 'true',
    enableWhaleTracking: process.env.ENABLE_WHALE_TRACKING === 'true',
    enableMLPredictions: process.env.ENABLE_ML_PREDICTIONS === 'true'
  }
};
