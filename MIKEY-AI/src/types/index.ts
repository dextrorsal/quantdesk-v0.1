// Core Types for Solana DeFi Trading Intelligence AI

export interface SolanaConfig {
  rpcUrl: string;
  wsUrl: string;
  privateKey: string;
  publicKey: string;
  cluster: 'mainnet-beta' | 'testnet' | 'devnet';
}

export interface AIConfig {
  openaiApiKey: string;
  anthropicApiKey?: string | undefined;
  modelName: string;
  temperature: number;
  maxTokens: number;
}

export interface DatabaseConfig {
  postgresUrl: string;
  redisUrl: string;
  influxdbUrl: string;
  influxdbToken: string;
  influxdbOrg: string;
  influxdbBucket: string;
  elasticsearchUrl: string;
}

export interface PriceData {
  symbol: string;
  price: number;
  change24h: number;
  volume24h: number;
  volume?: number;
  source: 'pyth' | 'switchboard' | 'coingecko' | 'aggregated';
  timestamp: Date;
  confidence?: number;
}

export interface HistoricalPriceData {
  symbol: string;
  interval: '1m' | '5m' | '15m' | '1h' | '4h' | '1d';
  data: {
    timestamp: Date;
    open: number;
    high: number;
    low: number;
    close: number;
    volume: number;
  }[];
}

export interface WalletData {
  address: string;
  label?: string;
  totalValue: number;
  portfolio: {
    [token: string]: {
      amount: number;
      value: number;
      percentage: number;
    };
  };
  recentActivity: {
    transactions24h: number;
    volume24h: number;
    largestTransaction: number;
  };
  positions?: PositionData[];
}

export interface PositionData {
  protocol: 'drift' | 'mango' | 'raydium' | 'orca' | 'jupiter';
  type: 'long' | 'short' | 'liquidity' | 'lending';
  size: number;
  entryPrice?: number;
  currentPrice?: number;
  pnl?: number;
  collateral?: number;
  leverage?: number;
}

export interface TransactionData {
  transactionId: string;
  timestamp: Date;
  type: 'swap' | 'transfer' | 'liquidation' | 'mint' | 'burn';
  fromToken?: string;
  toToken?: string;
  amountIn?: number;
  amountOut?: number;
  protocol?: string;
  gasFee: number;
  walletAddress: string;
}

export interface LiquidationData {
  liquidationId: string;
  timestamp: Date;
  protocol: 'drift' | 'mango' | 'raydium' | 'orca';
  walletAddress: string;
  positionType: 'long' | 'short';
  size: number;
  collateral: number;
  liquidationPrice: number;
  marketPrice: number;
  pnl: number;
}

export interface SentimentData {
  symbol: string;
  sentimentScore: number; // -1 to 1
  sentimentLabel: 'bearish' | 'neutral' | 'bullish';
  socialVolume: number;
  newsSentiment: number;
  twitterSentiment: number;
  redditSentiment: number;
  timestamp: Date;
}

export interface TechnicalAnalysis {
  symbol: string;
  trend: 'bullish' | 'bearish' | 'neutral';
  supportLevels: number[];
  resistanceLevels: number[];
  rsi: number;
  macd: {
    macd: number;
    signal: number;
    histogram: number;
  };
  bollingerBands: {
    upper: number;
    middle: number;
    lower: number;
  };
  volumeProfile: {
    highVolumeNodes: number[];
    lowVolumeNodes: number[];
  };
}

export interface MarketAnalysis {
  symbol: string;
  technicalAnalysis: TechnicalAnalysis;
  marketStructure: {
    regime: 'bull_market' | 'bear_market' | 'sideways';
    volatility: 'low' | 'medium' | 'high';
    liquidity: 'low' | 'medium' | 'high';
  };
  predictions: {
    shortTerm: 'bullish' | 'bearish' | 'neutral';
    mediumTerm: 'bullish' | 'bearish' | 'neutral';
    confidence: number; // 0 to 1
  };
}

export interface AIQuery {
  query: string;
  context?: {
    symbols?: string[];
    timeframe?: string;
    walletId?: string;
  };
}

export interface AIResponse {
  response: string;
  sources: string[];
  confidence: number;
  timestamp: Date;
  data?: any;
  provider?: string;
}

export interface AlertConfig {
  id: string;
  type: 'price' | 'volume' | 'liquidation' | 'wallet' | 'sentiment';
  symbol?: string;
  walletId?: string;
  condition: {
    operator: '>' | '<' | '=' | '>=' | '<=';
    value: number;
  };
  channels: ('email' | 'sms' | 'push' | 'discord' | 'telegram')[];
  enabled: boolean;
}

export interface WebSocketMessage {
  type: 'price_update' | 'wallet_activity' | 'liquidation' | 'sentiment_update' | 'alert';
  data: any;
  timestamp: Date;
}

export interface APIConfig {
  port: number;
  rateLimitWindow: number;
  rateLimitMax: number;
  corsOrigins: string[];
  jwtSecret: string;
  encryptionKey: string;
  quantdeskUrl?: string;
}

export interface TradingTool {
  name: string;
  description: string;
  parameters: {
    [key: string]: {
      type: 'string' | 'number' | 'boolean' | 'array';
      required: boolean;
      description: string;
    };
  };
  execute: (params: any) => Promise<any>;
}

export interface AgentConfig {
  name: string;
  description: string;
  personality: string;
  tools: TradingTool[];
  memory: boolean;
  streaming: boolean;
}

export interface ErrorResponse {
  success: false;
  error: {
    code: string;
    message: string;
    details?: any;
  };
  timestamp: Date;
}

export interface SuccessResponse<T = any> {
  success: true;
  data: T;
  timestamp: Date;
}

export type APIResponse<T = any> = SuccessResponse<T> | ErrorResponse;

// Utility types
export type Timestamp = Date | string | number;
export type Address = string;
export type TokenSymbol = string;
export type Protocol = 'drift' | 'mango' | 'raydium' | 'orca' | 'jupiter' | 'serum';
export type Timeframe = '1m' | '5m' | '15m' | '1h' | '4h' | '1d' | '1w' | '1M';
export type Sentiment = 'bearish' | 'neutral' | 'bullish';
export type Trend = 'bullish' | 'bearish' | 'neutral';
export type PositionType = 'long' | 'short' | 'liquidity' | 'lending';
export type TransactionType = 'swap' | 'transfer' | 'liquidation' | 'mint' | 'burn';
