// QuantDesk Admin Terminal
// Professional trading platform administration interface

import React, { useState, useEffect } from 'react';
import { Container, Row, Col, Card, Button, Alert, Badge, Table, Form, Nav, Tab } from 'react-bootstrap';
import { useAuth } from 'contexts/AuthContext';
import { 
  PlayCircle, 
  PauseCircle, 
  Settings, 
  Activity, 
  Users, 
  DollarSign, 
  TrendingUp,
  AlertTriangle,
  CheckCircle,
  XCircle,
  RefreshCw,
  Terminal,
  Database,
  Server,
  Shield,
  BarChart,
  Eye,
  Zap,
  Globe,
  Lock,
  FileText,
  Cpu,
  Wifi,
  Link
} from 'react-feather';

// Types
interface SystemStatus {
  mode: 'demo' | 'live';
  status: 'online' | 'offline' | 'maintenance';
  uptime: string;
  lastUpdate: string;
}

interface TradingMetrics {
  totalTrades: number;
  totalVolume: number;
  activeUsers: number;
  winRate: number;
  totalPnL: number;
  maxDrawdown: number;
  sharpeRatio: number;
  dataProcessed: number;
  exchangeCoverage: number;
  gpuSpeedup: number;
  systemUptime: number;
}

interface User {
  id: string;
  wallet: string;
  status: 'active' | 'inactive' | 'suspended';
  trades: number;
  volume: number;
  lastActive: string;
  pnl: number;
  riskLevel: 'low' | 'medium' | 'high';
  kycStatus: 'verified' | 'pending' | 'rejected' | 'not_required';
  accountCreated: string;
  verificationLevel: 'basic' | 'intermediate' | 'advanced';
  totalDeposits: number;
  totalWithdrawals: number;
  winRate: number;
  averageTradeSize: number;
  maxLeverage: number;
  marginRequirement: number;
  lastLogin: string;
  loginCount: number;
  ipAddress: string;
}

interface UserTrade {
  id: string;
  userId: string;
  asset: string;
  side: 'buy' | 'sell';
  size: number;
  price: number;
  timestamp: string;
  pnl: number;
  strategy: string;
  exchange: string;
}

interface UserActivity {
  id: string;
  userId: string;
  action: string;
  timestamp: string;
  ipAddress: string;
  userAgent: string;
  details: string;
}

interface RiskMetrics {
  totalExposure: number;
  marginCalls: number;
  liquidations: number;
  highRiskPositions: number;
  averageLeverage: number;
}

interface Position {
  id: string;
  user: string;
  asset: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  pnl: number;
  marginRatio: number;
  riskLevel: 'low' | 'medium' | 'high' | 'critical';
  liquidationPrice: number;
  lastUpdate: string;
}

interface MarginCall {
  id: string;
  user: string;
  position: string;
  currentMargin: number;
  requiredMargin: number;
  timeToLiquidation: string;
  status: 'warning' | 'critical' | 'liquidated';
  timestamp: string;
}

interface MarketData {
  openInterest: number;
  fundingRate: number;
  volume24h: number;
  priceChange24h: number;
  orderBookDepth: number;
}

interface OrderBook {
  asset: string;
  bids: { price: number; size: number }[];
  asks: { price: number; size: number }[];
  lastUpdate: string;
}

interface RecentTrade {
  id: string;
  asset: string;
  side: 'buy' | 'sell';
  price: number;
  size: number;
  timestamp: string;
  user: string;
}

interface PriceFeed {
  asset: string;
  price: number;
  change24h: number;
  volume24h: number;
  lastUpdate: string;
  source: string;
}

interface SystemHealth {
  solanaRpc: 'healthy' | 'degraded' | 'down';
  smartContracts: 'healthy' | 'degraded' | 'down';
  pythOracle: 'healthy' | 'degraded' | 'down';
  exchanges: { [key: string]: 'healthy' | 'degraded' | 'down' };
  gpuStatus: 'healthy' | 'degraded' | 'down';
  dataPipeline: 'healthy' | 'degraded' | 'down';
}

interface SolanaMetrics {
  rpcLatency: number;
  transactionSuccessRate: number;
  averageGasFee: number;
  networkCongestion: number;
  validatorCount: number;
  slotTime: number;
  lastUpdate: string;
}

interface SmartContractMetrics {
  contractAddress: string;
  contractName: string;
  status: 'healthy' | 'degraded' | 'down';
  lastInteraction: string;
  totalInteractions: number;
  gasUsed: number;
  errorRate: number;
}

interface GasFeeMetrics {
  currentFee: number;
  averageFee: number;
  maxFee: number;
  priorityFee: number;
  lastUpdate: string;
  trend: 'up' | 'down' | 'stable';
}

interface TradingPerformanceReport {
  period: string;
  totalTrades: number;
  totalVolume: number;
  totalPnL: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  averageTradeSize: number;
  topPerformingStrategy: string;
  topPerformingAsset: string;
  averageHoldingTime: string;
  profitFactor: number;
  calmarRatio: number;
  sortinoRatio: number;
}

interface UserBehaviorAnalytics {
  totalUsers: number;
  activeUsers: number;
  newUsers: number;
  churnRate: number;
  averageSessionDuration: string;
  averageTradesPerUser: number;
  userRetentionRate: number;
  topUserSegments: { segment: string; count: number; percentage: number }[];
  userActivityPatterns: { hour: number; activity: number }[];
  geographicDistribution: { region: string; users: number; percentage: number }[];
}

interface StrategyPerformance {
  strategyName: string;
  totalTrades: number;
  winRate: number;
  totalPnL: number;
  sharpeRatio: number;
  maxDrawdown: number;
  averageTradeSize: number;
  profitFactor: number;
  lastUpdated: string;
}

interface AssetPerformance {
  asset: string;
  totalVolume: number;
  totalTrades: number;
  averagePrice: number;
  priceChange24h: number;
  volatility: number;
  liquidity: number;
  lastUpdated: string;
}

interface AuditLog {
  id: string;
  timestamp: string;
  userId: string;
  action: string;
  resource: string;
  ipAddress: string;
  userAgent: string;
  status: 'success' | 'failed' | 'blocked';
  details: string;
}

interface SuspiciousActivity {
  id: string;
  userId: string;
  activityType: string;
  severity: 'low' | 'medium' | 'high' | 'critical';
  description: string;
  timestamp: string;
  status: 'investigating' | 'resolved' | 'false_positive';
  riskScore: number;
}

interface ComplianceReport {
  period: string;
  totalTransactions: number;
  flaggedTransactions: number;
  kycCompliance: number;
  amlChecks: number;
  suspiciousActivityCount: number;
  auditTrailCompleteness: number;
  lastReportDate: string;
}

interface MLModelPerformance {
  modelName: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  trainingTime: string;
  lastTraining: string;
  status: 'active' | 'training' | 'idle' | 'error';
  predictionsToday: number;
  averageLatency: number;
}

interface GPUMetrics {
  gpuName: string;
  utilization: number;
  memoryUsed: number;
  memoryTotal: number;
  temperature: number;
  powerDraw: number;
  status: 'healthy' | 'overloaded' | 'idle' | 'error';
  lastUpdate: string;
}

interface DataPipelineHealth {
  component: string;
  status: 'healthy' | 'degraded' | 'down';
  latency: number;
  throughput: number;
  errorRate: number;
  lastUpdate: string;
  uptime: number;
}

interface LorentzianClassifierMetrics {
  totalPredictions: number;
  accuracy: number;
  precision: number;
  recall: number;
  f1Score: number;
  trainingSamples: number;
  lastTraining: string;
  averagePredictionTime: number;
  status: 'active' | 'training' | 'idle';
}

interface ExchangeStatus {
  name: string;
  type: 'CEX' | 'DEX';
  status: 'healthy' | 'degraded' | 'down';
  latency: number;
  uptime: number;
  lastUpdate: string;
  apiCallsToday: number;
  errorRate: number;
  rateLimitRemaining: number;
  rateLimitTotal: number;
  supportedAssets: number;
  tradingVolume24h: number;
  orderBookDepth: number;
  lastTrade: string;
}

interface ExchangeMetrics {
  exchange: string;
  totalVolume: number;
  totalTrades: number;
  averageTradeSize: number;
  topTradingPairs: { pair: string; volume: number }[];
  latency: number;
  errorRate: number;
  lastUpdate: string;
}

interface ChainStatus {
  name: string;
  chainId: string;
  status: 'healthy' | 'degraded' | 'down';
  blockHeight: number;
  blockTime: number;
  gasPrice: number;
  gasPriceGwei: number;
  pendingTransactions: number;
  totalTransactions: number;
  networkHashRate: string;
  difficulty: string;
  lastUpdate: string;
  rpcEndpoints: number;
  healthyEndpoints: number;
  averageLatency: number;
  errorRate: number;
  uptime: number;
}

interface CrossChainMetrics {
  chain: string;
  totalVolume: number;
  totalTransactions: number;
  averageGasPrice: number;
  activeAddresses: number;
  newAddresses: number;
  topTokens: { symbol: string; volume: number; price: number }[];
  lastUpdate: string;
}

const AdminDashboard: React.FC = () => {
  const { user } = useAuth();
  // State management
  const [systemStatus, setSystemStatus] = useState<SystemStatus>({
    mode: 'demo',
    status: 'online',
    uptime: '2d 14h 32m',
    lastUpdate: new Date().toISOString()
  });

  const [tradingMetrics, setTradingMetrics] = useState<TradingMetrics>({
    totalTrades: 1247,
    totalVolume: 2847392.50,
    activeUsers: 156,
    winRate: 53.5,
    totalPnL: 18947.32,
    maxDrawdown: -4.7,
    sharpeRatio: 0.42,
    dataProcessed: 885391,
    exchangeCoverage: 6,
    gpuSpeedup: 3.2,
    systemUptime: 99.9
  });

  const [users, setUsers] = useState<User[]>([
    {
      id: '1',
      wallet: '0x742d...8a9b',
      status: 'active',
      trades: 45,
      volume: 125430.50,
      lastActive: '2 minutes ago',
      pnl: 2340.50,
      riskLevel: 'medium',
      kycStatus: 'verified',
      accountCreated: '2025-09-15',
      verificationLevel: 'advanced',
      totalDeposits: 50000,
      totalWithdrawals: 25000,
      winRate: 68.5,
      averageTradeSize: 2787.34,
      maxLeverage: 5.0,
      marginRequirement: 0.20,
      lastLogin: '2 minutes ago',
      loginCount: 127,
      ipAddress: '192.168.1.100'
    },
    {
      id: '2',
      wallet: '0x8f3a...2c7d',
      status: 'active',
      trades: 32,
      volume: 89230.75,
      lastActive: '5 minutes ago',
      pnl: -1200.25,
      riskLevel: 'high',
      kycStatus: 'pending',
      accountCreated: '2025-09-20',
      verificationLevel: 'basic',
      totalDeposits: 15000,
      totalWithdrawals: 5000,
      winRate: 45.2,
      averageTradeSize: 2788.46,
      maxLeverage: 10.0,
      marginRequirement: 0.10,
      lastLogin: '5 minutes ago',
      loginCount: 89,
      ipAddress: '192.168.1.101'
    },
    {
      id: '3',
      wallet: '0x5e9c...4f1a',
      status: 'inactive',
      trades: 18,
      volume: 45670.25,
      lastActive: '1 hour ago',
      pnl: 890.75,
      riskLevel: 'low',
      kycStatus: 'verified',
      accountCreated: '2025-09-25',
      verificationLevel: 'intermediate',
      totalDeposits: 25000,
      totalWithdrawals: 15000,
      winRate: 72.1,
      averageTradeSize: 2537.24,
      maxLeverage: 3.0,
      marginRequirement: 0.33,
      lastLogin: '1 hour ago',
      loginCount: 45,
      ipAddress: '192.168.1.102'
    }
  ]);

  const [userTrades, setUserTrades] = useState<UserTrade[]>([
    {
      id: '1',
      userId: '1',
      asset: 'BTC/USD',
      side: 'buy',
      size: 0.5,
      price: 43250.00,
      timestamp: '2 minutes ago',
      pnl: 435.00,
      strategy: 'Lorentzian Classifier',
      exchange: 'Binance'
    },
    {
      id: '2',
      userId: '1',
      asset: 'ETH/USD',
      side: 'sell',
      size: 2.0,
      price: 2650.00,
      timestamp: '15 minutes ago',
      pnl: -140.00,
      strategy: 'Lag-based Strategy',
      exchange: 'Coinbase'
    },
    {
      id: '3',
      userId: '2',
      asset: 'SOL/USD',
      side: 'buy',
      size: 10.0,
      price: 95.50,
      timestamp: '30 minutes ago',
      pnl: 27.00,
      strategy: 'Logistic Regression',
      exchange: 'MEXC'
    }
  ]);

  const [userActivities, setUserActivities] = useState<UserActivity[]>([
    {
      id: '1',
      userId: '1',
      action: 'Login',
      timestamp: '2 minutes ago',
      ipAddress: '192.168.1.100',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
      details: 'Successful login from desktop'
    },
    {
      id: '2',
      userId: '1',
      action: 'Trade Executed',
      timestamp: '2 minutes ago',
      ipAddress: '192.168.1.100',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
      details: 'BTC/USD buy order executed'
    },
    {
      id: '3',
      userId: '2',
      action: 'Margin Call',
      timestamp: '5 minutes ago',
      ipAddress: '192.168.1.101',
      userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
      details: 'Margin ratio below 50% threshold'
    }
  ]);

  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics>({
    totalExposure: 2847392.50,
    marginCalls: 3,
    liquidations: 1,
    highRiskPositions: 12,
    averageLeverage: 2.4
  });

  const [positions, setPositions] = useState<Position[]>([
    {
      id: '1',
      user: '0x742d...8a9b',
      asset: 'BTC/USD',
      side: 'long',
      size: 0.5,
      entryPrice: 43250.00,
      currentPrice: 44120.00,
      pnl: 435.00,
      marginRatio: 0.85,
      riskLevel: 'medium',
      liquidationPrice: 39800.00,
      lastUpdate: '2 minutes ago'
    },
    {
      id: '2',
      user: '0x8f3a...2c7d',
      asset: 'ETH/USD',
      side: 'short',
      size: 2.0,
      entryPrice: 2650.00,
      currentPrice: 2720.00,
      pnl: -140.00,
      marginRatio: 0.45,
      riskLevel: 'critical',
      liquidationPrice: 2780.00,
      lastUpdate: '1 minute ago'
    },
    {
      id: '3',
      user: '0x5e9c...4f1a',
      asset: 'SOL/USD',
      side: 'long',
      size: 10.0,
      entryPrice: 95.50,
      currentPrice: 98.20,
      pnl: 27.00,
      marginRatio: 0.92,
      riskLevel: 'low',
      liquidationPrice: 88.00,
      lastUpdate: '3 minutes ago'
    }
  ]);

  const [marginCalls, setMarginCalls] = useState<MarginCall[]>([
    {
      id: '1',
      user: '0x8f3a...2c7d',
      position: 'ETH/USD Short',
      currentMargin: 0.45,
      requiredMargin: 0.50,
      timeToLiquidation: '15 minutes',
      status: 'critical',
      timestamp: '5 minutes ago'
    },
    {
      id: '2',
      user: '0x9a2b...7e8f',
      position: 'BTC/USD Long',
      currentMargin: 0.48,
      requiredMargin: 0.50,
      timeToLiquidation: '2 hours',
      status: 'warning',
      timestamp: '10 minutes ago'
    }
  ]);

  const [marketData, setMarketData] = useState<MarketData>({
    openInterest: 1250000,
    fundingRate: 0.0001,
    volume24h: 2847392.50,
    priceChange24h: 2.4,
    orderBookDepth: 450000
  });

  const [orderBook, setOrderBook] = useState<OrderBook>({
    asset: 'BTC/USD',
    bids: [
      { price: 44120.00, size: 0.5 },
      { price: 44115.00, size: 1.2 },
      { price: 44110.00, size: 0.8 },
      { price: 44105.00, size: 2.1 },
      { price: 44100.00, size: 1.5 }
    ],
    asks: [
      { price: 44125.00, size: 0.7 },
      { price: 44130.00, size: 1.0 },
      { price: 44135.00, size: 0.9 },
      { price: 44140.00, size: 1.8 },
      { price: 44145.00, size: 1.2 }
    ],
    lastUpdate: '2 seconds ago'
  });

  const [recentTrades, setRecentTrades] = useState<RecentTrade[]>([
    {
      id: '1',
      asset: 'BTC/USD',
      side: 'buy',
      price: 44122.50,
      size: 0.3,
      timestamp: '2 seconds ago',
      user: '0x742d...8a9b'
    },
    {
      id: '2',
      asset: 'ETH/USD',
      side: 'sell',
      price: 2718.75,
      size: 1.5,
      timestamp: '5 seconds ago',
      user: '0x8f3a...2c7d'
    },
    {
      id: '3',
      asset: 'SOL/USD',
      side: 'buy',
      price: 98.25,
      size: 5.0,
      timestamp: '8 seconds ago',
      user: '0x5e9c...4f1a'
    }
  ]);

  const [priceFeeds, setPriceFeeds] = useState<PriceFeed[]>([
    {
      asset: 'BTC/USD',
      price: 44120.00,
      change24h: 2.4,
      volume24h: 1250000,
      lastUpdate: '1 second ago',
      source: 'Pyth'
    },
    {
      asset: 'ETH/USD',
      price: 2720.00,
      change24h: -1.2,
      volume24h: 890000,
      lastUpdate: '1 second ago',
      source: 'Pyth'
    },
    {
      asset: 'SOL/USD',
      price: 98.20,
      change24h: 5.8,
      volume24h: 450000,
      lastUpdate: '1 second ago',
      source: 'Pyth'
    }
  ]);

  const [systemHealth, setSystemHealth] = useState<SystemHealth>({
    solanaRpc: 'healthy',
    smartContracts: 'healthy',
    pythOracle: 'healthy',
    exchanges: {
      'Binance': 'healthy',
      'Coinbase': 'healthy',
      'MEXC': 'healthy',
      'KuCoin': 'degraded',
      'Kraken': 'healthy',
      'Bitget': 'healthy'
    },
    gpuStatus: 'healthy',
    dataPipeline: 'healthy'
  });

  const [solanaMetrics, setSolanaMetrics] = useState<SolanaMetrics>({
    rpcLatency: 45,
    transactionSuccessRate: 99.8,
    averageGasFee: 0.00025,
    networkCongestion: 12.5,
    validatorCount: 1847,
    slotTime: 400,
    lastUpdate: '1 second ago'
  });

  const [smartContractMetrics, setSmartContractMetrics] = useState<SmartContractMetrics[]>([
    {
      contractAddress: '0x742d...8a9b',
      contractName: 'Perpetual DEX',
      status: 'healthy',
      lastInteraction: '2 seconds ago',
      totalInteractions: 12547,
      gasUsed: 125000,
      errorRate: 0.2
    },
    {
      contractAddress: '0x8f3a...2c7d',
      contractName: 'Risk Management',
      status: 'healthy',
      lastInteraction: '5 seconds ago',
      totalInteractions: 8932,
      gasUsed: 89000,
      errorRate: 0.1
    },
    {
      contractAddress: '0x5e9c...4f1a',
      contractName: 'Liquidation Engine',
      status: 'degraded',
      lastInteraction: '1 minute ago',
      totalInteractions: 4567,
      gasUsed: 45000,
      errorRate: 1.2
    }
  ]);

  const [gasFeeMetrics, setGasFeeMetrics] = useState<GasFeeMetrics>({
    currentFee: 0.00025,
    averageFee: 0.00022,
    maxFee: 0.00045,
    priorityFee: 0.00005,
    lastUpdate: '1 second ago',
    trend: 'up'
  });

  const [tradingPerformanceReport, setTradingPerformanceReport] = useState<TradingPerformanceReport>({
    period: 'Last 30 Days',
    totalTrades: 1247,
    totalVolume: 2847392.50,
    totalPnL: 18947.32,
    winRate: 53.5,
    sharpeRatio: 0.42,
    maxDrawdown: -4.7,
    averageTradeSize: 2284.50,
    topPerformingStrategy: 'Lorentzian Classifier',
    topPerformingAsset: 'BTC/USD',
    averageHoldingTime: '2h 15m',
    profitFactor: 1.85,
    calmarRatio: 0.38,
    sortinoRatio: 0.65
  });

  const [userBehaviorAnalytics, setUserBehaviorAnalytics] = useState<UserBehaviorAnalytics>({
    totalUsers: 156,
    activeUsers: 89,
    newUsers: 12,
    churnRate: 3.2,
    averageSessionDuration: '1h 23m',
    averageTradesPerUser: 8.0,
    userRetentionRate: 78.5,
    topUserSegments: [
      { segment: 'High Volume Traders', count: 23, percentage: 14.7 },
      { segment: 'Day Traders', count: 45, percentage: 28.8 },
      { segment: 'Swing Traders', count: 34, percentage: 21.8 },
      { segment: 'Scalpers', count: 28, percentage: 17.9 },
      { segment: 'Long-term Holders', count: 26, percentage: 16.7 }
    ],
    userActivityPatterns: [
      { hour: 0, activity: 12 },
      { hour: 1, activity: 8 },
      { hour: 2, activity: 5 },
      { hour: 3, activity: 3 },
      { hour: 4, activity: 2 },
      { hour: 5, activity: 4 },
      { hour: 6, activity: 8 },
      { hour: 7, activity: 15 },
      { hour: 8, activity: 28 },
      { hour: 9, activity: 45 },
      { hour: 10, activity: 52 },
      { hour: 11, activity: 48 },
      { hour: 12, activity: 38 },
      { hour: 13, activity: 42 },
      { hour: 14, activity: 55 },
      { hour: 15, activity: 62 },
      { hour: 16, activity: 58 },
      { hour: 17, activity: 45 },
      { hour: 18, activity: 38 },
      { hour: 19, activity: 42 },
      { hour: 20, activity: 48 },
      { hour: 21, activity: 35 },
      { hour: 22, activity: 25 },
      { hour: 23, activity: 18 }
    ],
    geographicDistribution: [
      { region: 'North America', users: 67, percentage: 42.9 },
      { region: 'Europe', users: 45, percentage: 28.8 },
      { region: 'Asia', users: 28, percentage: 17.9 },
      { region: 'South America', users: 12, percentage: 7.7 },
      { region: 'Oceania', users: 4, percentage: 2.6 }
    ]
  });

  const [strategyPerformance, setStrategyPerformance] = useState<StrategyPerformance[]>([
    {
      strategyName: 'Lorentzian Classifier',
      totalTrades: 456,
      winRate: 68.5,
      totalPnL: 12450.75,
      sharpeRatio: 0.85,
      maxDrawdown: -2.1,
      averageTradeSize: 2850.50,
      profitFactor: 2.15,
      lastUpdated: '2 minutes ago'
    },
    {
      strategyName: 'Lag-based Strategy',
      totalTrades: 342,
      winRate: 58.2,
      totalPnL: 7890.25,
      sharpeRatio: 0.62,
      maxDrawdown: -3.8,
      averageTradeSize: 2150.75,
      profitFactor: 1.75,
      lastUpdated: '5 minutes ago'
    },
    {
      strategyName: 'Logistic Regression',
      totalTrades: 289,
      winRate: 52.1,
      totalPnL: 3456.80,
      sharpeRatio: 0.45,
      maxDrawdown: -5.2,
      averageTradeSize: 1890.25,
      profitFactor: 1.42,
      lastUpdated: '8 minutes ago'
    },
    {
      strategyName: 'Neural Network',
      totalTrades: 160,
      winRate: 61.8,
      totalPnL: 2149.52,
      sharpeRatio: 0.58,
      maxDrawdown: -4.1,
      averageTradeSize: 1650.00,
      profitFactor: 1.68,
      lastUpdated: '12 minutes ago'
    }
  ]);

  const [assetPerformance, setAssetPerformance] = useState<AssetPerformance[]>([
    {
      asset: 'BTC/USD',
      totalVolume: 1250000,
      totalTrades: 456,
      averagePrice: 43250.00,
      priceChange24h: 2.4,
      volatility: 15.2,
      liquidity: 95.8,
      lastUpdated: '1 minute ago'
    },
    {
      asset: 'ETH/USD',
      totalVolume: 890000,
      totalTrades: 342,
      averagePrice: 2650.00,
      priceChange24h: -1.2,
      volatility: 18.5,
      liquidity: 92.3,
      lastUpdated: '1 minute ago'
    },
    {
      asset: 'SOL/USD',
      totalVolume: 450000,
      totalTrades: 289,
      averagePrice: 95.50,
      priceChange24h: 5.8,
      volatility: 22.1,
      liquidity: 88.7,
      lastUpdated: '1 minute ago'
    },
    {
      asset: 'AVAX/USD',
      totalVolume: 180000,
      totalTrades: 160,
      averagePrice: 28.75,
      priceChange24h: 3.2,
      volatility: 25.8,
      liquidity: 85.2,
      lastUpdated: '2 minutes ago'
    }
  ]);

  const [auditLogs, setAuditLogs] = useState<AuditLog[]>([
    {
      id: '1',
      timestamp: '2 minutes ago',
      userId: '0x742d...8a9b',
      action: 'System Mode Change',
      resource: 'Admin Panel',
      ipAddress: '192.168.1.100',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
      status: 'success',
      details: 'Switched from demo to live mode'
    },
    {
      id: '2',
      timestamp: '5 minutes ago',
      userId: '0x8f3a...2c7d',
      action: 'Large Trade Execution',
      resource: 'Trading Engine',
      ipAddress: '192.168.1.101',
      userAgent: 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)',
      status: 'success',
      details: 'Executed $50,000 BTC/USD trade'
    },
    {
      id: '3',
      timestamp: '10 minutes ago',
      userId: '0x5e9c...4f1a',
      action: 'Failed Login Attempt',
      resource: 'Authentication',
      ipAddress: '192.168.1.102',
      userAgent: 'Mozilla/5.0 (X11; Linux x86_64)',
      status: 'failed',
      details: 'Invalid credentials provided'
    },
    {
      id: '4',
      timestamp: '15 minutes ago',
      userId: '0x9a2b...7e8f',
      action: 'Risk Limit Update',
      resource: 'Risk Management',
      ipAddress: '192.168.1.103',
      userAgent: 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)',
      status: 'success',
      details: 'Updated max leverage to 5x'
    }
  ]);

  const [suspiciousActivities, setSuspiciousActivities] = useState<SuspiciousActivity[]>([
    {
      id: '1',
      userId: '0x8f3a...2c7d',
      activityType: 'Unusual Trading Pattern',
      severity: 'high',
      description: 'User executed 15 trades in 5 minutes with identical sizes',
      timestamp: '5 minutes ago',
      status: 'investigating',
      riskScore: 85
    },
    {
      id: '2',
      userId: '0x5e9c...4f1a',
      activityType: 'Multiple Failed Logins',
      severity: 'medium',
      description: '5 failed login attempts from different IP addresses',
      timestamp: '10 minutes ago',
      status: 'investigating',
      riskScore: 65
    },
    {
      id: '3',
      userId: '0x9a2b...7e8f',
      activityType: 'Large Withdrawal Request',
      severity: 'critical',
      description: 'User requested withdrawal of $100,000 without KYC verification',
      timestamp: '20 minutes ago',
      status: 'investigating',
      riskScore: 95
    }
  ]);

  const [complianceReport, setComplianceReport] = useState<ComplianceReport>({
    period: 'Last 30 Days',
    totalTransactions: 1247,
    flaggedTransactions: 23,
    kycCompliance: 98.5,
    amlChecks: 100,
    suspiciousActivityCount: 8,
    auditTrailCompleteness: 99.8,
    lastReportDate: '2025-10-15'
  });

  const [mlModelPerformance, setMLModelPerformance] = useState<MLModelPerformance[]>([
    {
      modelName: 'Lorentzian Classifier',
      accuracy: 68.5,
      precision: 72.3,
      recall: 65.8,
      f1Score: 68.9,
      trainingTime: '2h 15m',
      lastTraining: '2025-10-15 14:30',
      status: 'active',
      predictionsToday: 1247,
      averageLatency: 45
    },
    {
      modelName: 'Lag-based Strategy',
      accuracy: 58.2,
      precision: 61.5,
      recall: 55.2,
      f1Score: 58.2,
      trainingTime: '1h 45m',
      lastTraining: '2025-10-14 16:20',
      status: 'active',
      predictionsToday: 892,
      averageLatency: 38
    },
    {
      modelName: 'Logistic Regression',
      accuracy: 52.1,
      precision: 54.8,
      recall: 49.5,
      f1Score: 52.0,
      trainingTime: '45m',
      lastTraining: '2025-10-13 10:15',
      status: 'idle',
      predictionsToday: 456,
      averageLatency: 25
    },
    {
      modelName: 'Neural Network',
      accuracy: 61.8,
      precision: 64.2,
      recall: 59.1,
      f1Score: 61.5,
      trainingTime: '3h 30m',
      lastTraining: '2025-10-12 08:45',
      status: 'training',
      predictionsToday: 234,
      averageLatency: 67
    }
  ]);

  const [gpuMetrics, setGpuMetrics] = useState<GPUMetrics[]>([
    {
      gpuName: 'NVIDIA RTX 4090',
      utilization: 89,
      memoryUsed: 18.5,
      memoryTotal: 24.0,
      temperature: 72,
      powerDraw: 320,
      status: 'healthy',
      lastUpdate: '1 second ago'
    },
    {
      gpuName: 'NVIDIA RTX 4080',
      utilization: 76,
      memoryUsed: 14.2,
      memoryTotal: 16.0,
      temperature: 68,
      powerDraw: 280,
      status: 'healthy',
      lastUpdate: '1 second ago'
    },
    {
      gpuName: 'NVIDIA RTX 4070',
      utilization: 45,
      memoryUsed: 8.1,
      memoryTotal: 12.0,
      temperature: 55,
      powerDraw: 180,
      status: 'idle',
      lastUpdate: '1 second ago'
    }
  ]);

  const [dataPipelineHealth, setDataPipelineHealth] = useState<DataPipelineHealth[]>([
    {
      component: 'Market Data Ingestion',
      status: 'healthy',
      latency: 12,
      throughput: 1250,
      errorRate: 0.1,
      lastUpdate: '1 second ago',
      uptime: 99.9
    },
    {
      component: 'Feature Engineering',
      status: 'healthy',
      latency: 8,
      throughput: 1180,
      errorRate: 0.05,
      lastUpdate: '1 second ago',
      uptime: 99.8
    },
    {
      component: 'Model Inference',
      status: 'healthy',
      latency: 45,
      throughput: 890,
      errorRate: 0.2,
      lastUpdate: '1 second ago',
      uptime: 99.7
    },
    {
      component: 'Signal Generation',
      status: 'degraded',
      latency: 67,
      throughput: 650,
      errorRate: 1.2,
      lastUpdate: '2 seconds ago',
      uptime: 98.5
    },
    {
      component: 'Order Execution',
      status: 'healthy',
      latency: 23,
      throughput: 450,
      errorRate: 0.3,
      lastUpdate: '1 second ago',
      uptime: 99.6
    }
  ]);

  const [lorentzianClassifierMetrics, setLorentzianClassifierMetrics] = useState<LorentzianClassifierMetrics>({
    totalPredictions: 12547,
    accuracy: 68.5,
    precision: 72.3,
    recall: 65.8,
    f1Score: 68.9,
    trainingSamples: 125000,
    lastTraining: '2025-10-15 14:30',
    averagePredictionTime: 45,
    status: 'active'
  });

  const [exchangeStatus, setExchangeStatus] = useState<ExchangeStatus[]>([
    // Centralized Exchanges (CEX)
    {
      name: 'Binance',
      type: 'CEX',
      status: 'healthy',
      latency: 45,
      uptime: 99.9,
      lastUpdate: '1 second ago',
      apiCallsToday: 12547,
      errorRate: 0.1,
      rateLimitRemaining: 1150,
      rateLimitTotal: 1200,
      supportedAssets: 350,
      tradingVolume24h: 12500000,
      orderBookDepth: 2500000,
      lastTrade: '2 seconds ago'
    },
    {
      name: 'Coinbase',
      type: 'CEX',
      status: 'healthy',
      latency: 67,
      uptime: 99.8,
      lastUpdate: '1 second ago',
      apiCallsToday: 8932,
      errorRate: 0.2,
      rateLimitRemaining: 450,
      rateLimitTotal: 500,
      supportedAssets: 180,
      tradingVolume24h: 8900000,
      orderBookDepth: 1800000,
      lastTrade: '3 seconds ago'
    },
    {
      name: 'MEXC',
      type: 'CEX',
      status: 'healthy',
      latency: 52,
      uptime: 99.7,
      lastUpdate: '1 second ago',
      apiCallsToday: 4567,
      errorRate: 0.3,
      rateLimitRemaining: 280,
      rateLimitTotal: 300,
      supportedAssets: 220,
      tradingVolume24h: 4500000,
      orderBookDepth: 1200000,
      lastTrade: '1 second ago'
    },
    {
      name: 'KuCoin',
      type: 'CEX',
      status: 'degraded',
      latency: 125,
      uptime: 98.5,
      lastUpdate: '5 seconds ago',
      apiCallsToday: 2345,
      errorRate: 1.2,
      rateLimitRemaining: 180,
      rateLimitTotal: 200,
      supportedAssets: 150,
      tradingVolume24h: 2800000,
      orderBookDepth: 800000,
      lastTrade: '8 seconds ago'
    },
    {
      name: 'Kraken',
      type: 'CEX',
      status: 'healthy',
      latency: 38,
      uptime: 99.9,
      lastUpdate: '1 second ago',
      apiCallsToday: 3456,
      errorRate: 0.1,
      rateLimitRemaining: 220,
      rateLimitTotal: 250,
      supportedAssets: 120,
      tradingVolume24h: 3200000,
      orderBookDepth: 950000,
      lastTrade: '2 seconds ago'
    },
    {
      name: 'Bitget',
      type: 'CEX',
      status: 'healthy',
      latency: 58,
      uptime: 99.6,
      lastUpdate: '1 second ago',
      apiCallsToday: 1890,
      errorRate: 0.4,
      rateLimitRemaining: 150,
      rateLimitTotal: 180,
      supportedAssets: 200,
      tradingVolume24h: 2100000,
      orderBookDepth: 650000,
      lastTrade: '4 seconds ago'
    },
    // Decentralized Exchanges (DEX) - Solana
    {
      name: 'Jupiter',
      type: 'DEX',
      status: 'healthy',
      latency: 12,
      uptime: 99.8,
      lastUpdate: '1 second ago',
      apiCallsToday: 15678,
      errorRate: 0.05,
      rateLimitRemaining: 0,
      rateLimitTotal: 0,
      supportedAssets: 85,
      tradingVolume24h: 4500000,
      orderBookDepth: 0,
      lastTrade: '1 second ago'
    },
    {
      name: 'Drift',
      type: 'DEX',
      status: 'healthy',
      latency: 15,
      uptime: 99.9,
      lastUpdate: '1 second ago',
      apiCallsToday: 8934,
      errorRate: 0.1,
      rateLimitRemaining: 0,
      rateLimitTotal: 0,
      supportedAssets: 25,
      tradingVolume24h: 3200000,
      orderBookDepth: 0,
      lastTrade: '2 seconds ago'
    },
    {
      name: 'Orca',
      type: 'DEX',
      status: 'healthy',
      latency: 18,
      uptime: 99.7,
      lastUpdate: '1 second ago',
      apiCallsToday: 5678,
      errorRate: 0.2,
      rateLimitRemaining: 0,
      rateLimitTotal: 0,
      supportedAssets: 45,
      tradingVolume24h: 2800000,
      orderBookDepth: 0,
      lastTrade: '3 seconds ago'
    },
    {
      name: 'Raydium',
      type: 'DEX',
      status: 'healthy',
      latency: 22,
      uptime: 99.6,
      lastUpdate: '1 second ago',
      apiCallsToday: 4234,
      errorRate: 0.3,
      rateLimitRemaining: 0,
      rateLimitTotal: 0,
      supportedAssets: 120,
      tradingVolume24h: 2100000,
      orderBookDepth: 0,
      lastTrade: '4 seconds ago'
    },
    {
      name: 'Serum',
      type: 'DEX',
      status: 'degraded',
      latency: 45,
      uptime: 98.2,
      lastUpdate: '3 seconds ago',
      apiCallsToday: 1890,
      errorRate: 0.8,
      rateLimitRemaining: 0,
      rateLimitTotal: 0,
      supportedAssets: 35,
      tradingVolume24h: 1200000,
      orderBookDepth: 0,
      lastTrade: '6 seconds ago'
    },
    {
      name: 'Meteora',
      type: 'DEX',
      status: 'healthy',
      latency: 25,
      uptime: 99.5,
      lastUpdate: '1 second ago',
      apiCallsToday: 2345,
      errorRate: 0.4,
      rateLimitRemaining: 0,
      rateLimitTotal: 0,
      supportedAssets: 60,
      tradingVolume24h: 1800000,
      orderBookDepth: 0,
      lastTrade: '5 seconds ago'
    }
  ]);

  const [exchangeMetrics, setExchangeMetrics] = useState<ExchangeMetrics[]>([
    {
      exchange: 'Binance',
      totalVolume: 12500000,
      totalTrades: 4567,
      averageTradeSize: 2738.50,
      topTradingPairs: [
        { pair: 'BTC/USDT', volume: 4500000 },
        { pair: 'ETH/USDT', volume: 3200000 },
        { pair: 'SOL/USDT', volume: 1800000 }
      ],
      latency: 45,
      errorRate: 0.1,
      lastUpdate: '1 second ago'
    },
    {
      exchange: 'Jupiter',
      totalVolume: 4500000,
      totalTrades: 8934,
      averageTradeSize: 503.25,
      topTradingPairs: [
        { pair: 'SOL/USDC', volume: 1800000 },
        { pair: 'RAY/USDC', volume: 890000 },
        { pair: 'ORCA/USDC', volume: 650000 }
      ],
      latency: 12,
      errorRate: 0.05,
      lastUpdate: '1 second ago'
    },
    {
      exchange: 'Drift',
      totalVolume: 3200000,
      totalTrades: 2345,
      averageTradeSize: 1364.50,
      topTradingPairs: [
        { pair: 'SOL-PERP', volume: 1500000 },
        { pair: 'BTC-PERP', volume: 980000 },
        { pair: 'ETH-PERP', volume: 720000 }
      ],
      latency: 15,
      errorRate: 0.1,
      lastUpdate: '2 seconds ago'
    }
  ]);

  const [chainStatus, setChainStatus] = useState<ChainStatus[]>([
    {
      name: 'Ethereum',
      chainId: '1',
      status: 'healthy',
      blockHeight: 18945678,
      blockTime: 12.1,
      gasPrice: 25000000000,
      gasPriceGwei: 25,
      pendingTransactions: 156789,
      totalTransactions: 2345678901,
      networkHashRate: '892.5 TH/s',
      difficulty: '15.2 P',
      lastUpdate: '1 second ago',
      rpcEndpoints: 8,
      healthyEndpoints: 7,
      averageLatency: 245,
      errorRate: 0.8,
      uptime: 99.9
    },
    {
      name: 'BSC',
      chainId: '56',
      status: 'healthy',
      blockHeight: 34567890,
      blockTime: 3.2,
      gasPrice: 5000000000,
      gasPriceGwei: 5,
      pendingTransactions: 23456,
      totalTransactions: 4567890123,
      networkHashRate: '156.8 TH/s',
      difficulty: '2.8 P',
      lastUpdate: '1 second ago',
      rpcEndpoints: 6,
      healthyEndpoints: 6,
      averageLatency: 89,
      errorRate: 0.2,
      uptime: 99.8
    },
    {
      name: 'Solana',
      chainId: '101',
      status: 'healthy',
      blockHeight: 234567890,
      blockTime: 0.4,
      gasPrice: 5000,
      gasPriceGwei: 0.005,
      pendingTransactions: 0,
      totalTransactions: 12345678901,
      networkHashRate: 'N/A',
      difficulty: 'N/A',
      lastUpdate: '1 second ago',
      rpcEndpoints: 12,
      healthyEndpoints: 11,
      averageLatency: 45,
      errorRate: 0.1,
      uptime: 99.7
    },
    {
      name: 'Polygon',
      chainId: '137',
      status: 'degraded',
      blockHeight: 45678901,
      blockTime: 2.1,
      gasPrice: 30000000000,
      gasPriceGwei: 30,
      pendingTransactions: 45678,
      totalTransactions: 3456789012,
      networkHashRate: 'N/A',
      difficulty: 'N/A',
      lastUpdate: '3 seconds ago',
      rpcEndpoints: 5,
      healthyEndpoints: 3,
      averageLatency: 156,
      errorRate: 2.1,
      uptime: 98.5
    }
  ]);

  const [crossChainMetrics, setCrossChainMetrics] = useState<CrossChainMetrics[]>([
    {
      chain: 'Ethereum',
      totalVolume: 45000000,
      totalTransactions: 1234567,
      averageGasPrice: 25,
      activeAddresses: 2345678,
      newAddresses: 12345,
      topTokens: [
        { symbol: 'ETH', volume: 25000000, price: 2450.50 },
        { symbol: 'USDC', volume: 12000000, price: 1.00 },
        { symbol: 'USDT', volume: 8000000, price: 1.00 }
      ],
      lastUpdate: '1 second ago'
    },
    {
      chain: 'BSC',
      totalVolume: 18000000,
      totalTransactions: 2345678,
      averageGasPrice: 5,
      activeAddresses: 3456789,
      newAddresses: 23456,
      topTokens: [
        { symbol: 'BNB', volume: 8000000, price: 320.25 },
        { symbol: 'USDT', volume: 6000000, price: 1.00 },
        { symbol: 'BUSD', volume: 4000000, price: 1.00 }
      ],
      lastUpdate: '1 second ago'
    },
    {
      chain: 'Solana',
      totalVolume: 12000000,
      totalTransactions: 4567890,
      averageGasPrice: 0.005,
      activeAddresses: 1234567,
      newAddresses: 34567,
      topTokens: [
        { symbol: 'SOL', volume: 6000000, price: 95.50 },
        { symbol: 'USDC', volume: 4000000, price: 1.00 },
        { symbol: 'RAY', volume: 2000000, price: 2.85 }
      ],
      lastUpdate: '1 second ago'
    },
    {
      chain: 'Polygon',
      totalVolume: 8000000,
      totalTransactions: 1234567,
      averageGasPrice: 30,
      activeAddresses: 987654,
      newAddresses: 12345,
      topTokens: [
        { symbol: 'MATIC', volume: 3000000, price: 0.85 },
        { symbol: 'USDC', volume: 2500000, price: 1.00 },
        { symbol: 'USDT', volume: 2500000, price: 1.00 }
      ],
      lastUpdate: '2 seconds ago'
    }
  ]);

  const [isLoading, setIsLoading] = useState(false);
  const [alert, setAlert] = useState<{ type: 'success' | 'warning' | 'danger' | 'info', message: string } | null>(null);
  const [activeTab, setActiveTab] = useState('overview');

  // Toggle between demo and live mode
  const toggleMode = async () => {
    setIsLoading(true);
    try {
      const newMode = systemStatus.mode === 'demo' ? 'live' : 'demo';
      
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      setSystemStatus(prev => ({
        ...prev,
        mode: newMode,
        lastUpdate: new Date().toISOString()
      }));
      
      setAlert({
        type: 'success',
        message: `Successfully switched to ${newMode.toUpperCase()} mode`
      });
    } catch (error) {
      setAlert({
        type: 'danger',
        message: 'Failed to switch mode. Please try again.'
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Refresh system data
  const refreshData = async () => {
    setIsLoading(true);
    try {
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 500));
      
      setTradingMetrics(prev => ({
        ...prev,
        totalTrades: prev.totalTrades + Math.floor(Math.random() * 10),
        totalVolume: prev.totalVolume + Math.random() * 10000,
        activeUsers: prev.activeUsers + Math.floor(Math.random() * 5) - 2
      }));
      
      setSystemStatus(prev => ({
        ...prev,
        lastUpdate: new Date().toISOString()
      }));
    } catch (error) {
      setAlert({
        type: 'danger',
        message: 'Failed to refresh data'
      });
    } finally {
      setIsLoading(false);
    }
  };

  // Auto-refresh every 30 seconds
  useEffect(() => {
    const interval = setInterval(refreshData, 30000);
    return () => clearInterval(interval);
  }, []);

  // Clear alert after 5 seconds
  useEffect(() => {
    if (alert) {
      const timer = setTimeout(() => setAlert(null), 5000);
      return () => clearTimeout(timer);
    }
  }, [alert]);

  return (
    <div className="admin-dashboard">
      {/* Terminal Header */}
      <div className="terminal-header bg-dark text-success p-3 border-bottom border-success">
        <Container fluid>
          <Row className="align-items-center">
            <Col>
              <h1 className="terminal-heading mb-0">
                <Terminal className="me-2" />
                QuantDesk Admin Terminal
              </h1>
              <p className="terminal-text mb-0">
                System Status: <span className="text-cyan">{systemStatus.status.toUpperCase()}</span> | 
                Mode: <span className="text-warning">{systemStatus.mode.toUpperCase()}</span> | 
                Uptime: <span className="text-info">{systemStatus.uptime}</span> |
                User: <span className="text-success">{user?.username || 'dex'}</span>
              </p>
            </Col>
            <Col xs="auto">
              <Button 
                variant="outline-success" 
                onClick={refreshData}
                disabled={isLoading}
                className="me-2"
              >
                <RefreshCw className={isLoading ? 'spin' : ''} />
              </Button>
              <Button variant="outline-info">
                <Settings />
              </Button>
            </Col>
          </Row>
        </Container>
      </div>

      <Container fluid className="p-4">
        {/* Alert */}
        {alert && (
          <Alert 
            variant={alert.type === 'success' ? 'success' : alert.type === 'warning' ? 'warning' : alert.type === 'danger' ? 'danger' : 'info'}
            className="terminal-alert"
            dismissible
            onClose={() => setAlert(null)}
          >
            {alert.message}
          </Alert>
        )}

        {/* Tab Navigation */}
            <Tab.Container activeKey={activeTab} onSelect={(k) => setActiveTab(k || 'overview')}>
              <Row>
                <Col md={2}>
                  <Nav variant="pills" className="flex-column">
                <Nav.Item>
                  <Nav.Link eventKey="overview" className="terminal-text">
                    <BarChart className="me-2" />
                    Overview
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="trading" className="terminal-text">
                    <TrendingUp className="me-2" />
                    Trading Operations
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="risk" className="terminal-text">
                    <AlertTriangle className="me-2" />
                    Risk Management
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="users" className="terminal-text">
                    <Users className="me-2" />
                    User Management
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="system" className="terminal-text">
                    <Server className="me-2" />
                    System Health
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="market" className="terminal-text">
                    <Globe className="me-2" />
                    Market Data
                  </Nav.Link>
                </Nav.Item>
                <Nav.Item>
                  <Nav.Link eventKey="compliance" className="terminal-text">
                    <Lock className="me-2" />
                    Compliance
                  </Nav.Link>
                </Nav.Item>
                    <Nav.Item>
                      <Nav.Link eventKey="analytics" className="terminal-text">
                        <FileText className="me-2" />
                        Analytics
                      </Nav.Link>
                    </Nav.Item>
                    <Nav.Item>
                      <Nav.Link eventKey="quantdesk" className="terminal-text">
                        <Cpu className="me-2" />
                        QuantDesk Core
                      </Nav.Link>
                    </Nav.Item>
                    <Nav.Item>
                      <Nav.Link eventKey="exchanges" className="terminal-text">
                        <Globe className="me-2" />
                        Exchange Status
                      </Nav.Link>
                    </Nav.Item>
                    <Nav.Item>
                      <Nav.Link eventKey="chains" className="terminal-text">
                        <Link className="me-2" />
                        Cross-Chain
                      </Nav.Link>
                    </Nav.Item>
              </Nav>
                </Col>
                <Col md={10}>
                  <Tab.Content>
                {/* Overview Tab */}
                <Tab.Pane eventKey="overview">
                  {/* Mode Toggle Section */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header className="d-flex justify-content-between align-items-center">
                          <h5 className="terminal-heading mb-0">
                            <Activity className="me-2" />
                            System Mode Control
                          </h5>
                          <Badge 
                            bg={systemStatus.mode === 'live' ? 'danger' : 'warning'}
                            className="terminal-glow"
                          >
                            {systemStatus.mode.toUpperCase()}
                          </Badge>
                        </Card.Header>
                        <Card.Body>
                          <Row className="align-items-center">
                            <Col md={8}>
                              <div className="d-flex align-items-center mb-3">
                                <span className="terminal-text me-3">Current Mode:</span>
                                <Badge 
                                  bg={systemStatus.mode === 'live' ? 'danger' : 'warning'}
                                  className="fs-6 px-3 py-2"
                                >
                                  {systemStatus.mode.toUpperCase()}
                                </Badge>
                              </div>
                              <p className="terminal-text mb-3">
                                {systemStatus.mode === 'demo' 
                                  ? 'Demo mode: All trades are simulated. No real funds are at risk.'
                                  : 'Live mode: Real trading with actual funds. Use with caution.'
                                }
                              </p>
                              <div className="d-flex align-items-center">
                                <span className="terminal-text me-3">Switch Mode:</span>
                                <label className="terminal-toggle me-3">
                                  <input 
                                    type="checkbox" 
                                    checked={systemStatus.mode === 'live'}
                                    onChange={toggleMode}
                                    disabled={isLoading}
                                  />
                                  <span className="slider"></span>
                                </label>
                                <span className="terminal-text">
                                  {systemStatus.mode === 'demo' ? 'Demo' : 'Live'}
                                </span>
                              </div>
                            </Col>
                            <Col md={4} className="text-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="terminal-text">System Online</span>
                              <br />
                              <small className="text-muted">
                                Last update: {new Date(systemStatus.lastUpdate).toLocaleTimeString()}
                              </small>
                            </Col>
                          </Row>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Key Metrics */}
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <TrendingUp className="me-2" />
                        Key Metrics
                      </h4>
                    </Col>
                  </Row>
                  <Row className="mb-4">
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{tradingMetrics.totalTrades.toLocaleString()}</div>
                        <div className="metric-label">Total Trades</div>
                        <div className="metric-change positive">+12.5%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">${tradingMetrics.totalVolume.toLocaleString()}</div>
                        <div className="metric-label">Total Volume</div>
                        <div className="metric-change positive">+8.3%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{tradingMetrics.activeUsers}</div>
                        <div className="metric-label">Active Users</div>
                        <div className="metric-change positive">+5.2%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{tradingMetrics.winRate}%</div>
                        <div className="metric-label">Win Rate</div>
                        <div className="metric-change positive">+2.1%</div>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* Trading Operations Tab */}
                <Tab.Pane eventKey="trading">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <TrendingUp className="me-2" />
                        Trading Operations
                      </h4>
                    </Col>
                  </Row>
                  <Row className="mb-4">
                    <Col md={4}>
                      <Card className="metric-card">
                        <div className="metric-value">{tradingMetrics.sharpeRatio}</div>
                        <div className="metric-label">Sharpe Ratio</div>
                        <div className="metric-change positive">+0.05</div>
                      </Card>
                    </Col>
                    <Col md={4}>
                      <Card className="metric-card">
                        <div className="metric-value">{tradingMetrics.maxDrawdown}%</div>
                        <div className="metric-label">Max Drawdown</div>
                        <div className="metric-change negative">-0.2%</div>
                      </Card>
                    </Col>
                    <Col md={4}>
                      <Card className="metric-card">
                        <div className="metric-value">{tradingMetrics.gpuSpeedup}x</div>
                        <div className="metric-label">GPU Speedup</div>
                        <div className="metric-change positive">+0.3x</div>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* Risk Management Tab */}
                <Tab.Pane eventKey="risk">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <AlertTriangle className="me-2" />
                        Risk Management
                      </h4>
                    </Col>
                  </Row>
                  
                  {/* Risk Metrics Overview */}
                  <Row className="mb-4">
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">${riskMetrics.totalExposure.toLocaleString()}</div>
                        <div className="metric-label">Total Exposure</div>
                        <div className="metric-change positive">+2.1%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{riskMetrics.marginCalls}</div>
                        <div className="metric-label">Margin Calls</div>
                        <div className="metric-change negative">+1</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{riskMetrics.liquidations}</div>
                        <div className="metric-label">Liquidations</div>
                        <div className="metric-change positive">-1</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{riskMetrics.highRiskPositions}</div>
                        <div className="metric-label">High Risk Positions</div>
                        <div className="metric-change negative">+2</div>
                      </Card>
                    </Col>
                  </Row>

                  {/* Margin Call Alerts */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <AlertTriangle className="me-2" />
                            Margin Call Alerts
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          {marginCalls.length > 0 ? (
                            <Table className="terminal-table">
                              <thead>
                                <tr>
                                  <th>User</th>
                                  <th>Position</th>
                                  <th>Current Margin</th>
                                  <th>Required Margin</th>
                                  <th>Time to Liquidation</th>
                                  <th>Status</th>
                                  <th>Timestamp</th>
                                </tr>
                              </thead>
                              <tbody>
                                {marginCalls.map(call => (
                                  <tr key={call.id}>
                                    <td className="terminal-text">{call.user}</td>
                                    <td className="terminal-text">{call.position}</td>
                                    <td className="terminal-text">{(call.currentMargin * 100).toFixed(1)}%</td>
                                    <td className="terminal-text">{(call.requiredMargin * 100).toFixed(1)}%</td>
                                    <td className="terminal-text">{call.timeToLiquidation}</td>
                                    <td>
                                      <Badge 
                                        bg={call.status === 'critical' ? 'danger' : call.status === 'warning' ? 'warning' : 'secondary'}
                                        className="terminal-glow"
                                      >
                                        {call.status}
                                      </Badge>
                                    </td>
                                    <td className="terminal-text">{call.timestamp}</td>
                                  </tr>
                                ))}
                              </tbody>
                            </Table>
                          ) : (
                            <div className="text-center py-4">
                              <CheckCircle className="text-success mb-2" size={48} />
                              <p className="terminal-text">No active margin calls</p>
                            </div>
                          )}
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Active Positions */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Activity className="me-2" />
                            Active Positions
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>User</th>
                                <th>Asset</th>
                                <th>Side</th>
                                <th>Size</th>
                                <th>Entry Price</th>
                                <th>Current Price</th>
                                <th>P&L</th>
                                <th>Margin Ratio</th>
                                <th>Risk Level</th>
                                <th>Liquidation Price</th>
                                <th>Last Update</th>
                              </tr>
                            </thead>
                            <tbody>
                              {positions.map(position => (
                                <tr key={position.id}>
                                  <td className="terminal-text">{position.user}</td>
                                  <td className="terminal-text">{position.asset}</td>
                                  <td>
                                    <Badge 
                                      bg={position.side === 'long' ? 'success' : 'danger'}
                                    >
                                      {position.side}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{position.size}</td>
                                  <td className="terminal-text">${position.entryPrice.toLocaleString()}</td>
                                  <td className="terminal-text">${position.currentPrice.toLocaleString()}</td>
                                  <td className={`terminal-text ${position.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                                    ${position.pnl.toLocaleString()}
                                  </td>
                                  <td className="terminal-text">{(position.marginRatio * 100).toFixed(1)}%</td>
                                  <td>
                                    <Badge 
                                      bg={position.riskLevel === 'low' ? 'success' : position.riskLevel === 'medium' ? 'warning' : position.riskLevel === 'high' ? 'danger' : 'dark'}
                                    >
                                      {position.riskLevel}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">${position.liquidationPrice.toLocaleString()}</td>
                                  <td className="terminal-text">{position.lastUpdate}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Risk Exposure Breakdown */}
                  <Row>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <BarChart className="me-2" />
                            Risk Exposure by Asset
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="terminal-text">
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>BTC/USD</span>
                              <span className="text-warning">$1,250,000 (44%)</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>ETH/USD</span>
                              <span className="text-danger">$890,000 (31%)</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>SOL/USD</span>
                              <span className="text-success">$450,000 (16%)</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center">
                              <span>Other</span>
                              <span className="text-info">$253,392 (9%)</span>
                            </div>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Users className="me-2" />
                            Risk by User Tier
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="terminal-text">
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>High Risk Users</span>
                              <span className="text-danger">12 users</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>Medium Risk Users</span>
                              <span className="text-warning">45 users</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>Low Risk Users</span>
                              <span className="text-success">99 users</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center">
                              <span>Average Leverage</span>
                              <span className="text-info">{riskMetrics.averageLeverage}x</span>
                            </div>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* User Management Tab */}
                <Tab.Pane eventKey="users">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <Users className="me-2" />
                        User Management
                      </h4>
                    </Col>
                  </Row>

                  {/* User Overview Metrics */}
                  <Row className="mb-4">
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{users.length}</div>
                        <div className="metric-label">Total Users</div>
                        <div className="metric-change positive">+3</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{users.filter(u => u.status === 'active').length}</div>
                        <div className="metric-label">Active Users</div>
                        <div className="metric-change positive">+2</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{users.filter(u => u.kycStatus === 'verified').length}</div>
                        <div className="metric-label">Verified Users</div>
                        <div className="metric-change positive">+1</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">${users.reduce((sum, u) => sum + u.totalDeposits, 0).toLocaleString()}</div>
                        <div className="metric-label">Total Deposits</div>
                        <div className="metric-change positive">+12.5%</div>
                      </Card>
                    </Col>
                  </Row>

                  {/* Enhanced User Table */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Users className="me-2" />
                            User Details
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Wallet Address</th>
                                <th>Status</th>
                                <th>KYC Status</th>
                                <th>Trades</th>
                                <th>Volume</th>
                                <th>P&L</th>
                                <th>Win Rate</th>
                                <th>Risk Level</th>
                                <th>Last Active</th>
                                <th>Actions</th>
                              </tr>
                            </thead>
                            <tbody>
                              {users.map(user => (
                                <tr key={user.id}>
                                  <td className="terminal-text">{user.wallet}</td>
                                  <td>
                                    <Badge 
                                      bg={user.status === 'active' ? 'success' : user.status === 'inactive' ? 'secondary' : 'danger'}
                                      className="terminal-glow"
                                    >
                                      {user.status}
                                    </Badge>
                                  </td>
                                  <td>
                                    <Badge 
                                      bg={user.kycStatus === 'verified' ? 'success' : user.kycStatus === 'pending' ? 'warning' : user.kycStatus === 'rejected' ? 'danger' : 'info'}
                                    >
                                      {user.kycStatus}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{user.trades}</td>
                                  <td className="terminal-text">${user.volume.toLocaleString()}</td>
                                  <td className={`terminal-text ${user.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                                    ${user.pnl.toLocaleString()}
                                  </td>
                                  <td className="terminal-text">{user.winRate}%</td>
                                  <td>
                                    <Badge 
                                      bg={user.riskLevel === 'low' ? 'success' : user.riskLevel === 'medium' ? 'warning' : 'danger'}
                                    >
                                      {user.riskLevel}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{user.lastActive}</td>
                                  <td>
                                    <Button size="sm" variant="outline-primary" className="me-1">
                                      <Eye size={14} />
                                    </Button>
                                    <Button size="sm" variant="outline-warning">
                                      <Settings size={14} />
                                    </Button>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* User Trading History */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Activity className="me-2" />
                            Recent Trading Activity
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>User</th>
                                <th>Asset</th>
                                <th>Side</th>
                                <th>Size</th>
                                <th>Price</th>
                                <th>P&L</th>
                                <th>Strategy</th>
                                <th>Exchange</th>
                                <th>Timestamp</th>
                              </tr>
                            </thead>
                            <tbody>
                              {userTrades.map(trade => (
                                <tr key={trade.id}>
                                  <td className="terminal-text">{users.find(u => u.id === trade.userId)?.wallet}</td>
                                  <td className="terminal-text">{trade.asset}</td>
                                  <td>
                                    <Badge bg={trade.side === 'buy' ? 'success' : 'danger'}>
                                      {trade.side}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{trade.size}</td>
                                  <td className="terminal-text">${trade.price.toLocaleString()}</td>
                                  <td className={`terminal-text ${trade.pnl >= 0 ? 'text-success' : 'text-danger'}`}>
                                    ${trade.pnl.toLocaleString()}
                                  </td>
                                  <td className="terminal-text">{trade.strategy}</td>
                                  <td>
                                    <Badge bg="info">{trade.exchange}</Badge>
                                  </td>
                                  <td className="terminal-text">{trade.timestamp}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* User Activity Logs */}
                  <Row>
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <FileText className="me-2" />
                            User Activity Logs
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>User</th>
                                <th>Action</th>
                                <th>Details</th>
                                <th>IP Address</th>
                                <th>User Agent</th>
                                <th>Timestamp</th>
                              </tr>
                            </thead>
                            <tbody>
                              {userActivities.map(activity => (
                                <tr key={activity.id}>
                                  <td className="terminal-text">{users.find(u => u.id === activity.userId)?.wallet}</td>
                                  <td>
                                    <Badge 
                                      bg={activity.action === 'Login' ? 'success' : activity.action === 'Trade Executed' ? 'info' : 'warning'}
                                    >
                                      {activity.action}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{activity.details}</td>
                                  <td className="terminal-text">{activity.ipAddress}</td>
                                  <td className="terminal-text" style={{maxWidth: '200px', overflow: 'hidden', textOverflow: 'ellipsis'}}>
                                    {activity.userAgent}
                                  </td>
                                  <td className="terminal-text">{activity.timestamp}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* System Health Tab */}
                <Tab.Pane eventKey="system">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <Server className="me-2" />
                        System Health
                      </h4>
                    </Col>
                  </Row>

                  {/* System Overview Metrics */}
                  <Row className="mb-4">
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{solanaMetrics.rpcLatency}ms</div>
                        <div className="metric-label">RPC Latency</div>
                        <div className="metric-change positive">-5ms</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{solanaMetrics.transactionSuccessRate}%</div>
                        <div className="metric-label">Tx Success Rate</div>
                        <div className="metric-change positive">+0.2%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{solanaMetrics.averageGasFee.toFixed(5)} SOL</div>
                        <div className="metric-label">Avg Gas Fee</div>
                        <div className="metric-change negative">+0.00001</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{solanaMetrics.networkCongestion}%</div>
                        <div className="metric-label">Network Congestion</div>
                        <div className="metric-change positive">-2.1%</div>
                      </Card>
                    </Col>
                  </Row>

                  {/* Blockchain Status */}
                  <Row className="mb-4">
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Wifi className="me-2" />
                            Blockchain Status
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Solana RPC</span>
                            <div className="d-flex align-items-center">
                              <div className={`status-indicator status-${systemHealth.solanaRpc === 'healthy' ? 'online' : systemHealth.solanaRpc === 'degraded' ? 'warning' : 'offline'} me-2`}></div>
                              <span className={`text-${systemHealth.solanaRpc === 'healthy' ? 'success' : systemHealth.solanaRpc === 'degraded' ? 'warning' : 'danger'}`}>
                                {systemHealth.solanaRpc}
                              </span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Smart Contracts</span>
                            <div className="d-flex align-items-center">
                              <div className={`status-indicator status-${systemHealth.smartContracts === 'healthy' ? 'online' : systemHealth.smartContracts === 'degraded' ? 'warning' : 'offline'} me-2`}></div>
                              <span className={`text-${systemHealth.smartContracts === 'healthy' ? 'success' : systemHealth.smartContracts === 'degraded' ? 'warning' : 'danger'}`}>
                                {systemHealth.smartContracts}
                              </span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Pyth Oracle</span>
                            <div className="d-flex align-items-center">
                              <div className={`status-indicator status-${systemHealth.pythOracle === 'healthy' ? 'online' : systemHealth.pythOracle === 'degraded' ? 'warning' : 'offline'} me-2`}></div>
                              <span className={`text-${systemHealth.pythOracle === 'healthy' ? 'success' : systemHealth.pythOracle === 'degraded' ? 'warning' : 'danger'}`}>
                                {systemHealth.pythOracle}
                              </span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">GPU Status</span>
                            <div className="d-flex align-items-center">
                              <div className={`status-indicator status-${systemHealth.gpuStatus === 'healthy' ? 'online' : systemHealth.gpuStatus === 'degraded' ? 'warning' : 'offline'} me-2`}></div>
                              <span className={`text-${systemHealth.gpuStatus === 'healthy' ? 'success' : systemHealth.gpuStatus === 'degraded' ? 'warning' : 'danger'}`}>
                                {systemHealth.gpuStatus}
                              </span>
                            </div>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Globe className="me-2" />
                            Exchange Status
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          {Object.entries(systemHealth.exchanges).map(([exchange, status]) => (
                            <div key={exchange} className="d-flex justify-content-between align-items-center mb-3">
                              <span className="terminal-text">{exchange}</span>
                              <div className="d-flex align-items-center">
                                <div className={`status-indicator status-${status === 'healthy' ? 'online' : status === 'degraded' ? 'warning' : 'offline'} me-2`}></div>
                                <span className={`text-${status === 'healthy' ? 'success' : status === 'degraded' ? 'warning' : 'danger'}`}>
                                  {status}
                                </span>
                              </div>
                            </div>
                          ))}
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Solana Network Metrics */}
                  <Row className="mb-4">
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Activity className="me-2" />
                            Solana Network Metrics
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">RPC Latency</span>
                            <span className="text-info">{solanaMetrics.rpcLatency}ms</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Transaction Success Rate</span>
                            <span className="text-success">{solanaMetrics.transactionSuccessRate}%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Network Congestion</span>
                            <span className="text-warning">{solanaMetrics.networkCongestion}%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Validator Count</span>
                            <span className="text-info">{solanaMetrics.validatorCount.toLocaleString()}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">Slot Time</span>
                            <span className="text-info">{solanaMetrics.slotTime}ms</span>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <DollarSign className="me-2" />
                            Gas Fee Metrics
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Current Fee</span>
                            <span className="text-info">{gasFeeMetrics.currentFee.toFixed(5)} SOL</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Average Fee</span>
                            <span className="text-success">{gasFeeMetrics.averageFee.toFixed(5)} SOL</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Max Fee</span>
                            <span className="text-warning">{gasFeeMetrics.maxFee.toFixed(5)} SOL</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Priority Fee</span>
                            <span className="text-info">{gasFeeMetrics.priorityFee.toFixed(5)} SOL</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">Trend</span>
                            <span className={`text-${gasFeeMetrics.trend === 'up' ? 'danger' : gasFeeMetrics.trend === 'down' ? 'success' : 'info'}`}>
                              {gasFeeMetrics.trend}
                            </span>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Smart Contract Status */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Terminal className="me-2" />
                            Smart Contract Status
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Contract Name</th>
                                <th>Address</th>
                                <th>Status</th>
                                <th>Last Interaction</th>
                                <th>Total Interactions</th>
                                <th>Gas Used</th>
                                <th>Error Rate</th>
                              </tr>
                            </thead>
                            <tbody>
                              {smartContractMetrics.map((contract, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{contract.contractName}</td>
                                  <td className="terminal-text">{contract.contractAddress}</td>
                                  <td>
                                    <Badge 
                                      bg={contract.status === 'healthy' ? 'success' : contract.status === 'degraded' ? 'warning' : 'danger'}
                                    >
                                      {contract.status}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{contract.lastInteraction}</td>
                                  <td className="terminal-text">{contract.totalInteractions.toLocaleString()}</td>
                                  <td className="terminal-text">{contract.gasUsed.toLocaleString()}</td>
                                  <td className={`terminal-text ${contract.errorRate < 1 ? 'text-success' : contract.errorRate < 5 ? 'text-warning' : 'text-danger'}`}>
                                    {contract.errorRate}%
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* System Performance */}
                  <Row>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Cpu className="me-2" />
                            System Performance
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">CPU Usage</span>
                            <span className="text-success">45%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Memory Usage</span>
                            <span className="text-warning">78%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Disk Usage</span>
                            <span className="text-info">62%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Network I/O</span>
                            <span className="text-success">125 MB/s</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">GPU Usage</span>
                            <span className="text-success">89%</span>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Database className="me-2" />
                            Data Pipeline Health
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Data Pipeline</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">healthy</span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">TimescaleDB</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">healthy</span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Supabase</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">healthy</span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Grafana</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">healthy</span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">WebSocket</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">healthy</span>
                            </div>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* Market Data Tab */}
                <Tab.Pane eventKey="market">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <Globe className="me-2" />
                        Market Data
                      </h4>
                    </Col>
                  </Row>
                  
                  {/* Market Overview Metrics */}
                  <Row className="mb-4">
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">${marketData.openInterest.toLocaleString()}</div>
                        <div className="metric-label">Open Interest</div>
                        <div className="metric-change positive">+5.2%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{(marketData.fundingRate * 100).toFixed(4)}%</div>
                        <div className="metric-label">Funding Rate</div>
                        <div className="metric-change positive">+0.0001%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">${marketData.orderBookDepth.toLocaleString()}</div>
                        <div className="metric-label">Order Book Depth</div>
                        <div className="metric-change positive">+12.3%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{marketData.priceChange24h}%</div>
                        <div className="metric-label">24h Price Change</div>
                        <div className="metric-change positive">+0.8%</div>
                      </Card>
                    </Col>
                  </Row>

                  {/* Price Feeds */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <TrendingUp className="me-2" />
                            Live Price Feeds
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Asset</th>
                                <th>Price</th>
                                <th>24h Change</th>
                                <th>24h Volume</th>
                                <th>Source</th>
                                <th>Last Update</th>
                              </tr>
                            </thead>
                            <tbody>
                              {priceFeeds.map(feed => (
                                <tr key={feed.asset}>
                                  <td className="terminal-text">{feed.asset}</td>
                                  <td className="terminal-text">${feed.price.toLocaleString()}</td>
                                  <td className={`terminal-text ${feed.change24h >= 0 ? 'text-success' : 'text-danger'}`}>
                                    {feed.change24h >= 0 ? '+' : ''}{feed.change24h}%
                                  </td>
                                  <td className="terminal-text">${feed.volume24h.toLocaleString()}</td>
                                  <td>
                                    <Badge bg="info">{feed.source}</Badge>
                                  </td>
                                  <td className="terminal-text">{feed.lastUpdate}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Order Book and Recent Trades */}
                  <Row className="mb-4">
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <BarChart className="me-2" />
                            Order Book - {orderBook.asset}
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="mb-3">
                            <h6 className="text-danger mb-2">Asks (Sell Orders)</h6>
                            {orderBook.asks.map((ask, index) => (
                              <div key={index} className="d-flex justify-content-between align-items-center mb-1">
                                <span className="terminal-text text-danger">${ask.price.toLocaleString()}</span>
                                <span className="terminal-text">{ask.size}</span>
                              </div>
                            ))}
                          </div>
                          <hr />
                          <div className="mb-3">
                            <h6 className="text-success mb-2">Bids (Buy Orders)</h6>
                            {orderBook.bids.map((bid, index) => (
                              <div key={index} className="d-flex justify-content-between align-items-center mb-1">
                                <span className="terminal-text text-success">${bid.price.toLocaleString()}</span>
                                <span className="terminal-text">{bid.size}</span>
                              </div>
                            ))}
                          </div>
                          <small className="text-muted">Last update: {orderBook.lastUpdate}</small>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Activity className="me-2" />
                            Recent Trades
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Asset</th>
                                <th>Side</th>
                                <th>Price</th>
                                <th>Size</th>
                                <th>Time</th>
                              </tr>
                            </thead>
                            <tbody>
                              {recentTrades.map(trade => (
                                <tr key={trade.id}>
                                  <td className="terminal-text">{trade.asset}</td>
                                  <td>
                                    <Badge bg={trade.side === 'buy' ? 'success' : 'danger'}>
                                      {trade.side}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">${trade.price.toLocaleString()}</td>
                                  <td className="terminal-text">{trade.size}</td>
                                  <td className="terminal-text">{trade.timestamp}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Market Depth Analysis */}
                  <Row>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <BarChart className="me-2" />
                            Market Depth Analysis
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="terminal-text">
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>Bid Depth (0-1%)</span>
                              <span className="text-success">$125,000</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>Ask Depth (0-1%)</span>
                              <span className="text-danger">$98,000</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>Spread</span>
                              <span className="text-warning">$5.00 (0.011%)</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center">
                              <span>Market Impact (1%)</span>
                              <span className="text-info">$223,000</span>
                            </div>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <TrendingUp className="me-2" />
                            Funding Rate History
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="terminal-text">
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>Current Rate</span>
                              <span className="text-info">0.0100%</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>8h Average</span>
                              <span className="text-success">0.0085%</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center mb-3">
                              <span>24h Average</span>
                              <span className="text-warning">0.0120%</span>
                            </div>
                            <div className="d-flex justify-content-between align-items-center">
                              <span>Next Funding</span>
                              <span className="text-info">2h 15m</span>
                            </div>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* Compliance Tab */}
                <Tab.Pane eventKey="compliance">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <Lock className="me-2" />
                        Compliance & Security
                      </h4>
                    </Col>
                  </Row>

                  {/* Compliance Overview Metrics */}
                  <Row className="mb-4">
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{complianceReport.kycCompliance}%</div>
                        <div className="metric-label">KYC Compliance</div>
                        <div className="metric-change positive">+0.5%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{complianceReport.amlChecks}%</div>
                        <div className="metric-label">AML Checks</div>
                        <div className="metric-change positive">+0.2%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{complianceReport.auditTrailCompleteness}%</div>
                        <div className="metric-label">Audit Trail</div>
                        <div className="metric-change positive">+0.1%</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{complianceReport.flaggedTransactions}</div>
                        <div className="metric-label">Flagged Transactions</div>
                        <div className="metric-change negative">+3</div>
                      </Card>
                    </Col>
                  </Row>

                  {/* Security Status */}
                  <Row className="mb-4">
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Shield className="me-2" />
                            Security Status
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Rate Limiting</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">Active</span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Authentication</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">Secure</span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">SSL/TLS</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">Enabled</span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Firewall</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">Protected</span>
                            </div>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">Encryption</span>
                            <div className="d-flex align-items-center">
                              <div className="status-indicator status-online me-2"></div>
                              <span className="text-success">AES-256</span>
                            </div>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <FileText className="me-2" />
                            Compliance Report - {complianceReport.period}
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Total Transactions:</span>
                            <span className="text-info">{complianceReport.totalTransactions.toLocaleString()}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Flagged Transactions:</span>
                            <span className="text-warning">{complianceReport.flaggedTransactions}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Suspicious Activities:</span>
                            <span className="text-danger">{complianceReport.suspiciousActivityCount}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">KYC Compliance:</span>
                            <span className="text-success">{complianceReport.kycCompliance}%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">Last Report:</span>
                            <span className="text-info">{complianceReport.lastReportDate}</span>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Suspicious Activity Alerts */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <AlertTriangle className="me-2" />
                            Suspicious Activity Alerts
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>User</th>
                                <th>Activity Type</th>
                                <th>Severity</th>
                                <th>Risk Score</th>
                                <th>Description</th>
                                <th>Status</th>
                                <th>Timestamp</th>
                                <th>Actions</th>
                              </tr>
                            </thead>
                            <tbody>
                              {suspiciousActivities.map(activity => (
                                <tr key={activity.id}>
                                  <td className="terminal-text">{activity.userId}</td>
                                  <td className="terminal-text">{activity.activityType}</td>
                                  <td>
                                    <Badge 
                                      bg={activity.severity === 'critical' ? 'danger' : activity.severity === 'high' ? 'warning' : activity.severity === 'medium' ? 'info' : 'secondary'}
                                    >
                                      {activity.severity}
                                    </Badge>
                                  </td>
                                  <td className={`terminal-text ${activity.riskScore > 80 ? 'text-danger' : activity.riskScore > 60 ? 'text-warning' : 'text-success'}`}>
                                    {activity.riskScore}
                                  </td>
                                  <td className="terminal-text">{activity.description}</td>
                                  <td>
                                    <Badge 
                                      bg={activity.status === 'investigating' ? 'warning' : activity.status === 'resolved' ? 'success' : 'info'}
                                    >
                                      {activity.status}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{activity.timestamp}</td>
                                  <td>
                                    <Button size="sm" variant="outline-primary" className="me-1">
                                      <Eye size={14} />
                                    </Button>
                                    <Button size="sm" variant="outline-success">
                                      <CheckCircle size={14} />
                                    </Button>
                                  </td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Audit Logs */}
                  <Row>
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <FileText className="me-2" />
                            Audit Logs
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Timestamp</th>
                                <th>User</th>
                                <th>Action</th>
                                <th>Resource</th>
                                <th>IP Address</th>
                                <th>Status</th>
                                <th>Details</th>
                              </tr>
                            </thead>
                            <tbody>
                              {auditLogs.map(log => (
                                <tr key={log.id}>
                                  <td className="terminal-text">{log.timestamp}</td>
                                  <td className="terminal-text">{log.userId}</td>
                                  <td className="terminal-text">{log.action}</td>
                                  <td className="terminal-text">{log.resource}</td>
                                  <td className="terminal-text">{log.ipAddress}</td>
                                  <td>
                                    <Badge 
                                      bg={log.status === 'success' ? 'success' : log.status === 'failed' ? 'danger' : 'warning'}
                                    >
                                      {log.status}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{log.details}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* Analytics Tab */}
                <Tab.Pane eventKey="analytics">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <FileText className="me-2" />
                        Analytics & Reporting
                      </h4>
                    </Col>
                  </Row>

                  {/* Trading Performance Overview */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <TrendingUp className="me-2" />
                            Trading Performance Report - {tradingPerformanceReport.period}
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Row>
                            <Col md={3}>
                              <div className="text-center">
                                <h3 className="text-primary">{tradingPerformanceReport.totalTrades.toLocaleString()}</h3>
                                <p className="terminal-text mb-0">Total Trades</p>
                              </div>
                            </Col>
                            <Col md={3}>
                              <div className="text-center">
                                <h3 className="text-success">${tradingPerformanceReport.totalPnL.toLocaleString()}</h3>
                                <p className="terminal-text mb-0">Total P&L</p>
                              </div>
                            </Col>
                            <Col md={3}>
                              <div className="text-center">
                                <h3 className="text-info">{tradingPerformanceReport.winRate}%</h3>
                                <p className="terminal-text mb-0">Win Rate</p>
                              </div>
                            </Col>
                            <Col md={3}>
                              <div className="text-center">
                                <h3 className="text-warning">{tradingPerformanceReport.sharpeRatio}</h3>
                                <p className="terminal-text mb-0">Sharpe Ratio</p>
                              </div>
                            </Col>
                          </Row>
                          <hr />
                          <Row>
                            <Col md={4}>
                              <div className="d-flex justify-content-between align-items-center mb-2">
                                <span className="terminal-text">Max Drawdown:</span>
                                <span className="text-danger">{tradingPerformanceReport.maxDrawdown}%</span>
                              </div>
                              <div className="d-flex justify-content-between align-items-center mb-2">
                                <span className="terminal-text">Profit Factor:</span>
                                <span className="text-success">{tradingPerformanceReport.profitFactor}</span>
                              </div>
                              <div className="d-flex justify-content-between align-items-center">
                                <span className="terminal-text">Calmar Ratio:</span>
                                <span className="text-info">{tradingPerformanceReport.calmarRatio}</span>
                              </div>
                            </Col>
                            <Col md={4}>
                              <div className="d-flex justify-content-between align-items-center mb-2">
                                <span className="terminal-text">Average Trade Size:</span>
                                <span className="text-info">${tradingPerformanceReport.averageTradeSize.toLocaleString()}</span>
                              </div>
                              <div className="d-flex justify-content-between align-items-center mb-2">
                                <span className="terminal-text">Average Holding Time:</span>
                                <span className="text-info">{tradingPerformanceReport.averageHoldingTime}</span>
                              </div>
                              <div className="d-flex justify-content-between align-items-center">
                                <span className="terminal-text">Sortino Ratio:</span>
                                <span className="text-success">{tradingPerformanceReport.sortinoRatio}</span>
                              </div>
                            </Col>
                            <Col md={4}>
                              <div className="d-flex justify-content-between align-items-center mb-2">
                                <span className="terminal-text">Top Strategy:</span>
                                <span className="text-success">{tradingPerformanceReport.topPerformingStrategy}</span>
                              </div>
                              <div className="d-flex justify-content-between align-items-center mb-2">
                                <span className="terminal-text">Top Asset:</span>
                                <span className="text-warning">{tradingPerformanceReport.topPerformingAsset}</span>
                              </div>
                              <div className="d-flex justify-content-between align-items-center">
                                <span className="terminal-text">Total Volume:</span>
                                <span className="text-primary">${tradingPerformanceReport.totalVolume.toLocaleString()}</span>
                              </div>
                            </Col>
                          </Row>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* System Analytics Metrics */}
                  <Row className="mb-4">
                    <Col md={4}>
                      <Card className="metric-card">
                        <div className="metric-value">{tradingMetrics.dataProcessed.toLocaleString()}</div>
                        <div className="metric-label">Data Processed</div>
                        <div className="metric-change positive">+15.3%</div>
                      </Card>
                    </Col>
                    <Col md={4}>
                      <Card className="metric-card">
                        <div className="metric-value">{tradingMetrics.exchangeCoverage}</div>
                        <div className="metric-label">Exchange Coverage</div>
                        <div className="metric-change positive">+1</div>
                      </Card>
                    </Col>
                    <Col md={4}>
                      <Card className="metric-card">
                        <div className="metric-value">{tradingMetrics.systemUptime}%</div>
                        <div className="metric-label">System Uptime</div>
                        <div className="metric-change positive">+0.1%</div>
                      </Card>
                    </Col>
                  </Row>

                  {/* Strategy Performance */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Zap className="me-2" />
                            Strategy Performance Analysis
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Strategy</th>
                                <th>Trades</th>
                                <th>Win Rate</th>
                                <th>Total P&L</th>
                                <th>Sharpe Ratio</th>
                                <th>Max Drawdown</th>
                                <th>Profit Factor</th>
                                <th>Last Updated</th>
                              </tr>
                            </thead>
                            <tbody>
                              {strategyPerformance.map((strategy, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{strategy.strategyName}</td>
                                  <td className="terminal-text">{strategy.totalTrades}</td>
                                  <td className={`terminal-text ${strategy.winRate > 60 ? 'text-success' : strategy.winRate > 50 ? 'text-warning' : 'text-danger'}`}>
                                    {strategy.winRate}%
                                  </td>
                                  <td className={`terminal-text ${strategy.totalPnL >= 0 ? 'text-success' : 'text-danger'}`}>
                                    ${strategy.totalPnL.toLocaleString()}
                                  </td>
                                  <td className={`terminal-text ${strategy.sharpeRatio > 0.5 ? 'text-success' : strategy.sharpeRatio > 0.3 ? 'text-warning' : 'text-danger'}`}>
                                    {strategy.sharpeRatio}
                                  </td>
                                  <td className={`terminal-text ${strategy.maxDrawdown > -3 ? 'text-success' : strategy.maxDrawdown > -5 ? 'text-warning' : 'text-danger'}`}>
                                    {strategy.maxDrawdown}%
                                  </td>
                                  <td className={`terminal-text ${strategy.profitFactor > 1.5 ? 'text-success' : strategy.profitFactor > 1.2 ? 'text-warning' : 'text-danger'}`}>
                                    {strategy.profitFactor}
                                  </td>
                                  <td className="terminal-text">{strategy.lastUpdated}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* QuantDesk Core Tab */}
                <Tab.Pane eventKey="quantdesk">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <Cpu className="me-2" />
                        QuantDesk Core Systems
                      </h4>
                    </Col>
                  </Row>

                  {/* ML Model Performance Overview */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Zap className="me-2" />
                            ML Model Performance
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Model</th>
                                <th>Accuracy</th>
                                <th>Precision</th>
                                <th>Recall</th>
                                <th>F1 Score</th>
                                <th>Status</th>
                                <th>Predictions Today</th>
                                <th>Avg Latency</th>
                                <th>Last Training</th>
                              </tr>
                            </thead>
                            <tbody>
                              {mlModelPerformance.map((model, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{model.modelName}</td>
                                  <td className={`terminal-text ${model.accuracy > 65 ? 'text-success' : model.accuracy > 55 ? 'text-warning' : 'text-danger'}`}>
                                    {model.accuracy}%
                                  </td>
                                  <td className={`terminal-text ${model.precision > 70 ? 'text-success' : model.precision > 60 ? 'text-warning' : 'text-danger'}`}>
                                    {model.precision}%
                                  </td>
                                  <td className={`terminal-text ${model.recall > 65 ? 'text-success' : model.recall > 55 ? 'text-warning' : 'text-danger'}`}>
                                    {model.recall}%
                                  </td>
                                  <td className={`terminal-text ${model.f1Score > 65 ? 'text-success' : model.f1Score > 55 ? 'text-warning' : 'text-danger'}`}>
                                    {model.f1Score}%
                                  </td>
                                  <td>
                                    <Badge 
                                      bg={model.status === 'active' ? 'success' : model.status === 'training' ? 'warning' : model.status === 'idle' ? 'secondary' : 'danger'}
                                    >
                                      {model.status}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{model.predictionsToday.toLocaleString()}</td>
                                  <td className="terminal-text">{model.averageLatency}ms</td>
                                  <td className="terminal-text">{model.lastTraining}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* GPU Monitoring */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Cpu className="me-2" />
                            GPU Monitoring
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>GPU</th>
                                <th>Utilization</th>
                                <th>Memory</th>
                                <th>Temperature</th>
                                <th>Power Draw</th>
                                <th>Status</th>
                                <th>Last Update</th>
                              </tr>
                            </thead>
                            <tbody>
                              {gpuMetrics.map((gpu, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{gpu.gpuName}</td>
                                  <td className={`terminal-text ${gpu.utilization > 80 ? 'text-success' : gpu.utilization > 50 ? 'text-warning' : 'text-info'}`}>
                                    {gpu.utilization}%
                                  </td>
                                  <td className="terminal-text">
                                    {gpu.memoryUsed}GB / {gpu.memoryTotal}GB
                                    <div className="progress mt-1" style={{height: '4px'}}>
                                      <div 
                                        className="progress-bar" 
                                        style={{width: `${(gpu.memoryUsed / gpu.memoryTotal) * 100}%`}}
                                      ></div>
                                    </div>
                                  </td>
                                  <td className={`terminal-text ${gpu.temperature > 80 ? 'text-danger' : gpu.temperature > 70 ? 'text-warning' : 'text-success'}`}>
                                    {gpu.temperature}°C
                                  </td>
                                  <td className="terminal-text">{gpu.powerDraw}W</td>
                                  <td>
                                    <Badge 
                                      bg={gpu.status === 'healthy' ? 'success' : gpu.status === 'overloaded' ? 'warning' : gpu.status === 'idle' ? 'secondary' : 'danger'}
                                    >
                                      {gpu.status}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{gpu.lastUpdate}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Data Pipeline Health */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Database className="me-2" />
                            Data Pipeline Health
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Component</th>
                                <th>Status</th>
                                <th>Latency</th>
                                <th>Throughput</th>
                                <th>Error Rate</th>
                                <th>Uptime</th>
                                <th>Last Update</th>
                              </tr>
                            </thead>
                            <tbody>
                              {dataPipelineHealth.map((component, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{component.component}</td>
                                  <td>
                                    <Badge 
                                      bg={component.status === 'healthy' ? 'success' : component.status === 'degraded' ? 'warning' : 'danger'}
                                    >
                                      {component.status}
                                    </Badge>
                                  </td>
                                  <td className={`terminal-text ${component.latency < 50 ? 'text-success' : component.latency < 100 ? 'text-warning' : 'text-danger'}`}>
                                    {component.latency}ms
                                  </td>
                                  <td className="terminal-text">{component.throughput.toLocaleString()}/s</td>
                                  <td className={`terminal-text ${component.errorRate < 1 ? 'text-success' : component.errorRate < 5 ? 'text-warning' : 'text-danger'}`}>
                                    {component.errorRate}%
                                  </td>
                                  <td className={`terminal-text ${component.uptime > 99 ? 'text-success' : component.uptime > 95 ? 'text-warning' : 'text-danger'}`}>
                                    {component.uptime}%
                                  </td>
                                  <td className="terminal-text">{component.lastUpdate}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Lorentzian Classifier Deep Dive */}
                  <Row className="mb-4">
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Zap className="me-2" />
                            Lorentzian Classifier Metrics
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Total Predictions:</span>
                            <span className="text-primary">{lorentzianClassifierMetrics.totalPredictions.toLocaleString()}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Accuracy:</span>
                            <span className="text-success">{lorentzianClassifierMetrics.accuracy}%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Precision:</span>
                            <span className="text-info">{lorentzianClassifierMetrics.precision}%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Recall:</span>
                            <span className="text-warning">{lorentzianClassifierMetrics.recall}%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">F1 Score:</span>
                            <span className="text-success">{lorentzianClassifierMetrics.f1Score}%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Training Samples:</span>
                            <span className="text-info">{lorentzianClassifierMetrics.trainingSamples.toLocaleString()}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Avg Prediction Time:</span>
                            <span className="text-primary">{lorentzianClassifierMetrics.averagePredictionTime}ms</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">Status:</span>
                            <Badge bg="success">{lorentzianClassifierMetrics.status}</Badge>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Activity className="me-2" />
                            System Performance Metrics
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Total GPU Utilization:</span>
                            <span className="text-success">70%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Total Memory Used:</span>
                            <span className="text-info">40.8GB / 52GB</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Average Temperature:</span>
                            <span className="text-warning">65°C</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Total Power Draw:</span>
                            <span className="text-primary">780W</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Data Processed Today:</span>
                            <span className="text-success">2.8M records</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Model Training Jobs:</span>
                            <span className="text-info">3 active</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">System Health:</span>
                            <Badge bg="success">Excellent</Badge>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* QuantDesk Architecture Overview */}
                  <Row>
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Terminal className="me-2" />
                            QuantDesk Architecture Overview
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Row>
                            <Col md={4}>
                              <h6 className="text-primary mb-3">Data Layer</h6>
                              <div className="terminal-text">
                                <div className="mb-2">• Pyth Oracle Integration</div>
                                <div className="mb-2">• Multi-exchange Data Feeds</div>
                                <div className="mb-2">• Real-time Market Data</div>
                                <div className="mb-2">• Historical Data Storage</div>
                              </div>
                            </Col>
                            <Col md={4}>
                              <h6 className="text-success mb-3">ML Layer</h6>
                              <div className="terminal-text">
                                <div className="mb-2">• Lorentzian Classifier</div>
                                <div className="mb-2">• Lag-based Strategies</div>
                                <div className="mb-2">• Neural Networks</div>
                                <div className="mb-2">• Feature Engineering</div>
                              </div>
                            </Col>
                            <Col md={4}>
                              <h6 className="text-warning mb-3">Execution Layer</h6>
                              <div className="terminal-text">
                                <div className="mb-2">• Solana Smart Contracts</div>
                                <div className="mb-2">• Risk Management</div>
                                <div className="mb-2">• Order Execution</div>
                                <div className="mb-2">• Position Management</div>
                              </div>
                            </Col>
                          </Row>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* Exchange Status Tab */}
                <Tab.Pane eventKey="exchanges">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <Globe className="me-2" />
                        Exchange & DEX Status Monitoring
                      </h4>
                    </Col>
                  </Row>

                  {/* Exchange Overview Metrics */}
                  <Row className="mb-4">
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{exchangeStatus.length}</div>
                        <div className="metric-label">Total Exchanges</div>
                        <div className="metric-change positive">+2</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{exchangeStatus.filter(e => e.status === 'healthy').length}</div>
                        <div className="metric-label">Healthy</div>
                        <div className="metric-change positive">+1</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{exchangeStatus.filter(e => e.type === 'DEX').length}</div>
                        <div className="metric-label">DEXs</div>
                        <div className="metric-change positive">+1</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{exchangeStatus.filter(e => e.type === 'CEX').length}</div>
                        <div className="metric-label">CEXs</div>
                        <div className="metric-change positive">+1</div>
                      </Card>
                    </Col>
                  </Row>

                  {/* Centralized Exchanges (CEX) */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Globe className="me-2" />
                            Centralized Exchanges (CEX)
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Exchange</th>
                                <th>Status</th>
                                <th>Latency</th>
                                <th>Uptime</th>
                                <th>API Calls Today</th>
                                <th>Error Rate</th>
                                <th>Rate Limit</th>
                                <th>24h Volume</th>
                                <th>Last Trade</th>
                              </tr>
                            </thead>
                            <tbody>
                              {exchangeStatus.filter(e => e.type === 'CEX').map((exchange, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{exchange.name}</td>
                                  <td>
                                    <Badge 
                                      bg={exchange.status === 'healthy' ? 'success' : exchange.status === 'degraded' ? 'warning' : 'danger'}
                                    >
                                      {exchange.status}
                                    </Badge>
                                  </td>
                                  <td className={`terminal-text ${exchange.latency < 50 ? 'text-success' : exchange.latency < 100 ? 'text-warning' : 'text-danger'}`}>
                                    {exchange.latency}ms
                                  </td>
                                  <td className={`terminal-text ${exchange.uptime > 99 ? 'text-success' : exchange.uptime > 98 ? 'text-warning' : 'text-danger'}`}>
                                    {exchange.uptime}%
                                  </td>
                                  <td className="terminal-text">{exchange.apiCallsToday.toLocaleString()}</td>
                                  <td className={`terminal-text ${exchange.errorRate < 1 ? 'text-success' : exchange.errorRate < 5 ? 'text-warning' : 'text-danger'}`}>
                                    {exchange.errorRate}%
                                  </td>
                                  <td className="terminal-text">
                                    {exchange.rateLimitRemaining}/{exchange.rateLimitTotal}
                                    <div className="progress mt-1" style={{height: '4px'}}>
                                      <div 
                                        className="progress-bar" 
                                        style={{width: `${(exchange.rateLimitRemaining / exchange.rateLimitTotal) * 100}%`}}
                                      ></div>
                                    </div>
                                  </td>
                                  <td className="terminal-text">${exchange.tradingVolume24h.toLocaleString()}</td>
                                  <td className="terminal-text">{exchange.lastTrade}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Decentralized Exchanges (DEX) */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Zap className="me-2" />
                            Decentralized Exchanges (DEX) - Solana
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>DEX</th>
                                <th>Status</th>
                                <th>Latency</th>
                                <th>Uptime</th>
                                <th>API Calls Today</th>
                                <th>Error Rate</th>
                                <th>Supported Assets</th>
                                <th>24h Volume</th>
                                <th>Last Trade</th>
                              </tr>
                            </thead>
                            <tbody>
                              {exchangeStatus.filter(e => e.type === 'DEX').map((exchange, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{exchange.name}</td>
                                  <td>
                                    <Badge 
                                      bg={exchange.status === 'healthy' ? 'success' : exchange.status === 'degraded' ? 'warning' : 'danger'}
                                    >
                                      {exchange.status}
                                    </Badge>
                                  </td>
                                  <td className={`terminal-text ${exchange.latency < 30 ? 'text-success' : exchange.latency < 50 ? 'text-warning' : 'text-danger'}`}>
                                    {exchange.latency}ms
                                  </td>
                                  <td className={`terminal-text ${exchange.uptime > 99 ? 'text-success' : exchange.uptime > 98 ? 'text-warning' : 'text-danger'}`}>
                                    {exchange.uptime}%
                                  </td>
                                  <td className="terminal-text">{exchange.apiCallsToday.toLocaleString()}</td>
                                  <td className={`terminal-text ${exchange.errorRate < 1 ? 'text-success' : exchange.errorRate < 5 ? 'text-warning' : 'text-danger'}`}>
                                    {exchange.errorRate}%
                                  </td>
                                  <td className="terminal-text">{exchange.supportedAssets}</td>
                                  <td className="terminal-text">${exchange.tradingVolume24h.toLocaleString()}</td>
                                  <td className="terminal-text">{exchange.lastTrade}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Exchange Performance Metrics */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <BarChart className="me-2" />
                            Exchange Performance Metrics
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Exchange</th>
                                <th>Total Volume</th>
                                <th>Total Trades</th>
                                <th>Average Trade Size</th>
                                <th>Top Trading Pairs</th>
                                <th>Latency</th>
                                <th>Error Rate</th>
                                <th>Last Update</th>
                              </tr>
                            </thead>
                            <tbody>
                              {exchangeMetrics.map((metric, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{metric.exchange}</td>
                                  <td className="terminal-text">${metric.totalVolume.toLocaleString()}</td>
                                  <td className="terminal-text">{metric.totalTrades.toLocaleString()}</td>
                                  <td className="terminal-text">${metric.averageTradeSize.toLocaleString()}</td>
                                  <td className="terminal-text">
                                    {metric.topTradingPairs.map((pair, i) => (
                                      <div key={i} className="mb-1">
                                        <span className="text-info">{pair.pair}</span>: ${pair.volume.toLocaleString()}
                                      </div>
                                    ))}
                                  </td>
                                  <td className={`terminal-text ${metric.latency < 50 ? 'text-success' : metric.latency < 100 ? 'text-warning' : 'text-danger'}`}>
                                    {metric.latency}ms
                                  </td>
                                  <td className={`terminal-text ${metric.errorRate < 1 ? 'text-success' : metric.errorRate < 5 ? 'text-warning' : 'text-danger'}`}>
                                    {metric.errorRate}%
                                  </td>
                                  <td className="terminal-text">{metric.lastUpdate}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Exchange Health Summary */}
                  <Row>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Activity className="me-2" />
                            Exchange Health Summary
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Healthy Exchanges:</span>
                            <span className="text-success">{exchangeStatus.filter(e => e.status === 'healthy').length}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Degraded Exchanges:</span>
                            <span className="text-warning">{exchangeStatus.filter(e => e.status === 'degraded').length}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Down Exchanges:</span>
                            <span className="text-danger">{exchangeStatus.filter(e => e.status === 'down').length}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Average Latency:</span>
                            <span className="text-info">{Math.round(exchangeStatus.reduce((sum, e) => sum + e.latency, 0) / exchangeStatus.length)}ms</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Average Uptime:</span>
                            <span className="text-success">{(exchangeStatus.reduce((sum, e) => sum + e.uptime, 0) / exchangeStatus.length).toFixed(1)}%</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">Total API Calls Today:</span>
                            <span className="text-primary">{exchangeStatus.reduce((sum, e) => sum + e.apiCallsToday, 0).toLocaleString()}</span>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <TrendingUp className="me-2" />
                            Top Performing Exchanges
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          {exchangeStatus
                            .sort((a, b) => b.tradingVolume24h - a.tradingVolume24h)
                            .slice(0, 5)
                            .map((exchange, index) => (
                              <div key={index} className="d-flex justify-content-between align-items-center mb-3">
                                <span className="terminal-text">{exchange.name}</span>
                                <div className="d-flex align-items-center">
                                  <span className="text-info me-2">${exchange.tradingVolume24h.toLocaleString()}</span>
                                  <Badge 
                                    bg={exchange.status === 'healthy' ? 'success' : exchange.status === 'degraded' ? 'warning' : 'danger'}
                                    className="me-2"
                                  >
                                    {exchange.status}
                                  </Badge>
                                  <span className="text-muted">{exchange.latency}ms</span>
                                </div>
                              </div>
                            ))}
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>

                {/* Cross-Chain Status Tab */}
                <Tab.Pane eventKey="chains">
                  <Row className="mb-4">
                    <Col>
                      <h4 className="terminal-heading mb-3">
                        <Link className="me-2" />
                        Cross-Chain Status Monitoring
                      </h4>
                    </Col>
                  </Row>

                  {/* Chain Overview Metrics */}
                  <Row className="mb-4">
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{chainStatus.length}</div>
                        <div className="metric-label">Total Chains</div>
                        <div className="metric-change positive">+1</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{chainStatus.filter(c => c.status === 'healthy').length}</div>
                        <div className="metric-label">Healthy</div>
                        <div className="metric-change positive">+1</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{chainStatus.reduce((sum, c) => sum + c.healthyEndpoints, 0)}</div>
                        <div className="metric-label">Healthy RPCs</div>
                        <div className="metric-change positive">+2</div>
                      </Card>
                    </Col>
                    <Col md={3}>
                      <Card className="metric-card">
                        <div className="metric-value">{Math.round(chainStatus.reduce((sum, c) => sum + c.averageLatency, 0) / chainStatus.length)}ms</div>
                        <div className="metric-label">Avg Latency</div>
                        <div className="metric-change negative">-15ms</div>
                      </Card>
                    </Col>
                  </Row>

                  {/* Chain Status Table */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Activity className="me-2" />
                            Blockchain Network Status
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Chain</th>
                                <th>Status</th>
                                <th>Block Height</th>
                                <th>Block Time</th>
                                <th>Gas Price</th>
                                <th>Pending TXs</th>
                                <th>RPC Health</th>
                                <th>Latency</th>
                                <th>Uptime</th>
                                <th>Last Update</th>
                              </tr>
                            </thead>
                            <tbody>
                              {chainStatus.map((chain, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{chain.name}</td>
                                  <td>
                                    <Badge 
                                      bg={chain.status === 'healthy' ? 'success' : chain.status === 'degraded' ? 'warning' : 'danger'}
                                    >
                                      {chain.status}
                                    </Badge>
                                  </td>
                                  <td className="terminal-text">{chain.blockHeight.toLocaleString()}</td>
                                  <td className="terminal-text">{chain.blockTime}s</td>
                                  <td className="terminal-text">
                                    {chain.gasPriceGwei} gwei
                                    <div className="text-muted small">
                                      ${(chain.gasPrice / 1000000000).toFixed(2)}
                                    </div>
                                  </td>
                                  <td className="terminal-text">{chain.pendingTransactions.toLocaleString()}</td>
                                  <td className="terminal-text">
                                    {chain.healthyEndpoints}/{chain.rpcEndpoints}
                                    <div className="progress mt-1" style={{height: '4px'}}>
                                      <div 
                                        className="progress-bar" 
                                        style={{width: `${(chain.healthyEndpoints / chain.rpcEndpoints) * 100}%`}}
                                      ></div>
                                    </div>
                                  </td>
                                  <td className={`terminal-text ${chain.averageLatency < 100 ? 'text-success' : chain.averageLatency < 200 ? 'text-warning' : 'text-danger'}`}>
                                    {chain.averageLatency}ms
                                  </td>
                                  <td className={`terminal-text ${chain.uptime > 99 ? 'text-success' : chain.uptime > 98 ? 'text-warning' : 'text-danger'}`}>
                                    {chain.uptime}%
                                  </td>
                                  <td className="terminal-text">{chain.lastUpdate}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Cross-Chain Performance Metrics */}
                  <Row className="mb-4">
                    <Col>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <BarChart className="me-2" />
                            Cross-Chain Performance Metrics
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <Table className="terminal-table">
                            <thead>
                              <tr>
                                <th>Chain</th>
                                <th>Total Volume</th>
                                <th>Total Transactions</th>
                                <th>Average Gas Price</th>
                                <th>Active Addresses</th>
                                <th>New Addresses</th>
                                <th>Top Tokens</th>
                                <th>Last Update</th>
                              </tr>
                            </thead>
                            <tbody>
                              {crossChainMetrics.map((metric, index) => (
                                <tr key={index}>
                                  <td className="terminal-text">{metric.chain}</td>
                                  <td className="terminal-text">${metric.totalVolume.toLocaleString()}</td>
                                  <td className="terminal-text">{metric.totalTransactions.toLocaleString()}</td>
                                  <td className="terminal-text">
                                    {metric.averageGasPrice} gwei
                                    <div className="text-muted small">
                                      ${(metric.averageGasPrice * 0.000000001).toFixed(6)}
                                    </div>
                                  </td>
                                  <td className="terminal-text">{metric.activeAddresses.toLocaleString()}</td>
                                  <td className="terminal-text">{metric.newAddresses.toLocaleString()}</td>
                                  <td className="terminal-text">
                                    {metric.topTokens.map((token, i) => (
                                      <div key={i} className="mb-1">
                                        <span className="text-info">{token.symbol}</span>: ${token.volume.toLocaleString()} 
                                        <span className="text-muted">(${token.price})</span>
                                      </div>
                                    ))}
                                  </td>
                                  <td className="terminal-text">{metric.lastUpdate}</td>
                                </tr>
                              ))}
                            </tbody>
                          </Table>
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>

                  {/* Chain Health Summary */}
                  <Row>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <Activity className="me-2" />
                            Chain Health Summary
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Healthy Chains:</span>
                            <span className="text-success">{chainStatus.filter(c => c.status === 'healthy').length}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Degraded Chains:</span>
                            <span className="text-warning">{chainStatus.filter(c => c.status === 'degraded').length}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Down Chains:</span>
                            <span className="text-danger">{chainStatus.filter(c => c.status === 'down').length}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Total RPC Endpoints:</span>
                            <span className="text-info">{chainStatus.reduce((sum, c) => sum + c.rpcEndpoints, 0)}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center mb-3">
                            <span className="terminal-text">Healthy RPC Endpoints:</span>
                            <span className="text-success">{chainStatus.reduce((sum, c) => sum + c.healthyEndpoints, 0)}</span>
                          </div>
                          <div className="d-flex justify-content-between align-items-center">
                            <span className="terminal-text">Average Uptime:</span>
                            <span className="text-success">{(chainStatus.reduce((sum, c) => sum + c.uptime, 0) / chainStatus.length).toFixed(1)}%</span>
                          </div>
                        </Card.Body>
                      </Card>
                    </Col>
                    <Col md={6}>
                      <Card className="terminal-card">
                        <Card.Header>
                          <h5 className="terminal-heading mb-0">
                            <TrendingUp className="me-2" />
                            Top Performing Chains
                          </h5>
                        </Card.Header>
                        <Card.Body>
                          {crossChainMetrics
                            .sort((a, b) => b.totalVolume - a.totalVolume)
                            .map((chain, index) => (
                              <div key={index} className="d-flex justify-content-between align-items-center mb-3">
                                <span className="terminal-text">{chain.chain}</span>
                                <div className="d-flex align-items-center">
                                  <span className="text-info me-2">${chain.totalVolume.toLocaleString()}</span>
                                  <span className="text-muted me-2">{chain.totalTransactions.toLocaleString()} TXs</span>
                                  <span className="text-success">{chain.averageGasPrice} gwei</span>
                                </div>
                              </div>
                            ))}
                        </Card.Body>
                      </Card>
                    </Col>
                  </Row>
                </Tab.Pane>
              </Tab.Content>
            </Col>
          </Row>
        </Tab.Container>
      </Container>
    </div>
  );
};

export default AdminDashboard;
