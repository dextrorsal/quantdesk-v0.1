/**
 * Core types for QuantDesk Protocol
 */

export interface Market {
  symbol: string
  name: string
  baseAsset: string
  quoteAsset: string
  price: number
  change24h: number
  changePercent24h: number
  volume24h: number
  marketCap: number
  high24h: number
  low24h: number
  category: 'Large Cap' | 'Mid Cap' | 'Small Cap'
}

export interface Order {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  type: 'market' | 'limit' | 'stop' | 'stop-limit' | 'take-profit' | 'trailing-stop' | 'iceberg' | 'twap'
  amount: number
  price?: number
  stopPrice?: number
  trailingDistance?: number
  timeInForce: 'GTC' | 'IOC' | 'FOK' | 'GTD'
  status: 'pending' | 'open' | 'filled' | 'cancelled' | 'rejected'
  createdAt: string
  updatedAt: string
}

export interface Position {
  id: string
  symbol: string
  side: 'long' | 'short'
  size: number
  entryPrice: number
  currentPrice: number
  pnl: number
  pnlPercent: number
  margin: number
  leverage: number
  liquidationPrice: number
  createdAt: string
}

export interface PortfolioSummary {
  totalValue: number
  totalPnl: number
  totalPnlPercent: number
  availableBalance: number
  usedMargin: number
  marginRatio: number
  positions: Position[]
  riskScore: number
}

export interface RiskMetrics {
  portfolioVaR: number
  portfolioCVaR: number
  maxDrawdown: number
  leverageRatio: number
  concentrationRisk: number
  correlationRisk: number
  liquidityRisk: number
  volatilityRisk: number
  overallRiskScore: number
}

export interface CollateralAccount {
  id: string
  owner: string
  assets: CollateralAsset[]
  totalValue: number
  availableMargin: number
  usedMargin: number
  createdAt: string
  updatedAt: string
}

export interface CollateralAsset {
  symbol: string
  amount: number
  value: number
  weight: number
  isActive: boolean
}

export interface LiquidityAuction {
  id: string
  symbol: string
  size: number
  side: 'buy' | 'sell'
  minPrice: number
  maxPrice: number
  startTime: string
  endTime: string
  status: 'pending' | 'active' | 'closed' | 'cancelled'
  bids: LiquidityBid[]
  winningBid?: LiquidityBid
}

export interface LiquidityBid {
  id: string
  auctionId: string
  bidder: string
  price: number
  size: number
  timestamp: string
  isWinning: boolean
}

export interface ApiResponse<T> {
  success: boolean
  data?: T
  error?: string
  message?: string
  timestamp: string
}

export interface PaginatedResponse<T> extends ApiResponse<T[]> {
  pagination: {
    page: number
    limit: number
    total: number
    totalPages: number
  }
}
