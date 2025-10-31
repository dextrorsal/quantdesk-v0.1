import axios from 'axios';

// Portfolio Analytics Service
// Connects frontend to backend portfolio analytics APIs

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:3002';

export interface PositionAnalytics {
  symbol: string;
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  unrealizedPnLPercent: number;
  leverage: number;
  marginUsed: number;
  riskScore: number;
  correlationWithPortfolio: number;
  contributionToRisk: number;
  contributionToReturn: number;
}

export interface PortfolioAnalytics {
  totalValue: number;
  totalPnL: number;
  totalPnLPercent: number;
  totalMarginUsed: number;
  availableMargin: number;
  portfolioRiskScore: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  positions: PositionAnalytics[];
  historicalReturns: number[];
  marketReturns: number[];
  correlationWithMarket: number;
  beta: number;
  alpha: number;
  volatility: number;
  var95: number;
  cvar95: number;
  lastUpdated: string;
}

export interface RiskMetrics {
  portfolioRiskScore: number;
  var95: number;
  cvar95: number;
  maxDrawdown: number;
  volatility: number;
  sharpeRatio: number;
  beta: number;
  alpha: number;
}

export interface PerformanceMetrics {
  totalReturn: number;
  annualizedReturn: number;
  volatility: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  profitFactor: number;
  calmarRatio: number;
}

class PortfolioAnalyticsService {
  private baseURL: string;

  constructor() {
    this.baseURL = `${API_BASE_URL}/api/portfolio`;
  }

  /**
   * Get comprehensive portfolio analytics
   */
  async getPortfolioAnalytics(): Promise<PortfolioAnalytics> {
    try {
      const response = await axios.get(`${this.baseURL}/analytics`);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching portfolio analytics:', error);
      throw new Error('Failed to fetch portfolio analytics');
    }
  }

  /**
   * Get portfolio risk metrics
   */
  async getRiskMetrics(): Promise<RiskMetrics> {
    try {
      const response = await axios.get(`${this.baseURL}/risk-metrics`);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching risk metrics:', error);
      throw new Error('Failed to fetch risk metrics');
    }
  }

  /**
   * Get portfolio performance metrics
   */
  async getPerformanceMetrics(): Promise<PerformanceMetrics> {
    try {
      const response = await axios.get(`${this.baseURL}/performance-metrics`);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching performance metrics:', error);
      throw new Error('Failed to fetch performance metrics');
    }
  }

  /**
   * Get portfolio positions
   */
  async getPositions(): Promise<PositionAnalytics[]> {
    try {
      const response = await axios.get(`${this.baseURL}/positions`);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching positions:', error);
      throw new Error('Failed to fetch positions');
    }
  }

  /**
   * Get portfolio historical returns
   */
  async getHistoricalReturns(): Promise<number[]> {
    try {
      const response = await axios.get(`${this.baseURL}/historical-returns`);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching historical returns:', error);
      throw new Error('Failed to fetch historical returns');
    }
  }

  /**
   * Get portfolio correlation analysis
   */
  async getCorrelationAnalysis(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/correlation-analysis`);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching correlation analysis:', error);
      throw new Error('Failed to fetch correlation analysis');
    }
  }

  /**
   * Get portfolio stress test results
   */
  async getStressTestResults(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/stress-test`);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching stress test results:', error);
      throw new Error('Failed to fetch stress test results');
    }
  }

  /**
   * Get portfolio optimization suggestions
   */
  async getOptimizationSuggestions(): Promise<any> {
    try {
      const response = await axios.get(`${this.baseURL}/optimization-suggestions`);
      return response.data.data;
    } catch (error) {
      console.error('Error fetching optimization suggestions:', error);
      throw new Error('Failed to fetch optimization suggestions');
    }
  }
}

export const portfolioAnalyticsService = new PortfolioAnalyticsService();
