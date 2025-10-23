// Portfolio analytics interfaces
export interface PortfolioMetrics {
  totalValue: number;
  totalPnL: number;
  totalPnLPercent: number;
  dailyPnL: number;
  dailyPnLPercent: number;
  sharpeRatio: number;
  maxDrawdown: number;
  maxDrawdownPercent: number;
  volatility: number;
  var95: number; // Value at Risk (95% confidence)
  var99: number; // Value at Risk (99% confidence)
  beta: number;
  alpha: number;
  informationRatio: number;
  calmarRatio: number;
  sortinoRatio: number;
  treynorRatio: number;
}

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

export interface CorrelationMatrix {
  [symbol: string]: {
    [otherSymbol: string]: number;
  };
}

export interface RiskMetrics {
  portfolioVaR: number;
  portfolioCVaR: number; // Conditional Value at Risk
  concentrationRisk: number;
  liquidityRisk: number;
  leverageRisk: number;
  correlationRisk: number;
  stressTestResults: {
    scenario: string;
    portfolioValue: number;
    portfolioPnL: number;
    worstPosition: string;
  }[];
}

export interface PerformanceAnalytics {
  returns: {
    daily: number[];
    weekly: number[];
    monthly: number[];
    yearly: number[];
  };
  cumulativeReturns: number[];
  rollingSharpe: number[];
  rollingVolatility: number[];
  rollingBeta: number[];
  drawdowns: {
    peak: number;
    trough: number;
    recovery: number;
    duration: number;
  }[];
}

class PortfolioAnalyticsService {
  private static instance: PortfolioAnalyticsService;
  
  private constructor() {}
  
  public static getInstance(): PortfolioAnalyticsService {
    if (!PortfolioAnalyticsService.instance) {
      PortfolioAnalyticsService.instance = new PortfolioAnalyticsService();
    }
    return PortfolioAnalyticsService.instance;
  }
  
  // Calculate Sharpe Ratio
  public calculateSharpeRatio(returns: number[], riskFreeRate: number = 0.02): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
    const volatility = Math.sqrt(variance);
    
    if (volatility === 0) return 0;
    
    return (avgReturn - riskFreeRate / 365) / volatility; // Daily Sharpe ratio
  }
  
  // Calculate Value at Risk (VaR)
  public calculateVaR(returns: number[], confidenceLevel: number = 0.95): number {
    if (returns.length === 0) return 0;
    
    const sortedReturns = [...returns].sort((a, b) => a - b);
    const index = Math.floor((1 - confidenceLevel) * sortedReturns.length);
    
    return Math.abs(sortedReturns[index] || 0);
  }
  
  // Calculate Conditional Value at Risk (CVaR)
  public calculateCVaR(returns: number[], confidenceLevel: number = 0.95): number {
    if (returns.length === 0) return 0;
    
    const varValue = this.calculateVaR(returns, confidenceLevel);
    const tailReturns = returns.filter(ret => ret <= -varValue);
    
    if (tailReturns.length === 0) return varValue;
    
    return Math.abs(tailReturns.reduce((sum, ret) => sum + ret, 0) / tailReturns.length);
  }
  
  // Calculate Maximum Drawdown
  public calculateMaxDrawdown(values: number[]): { maxDrawdown: number; maxDrawdownPercent: number } {
    if (values.length === 0) return { maxDrawdown: 0, maxDrawdownPercent: 0 };
    
    let peak = values[0];
    let maxDrawdown = 0;
    let maxDrawdownPercent = 0;
    
    for (const value of values) {
      if (value > peak) {
        peak = value;
      }
      
      const drawdown = peak - value;
      const drawdownPercent = (drawdown / peak) * 100;
      
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
        maxDrawdownPercent = drawdownPercent;
      }
    }
    
    return { maxDrawdown, maxDrawdownPercent };
  }
  
  // Calculate Volatility (standard deviation of returns)
  public calculateVolatility(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
    
    return Math.sqrt(variance);
  }
  
  // Calculate Beta (correlation with market)
  public calculateBeta(portfolioReturns: number[], marketReturns: number[]): number {
    if (portfolioReturns.length === 0 || marketReturns.length === 0) return 1;
    
    const minLength = Math.min(portfolioReturns.length, marketReturns.length);
    const portfolioSlice = portfolioReturns.slice(0, minLength);
    const marketSlice = marketReturns.slice(0, minLength);
    
    const portfolioAvg = portfolioSlice.reduce((sum, ret) => sum + ret, 0) / minLength;
    const marketAvg = marketSlice.reduce((sum, ret) => sum + ret, 0) / minLength;
    
    let covariance = 0;
    let marketVariance = 0;
    
    for (let i = 0; i < minLength; i++) {
      covariance += (portfolioSlice[i] - portfolioAvg) * (marketSlice[i] - marketAvg);
      marketVariance += Math.pow(marketSlice[i] - marketAvg, 2);
    }
    
    covariance /= minLength;
    marketVariance /= minLength;
    
    return marketVariance === 0 ? 1 : covariance / marketVariance;
  }
  
  // Calculate Alpha (excess return over expected return)
  public calculateAlpha(portfolioReturn: number, riskFreeRate: number, beta: number, marketReturn: number): number {
    const expectedReturn = riskFreeRate + beta * (marketReturn - riskFreeRate);
    return portfolioReturn - expectedReturn;
  }
  
  // Calculate Information Ratio
  public calculateInformationRatio(portfolioReturns: number[], benchmarkReturns: number[]): number {
    if (portfolioReturns.length === 0 || benchmarkReturns.length === 0) return 0;
    
    const minLength = Math.min(portfolioReturns.length, benchmarkReturns.length);
    const portfolioSlice = portfolioReturns.slice(0, minLength);
    const benchmarkSlice = benchmarkReturns.slice(0, minLength);
    
    const excessReturns = portfolioSlice.map((ret, i) => ret - benchmarkSlice[i]);
    const avgExcessReturn = excessReturns.reduce((sum, ret) => sum + ret, 0) / minLength;
    const trackingError = this.calculateVolatility(excessReturns);
    
    return trackingError === 0 ? 0 : avgExcessReturn / trackingError;
  }
  
  // Calculate Calmar Ratio (annual return / max drawdown)
  public calculateCalmarRatio(annualReturn: number, maxDrawdownPercent: number): number {
    return maxDrawdownPercent === 0 ? 0 : annualReturn / maxDrawdownPercent;
  }
  
  // Calculate Sortino Ratio (return / downside deviation)
  public calculateSortinoRatio(returns: number[], riskFreeRate: number = 0.02): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const downsideReturns = returns.filter(ret => ret < riskFreeRate / 365);
    
    if (downsideReturns.length === 0) return 0;
    
    const downsideVariance = downsideReturns.reduce((sum, ret) => sum + Math.pow(ret - riskFreeRate / 365, 2), 0) / downsideReturns.length;
    const downsideDeviation = Math.sqrt(downsideVariance);
    
    return downsideDeviation === 0 ? 0 : (avgReturn - riskFreeRate / 365) / downsideDeviation;
  }
  
  // Calculate Treynor Ratio (excess return / beta)
  public calculateTreynorRatio(portfolioReturn: number, riskFreeRate: number, beta: number): number {
    return beta === 0 ? 0 : (portfolioReturn - riskFreeRate) / beta;
  }
  
  // Calculate correlation matrix
  public calculateCorrelationMatrix(positions: PositionAnalytics[]): CorrelationMatrix {
    const matrix: CorrelationMatrix = {};
    
    for (const position of positions) {
      matrix[position.symbol] = {};
      for (const otherPosition of positions) {
        if (position.symbol === otherPosition.symbol) {
          matrix[position.symbol][otherPosition.symbol] = 1;
        } else {
          // Simplified correlation calculation
          // In a real implementation, you'd use historical price data
          matrix[position.symbol][otherPosition.symbol] = Math.random() * 0.8 - 0.4; // Random correlation between -0.4 and 0.4
        }
      }
    }
    
    return matrix;
  }
  
  // Calculate portfolio risk metrics
  public calculateRiskMetrics(positions: PositionAnalytics[], portfolioValue: number): RiskMetrics {
    const totalLeverage = positions.reduce((sum, pos) => sum + pos.leverage * pos.size, 0) / portfolioValue;
    const maxPositionSize = Math.max(...positions.map(pos => pos.size));
    const concentrationRisk = maxPositionSize / portfolioValue;
    
    // Stress test scenarios
    const stressTestResults = [
      {
        scenario: 'Market Crash (-20%)',
        portfolioValue: portfolioValue * 0.8,
        portfolioPnL: portfolioValue * -0.2,
        worstPosition: positions.reduce((worst, pos) => 
          pos.unrealizedPnL < worst.unrealizedPnL ? pos : worst, positions[0])?.symbol || 'N/A'
      },
      {
        scenario: 'Volatility Spike (+50%)',
        portfolioValue: portfolioValue * 0.9,
        portfolioPnL: portfolioValue * -0.1,
        worstPosition: positions.reduce((worst, pos) => 
          pos.unrealizedPnL < worst.unrealizedPnL ? pos : worst, positions[0])?.symbol || 'N/A'
      },
      {
        scenario: 'Liquidity Crisis',
        portfolioValue: portfolioValue * 0.85,
        portfolioPnL: portfolioValue * -0.15,
        worstPosition: positions.reduce((worst, pos) => 
          pos.unrealizedPnL < worst.unrealizedPnL ? pos : worst, positions[0])?.symbol || 'N/A'
      }
    ];
    
    return {
      portfolioVaR: this.calculateVaR(positions.map(pos => pos.unrealizedPnLPercent / 100)),
      portfolioCVaR: this.calculateCVaR(positions.map(pos => pos.unrealizedPnLPercent / 100)),
      concentrationRisk: concentrationRisk,
      liquidityRisk: Math.min(concentrationRisk * 2, 1),
      leverageRisk: Math.min(totalLeverage / 10, 1),
      correlationRisk: 0.3, // Simplified calculation
      stressTestResults
    };
  }
  
  // Calculate comprehensive portfolio metrics
  public calculatePortfolioMetrics(
    positions: PositionAnalytics[],
    historicalReturns: number[],
    marketReturns: number[] = [],
    riskFreeRate: number = 0.02
  ): PortfolioMetrics {
    const totalValue = positions.reduce((sum, pos) => sum + pos.size * pos.currentPrice, 0);
    const totalPnL = positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0);
    const totalPnLPercent = totalValue === 0 ? 0 : (totalPnL / totalValue) * 100;
    
    // Calculate daily PnL (simplified)
    const dailyPnL = totalPnL * 0.1; // Assume 10% of total PnL is daily
    const dailyPnLPercent = totalValue === 0 ? 0 : (dailyPnL / totalValue) * 100;
    
    const sharpeRatio = this.calculateSharpeRatio(historicalReturns, riskFreeRate);
    const { maxDrawdown, maxDrawdownPercent } = this.calculateMaxDrawdown(
      historicalReturns.map((_, i) => totalValue * (1 + historicalReturns.slice(0, i + 1).reduce((sum, ret) => sum + ret, 0)))
    );
    const volatility = this.calculateVolatility(historicalReturns);
    const var95 = this.calculateVaR(historicalReturns, 0.95) * totalValue;
    const var99 = this.calculateVaR(historicalReturns, 0.99) * totalValue;
    const beta = this.calculateBeta(historicalReturns, marketReturns);
    const alpha = this.calculateAlpha(
      historicalReturns.reduce((sum, ret) => sum + ret, 0) / historicalReturns.length,
      riskFreeRate,
      beta,
      marketReturns.reduce((sum, ret) => sum + ret, 0) / marketReturns.length
    );
    const informationRatio = this.calculateInformationRatio(historicalReturns, marketReturns);
    const calmarRatio = this.calculateCalmarRatio(totalPnLPercent, maxDrawdownPercent);
    const sortinoRatio = this.calculateSortinoRatio(historicalReturns, riskFreeRate);
    const treynorRatio = this.calculateTreynorRatio(
      historicalReturns.reduce((sum, ret) => sum + ret, 0) / historicalReturns.length,
      riskFreeRate,
      beta
    );
    
    return {
      totalValue,
      totalPnL,
      totalPnLPercent,
      dailyPnL,
      dailyPnLPercent,
      sharpeRatio,
      maxDrawdown,
      maxDrawdownPercent,
      volatility,
      var95,
      var99,
      beta,
      alpha,
      informationRatio,
      calmarRatio,
      sortinoRatio,
      treynorRatio
    };
  }
  
  // Generate performance analytics
  public generatePerformanceAnalytics(historicalReturns: number[]): PerformanceAnalytics {
    const returns = {
      daily: historicalReturns,
      weekly: this.aggregateReturns(historicalReturns, 7),
      monthly: this.aggregateReturns(historicalReturns, 30),
      yearly: this.aggregateReturns(historicalReturns, 365)
    };
    
    const cumulativeReturns = historicalReturns.reduce((acc, ret, i) => {
      acc.push(i === 0 ? ret : acc[i - 1] + ret);
      return acc;
    }, [] as number[]);
    
    const rollingSharpe = this.calculateRollingMetrics(historicalReturns, 30, (rets) => this.calculateSharpeRatio(rets));
    const rollingVolatility = this.calculateRollingMetrics(historicalReturns, 30, (rets) => this.calculateVolatility(rets));
    const rollingBeta = this.calculateRollingMetrics(historicalReturns, 30, (rets) => this.calculateBeta(rets, historicalReturns));
    
    const drawdowns = this.calculateDrawdowns(cumulativeReturns);
    
    return {
      returns,
      cumulativeReturns,
      rollingSharpe,
      rollingVolatility,
      rollingBeta,
      drawdowns
    };
  }
  
  // Helper methods
  private aggregateReturns(returns: number[], period: number): number[] {
    const aggregated: number[] = [];
    for (let i = 0; i < returns.length; i += period) {
      const periodReturns = returns.slice(i, i + period);
      aggregated.push(periodReturns.reduce((sum, ret) => sum + ret, 0));
    }
    return aggregated;
  }
  
  private calculateRollingMetrics(returns: number[], window: number, metricFn: (rets: number[]) => number): number[] {
    const metrics: number[] = [];
    for (let i = window; i <= returns.length; i++) {
      const windowReturns = returns.slice(i - window, i);
      metrics.push(metricFn(windowReturns));
    }
    return metrics;
  }
  
  private calculateDrawdowns(cumulativeReturns: number[]): Array<{ peak: number; trough: number; recovery: number; duration: number }> {
    const drawdowns: Array<{ peak: number; trough: number; recovery: number; duration: number }> = [];
    let peak = cumulativeReturns[0];
    let peakIndex = 0;
    
    for (let i = 1; i < cumulativeReturns.length; i++) {
      if (cumulativeReturns[i] > peak) {
        peak = cumulativeReturns[i];
        peakIndex = i;
      } else if (cumulativeReturns[i] < peak) {
        // We're in a drawdown
        const trough = Math.min(...cumulativeReturns.slice(peakIndex, i + 1));
        const recovery = cumulativeReturns[i] - trough;
        const duration = i - peakIndex;
        
        drawdowns.push({
          peak,
          trough,
          recovery,
          duration
        });
      }
    }
    
    return drawdowns;
  }
}

export const portfolioAnalyticsService = PortfolioAnalyticsService.getInstance();
export default portfolioAnalyticsService;
