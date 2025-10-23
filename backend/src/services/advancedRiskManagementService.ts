import { Logger } from '../utils/logger';

const logger = new Logger();

// Risk management interfaces
export interface RiskLimits {
  maxPortfolioVaR: number;           // Maximum portfolio VaR
  maxPositionSize: number;           // Maximum position size as % of portfolio
  maxLeverage: number;              // Maximum leverage allowed
  maxDrawdown: number;              // Maximum drawdown threshold
  maxCorrelation: number;           // Maximum correlation between positions
  maxConcentration: number;        // Maximum concentration in single asset
  minLiquidity: number;            // Minimum liquidity requirement
  maxDailyLoss: number;            // Maximum daily loss limit
}

export interface RiskAlert {
  id: string;
  userId: string;
  type: RiskAlertType;
  severity: RiskSeverity;
  message: string;
  currentValue: number;
  threshold: number;
  timestamp: number;
  acknowledged: boolean;
  resolved: boolean;
}

export enum RiskAlertType {
  PORTFOLIO_VAR_BREACH = 'portfolio_var_breach',
  POSITION_SIZE_EXCEEDED = 'position_size_exceeded',
  LEVERAGE_EXCEEDED = 'leverage_exceeded',
  DRAWDOWN_THRESHOLD = 'drawdown_threshold',
  CORRELATION_HIGH = 'correlation_high',
  CONCENTRATION_HIGH = 'concentration_high',
  LIQUIDITY_LOW = 'liquidity_low',
  DAILY_LOSS_LIMIT = 'daily_loss_limit',
  VOLATILITY_SPIKE = 'volatility_spike',
  MARKET_STRESS = 'market_stress'
}

export enum RiskSeverity {
  LOW = 'low',
  MEDIUM = 'medium',
  HIGH = 'high',
  CRITICAL = 'critical'
}

export interface RiskMetrics {
  portfolioVaR: number;
  portfolioCVaR: number;
  maxDrawdown: number;
  currentDrawdown: number;
  leverageRatio: number;
  concentrationRisk: number;
  correlationRisk: number;
  liquidityRisk: number;
  volatilityRisk: number;
  stressTestScore: number;
  overallRiskScore: number; // 0-100 scale
}

export interface PositionRisk {
  symbol: string;
  size: number;
  leverage: number;
  varContribution: number;
  correlationRisk: number;
  liquidityRisk: number;
  concentrationRisk: number;
  individualRiskScore: number;
}

export interface RiskScenario {
  id: string;
  name: string;
  description: string;
  marketChange: number;      // % change in market
  volatilityChange: number;  // % change in volatility
  correlationChange: number; // % change in correlations
  liquidityChange: number;  // % change in liquidity
  expectedImpact: number;   // Expected portfolio impact %
}

class AdvancedRiskManagementService {
  private static instance: AdvancedRiskManagementService;
  private riskAlerts: Map<string, RiskAlert[]> = new Map();
  private riskLimits: Map<string, RiskLimits> = new Map();
  private riskScenarios: RiskScenario[] = [];
  
  private constructor() {
    this.initializeDefaultRiskLimits();
    this.initializeRiskScenarios();
  }
  
  public static getInstance(): AdvancedRiskManagementService {
    if (!AdvancedRiskManagementService.instance) {
      AdvancedRiskManagementService.instance = new AdvancedRiskManagementService();
    }
    return AdvancedRiskManagementService.instance;
  }
  
  // Initialize default risk limits
  private initializeDefaultRiskLimits(): void {
    const defaultLimits: RiskLimits = {
      maxPortfolioVaR: 0.05,        // 5% portfolio VaR
      maxPositionSize: 0.20,        // 20% max position size
      maxLeverage: 10,              // 10x max leverage
      maxDrawdown: 0.15,            // 15% max drawdown
      maxCorrelation: 0.8,          // 80% max correlation
      maxConcentration: 0.30,       // 30% max concentration
      minLiquidity: 0.10,           // 10% min liquidity
      maxDailyLoss: 0.10            // 10% max daily loss
    };
    
    this.riskLimits.set('default', defaultLimits);
  }
  
  // Initialize risk scenarios
  private initializeRiskScenarios(): void {
    this.riskScenarios = [
      {
        id: 'crypto_crash',
        name: 'Crypto Market Crash',
        description: 'Severe cryptocurrency market downturn',
        marketChange: -0.4,
        volatilityChange: 0.8,
        correlationChange: 0.3,
        liquidityChange: -0.5,
        expectedImpact: -0.35
      },
      {
        id: 'fed_rate_hike',
        name: 'Federal Reserve Rate Hike',
        description: 'Aggressive interest rate increases',
        marketChange: -0.2,
        volatilityChange: 0.4,
        correlationChange: 0.2,
        liquidityChange: -0.2,
        expectedImpact: -0.18
      },
      {
        id: 'regulatory_crackdown',
        name: 'Regulatory Crackdown',
        description: 'Strict cryptocurrency regulations',
        marketChange: -0.3,
        volatilityChange: 0.6,
        correlationChange: 0.4,
        liquidityChange: -0.4,
        expectedImpact: -0.28
      },
      {
        id: 'liquidity_crisis',
        name: 'Liquidity Crisis',
        description: 'Severe liquidity shortage',
        marketChange: -0.15,
        volatilityChange: 0.5,
        correlationChange: 0.1,
        liquidityChange: -0.7,
        expectedImpact: -0.22
      },
      {
        id: 'volatility_spike',
        name: 'Volatility Spike',
        description: 'Extreme market volatility',
        marketChange: -0.1,
        volatilityChange: 1.0,
        correlationChange: 0.2,
        liquidityChange: -0.3,
        expectedImpact: -0.15
      }
    ];
  }
  
  // Calculate comprehensive risk metrics
  public calculateRiskMetrics(
    positions: PositionRisk[],
    portfolioValue: number,
    historicalReturns: number[]
  ): RiskMetrics {
    const portfolioVaR = this.calculatePortfolioVaR(positions, portfolioValue);
    const portfolioCVaR = this.calculatePortfolioCVaR(positions, portfolioValue);
    const maxDrawdown = this.calculateMaxDrawdown(historicalReturns);
    const currentDrawdown = this.calculateCurrentDrawdown(historicalReturns);
    const leverageRatio = this.calculateLeverageRatio(positions, portfolioValue);
    const concentrationRisk = this.calculateConcentrationRisk(positions, portfolioValue);
    const correlationRisk = this.calculateCorrelationRisk(positions);
    const liquidityRisk = this.calculateLiquidityRisk(positions);
    const volatilityRisk = this.calculateVolatilityRisk(historicalReturns);
    const stressTestScore = this.calculateStressTestScore(positions, portfolioValue);
    
    // Calculate overall risk score (0-100)
    const overallRiskScore = this.calculateOverallRiskScore({
      portfolioVaR,
      portfolioCVaR,
      maxDrawdown,
      currentDrawdown,
      leverageRatio,
      concentrationRisk,
      correlationRisk,
      liquidityRisk,
      volatilityRisk,
      stressTestScore
    });
    
    return {
      portfolioVaR,
      portfolioCVaR,
      maxDrawdown,
      currentDrawdown,
      leverageRatio,
      concentrationRisk,
      correlationRisk,
      liquidityRisk,
      volatilityRisk,
      stressTestScore,
      overallRiskScore
    };
  }
  
  // Calculate portfolio VaR
  private calculatePortfolioVaR(positions: PositionRisk[], portfolioValue: number): number {
    const totalVaR = positions.reduce((sum, pos) => sum + pos.varContribution, 0);
    return totalVaR / portfolioValue;
  }
  
  // Calculate portfolio CVaR
  private calculatePortfolioCVaR(positions: PositionRisk[], portfolioValue: number): number {
    // Simplified CVaR calculation
    const portfolioVaR = this.calculatePortfolioVaR(positions, portfolioValue);
    return portfolioVaR * 1.3; // CVaR is typically 30% higher than VaR
  }
  
  // Calculate maximum drawdown
  private calculateMaxDrawdown(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    let peak = 0;
    let maxDrawdown = 0;
    let cumulativeReturn = 0;
    
    for (const ret of returns) {
      cumulativeReturn += ret;
      if (cumulativeReturn > peak) {
        peak = cumulativeReturn;
      }
      const drawdown = peak - cumulativeReturn;
      if (drawdown > maxDrawdown) {
        maxDrawdown = drawdown;
      }
    }
    
    return maxDrawdown;
  }
  
  // Calculate current drawdown
  private calculateCurrentDrawdown(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const cumulativeReturn = returns.reduce((sum, ret) => sum + ret, 0);
    const recentReturns = returns.slice(-30); // Last 30 days
    const recentPeak = Math.max(...recentReturns.map((_, i) => 
      recentReturns.slice(0, i + 1).reduce((sum, ret) => sum + ret, 0)
    ));
    
    return Math.max(0, recentPeak - cumulativeReturn);
  }
  
  // Calculate leverage ratio
  private calculateLeverageRatio(positions: PositionRisk[], portfolioValue: number): number {
    const totalExposure = positions.reduce((sum, pos) => sum + pos.size * pos.leverage, 0);
    return totalExposure / portfolioValue;
  }
  
  // Calculate concentration risk
  private calculateConcentrationRisk(positions: PositionRisk[], portfolioValue: number): number {
    if (positions.length === 0) return 0;
    
    const positionValues = positions.map(pos => pos.size);
    const maxPosition = Math.max(...positionValues);
    return maxPosition / portfolioValue;
  }
  
  // Calculate correlation risk
  private calculateCorrelationRisk(positions: PositionRisk[]): number {
    if (positions.length <= 1) return 0;
    
    const avgCorrelation = positions.reduce((sum, pos) => sum + pos.correlationRisk, 0) / positions.length;
    return avgCorrelation;
  }
  
  // Calculate liquidity risk
  private calculateLiquidityRisk(positions: PositionRisk[]): number {
    if (positions.length === 0) return 0;
    
    const avgLiquidityRisk = positions.reduce((sum, pos) => sum + pos.liquidityRisk, 0) / positions.length;
    return avgLiquidityRisk;
  }
  
  // Calculate volatility risk
  private calculateVolatilityRisk(returns: number[]): number {
    if (returns.length === 0) return 0;
    
    const avgReturn = returns.reduce((sum, ret) => sum + ret, 0) / returns.length;
    const variance = returns.reduce((sum, ret) => sum + Math.pow(ret - avgReturn, 2), 0) / returns.length;
    return Math.sqrt(variance);
  }
  
  // Calculate stress test score
  private calculateStressTestScore(positions: PositionRisk[], portfolioValue: number): number {
    let totalStressImpact = 0;
    
    for (const scenario of this.riskScenarios) {
      const scenarioImpact = this.calculateScenarioImpact(positions, portfolioValue, scenario);
      totalStressImpact += Math.abs(scenarioImpact);
    }
    
    const avgStressImpact = totalStressImpact / this.riskScenarios.length;
    return Math.min(1, avgStressImpact); // Normalize to 0-1
  }
  
  // Calculate scenario impact
  private calculateScenarioImpact(
    positions: PositionRisk[],
    portfolioValue: number,
    scenario: RiskScenario
  ): number {
    let totalImpact = 0;
    
    for (const position of positions) {
      // Simplified impact calculation
      const positionValue = position.size;
      const marketImpact = positionValue * scenario.marketChange;
      const volatilityImpact = positionValue * scenario.volatilityChange * 0.1;
      const correlationImpact = positionValue * scenario.correlationChange * 0.05;
      
      totalImpact += marketImpact + volatilityImpact + correlationImpact;
    }
    
    return totalImpact / portfolioValue;
  }
  
  // Calculate overall risk score
  private calculateOverallRiskScore(metrics: Partial<RiskMetrics>): number {
    const weights = {
      portfolioVaR: 0.25,
      maxDrawdown: 0.20,
      leverageRatio: 0.15,
      concentrationRisk: 0.15,
      correlationRisk: 0.10,
      liquidityRisk: 0.10,
      volatilityRisk: 0.05
    };
    
    let weightedScore = 0;
    let totalWeight = 0;
    
    for (const [metric, weight] of Object.entries(weights)) {
      const value = metrics[metric as keyof RiskMetrics];
      if (value !== undefined) {
        // Normalize values to 0-100 scale
        const normalizedValue = Math.min(100, Math.max(0, value * 100));
        weightedScore += normalizedValue * weight;
        totalWeight += weight;
      }
    }
    
    return totalWeight > 0 ? weightedScore / totalWeight : 0;
  }
  
  // Check risk limits and generate alerts
  public checkRiskLimits(
    userId: string,
    riskMetrics: RiskMetrics,
    positions: PositionRisk[],
    portfolioValue: number
  ): RiskAlert[] {
    const limits = this.riskLimits.get(userId) || this.riskLimits.get('default')!;
    const alerts: RiskAlert[] = [];
    
    // Check portfolio VaR
    if (riskMetrics.portfolioVaR > limits.maxPortfolioVaR) {
      alerts.push(this.createAlert(
        userId,
        RiskAlertType.PORTFOLIO_VAR_BREACH,
        RiskSeverity.HIGH,
        `Portfolio VaR exceeded: ${(riskMetrics.portfolioVaR * 100).toFixed(2)}% > ${(limits.maxPortfolioVaR * 100).toFixed(2)}%`,
        riskMetrics.portfolioVaR,
        limits.maxPortfolioVaR
      ));
    }
    
    // Check leverage
    if (riskMetrics.leverageRatio > limits.maxLeverage) {
      alerts.push(this.createAlert(
        userId,
        RiskAlertType.LEVERAGE_EXCEEDED,
        RiskSeverity.CRITICAL,
        `Leverage exceeded: ${riskMetrics.leverageRatio.toFixed(2)}x > ${limits.maxLeverage}x`,
        riskMetrics.leverageRatio,
        limits.maxLeverage
      ));
    }
    
    // Check drawdown
    if (riskMetrics.currentDrawdown > limits.maxDrawdown) {
      alerts.push(this.createAlert(
        userId,
        RiskAlertType.DRAWDOWN_THRESHOLD,
        RiskSeverity.HIGH,
        `Drawdown threshold exceeded: ${(riskMetrics.currentDrawdown * 100).toFixed(2)}% > ${(limits.maxDrawdown * 100).toFixed(2)}%`,
        riskMetrics.currentDrawdown,
        limits.maxDrawdown
      ));
    }
    
    // Check concentration
    if (riskMetrics.concentrationRisk > limits.maxConcentration) {
      alerts.push(this.createAlert(
        userId,
        RiskAlertType.CONCENTRATION_HIGH,
        RiskSeverity.MEDIUM,
        `Concentration risk high: ${(riskMetrics.concentrationRisk * 100).toFixed(2)}% > ${(limits.maxConcentration * 100).toFixed(2)}%`,
        riskMetrics.concentrationRisk,
        limits.maxConcentration
      ));
    }
    
    // Check correlation
    if (riskMetrics.correlationRisk > limits.maxCorrelation) {
      alerts.push(this.createAlert(
        userId,
        RiskAlertType.CORRELATION_HIGH,
        RiskSeverity.MEDIUM,
        `Correlation risk high: ${(riskMetrics.correlationRisk * 100).toFixed(2)}% > ${(limits.maxCorrelation * 100).toFixed(2)}%`,
        riskMetrics.correlationRisk,
        limits.maxCorrelation
      ));
    }
    
    // Store alerts
    if (!this.riskAlerts.has(userId)) {
      this.riskAlerts.set(userId, []);
    }
    this.riskAlerts.get(userId)!.push(...alerts);
    
    return alerts;
  }
  
  // Create risk alert
  private createAlert(
    userId: string,
    type: RiskAlertType,
    severity: RiskSeverity,
    message: string,
    currentValue: number,
    threshold: number
  ): RiskAlert {
    return {
      id: `alert_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
      userId,
      type,
      severity,
      message,
      currentValue,
      threshold,
      timestamp: Date.now(),
      acknowledged: false,
      resolved: false
    };
  }
  
  // Get user's risk alerts
  public getUserRiskAlerts(userId: string): RiskAlert[] {
    return this.riskAlerts.get(userId) || [];
  }
  
  // Acknowledge risk alert
  public acknowledgeAlert(userId: string, alertId: string): boolean {
    const alerts = this.riskAlerts.get(userId);
    if (!alerts) return false;
    
    const alert = alerts.find(a => a.id === alertId);
    if (!alert) return false;
    
    alert.acknowledged = true;
    return true;
  }
  
  // Resolve risk alert
  public resolveAlert(userId: string, alertId: string): boolean {
    const alerts = this.riskAlerts.get(userId);
    if (!alerts) return false;
    
    const alert = alerts.find(a => a.id === alertId);
    if (!alert) return false;
    
    alert.resolved = true;
    return true;
  }
  
  // Set user risk limits
  public setUserRiskLimits(userId: string, limits: RiskLimits): void {
    this.riskLimits.set(userId, limits);
    logger.info(`Risk limits updated for user ${userId}`);
  }
  
  // Get user risk limits
  public getUserRiskLimits(userId: string): RiskLimits {
    return this.riskLimits.get(userId) || this.riskLimits.get('default')!;
  }
  
  // Run stress test
  public runStressTest(
    positions: PositionRisk[],
    portfolioValue: number,
    scenarioId?: string
  ): any[] {
    const scenarios = scenarioId 
      ? this.riskScenarios.filter(s => s.id === scenarioId)
      : this.riskScenarios;
    
    return scenarios.map(scenario => {
      const impact = this.calculateScenarioImpact(positions, portfolioValue, scenario);
      return {
        scenario: scenario.name,
        scenarioId: scenario.id,
        marketChange: scenario.marketChange * 100,
        volatilityChange: scenario.volatilityChange * 100,
        expectedImpact: scenario.expectedImpact * 100,
        calculatedImpact: impact * 100,
        portfolioValueAfter: portfolioValue * (1 + impact),
        portfolioPnL: portfolioValue * impact,
        severity: Math.abs(impact) > 0.2 ? 'HIGH' : Math.abs(impact) > 0.1 ? 'MEDIUM' : 'LOW'
      };
    });
  }
  
  // Get risk scenarios
  public getRiskScenarios(): RiskScenario[] {
    return this.riskScenarios;
  }
  
  // Generate risk report
  public generateRiskReport(
    userId: string,
    riskMetrics: RiskMetrics,
    positions: PositionRisk[],
    portfolioValue: number
  ): any {
    const limits = this.getUserRiskLimits(userId);
    const alerts = this.getUserRiskAlerts(userId);
    const stressTestResults = this.runStressTest(positions, portfolioValue);
    
    return {
      userId,
      timestamp: Date.now(),
      portfolioValue,
      riskMetrics,
      riskLimits: limits,
      activeAlerts: alerts.filter(a => !a.resolved),
      resolvedAlerts: alerts.filter(a => a.resolved),
      stressTestResults,
      riskScore: riskMetrics.overallRiskScore,
      riskLevel: this.getRiskLevel(riskMetrics.overallRiskScore),
      recommendations: this.generateRecommendations(riskMetrics, limits)
    };
  }
  
  // Get risk level based on score
  private getRiskLevel(score: number): string {
    if (score >= 80) return 'CRITICAL';
    if (score >= 60) return 'HIGH';
    if (score >= 40) return 'MEDIUM';
    if (score >= 20) return 'LOW';
    return 'VERY_LOW';
  }
  
  // Generate risk recommendations
  private generateRecommendations(metrics: RiskMetrics, limits: RiskLimits): string[] {
    const recommendations: string[] = [];
    
    if (metrics.portfolioVaR > limits.maxPortfolioVaR) {
      recommendations.push('Consider reducing position sizes to lower portfolio VaR');
    }
    
    if (metrics.leverageRatio > limits.maxLeverage) {
      recommendations.push('Reduce leverage to stay within risk limits');
    }
    
    if (metrics.concentrationRisk > limits.maxConcentration) {
      recommendations.push('Diversify portfolio to reduce concentration risk');
    }
    
    if (metrics.correlationRisk > limits.maxCorrelation) {
      recommendations.push('Consider adding uncorrelated assets to reduce correlation risk');
    }
    
    if (metrics.overallRiskScore > 70) {
      recommendations.push('Overall risk is high - consider reducing exposure');
    }
    
    return recommendations;
  }
}

export const advancedRiskManagementService = AdvancedRiskManagementService.getInstance();
export default advancedRiskManagementService;
