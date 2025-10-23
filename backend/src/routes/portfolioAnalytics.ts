import { Router, Request, Response } from 'express';
import { portfolioAnalyticsService, PositionAnalytics } from '../services/portfolioAnalyticsService';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

// Mock data generator for testing
function generateMockPositions(): PositionAnalytics[] {
  return [
    {
      symbol: 'BTC-PERP',
      size: 0.5,
      entryPrice: 45000,
      currentPrice: 47000,
      unrealizedPnL: 1000,
      unrealizedPnLPercent: 4.44,
      leverage: 10,
      marginUsed: 2250,
      riskScore: 0.7,
      correlationWithPortfolio: 0.8,
      contributionToRisk: 0.3,
      contributionToReturn: 0.4
    },
    {
      symbol: 'ETH-PERP',
      size: 2.0,
      entryPrice: 3000,
      currentPrice: 3200,
      unrealizedPnL: 400,
      unrealizedPnLPercent: 6.67,
      leverage: 5,
      marginUsed: 1200,
      riskScore: 0.5,
      correlationWithPortfolio: 0.6,
      contributionToRisk: 0.2,
      contributionToReturn: 0.3
    },
    {
      symbol: 'SOL-PERP',
      size: 10.0,
      entryPrice: 180,
      currentPrice: 190,
      unrealizedPnL: 100,
      unrealizedPnLPercent: 5.56,
      leverage: 3,
      marginUsed: 600,
      riskScore: 0.3,
      correlationWithPortfolio: 0.4,
      contributionToRisk: 0.1,
      contributionToReturn: 0.2
    }
  ];
}

function generateMockHistoricalReturns(): number[] {
  // Generate 365 days of returns
  const returns: number[] = [];
  for (let i = 0; i < 365; i++) {
    // Generate realistic daily returns with some volatility
    const baseReturn = 0.0005; // 0.05% daily return
    const volatility = 0.02; // 2% daily volatility
    const randomReturn = baseReturn + (Math.random() - 0.5) * volatility;
    returns.push(randomReturn);
  }
  return returns;
}

function generateMockMarketReturns(): number[] {
  // Generate market returns (e.g., S&P 500 equivalent)
  const returns: number[] = [];
  for (let i = 0; i < 365; i++) {
    const baseReturn = 0.0003; // 0.03% daily return
    const volatility = 0.015; // 1.5% daily volatility
    const randomReturn = baseReturn + (Math.random() - 0.5) * volatility;
    returns.push(randomReturn);
  }
  return returns;
}

/**
 * GET /api/portfolio/analytics
 * Get comprehensive portfolio analytics
 */
router.get('/analytics', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    // Get mock data (in production, fetch from database)
    const positions = generateMockPositions();
    const historicalReturns = generateMockHistoricalReturns();
    const marketReturns = generateMockMarketReturns();
    
    // Calculate portfolio metrics
    const portfolioMetrics = portfolioAnalyticsService.calculatePortfolioMetrics(
      positions,
      historicalReturns,
      marketReturns
    );
    
    // Calculate risk metrics
    const riskMetrics = portfolioAnalyticsService.calculateRiskMetrics(
      positions,
      portfolioMetrics.totalValue
    );
    
    // Generate performance analytics
    const performanceAnalytics = portfolioAnalyticsService.generatePerformanceAnalytics(
      historicalReturns
    );
    
    // Calculate correlation matrix
    const correlationMatrix = portfolioAnalyticsService.calculateCorrelationMatrix(positions);
    
    res.json({
      success: true,
      data: {
        portfolioMetrics,
        riskMetrics,
        performanceAnalytics,
        correlationMatrix,
        positions,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error fetching portfolio analytics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch portfolio analytics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/portfolio/metrics
 * Get basic portfolio metrics
 */
router.get('/metrics', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const positions = generateMockPositions();
    const historicalReturns = generateMockHistoricalReturns();
    const marketReturns = generateMockMarketReturns();
    
    const portfolioMetrics = portfolioAnalyticsService.calculatePortfolioMetrics(
      positions,
      historicalReturns,
      marketReturns
    );
    
    res.json({
      success: true,
      data: portfolioMetrics
    });
    
  } catch (error) {
    logger.error('Error fetching portfolio metrics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch portfolio metrics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/portfolio/risk
 * Get portfolio risk analysis
 */
router.get('/risk', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const positions = generateMockPositions();
    const totalValue = positions.reduce((sum, pos) => sum + pos.size * pos.currentPrice, 0);
    
    const riskMetrics = portfolioAnalyticsService.calculateRiskMetrics(positions, totalValue);
    
    res.json({
      success: true,
      data: riskMetrics
    });
    
  } catch (error) {
    logger.error('Error fetching portfolio risk:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch portfolio risk',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/portfolio/performance
 * Get portfolio performance analytics
 */
router.get('/performance', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const historicalReturns = generateMockHistoricalReturns();
    const performanceAnalytics = portfolioAnalyticsService.generatePerformanceAnalytics(historicalReturns);
    
    res.json({
      success: true,
      data: performanceAnalytics
    });
    
  } catch (error) {
    logger.error('Error fetching portfolio performance:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch portfolio performance',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/portfolio/correlation
 * Get correlation matrix for portfolio positions
 */
router.get('/correlation', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const positions = generateMockPositions();
    const correlationMatrix = portfolioAnalyticsService.calculateCorrelationMatrix(positions);
    
    res.json({
      success: true,
      data: correlationMatrix
    });
    
  } catch (error) {
    logger.error('Error fetching correlation matrix:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch correlation matrix',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/portfolio/positions
 * Get detailed position analytics
 */
router.get('/positions', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const positions = generateMockPositions();
    
    res.json({
      success: true,
      data: positions
    });
    
  } catch (error) {
    logger.error('Error fetching position analytics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch position analytics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/portfolio/stress-test
 * Run stress test scenarios
 */
router.post('/stress-test', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const { scenarios } = req.body;
    
    if (!scenarios || !Array.isArray(scenarios)) {
      return res.status(400).json({
        success: false,
        error: 'Scenarios array is required'
      });
    }
    
    const positions = generateMockPositions();
    const totalValue = positions.reduce((sum, pos) => sum + pos.size * pos.currentPrice, 0);
    
    // Run custom stress test scenarios
    const stressTestResults = scenarios.map((scenario: any) => {
      const { name, marketChange, volatilityChange } = scenario;
      
      // Calculate portfolio impact based on scenario
      const portfolioImpact = marketChange * 0.8 + volatilityChange * 0.2;
      const newPortfolioValue = totalValue * (1 + portfolioImpact);
      const portfolioPnL = newPortfolioValue - totalValue;
      
      return {
        scenario: name,
        portfolioValue: newPortfolioValue,
        portfolioPnL: portfolioPnL,
        portfolioPnLPercent: (portfolioPnL / totalValue) * 100,
        worstPosition: positions.reduce((worst, pos) => 
          pos.unrealizedPnL < worst.unrealizedPnL ? pos : worst, positions[0])?.symbol || 'N/A'
      };
    });
    
    res.json({
      success: true,
      data: {
        stressTestResults,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error running stress test:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to run stress test',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/portfolio/benchmark
 * Get portfolio benchmark comparison
 */
router.get('/benchmark', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const historicalReturns = generateMockHistoricalReturns();
    const marketReturns = generateMockMarketReturns();
    
    const portfolioReturn = historicalReturns.reduce((sum, ret) => sum + ret, 0) / historicalReturns.length;
    const marketReturn = marketReturns.reduce((sum, ret) => sum + ret, 0) / marketReturns.length;
    
    const beta = portfolioAnalyticsService.calculateBeta(historicalReturns, marketReturns);
    const alpha = portfolioAnalyticsService.calculateAlpha(portfolioReturn, 0.02, beta, marketReturn);
    const informationRatio = portfolioAnalyticsService.calculateInformationRatio(historicalReturns, marketReturns);
    
    res.json({
      success: true,
      data: {
        portfolioReturn: portfolioReturn * 100, // Convert to percentage
        marketReturn: marketReturn * 100,
        beta,
        alpha: alpha * 100,
        informationRatio,
        outperformance: (portfolioReturn - marketReturn) * 100,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error fetching benchmark comparison:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch benchmark comparison',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
