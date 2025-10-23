import { Router, Request, Response } from 'express';
import { 
  advancedRiskManagementService, 
  RiskLimits, 
  PositionRisk
} from '../services/advancedRiskManagementService';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

// Mock data generator for testing
function generateMockPositions(): PositionRisk[] {
  return [
    {
      symbol: 'BTC-PERP',
      size: 0.5,
      leverage: 10,
      varContribution: 0.02,
      correlationRisk: 0.8,
      liquidityRisk: 0.3,
      concentrationRisk: 0.4,
      individualRiskScore: 75
    },
    {
      symbol: 'ETH-PERP',
      size: 2.0,
      leverage: 5,
      varContribution: 0.015,
      correlationRisk: 0.6,
      liquidityRisk: 0.2,
      concentrationRisk: 0.3,
      individualRiskScore: 60
    },
    {
      symbol: 'SOL-PERP',
      size: 10.0,
      leverage: 3,
      varContribution: 0.01,
      correlationRisk: 0.4,
      liquidityRisk: 0.1,
      concentrationRisk: 0.2,
      individualRiskScore: 45
    }
  ];
}

function generateMockHistoricalReturns(): number[] {
  const returns: number[] = [];
  for (let i = 0; i < 365; i++) {
    const baseReturn = 0.0005;
    const volatility = 0.02;
    const randomReturn = baseReturn + (Math.random() - 0.5) * volatility;
    returns.push(randomReturn);
  }
  return returns;
}

/**
 * GET /api/risk/metrics
 * Get comprehensive risk metrics
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
    const portfolioValue = 100000; // Mock portfolio value
    
    const riskMetrics = advancedRiskManagementService.calculateRiskMetrics(
      positions,
      portfolioValue,
      historicalReturns
    );
    
    res.json({
      success: true,
      data: {
        riskMetrics,
        portfolioValue,
        positions,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error fetching risk metrics:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch risk metrics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/risk/alerts
 * Get user's risk alerts
 */
router.get('/alerts', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const alerts = advancedRiskManagementService.getUserRiskAlerts(userId);
    
    res.json({
      success: true,
      data: {
        alerts,
        activeAlerts: alerts.filter(a => !a.resolved),
        resolvedAlerts: alerts.filter(a => a.resolved),
        totalAlerts: alerts.length,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error fetching risk alerts:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch risk alerts',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/risk/alerts/:alertId/acknowledge
 * Acknowledge a risk alert
 */
router.post('/alerts/:alertId/acknowledge', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    const { alertId } = req.params;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const acknowledged = advancedRiskManagementService.acknowledgeAlert(userId, alertId);
    
    if (acknowledged) {
      res.json({
        success: true,
        message: 'Alert acknowledged successfully'
      });
    } else {
      res.status(404).json({
        success: false,
        error: 'Alert not found'
      });
    }
    
  } catch (error) {
    logger.error('Error acknowledging alert:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to acknowledge alert',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/risk/alerts/:alertId/resolve
 * Resolve a risk alert
 */
router.post('/alerts/:alertId/resolve', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    const { alertId } = req.params;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const resolved = advancedRiskManagementService.resolveAlert(userId, alertId);
    
    if (resolved) {
      res.json({
        success: true,
        message: 'Alert resolved successfully'
      });
    } else {
      res.status(404).json({
        success: false,
        error: 'Alert not found'
      });
    }
    
  } catch (error) {
    logger.error('Error resolving alert:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to resolve alert',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/risk/limits
 * Get user's risk limits
 */
router.get('/limits', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const limits = advancedRiskManagementService.getUserRiskLimits(userId);
    
    res.json({
      success: true,
      data: limits
    });
    
  } catch (error) {
    logger.error('Error fetching risk limits:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch risk limits',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * PUT /api/risk/limits
 * Update user's risk limits
 */
router.put('/limits', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    const limits: RiskLimits = req.body;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    // Validate limits
    if (limits.maxPortfolioVaR < 0 || limits.maxPortfolioVaR > 1) {
      return res.status(400).json({
        success: false,
        error: 'Invalid maxPortfolioVaR (must be between 0 and 1)'
      });
    }
    
    if (limits.maxLeverage < 1 || limits.maxLeverage > 100) {
      return res.status(400).json({
        success: false,
        error: 'Invalid maxLeverage (must be between 1 and 100)'
      });
    }
    
    advancedRiskManagementService.setUserRiskLimits(userId, limits);
    
    res.json({
      success: true,
      message: 'Risk limits updated successfully',
      data: limits
    });
    
  } catch (error) {
    logger.error('Error updating risk limits:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to update risk limits',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/risk/stress-test
 * Run stress test scenarios
 */
router.post('/stress-test', async (req: Request, res: Response) => {
  try {
    const userId = (req as any).user?.id;
    const { scenarioId } = req.body;
    
    if (!userId) {
      return res.status(401).json({
        success: false,
        error: 'Authentication required'
      });
    }
    
    const positions = generateMockPositions();
    const portfolioValue = 100000;
    
    const stressTestResults = advancedRiskManagementService.runStressTest(
      positions,
      portfolioValue,
      scenarioId
    );
    
    res.json({
      success: true,
      data: {
        stressTestResults,
        portfolioValue,
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
 * GET /api/risk/scenarios
 * Get available risk scenarios
 */
router.get('/scenarios', async (req: Request, res: Response) => {
  try {
    const scenarios = advancedRiskManagementService.getRiskScenarios();
    
    res.json({
      success: true,
      data: scenarios
    });
    
  } catch (error) {
    logger.error('Error fetching risk scenarios:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch risk scenarios',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/risk/report
 * Generate comprehensive risk report
 */
router.get('/report', async (req: Request, res: Response) => {
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
    const portfolioValue = 100000;
    
    const riskMetrics = advancedRiskManagementService.calculateRiskMetrics(
      positions,
      portfolioValue,
      historicalReturns
    );
    
    const riskReport = advancedRiskManagementService.generateRiskReport(
      userId,
      riskMetrics,
      positions,
      portfolioValue
    );
    
    res.json({
      success: true,
      data: riskReport
    });
    
  } catch (error) {
    logger.error('Error generating risk report:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to generate risk report',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/risk/check
 * Check risk limits and generate alerts
 */
router.post('/check', async (req: Request, res: Response) => {
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
    const portfolioValue = 100000;
    
    const riskMetrics = advancedRiskManagementService.calculateRiskMetrics(
      positions,
      portfolioValue,
      historicalReturns
    );
    
    const alerts = advancedRiskManagementService.checkRiskLimits(
      userId,
      riskMetrics,
      positions,
      portfolioValue
    );
    
    res.json({
      success: true,
      data: {
        riskMetrics,
        alerts,
        alertsGenerated: alerts.length,
        timestamp: Date.now()
      }
    });
    
  } catch (error) {
    logger.error('Error checking risk limits:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to check risk limits',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/risk/monitor
 * Real-time risk monitoring endpoint
 */
router.get('/monitor', async (req: Request, res: Response) => {
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
    const portfolioValue = 100000;
    
    const riskMetrics = advancedRiskManagementService.calculateRiskMetrics(
      positions,
      portfolioValue,
      historicalReturns
    );
    
    const alerts = advancedRiskManagementService.checkRiskLimits(
      userId,
      riskMetrics,
      positions,
      portfolioValue
    );
    
    res.json({
      success: true,
      data: {
        riskMetrics,
        activeAlerts: alerts.filter(a => !a.resolved),
        riskLevel: riskMetrics.overallRiskScore >= 80 ? 'CRITICAL' : 
                  riskMetrics.overallRiskScore >= 60 ? 'HIGH' : 
                  riskMetrics.overallRiskScore >= 40 ? 'MEDIUM' : 'LOW',
        timestamp: Date.now(),
        monitoringActive: true
      }
    });
    
  } catch (error) {
    logger.error('Error monitoring risk:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to monitor risk',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
