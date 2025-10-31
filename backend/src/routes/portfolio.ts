import express from 'express';
import { PortfolioCalculationService } from '../services/portfolioCalculationService';
import { Logger } from '../utils/logger';
import { asyncHandler } from '../middleware/errorHandling';
import { AuthenticatedRequest } from '../middleware/auth';

const router = express.Router();
const logger = new Logger();
const portfolioService = PortfolioCalculationService.getInstance();

/**
 * GET /api/portfolio
 * Get user's complete portfolio data
 */
router.get('/', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const portfolio = await portfolioService.calculatePortfolio(req.userId);
    
    if (!portfolio) {
      return res.status(404).json({
        success: false,
        error: 'Portfolio not found',
        code: 'PORTFOLIO_NOT_FOUND'
      });
    }

    res.json({
      success: true,
      data: portfolio
    });

  } catch (error) {
    logger.error('Error fetching portfolio:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch portfolio',
      code: 'FETCH_ERROR'
    });
  }
}));

/**
 * GET /api/portfolio/summary
 * Get portfolio summary only
 */
router.get('/summary', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const portfolio = await portfolioService.calculatePortfolio(req.userId);
    
    if (!portfolio) {
      return res.status(404).json({
        success: false,
        error: 'Portfolio not found',
        code: 'PORTFOLIO_NOT_FOUND'
      });
    }

    const summary = {
      userId: portfolio.userId,
      totalValue: portfolio.totalValue,
      totalUnrealizedPnl: portfolio.totalUnrealizedPnl,
      totalRealizedPnl: portfolio.totalRealizedPnl,
      marginRatio: portfolio.marginRatio,
      healthFactor: portfolio.healthFactor,
      totalCollateral: portfolio.totalCollateral,
      usedMargin: portfolio.usedMargin,
      availableMargin: portfolio.availableMargin,
      timestamp: portfolio.timestamp
    };

    res.json({
      success: true,
      data: summary
    });

  } catch (error) {
    logger.error('Error fetching portfolio summary:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch portfolio summary',
      code: 'FETCH_ERROR'
    });
  }
}));

/**
 * GET /api/portfolio/positions
 * Get portfolio positions only
 */
router.get('/positions', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const portfolio = await portfolioService.calculatePortfolio(req.userId);
    
    if (!portfolio) {
      return res.status(404).json({
        success: false,
        error: 'Portfolio not found',
        code: 'PORTFOLIO_NOT_FOUND'
      });
    }

    res.json({
      success: true,
      data: portfolio.positions
    });

  } catch (error) {
    logger.error('Error fetching portfolio positions:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch portfolio positions',
      code: 'FETCH_ERROR'
    });
  }
}));

/**
 * POST /api/portfolio/refresh
 * Force refresh portfolio data
 */
router.post('/refresh', asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const portfolio = await portfolioService.calculatePortfolio(req.userId, true); // Force refresh
    
    if (!portfolio) {
      return res.status(404).json({
        success: false,
        error: 'Portfolio not found',
        code: 'PORTFOLIO_NOT_FOUND'
      });
    }

    res.json({
      success: true,
      data: portfolio,
      refreshed: true
    });

  } catch (error) {
    logger.error('Error refreshing portfolio:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to refresh portfolio',
      code: 'REFRESH_ERROR'
    });
  }
}));

export default router;
