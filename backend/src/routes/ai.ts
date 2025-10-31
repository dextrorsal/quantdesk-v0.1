import express from 'express';
import { aiService } from '../services/aiService';
import { authMiddleware, AuthenticatedRequest } from '../middleware/auth';
import { asyncHandler } from '../middleware/errorHandling';
import { Logger } from '../utils/logger';

const router = express.Router();
const logger = new Logger();

// AI query endpoint (authenticated users only)
router.post('/query', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { query, context } = req.body;
    
    if (!query || typeof query !== 'string') {
      return res.status(400).json({
        success: false,
        error: 'Query is required and must be a string'
      });
    }

    const result = await aiService.queryAI(query, context);
    res.json({ success: true, data: result });
  } catch (error) {
    logger.error('AI query error:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'AI service error'
    });
  }
}));

// Market analysis endpoint
router.get('/analysis/:symbol', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { symbol } = req.params;
    
    if (!symbol) {
      return res.status(400).json({
        success: false,
        error: 'Symbol is required'
      });
    }

    const analysis = await aiService.getMarketAnalysis(symbol);
    res.json({ success: true, data: analysis });
  } catch (error) {
    logger.error('Market analysis error:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Analysis failed'
    });
  }
}));

// Whale activity endpoint
router.get('/whales', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const whales = await aiService.getWhaleActivity();
    res.json({ success: true, data: whales });
  } catch (error) {
    logger.error('Whale activity error:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Whale data unavailable'
    });
  }
}));

// Trading insights endpoint
router.post('/insights', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { marketData } = req.body;
    
    if (!marketData) {
      return res.status(400).json({
        success: false,
        error: 'Market data is required'
      });
    }

    const insights = await aiService.getTradingInsights(marketData);
    res.json({ success: true, data: insights });
  } catch (error) {
    logger.error('Trading insights error:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Insights unavailable'
    });
  }
}));

// Risk assessment endpoint
router.post('/risk-assessment', authMiddleware, asyncHandler(async (req: AuthenticatedRequest, res) => {
  try {
    const { positionData } = req.body;
    
    if (!positionData) {
      return res.status(400).json({
        success: false,
        error: 'Position data is required'
      });
    }

    const assessment = await aiService.getRiskAssessment(positionData);
    res.json({ success: true, data: assessment });
  } catch (error) {
    logger.error('Risk assessment error:', error);
    res.status(500).json({ 
      success: false, 
      error: error instanceof Error ? error.message : 'Risk assessment failed'
    });
  }
}));

// AI service health check
router.get('/health', asyncHandler(async (req, res) => {
  try {
    const isHealthy = await aiService.healthCheck();
    const status = await aiService.getServiceStatus();
    
    res.json({
      success: true,
      healthy: isHealthy,
      status: status.status || 'unknown',
      timestamp: new Date().toISOString()
    });
  } catch (error) {
    logger.error('AI health check error:', error);
    res.status(500).json({
      success: false,
      healthy: false,
      error: error instanceof Error ? error.message : 'Health check failed'
    });
  }
}));

// AI service status endpoint
router.get('/status', asyncHandler(async (req, res) => {
  try {
    const status = await aiService.getServiceStatus();
    res.json({ success: true, data: status });
  } catch (error) {
    logger.error('AI status error:', error);
    res.status(500).json({
      success: false,
      error: error instanceof Error ? error.message : 'Status unavailable'
    });
  }
}));

export default router;
