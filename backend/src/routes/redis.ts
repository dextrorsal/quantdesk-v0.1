import { Router, Request, Response } from 'express';
import { redisCacheService } from '../services/redisCache';
import { pingRedis } from '../services/redisClient';
import { Logger } from '../utils/logger';

const router = Router();
const logger = new Logger();

/**
 * GET /api/redis/health
 * Get Redis health status with latency
 */
router.get('/health', async (req: Request, res: Response) => {
  try {
    const healthStatus = await redisCacheService.getHealthStatus();
    
    const statusCode = healthStatus.status === 'healthy' ? 200 : 503;
    
    res.status(statusCode).json({
      success: healthStatus.status === 'healthy',
      status: healthStatus.status,
      latency: healthStatus.latency,
      available: healthStatus.available,
      cache: healthStatus.stats,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error checking Redis health:', error);
    res.status(503).json({
      success: false,
      status: 'unhealthy',
      error: 'Failed to check Redis health',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    });
  }
});

/**
 * GET /api/redis/stats
 * Get Redis cache statistics
 */
router.get('/stats', async (req: Request, res: Response) => {
  try {
    const stats = redisCacheService.getStats();
    const healthStatus = await redisCacheService.getHealthStatus();
    
    res.json({
      success: true,
      data: {
        cache: stats,
        redis: {
          status: healthStatus.status,
          latency: healthStatus.latency,
          available: healthStatus.available
        }
      },
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error fetching Redis stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to fetch Redis stats',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    });
  }
});

/**
 * POST /api/redis/clear
 * Clear Redis cache (admin only - should be protected in production)
 */
router.post('/clear', async (req: Request, res: Response) => {
  try {
    const pattern = (req.body.pattern as string) || '*';
    const cleared = await redisCacheService.clear(pattern);
    
    logger.info(`Redis cache cleared: ${cleared} keys removed (pattern: ${pattern})`);
    
    res.json({
      success: true,
      cleared,
      pattern,
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error clearing Redis cache:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to clear Redis cache',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    });
  }
});

/**
 * POST /api/redis/reset-stats
 * Reset cache statistics
 */
router.post('/reset-stats', async (req: Request, res: Response) => {
  try {
    redisCacheService.resetStats();
    
    res.json({
      success: true,
      message: 'Cache statistics reset',
      timestamp: Date.now()
    });
  } catch (error) {
    logger.error('Error resetting Redis stats:', error);
    res.status(500).json({
      success: false,
      error: 'Failed to reset stats',
      message: error instanceof Error ? error.message : 'Unknown error',
      timestamp: Date.now()
    });
  }
});

export default router;

