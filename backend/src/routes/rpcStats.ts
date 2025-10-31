import express from 'express';
import { SolanaService } from '../services/solana';
import { asyncHandler } from '../middleware/errorHandling';

const router = express.Router();
const solanaService = SolanaService.getInstance();

// Get RPC provider statistics (public for testing)
router.get('/stats', asyncHandler(async (req: any, res) => {
  try {
    const stats = solanaService.getRPCStats();
    
    res.json({
      success: true,
      stats: {
        ...stats,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to fetch RPC statistics',
      code: 'RPC_STATS_ERROR'
    });
  }
}));

// Get RPC health status (public for testing)
router.get('/health', asyncHandler(async (req: any, res) => {
  try {
    const isHealthy = await solanaService.healthCheck();
    const stats = solanaService.getRPCStats();
    
    res.json({
      success: true,
      health: {
        isHealthy,
        healthyProviders: stats.healthyProviders,
        totalProviders: stats.providers.length,
        timestamp: new Date().toISOString()
      }
    });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to check RPC health',
      code: 'RPC_HEALTH_ERROR'
    });
  }
}));

// Get detailed provider information (public for testing)
router.get('/providers', asyncHandler(async (req: any, res) => {
  try {
    const stats = solanaService.getRPCStats();
    
    const providers = stats.providers.map(provider => ({
      name: provider.name,
      isHealthy: provider.isHealthy,
      requestCount: provider.requestCount,
      errorCount: provider.errorCount,
      avgResponseTime: provider.avgResponseTime,
      lastUsed: provider.lastUsed,
      uptime: provider.isHealthy ? '100%' : '0%',
      status: provider.isHealthy ? 'active' : 'unhealthy'
    }));
    
    res.json({
      success: true,
      providers,
      summary: {
        total: stats.providers.length,
        healthy: stats.healthyProviders,
        unhealthy: stats.providers.length - stats.healthyProviders,
        totalRequests: stats.totalRequests
      }
    });
  } catch (error) {
    res.status(500).json({
      error: 'Failed to fetch provider information',
      code: 'PROVIDER_INFO_ERROR'
    });
  }
}));

export default router;
