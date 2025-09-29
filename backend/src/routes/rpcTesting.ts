import express from 'express';
import { RPCTester } from '../../scripts/test-rpc-performance';
import { asyncHandler } from '../middleware/errorHandler';

const router = express.Router();

// Run comprehensive RPC tests
router.post('/test', asyncHandler(async (req: any, res) => {
  try {
    const tester = new RPCTester();
    const results = await tester.runAllTests();
    
    res.json({
      success: true,
      message: 'RPC tests completed successfully',
      results: {
        timestamp: new Date().toISOString(),
        summary: {
          totalProviders: results.loadBalancerResults.length,
          successfulRequests: results.loadBalancerResults.filter(r => r.success).length,
          rateLimitedRequests: results.rateLimitResults.rateLimited,
          avgResponseTime: results.speedResults.getCurrentSlot?.average || 0
        }
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'RPC tests failed',
      message: error.message
    });
  }
}));

// Run stress test
router.post('/stress-test', asyncHandler(async (req: any, res) => {
  try {
    const { duration = 30, rps = 10 } = req.body;
    
    const tester = new RPCTester();
    const results = await tester.runStressTest(duration, rps);
    
    res.json({
      success: true,
      message: `Stress test completed: ${duration}s duration, ${rps} req/s`,
      results: {
        timestamp: new Date().toISOString(),
        duration: results.duration,
        totalRequests: results.totalRequests,
        successfulRequests: results.successfulRequests,
        rateLimitedRequests: results.rateLimitedRequests,
        successRate: results.successRate,
        actualRPS: results.actualRPS
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Stress test failed',
      message: error.message
    });
  }
}));

// Get current RPC performance metrics
router.get('/metrics', asyncHandler(async (req: any, res) => {
  try {
    const { SolanaService } = require('../services/solana');
    const solanaService = SolanaService.getInstance();
    const stats = solanaService.getRPCStats();
    
    // Calculate performance metrics
    const healthyProviders = stats.providers.filter(p => p.isHealthy);
    const avgResponseTime = healthyProviders.length > 0 
      ? healthyProviders.reduce((sum, p) => sum + p.avgResponseTime, 0) / healthyProviders.length 
      : 0;
    
    const totalErrors = stats.providers.reduce((sum, p) => sum + p.errorCount, 0);
    const totalRequests = stats.providers.reduce((sum, p) => sum + p.requestCount, 0);
    const successRate = totalRequests > 0 ? ((totalRequests - totalErrors) / totalRequests * 100) : 100;
    
    res.json({
      success: true,
      metrics: {
        timestamp: new Date().toISOString(),
        loadBalancer: {
          totalProviders: stats.providers.length,
          healthyProviders: stats.healthyProviders,
          unhealthyProviders: stats.providers.length - stats.healthyProviders
        },
        performance: {
          avgResponseTime: Math.round(avgResponseTime * 10) / 10,
          totalRequests: totalRequests,
          totalErrors: totalErrors,
          successRate: Math.round(successRate * 10) / 10
        },
        providers: stats.providers.map(provider => ({
          name: provider.name,
          isHealthy: provider.isHealthy,
          requestCount: provider.requestCount,
          errorCount: provider.errorCount,
          avgResponseTime: Math.round(provider.avgResponseTime * 10) / 10,
          successRate: provider.requestCount > 0 
            ? Math.round(((provider.requestCount - provider.errorCount) / provider.requestCount * 100) * 10) / 10
            : 100,
          lastUsed: provider.lastUsed ? new Date(provider.lastUsed).toISOString() : null
        }))
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: 'Failed to get RPC metrics',
      message: error.message
    });
  }
}));

export default router;
