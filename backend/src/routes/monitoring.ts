import { Router } from 'express';
import { monitoringService } from '../services/monitoringService';

const router = Router();

// Get performance metrics
router.get('/metrics', async (req, res) => {
  try {
    const summary = await monitoringService.getPerformanceSummary();
    const recentMetrics = monitoringService.getRecentMetrics(undefined, 50);
    
    res.json({
      success: true,
      data: {
        summary,
        recentMetrics
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

// Get service-specific metrics
router.get('/metrics/:service', async (req, res) => {
  try {
    const { service } = req.params;
    const limit = parseInt(req.query.limit as string) || 100;
    
    const metrics = monitoringService.getRecentMetrics(service, limit);
    const successRate = monitoringService.calculateSuccessRate(service);
    const averageLatency = monitoringService.calculateAverageLatency(service);
    
    res.json({
      success: true,
      data: {
        service,
        metrics,
        successRate,
        averageLatency
      }
    });
  } catch (error) {
    res.status(500).json({
      success: false,
      error: error.message
    });
  }
});

export default router;
