/**
 * Monitoring API Routes
 * Provides endpoints for monitoring metrics, alerts, and system status
 */

import { Router, Request, Response } from 'express';
import type { Express } from 'express';
import { MonitoringService } from '../services/MonitoringService';
import { systemLogger, errorLogger } from '../utils/logger';

const router: Express = Router() as any;
const monitoringService = new MonitoringService();

/**
 * GET /api/monitoring/metrics
 * Get current monitoring metrics
 */
router.get('/metrics', async (req: Request, res: Response) => {
  try {
    const { timeRange = '1h', provider, taskType } = req.query;
    
    const metrics = await (monitoringService as any).getMetrics({
      timeRange: timeRange as string,
      provider: provider as string,
      taskType: taskType as string
    });
    
    res.json({
      success: true,
      data: metrics,
      timestamp: new Date()
    });
  } catch (error) {
    errorLogger.aiError(error as Error, 'Monitoring metrics retrieval');
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve monitoring metrics',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/monitoring/alerts
 * Get active alerts
 */
router.get('/alerts', async (req: Request, res: Response) => {
  try {
    const { severity, resolved } = req.query;
    
    const alerts = await (monitoringService as any).getAllAlerts({
      severity: severity as string,
      resolved: resolved === 'true'
    });
    
    res.json({
      success: true,
      data: alerts,
      timestamp: new Date()
    });
  } catch (error) {
    errorLogger.aiError(error as Error, 'Monitoring alerts retrieval');
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve monitoring alerts',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/monitoring/status
 * Get system status
 */
router.get('/status', async (req: Request, res: Response) => {
  try {
    const status = await (monitoringService as any).getSystemStatus();
    
    res.json({
      success: true,
      data: status,
      timestamp: new Date()
    });
  } catch (error) {
    errorLogger.aiError(error as Error, 'System status retrieval');
    res.status(500).json({
      success: false,
      error: 'Failed to retrieve system status',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/monitoring/configure
 * Configure monitoring settings
 */
router.post('/configure', async (req: Request, res: Response) => {
  try {
      const { alertThresholds, metricsInterval, alertCheckInterval } = req.body;
      
      await (monitoringService as any).configureMonitoring({
      alertThresholds,
      metricsInterval,
      alertCheckInterval
    });
    
    systemLogger.startup('Monitoring API', 'Monitoring configuration updated');
    
    res.json({
      success: true,
      message: 'Monitoring configuration updated successfully',
      timestamp: new Date()
    });
  } catch (error) {
    errorLogger.aiError(error as Error, 'Monitoring configuration update');
    res.status(500).json({
      success: false,
      error: 'Failed to update monitoring configuration',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * POST /api/monitoring/alerts/:alertId/resolve
 * Resolve an alert
 */
router.post('/alerts/:alertId/resolve', async (req: Request, res: Response) => {
  try {
    const { alertId } = req.params;
    const { resolution } = req.body;
    
    await (monitoringService as any).resolveAlert(alertId);
    
    systemLogger.startup('Monitoring API', `Alert ${alertId} resolved`);
    
    res.json({
      success: true,
      message: `Alert ${alertId} resolved successfully`,
      timestamp: new Date()
    });
  } catch (error) {
    errorLogger.aiError(error as Error, 'Alert resolution');
    res.status(500).json({
      success: false,
      error: 'Failed to resolve alert',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

/**
 * GET /api/monitoring/health
 * Health check endpoint
 */
router.get('/health', async (req: Request, res: Response) => {
  try {
    const health = await (monitoringService as any).getHealthStatus?.() || {};
    
    res.json({
      success: true,
      data: health,
      timestamp: new Date()
    });
  } catch (error) {
    errorLogger.aiError(error as Error, 'Health check');
    res.status(500).json({
      success: false,
      error: 'Health check failed',
      message: error instanceof Error ? error.message : 'Unknown error'
    });
  }
});

export default router;
