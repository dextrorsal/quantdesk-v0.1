import { Request, Response, NextFunction } from 'express';
import { monitoringService } from '../services/monitoringService';

export const contractMonitoringMiddleware = (req: Request, res: Response, next: NextFunction) => {
  const startTime = Date.now();
  const originalSend = res.send;

  res.send = function(data) {
    const duration = Date.now() - startTime;
    const success = res.statusCode < 400;

    // Record contract operation metrics
    monitoringService.recordMetric({
      timestamp: new Date(),
      service: 'contract',
      operation: `${req.method} ${req.path}`,
      duration,
      success,
      error: success ? undefined : data,
      metadata: {
        statusCode: res.statusCode,
        userAgent: req.get('User-Agent'),
        ip: req.ip
      }
    });

    return originalSend.call(this, data);
  };

  next();
};
