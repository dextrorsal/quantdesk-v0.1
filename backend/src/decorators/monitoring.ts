import { Logger } from '../utils/logger';

const logger = new Logger();

/**
 * Order Placement Monitoring Decorator
 * Tracks order placement metrics and performance
 */
export function monitorOrderPlacement(target: any, propertyName: string, descriptor: PropertyDescriptor) {
  const method = descriptor.value;

  descriptor.value = async function (...args: any[]) {
    const startTime = Date.now();
    const orderInput = args[0]; // First argument should be PlaceOrderInput
    
    try {
      logger.info(`📊 Order placement started: ${orderInput?.symbol} ${orderInput?.side} ${orderInput?.size}`);
      
      // Call the original method
      const result = await method.apply(this, args);
      
      const duration = Date.now() - startTime;
      logger.info(`✅ Order placement completed in ${duration}ms: ${result?.orderId}`);
      
      // Track metrics (could be sent to analytics service)
      if (process.env.NODE_ENV === 'production') {
        // In production, you might want to send metrics to a monitoring service
        // await metricsService.trackOrderPlacement({
        //   symbol: orderInput?.symbol,
        //   side: orderInput?.side,
        //   size: orderInput?.size,
        //   duration,
        //   success: true,
        //   orderId: result?.orderId
        // });
      }
      
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      logger.error(`❌ Order placement failed in ${duration}ms:`, error);
      
      // Track error metrics
      if (process.env.NODE_ENV === 'production') {
        // await metricsService.trackOrderPlacement({
        //   symbol: orderInput?.symbol,
        //   side: orderInput?.side,
        //   size: orderInput?.size,
        //   duration,
        //   success: false,
        //   error: error.message
        // });
      }
      
      throw error;
    }
  };

  return descriptor;
}

/**
 * Performance Monitoring Decorator (generic)
 */
export function monitorPerformance(target: any, propertyName: string, descriptor: PropertyDescriptor) {
  const method = descriptor.value;

  descriptor.value = async function (...args: any[]) {
    const startTime = Date.now();
    
    try {
      const result = await method.apply(this, args);
      const duration = Date.now() - startTime;
      
      if (duration > 1000) { // Log slow operations (>1s)
        logger.warn(`⚠️ Slow operation: ${target.constructor.name}.${propertyName} took ${duration}ms`);
      }
      
      return result;
    } catch (error) {
      const duration = Date.now() - startTime;
      logger.error(`❌ Operation failed: ${target.constructor.name}.${propertyName} failed in ${duration}ms:`, error);
      throw error;
    }
  };

  return descriptor;
}
