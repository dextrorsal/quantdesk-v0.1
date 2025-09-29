import { grafanaMetricsService } from './grafanaMetrics';
import { Logger } from '../utils/logger';

const logger = new Logger();

export class MetricsCollector {
  private static instance: MetricsCollector;
  private tradingInterval?: NodeJS.Timeout;
  private systemInterval?: NodeJS.Timeout;
  private isRunning = false;

  private constructor() {}

  public static getInstance(): MetricsCollector {
    if (!MetricsCollector.instance) {
      MetricsCollector.instance = new MetricsCollector();
    }
    return MetricsCollector.instance;
  }

  /**
   * Start collecting metrics periodically
   */
  public start(): void {
    if (this.isRunning) {
      logger.warn('⚠️ Metrics collector is already running');
      return;
    }

    logger.info('📊 Starting metrics collection service...');

    // Collect trading metrics every 30 seconds
    this.tradingInterval = setInterval(async () => {
      try {
        const metrics = await grafanaMetricsService.collectTradingMetrics();
        await grafanaMetricsService.sendMetricsToGrafana(metrics);
      } catch (error) {
        logger.error('❌ Error collecting trading metrics:', error);
      }
    }, 30000); // 30 seconds

    // Collect system metrics every 10 seconds
    this.systemInterval = setInterval(async () => {
      try {
        const metrics = await grafanaMetricsService.collectSystemMetrics();
        await grafanaMetricsService.sendMetricsToGrafana(metrics);
      } catch (error) {
        logger.error('❌ Error collecting system metrics:', error);
      }
    }, 10000); // 10 seconds

    this.isRunning = true;
    logger.info('✅ Metrics collection service started');
  }

  /**
   * Stop collecting metrics
   */
  public stop(): void {
    if (!this.isRunning) {
      logger.warn('⚠️ Metrics collector is not running');
      return;
    }

    logger.info('🛑 Stopping metrics collection service...');

    if (this.tradingInterval) {
      clearInterval(this.tradingInterval);
      this.tradingInterval = undefined;
    }

    if (this.systemInterval) {
      clearInterval(this.systemInterval);
      this.systemInterval = undefined;
    }

    this.isRunning = false;
    logger.info('✅ Metrics collection service stopped');
  }

  /**
   * Get current status
   */
  public getStatus(): { isRunning: boolean; tradingInterval: boolean; systemInterval: boolean } {
    return {
      isRunning: this.isRunning,
      tradingInterval: !!this.tradingInterval,
      systemInterval: !!this.systemInterval
    };
  }
}

export const metricsCollector = MetricsCollector.getInstance();
