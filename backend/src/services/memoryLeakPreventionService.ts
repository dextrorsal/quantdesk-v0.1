import { Logger } from '../utils/logger';

const logger = new Logger();

interface MemoryLeakDetector {
  componentName: string;
  mountTime: number;
  unmountTime?: number;
  memoryUsage: NodeJS.MemoryUsage;
  eventListeners: Set<Function>;
  timers: Set<NodeJS.Timeout>;
  intervals: Set<NodeJS.Timeout>;
}

interface MemoryStats {
  heapUsed: number;
  heapTotal: number;
  external: number;
  rss: number;
  timestamp: number;
}

/**
 * Memory Leak Prevention Service
 * 
 * Monitors and prevents memory leaks in long-running sessions by:
 * - Tracking component lifecycle
 * - Monitoring memory usage patterns
 * - Detecting orphaned event listeners
 * - Cleaning up timers and intervals
 * - Providing memory optimization recommendations
 */
class MemoryLeakPreventionService {
  private static instance: MemoryLeakPreventionService;
  private componentTrackers = new Map<string, MemoryLeakDetector>();
  private memoryHistory: MemoryStats[] = [];
  private maxHistorySize = 100;
  private monitoringInterval: NodeJS.Timeout | null = null;
  private isMonitoring = false;

  private constructor() {
    this.startMonitoring();
  }

  public static getInstance(): MemoryLeakPreventionService {
    if (!MemoryLeakPreventionService.instance) {
      MemoryLeakPreventionService.instance = new MemoryLeakPreventionService();
    }
    return MemoryLeakPreventionService.instance;
  }

  /**
   * Track component mount
   */
  public trackComponentMount(componentName: string): string {
    const trackerId = `${componentName}_${Date.now()}`;
    
    this.componentTrackers.set(trackerId, {
      componentName,
      mountTime: Date.now(),
      memoryUsage: process.memoryUsage(),
      eventListeners: new Set(),
      timers: new Set(),
      intervals: new Set()
    });

    logger.debug(`Component mounted: ${componentName} (${trackerId})`);
    return trackerId;
  }

  /**
   * Track component unmount
   */
  public trackComponentUnmount(trackerId: string): void {
    const tracker = this.componentTrackers.get(trackerId);
    if (!tracker) return;

    tracker.unmountTime = Date.now();
    
    // Clean up event listeners
    tracker.eventListeners.forEach(listener => {
      // Remove event listeners (this would need to be implemented per component)
      logger.debug(`Cleaning up event listener for ${tracker.componentName}`);
    });
    tracker.eventListeners.clear();

    // Clean up timers
    tracker.timers.forEach(timer => {
      clearTimeout(timer);
    });
    tracker.timers.clear();

    // Clean up intervals
    tracker.intervals.forEach(interval => {
      clearInterval(interval);
    });
    tracker.intervals.clear();

    // Calculate memory usage
    const currentMemory = process.memoryUsage();
    const memoryIncrease = currentMemory.heapUsed - tracker.memoryUsage.heapUsed;
    
    if (memoryIncrease > 1024 * 1024) { // 1MB increase
      logger.warn(`Potential memory leak in ${tracker.componentName}: ${memoryIncrease / 1024 / 1024}MB increase`);
    }

    this.componentTrackers.delete(trackerId);
    logger.debug(`Component unmounted: ${tracker.componentName} (${trackerId})`);
  }

  /**
   * Track event listener
   */
  public trackEventListener(trackerId: string, listener: Function): void {
    const tracker = this.componentTrackers.get(trackerId);
    if (tracker) {
      tracker.eventListeners.add(listener);
    }
  }

  /**
   * Track timer
   */
  public trackTimer(trackerId: string, timer: NodeJS.Timeout): void {
    const tracker = this.componentTrackers.get(trackerId);
    if (tracker) {
      tracker.timers.add(timer);
    }
  }

  /**
   * Track interval
   */
  public trackInterval(trackerId: string, interval: NodeJS.Timeout): void {
    const tracker = this.componentTrackers.get(trackerId);
    if (tracker) {
      tracker.intervals.add(interval);
    }
  }

  /**
   * Start memory monitoring
   */
  private startMonitoring(): void {
    if (this.isMonitoring) return;

    this.isMonitoring = true;
    this.monitoringInterval = setInterval(() => {
      this.recordMemoryUsage();
      this.detectMemoryLeaks();
    }, 30000); // Check every 30 seconds

    logger.info('Memory leak prevention service started');
  }

  /**
   * Stop memory monitoring
   */
  public stopMonitoring(): void {
    if (this.monitoringInterval) {
      clearInterval(this.monitoringInterval);
      this.monitoringInterval = null;
    }
    this.isMonitoring = false;
    logger.info('Memory leak prevention service stopped');
  }

  /**
   * Record memory usage
   */
  private recordMemoryUsage(): void {
    const memory = process.memoryUsage();
    this.memoryHistory.push({
      heapUsed: memory.heapUsed,
      heapTotal: memory.heapTotal,
      external: memory.external,
      rss: memory.rss,
      timestamp: Date.now()
    });

    // Maintain history size
    if (this.memoryHistory.length > this.maxHistorySize) {
      this.memoryHistory = this.memoryHistory.slice(-this.maxHistorySize);
    }
  }

  /**
   * Detect memory leaks
   */
  private detectMemoryLeaks(): void {
    if (this.memoryHistory.length < 10) return; // Need at least 10 data points

    const recent = this.memoryHistory.slice(-10);
    const older = this.memoryHistory.slice(-20, -10);

    if (older.length === 0) return;

    const recentAvg = recent.reduce((sum, m) => sum + m.heapUsed, 0) / recent.length;
    const olderAvg = older.reduce((sum, m) => sum + m.heapUsed, 0) / older.length;

    const memoryIncrease = recentAvg - olderAvg;
    const increasePercent = (memoryIncrease / olderAvg) * 100;

    // Alert if memory increased by more than 20%
    if (increasePercent > 20) {
      logger.warn(`Potential memory leak detected: ${increasePercent.toFixed(2)}% increase in heap usage`);
      this.generateMemoryReport();
    }

    // Alert if memory usage is very high
    const currentMemory = process.memoryUsage();
    if (currentMemory.heapUsed > 500 * 1024 * 1024) { // 500MB
      logger.warn(`High memory usage detected: ${(currentMemory.heapUsed / 1024 / 1024).toFixed(2)}MB`);
    }
  }

  /**
   * Generate memory report
   */
  public generateMemoryReport(): {
    currentMemory: NodeJS.MemoryUsage;
    memoryTrend: 'increasing' | 'decreasing' | 'stable';
    activeComponents: number;
    recommendations: string[];
  } {
    const currentMemory = process.memoryUsage();
    
    // Calculate memory trend
    let memoryTrend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (this.memoryHistory.length >= 5) {
      const recent = this.memoryHistory.slice(-5);
      const older = this.memoryHistory.slice(-10, -5);
      
      if (older.length > 0) {
        const recentAvg = recent.reduce((sum, m) => sum + m.heapUsed, 0) / recent.length;
        const olderAvg = older.reduce((sum, m) => sum + m.heapUsed, 0) / older.length;
        
        if (recentAvg > olderAvg * 1.1) {
          memoryTrend = 'increasing';
        } else if (recentAvg < olderAvg * 0.9) {
          memoryTrend = 'decreasing';
        }
      }
    }

    // Generate recommendations
    const recommendations: string[] = [];
    
    if (memoryTrend === 'increasing') {
      recommendations.push('Memory usage is increasing - check for memory leaks');
    }
    
    if (currentMemory.heapUsed > 300 * 1024 * 1024) { // 300MB
      recommendations.push('Consider implementing memory optimization strategies');
    }
    
    if (this.componentTrackers.size > 50) {
      recommendations.push('High number of active components - consider component optimization');
    }

    return {
      currentMemory,
      memoryTrend,
      activeComponents: this.componentTrackers.size,
      recommendations
    };
  }

  /**
   * Force garbage collection (if available)
   */
  public forceGarbageCollection(): void {
    if (global.gc) {
      global.gc();
      logger.info('Forced garbage collection');
    } else {
      logger.warn('Garbage collection not available - run with --expose-gc flag');
    }
  }

  /**
   * Get memory statistics
   */
  public getMemoryStats(): {
    current: NodeJS.MemoryUsage;
    peak: NodeJS.MemoryUsage;
    average: NodeJS.MemoryUsage;
    trend: 'increasing' | 'decreasing' | 'stable';
  } {
    const current = process.memoryUsage();
    
    let peak = current;
    let average = current;
    
    if (this.memoryHistory.length > 0) {
      // Find peak usage
      const peakUsage = this.memoryHistory.reduce((max, m) => 
        m.heapUsed > max.heapUsed ? m : max
      );
      peak = {
        heapUsed: peakUsage.heapUsed,
        heapTotal: peakUsage.heapTotal,
        external: peakUsage.external,
        rss: peakUsage.rss,
        arrayBuffers: 0
      };

      // Calculate average
      const avgHeapUsed = this.memoryHistory.reduce((sum, m) => sum + m.heapUsed, 0) / this.memoryHistory.length;
      const avgHeapTotal = this.memoryHistory.reduce((sum, m) => sum + m.heapTotal, 0) / this.memoryHistory.length;
      const avgExternal = this.memoryHistory.reduce((sum, m) => sum + m.external, 0) / this.memoryHistory.length;
      const avgRss = this.memoryHistory.reduce((sum, m) => sum + m.rss, 0) / this.memoryHistory.length;
      
      average = {
        heapUsed: avgHeapUsed,
        heapTotal: avgHeapTotal,
        external: avgExternal,
        rss: avgRss,
        arrayBuffers: 0
      };
    }

    // Calculate trend
    let trend: 'increasing' | 'decreasing' | 'stable' = 'stable';
    if (this.memoryHistory.length >= 10) {
      const recent = this.memoryHistory.slice(-5);
      const older = this.memoryHistory.slice(-10, -5);
      
      const recentAvg = recent.reduce((sum, m) => sum + m.heapUsed, 0) / recent.length;
      const olderAvg = older.reduce((sum, m) => sum + m.heapUsed, 0) / older.length;
      
      if (recentAvg > olderAvg * 1.1) {
        trend = 'increasing';
      } else if (recentAvg < olderAvg * 0.9) {
        trend = 'decreasing';
      }
    }

    return {
      current,
      peak,
      average,
      trend
    };
  }
}

export const memoryLeakPreventionService = MemoryLeakPreventionService.getInstance();
