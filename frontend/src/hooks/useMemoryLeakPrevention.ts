import { useEffect, useRef, useCallback, useState } from 'react';
import { Logger } from '../utils/logger';

const logger = new Logger();

/**
 * React Hook for Memory Leak Prevention
 * 
 * Automatically tracks component lifecycle and cleans up resources
 */
export const useMemoryLeakPrevention = (componentName: string) => {
  const trackerIdRef = useRef<string | null>(null);
  const eventListenersRef = useRef<Set<Function>>(new Set());
  const timersRef = useRef<Set<NodeJS.Timeout>>(new Set());
  const intervalsRef = useRef<Set<NodeJS.Timeout>>(new Set());

  // Track component mount
  useEffect(() => {
    // This would integrate with the backend memory leak prevention service
    // For now, we'll just log the component mount
    logger.debug(`Component mounted: ${componentName}`);
    
    return () => {
      // Cleanup on unmount
      logger.debug(`Component unmounting: ${componentName}`);
      
      // Clear all timers
      timersRef.current.forEach(timer => {
        clearTimeout(timer);
      });
      timersRef.current.clear();

      // Clear all intervals
      intervalsRef.current.forEach(interval => {
        clearInterval(interval);
      });
      intervalsRef.current.clear();

      // Clear event listeners
      eventListenersRef.current.clear();
    };
  }, [componentName]);

  // Safe setTimeout wrapper
  const safeSetTimeout = useCallback((callback: () => void, delay: number): NodeJS.Timeout => {
    const timer = setTimeout(() => {
      timersRef.current.delete(timer);
      callback();
    }, delay);
    
    timersRef.current.add(timer);
    return timer;
  }, []);

  // Safe setInterval wrapper
  const safeSetInterval = useCallback((callback: () => void, delay: number): NodeJS.Timeout => {
    const interval = setInterval(callback, delay);
    intervalsRef.current.add(interval);
    return interval;
  }, []);

  // Safe clearTimeout wrapper
  const safeClearTimeout = useCallback((timer: NodeJS.Timeout) => {
    clearTimeout(timer);
    timersRef.current.delete(timer);
  }, []);

  // Safe clearInterval wrapper
  const safeClearInterval = useCallback((interval: NodeJS.Timeout) => {
    clearInterval(interval);
    intervalsRef.current.delete(interval);
  }, []);

  // Track event listener
  const trackEventListener = useCallback((listener: Function) => {
    eventListenersRef.current.add(listener);
    return listener;
  }, []);

  return {
    safeSetTimeout,
    safeSetInterval,
    safeClearTimeout,
    safeClearInterval,
    trackEventListener
  };
};

/**
 * Hook for debounced values with automatic cleanup
 */
export const useDebounce = <T>(value: T, delay: number): T => {
  const { safeSetTimeout, safeClearTimeout } = useMemoryLeakPrevention('useDebounce');
  const [debouncedValue, setDebouncedValue] = useState<T>(value);

  useEffect(() => {
    const timer = safeSetTimeout(() => {
      setDebouncedValue(value);
    }, delay);

    return () => {
      safeClearTimeout(timer);
    };
  }, [value, delay, safeSetTimeout, safeClearTimeout]);

  return debouncedValue;
};

/**
 * Hook for throttled values with automatic cleanup
 */
export const useThrottle = <T>(value: T, delay: number): T => {
  const { safeSetTimeout, safeClearTimeout } = useMemoryLeakPrevention('useThrottle');
  const [throttledValue, setThrottledValue] = useState<T>(value);
  const lastExecutedRef = useRef<number>(0);

  useEffect(() => {
    const now = Date.now();
    
    if (now - lastExecutedRef.current >= delay) {
      setThrottledValue(value);
      lastExecutedRef.current = now;
    } else {
      const timer = safeSetTimeout(() => {
        setThrottledValue(value);
        lastExecutedRef.current = Date.now();
      }, delay - (now - lastExecutedRef.current));

      return () => {
        safeClearTimeout(timer);
      };
    }
  }, [value, delay, safeSetTimeout, safeClearTimeout]);

  return throttledValue;
};

/**
 * Hook for managing WebSocket connections with automatic cleanup
 */
export const useWebSocketWithCleanup = (url: string, options?: any) => {
  const { safeSetTimeout, safeClearTimeout } = useMemoryLeakPrevention('useWebSocketWithCleanup');
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  const [connectionState, setConnectionState] = useState<'connecting' | 'connected' | 'disconnected'>('disconnected');

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return;
    }

    setConnectionState('connecting');
    
    try {
      wsRef.current = new WebSocket(url, options);
      
      wsRef.current.onopen = () => {
        setConnectionState('connected');
        logger.debug('WebSocket connected');
      };
      
      wsRef.current.onclose = () => {
        setConnectionState('disconnected');
        logger.debug('WebSocket disconnected');
        
        // Attempt to reconnect after 5 seconds
        reconnectTimeoutRef.current = safeSetTimeout(() => {
          connect();
        }, 5000);
      };
      
      wsRef.current.onerror = (error) => {
        logger.error('WebSocket error:', error);
        setConnectionState('disconnected');
      };
    } catch (error) {
      logger.error('Failed to create WebSocket:', error);
      setConnectionState('disconnected');
    }
  }, [url, options, safeSetTimeout]);

  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      safeClearTimeout(reconnectTimeoutRef.current);
      reconnectTimeoutRef.current = null;
    }
    
    if (wsRef.current) {
      wsRef.current.close();
      wsRef.current = null;
    }
    
    setConnectionState('disconnected');
  }, [safeClearTimeout]);

  useEffect(() => {
    connect();
    
    return () => {
      disconnect();
    };
  }, [connect, disconnect]);

  return {
    connectionState,
    connect,
    disconnect,
    send: (data: any) => {
      if (wsRef.current?.readyState === WebSocket.OPEN) {
        wsRef.current.send(JSON.stringify(data));
      }
    }
  };
};

/**
 * Hook for managing intervals with automatic cleanup
 */
export const useIntervalWithCleanup = (callback: () => void, delay: number | null) => {
  const { safeSetInterval, safeClearInterval } = useMemoryLeakPrevention('useIntervalWithCleanup');
  const intervalRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (delay !== null) {
      intervalRef.current = safeSetInterval(callback, delay);
      
      return () => {
        if (intervalRef.current) {
          safeClearInterval(intervalRef.current);
          intervalRef.current = null;
        }
      };
    }
  }, [callback, delay, safeSetInterval, safeClearInterval]);
};

/**
 * Hook for managing timeouts with automatic cleanup
 */
export const useTimeoutWithCleanup = (callback: () => void, delay: number | null) => {
  const { safeSetTimeout, safeClearTimeout } = useMemoryLeakPrevention('useTimeoutWithCleanup');
  const timeoutRef = useRef<NodeJS.Timeout | null>(null);

  useEffect(() => {
    if (delay !== null) {
      timeoutRef.current = safeSetTimeout(callback, delay);
      
      return () => {
        if (timeoutRef.current) {
          safeClearTimeout(timeoutRef.current);
          timeoutRef.current = null;
        }
      };
    }
  }, [callback, delay, safeSetTimeout, safeClearTimeout]);
};
