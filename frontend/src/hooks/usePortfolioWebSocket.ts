// React Hook for Portfolio WebSocket Service
// Provides easy integration with portfolio real-time updates

import { useEffect, useRef, useState, useCallback } from 'react';
import { portfolioWebSocketService, PortfolioUpdateData, ConnectionStats } from './portfolioWebSocketService';
import { Logger } from '../utils/logger';

const logger = new Logger();

export interface UsePortfolioWebSocketOptions {
  authToken?: string;
  userId?: string;
  autoConnect?: boolean;
  onError?: (error: Error) => void;
  onConnect?: () => void;
  onDisconnect?: () => void;
}

export interface UsePortfolioWebSocketReturn {
  portfolioData: PortfolioUpdateData | null;
  isConnected: boolean;
  connectionStats: ConnectionStats;
  connect: (authToken: string, userId: string) => Promise<void>;
  disconnect: () => void;
  forceReconnect: () => Promise<void>;
  error: Error | null;
}

export function usePortfolioWebSocket(options: UsePortfolioWebSocketOptions = {}): UsePortfolioWebSocketReturn {
  const {
    authToken,
    userId,
    autoConnect = true,
    onError,
    onConnect,
    onDisconnect
  } = options;

  const [portfolioData, setPortfolioData] = useState<PortfolioUpdateData | null>(null);
  const [isConnected, setIsConnected] = useState(false);
  const [connectionStats, setConnectionStats] = useState<ConnectionStats>(portfolioWebSocketService.getStats());
  const [error, setError] = useState<Error | null>(null);
  
  const unsubscribeRef = useRef<(() => void) | null>(null);
  const isInitializedRef = useRef(false);

  // Update connection stats periodically
  useEffect(() => {
    const interval = setInterval(() => {
      setConnectionStats(portfolioWebSocketService.getStats());
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  // Handle portfolio updates
  const handlePortfolioUpdate = useCallback((data: PortfolioUpdateData) => {
    logger.debug('📊 Portfolio update received in hook:', data.userId);
    setPortfolioData(data);
    setError(null);
  }, []);

  // Connect function
  const connect = useCallback(async (token: string, user: string) => {
    try {
      setError(null);
      await portfolioWebSocketService.connect(token, user);
      setIsConnected(true);
      onConnect?.();
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Connection failed');
      setError(error);
      onError?.(error);
      logger.error('Portfolio WebSocket connection error:', error);
    }
  }, [onConnect, onError]);

  // Disconnect function
  const disconnect = useCallback(() => {
    portfolioWebSocketService.disconnect();
    setIsConnected(false);
    onDisconnect?.();
  }, [onDisconnect]);

  // Force reconnect function
  const forceReconnect = useCallback(async () => {
    try {
      setError(null);
      await portfolioWebSocketService.forceReconnect();
      setIsConnected(true);
      onConnect?.();
    } catch (err) {
      const error = err instanceof Error ? err : new Error('Reconnection failed');
      setError(error);
      onError?.(error);
      logger.error('Portfolio WebSocket reconnection error:', error);
    }
  }, [onConnect, onError]);

  // Auto-connect effect
  useEffect(() => {
    if (autoConnect && authToken && userId && !isInitializedRef.current) {
      isInitializedRef.current = true;
      connect(authToken, userId);
    }
  }, [autoConnect, authToken, userId, connect]);

  // Subscribe to portfolio updates
  useEffect(() => {
    if (isConnected) {
      // Subscribe to portfolio updates
      unsubscribeRef.current = portfolioWebSocketService.subscribe(handlePortfolioUpdate);
      
      logger.debug('📊 Subscribed to portfolio updates');
    }

    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
        unsubscribeRef.current = null;
        logger.debug('📊 Unsubscribed from portfolio updates');
      }
    };
  }, [isConnected, handlePortfolioUpdate]);

  // Cleanup on unmount
  useEffect(() => {
    return () => {
      if (unsubscribeRef.current) {
        unsubscribeRef.current();
      }
      portfolioWebSocketService.disconnect();
    };
  }, []);

  return {
    portfolioData,
    isConnected,
    connectionStats,
    connect,
    disconnect,
    forceReconnect,
    error
  };
}

// Hook for portfolio data only (simplified version)
export function usePortfolioData(authToken?: string, userId?: string): {
  portfolioData: PortfolioUpdateData | null;
  isLoading: boolean;
  error: Error | null;
} {
  const { portfolioData, isConnected, error } = usePortfolioWebSocket({
    authToken,
    userId,
    autoConnect: true
  });

  return {
    portfolioData,
    isLoading: !isConnected && !error,
    error
  };
}

// Hook for connection status only
export function usePortfolioConnection(authToken?: string, userId?: string): {
  isConnected: boolean;
  connectionStats: ConnectionStats;
  reconnect: () => Promise<void>;
} {
  const { isConnected, connectionStats, forceReconnect } = usePortfolioWebSocket({
    authToken,
    userId,
    autoConnect: true
  });

  return {
    isConnected,
    connectionStats,
    reconnect: forceReconnect
  };
}
