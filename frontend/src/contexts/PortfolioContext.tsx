// Portfolio Context Provider
// Provides portfolio data and WebSocket connection management throughout the app

import React, { createContext, useContext, useEffect, ReactNode } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { usePortfolioStore, startPortfolioWebSocketSubscription, stopPortfolioWebSocketSubscription } from '../stores/portfolioStore';
import { Logger } from '../utils/logger';

const logger = new Logger();

interface PortfolioContextType {
  // Portfolio data
  portfolioData: any;
  positions: any[];
  summary: any;
  
  // Connection state
  isConnected: boolean;
  isConnecting: boolean;
  lastUpdate: number | null;
  
  // Error handling
  error: string | null;
  connectionError: string | null;
  
  // Actions
  connect: (authToken: string, userId: string) => Promise<void>;
  disconnect: () => void;
  forceReconnect: () => Promise<void>;
  clearError: () => void;
  
  // Computed values
  totalValue: number;
  totalPnl: number;
  pnlPercentage: number;
  healthStatus: 'healthy' | 'warning' | 'danger';
}

const PortfolioContext = createContext<PortfolioContextType | undefined>(undefined);

interface PortfolioProviderProps {
  children: ReactNode;
  authToken?: string;
}

export function PortfolioProvider({ children, authToken }: PortfolioProviderProps) {
  const { publicKey } = useWallet();
  
  // Get store state and actions
  const portfolioData = usePortfolioStore((state) => state.portfolioData);
  const positions = usePortfolioStore((state) => state.positions);
  const summary = usePortfolioStore((state) => state.summary);
  const isConnected = usePortfolioStore((state) => state.isConnected);
  const isConnecting = usePortfolioStore((state) => state.isConnecting);
  const lastUpdate = usePortfolioStore((state) => state.lastUpdate);
  const error = usePortfolioStore((state) => state.error);
  const connectionError = usePortfolioStore((state) => state.connectionError);
  
  const connect = usePortfolioStore((state) => state.connect);
  const disconnect = usePortfolioStore((state) => state.disconnect);
  const forceReconnect = usePortfolioStore((state) => state.forceReconnect);
  const clearError = usePortfolioStore((state) => state.clearError);
  
  const totalValue = usePortfolioStore((state) => state.getTotalValue());
  const totalPnl = usePortfolioStore((state) => state.getTotalPnl());
  const pnlPercentage = usePortfolioStore((state) => state.getPnlPercentage());
  const healthStatus = usePortfolioStore((state) => state.getHealthStatus());

  // Auto-connect when wallet is connected and auth token is available
  useEffect(() => {
    if (publicKey && authToken && !isConnected && !isConnecting) {
      const userId = publicKey.toString();
      logger.debug(`🔌 Auto-connecting portfolio WebSocket for user: ${userId}`);
      
      connect(authToken, userId).catch((error) => {
        logger.error('Failed to auto-connect portfolio WebSocket:', error);
      });
    }
  }, [publicKey, authToken, isConnected, isConnecting, connect]);

  // Auto-disconnect when wallet is disconnected
  useEffect(() => {
    if (!publicKey && isConnected) {
      logger.debug('🔌 Auto-disconnecting portfolio WebSocket (wallet disconnected)');
      disconnect();
    }
  }, [publicKey, isConnected, disconnect]);

  // Start WebSocket subscription on mount
  useEffect(() => {
    startPortfolioWebSocketSubscription();
    
    return () => {
      stopPortfolioWebSocketSubscription();
    };
  }, []);

  // Log connection status changes
  useEffect(() => {
    if (isConnected) {
      logger.info('✅ Portfolio WebSocket connected');
    } else if (isConnecting) {
      logger.info('🔄 Portfolio WebSocket connecting...');
    } else {
      logger.info('🔌 Portfolio WebSocket disconnected');
    }
  }, [isConnected, isConnecting]);

  // Log errors
  useEffect(() => {
    if (error) {
      logger.error('Portfolio error:', error);
    }
    if (connectionError) {
      logger.error('Portfolio connection error:', connectionError);
    }
  }, [error, connectionError]);

  const contextValue: PortfolioContextType = {
    // Portfolio data
    portfolioData,
    positions,
    summary,
    
    // Connection state
    isConnected,
    isConnecting,
    lastUpdate,
    
    // Error handling
    error,
    connectionError,
    
    // Actions
    connect,
    disconnect,
    forceReconnect,
    clearError,
    
    // Computed values
    totalValue,
    totalPnl,
    pnlPercentage,
    healthStatus,
  };

  return (
    <PortfolioContext.Provider value={contextValue}>
      {children}
    </PortfolioContext.Provider>
  );
}

// Hook to use portfolio context
export function usePortfolio(): PortfolioContextType {
  const context = useContext(PortfolioContext);
  if (context === undefined) {
    throw new Error('usePortfolio must be used within a PortfolioProvider');
  }
  return context;
}

// Hook for portfolio data only
export function usePortfolioData() {
  const { portfolioData, positions, summary, lastUpdate } = usePortfolio();
  return { portfolioData, positions, summary, lastUpdate };
}

// Hook for connection status only
export function usePortfolioConnection() {
  const { isConnected, isConnecting, error, connectionError, connect, disconnect, forceReconnect } = usePortfolio();
  return { isConnected, isConnecting, error, connectionError, connect, disconnect, forceReconnect };
}

// Hook for portfolio metrics only
export function usePortfolioMetrics() {
  const { totalValue, totalPnl, pnlPercentage, healthStatus } = usePortfolio();
  return { totalValue, totalPnl, pnlPercentage, healthStatus };
}

// Hook for specific position
export function usePosition(positionId: string) {
  const { positions } = usePortfolio();
  return positions.find(pos => pos.id === positionId);
}

// Hook for positions by symbol
export function usePositionsBySymbol(symbol: string) {
  const { positions } = usePortfolio();
  return positions.filter(pos => pos.symbol === symbol);
}
