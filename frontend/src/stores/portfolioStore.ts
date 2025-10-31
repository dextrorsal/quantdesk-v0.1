// Portfolio Store using Zustand
// Manages real-time portfolio data with WebSocket integration

import { create } from 'zustand';
import { subscribeWithSelector } from 'zustand/middleware';
import { PortfolioUpdateData } from '../services/portfolioWebSocketService';
import { portfolioWebSocketService } from '../services/portfolioWebSocketService';

export interface PortfolioPosition {
  id: string;
  symbol: string;
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  margin: number;
  leverage: number;
  side: 'long' | 'short';
}

export interface PortfolioSummary {
  userId: string;
  totalValue: number;
  totalUnrealizedPnl: number;
  totalRealizedPnl: number;
  marginRatio: number;
  healthFactor: number;
  totalCollateral: number;
  usedMargin: number;
  availableMargin: number;
  timestamp: number;
}

export interface PortfolioState {
  // Data
  portfolioData: PortfolioUpdateData | null;
  positions: PortfolioPosition[];
  summary: PortfolioSummary | null;
  
  // Connection state
  isConnected: boolean;
  isConnecting: boolean;
  lastUpdate: number | null;
  
  // Error handling
  error: string | null;
  connectionError: string | null;
  
  // Actions
  setPortfolioData: (data: PortfolioUpdateData) => void;
  updatePosition: (id: string, updates: Partial<PortfolioPosition>) => void;
  removePosition: (id: string) => void;
  setConnected: (connected: boolean) => void;
  setConnecting: (connecting: boolean) => void;
  setError: (error: string | null) => void;
  setConnectionError: (error: string | null) => void;
  clearError: () => void;
  
  // WebSocket actions
  connect: (authToken: string, userId: string) => Promise<void>;
  disconnect: () => void;
  forceReconnect: () => Promise<void>;
  
  // Computed values
  getTotalValue: () => number;
  getTotalPnl: () => number;
  getPnlPercentage: () => number;
  getHealthStatus: () => 'healthy' | 'warning' | 'danger';
  getPositionById: (id: string) => PortfolioPosition | undefined;
  getPositionsBySymbol: (symbol: string) => PortfolioPosition[];
}

export const usePortfolioStore = create<PortfolioState>()(
  subscribeWithSelector((set, get) => ({
    // Initial state
    portfolioData: null,
    positions: [],
    summary: null,
    isConnected: false,
    isConnecting: false,
    lastUpdate: null,
    error: null,
    connectionError: null,

    // Actions
    setPortfolioData: (data) => {
      const positions: PortfolioPosition[] = data.positions.map(pos => ({
        id: pos.id,
        symbol: pos.symbol,
        size: pos.size,
        entryPrice: pos.entryPrice,
        currentPrice: pos.currentPrice,
        unrealizedPnl: pos.unrealizedPnl,
        unrealizedPnlPercent: pos.unrealizedPnlPercent,
        margin: pos.margin,
        leverage: pos.leverage,
        side: pos.side
      }));

      const summary: PortfolioSummary = {
        userId: data.userId,
        totalValue: data.totalValue,
        totalUnrealizedPnl: data.totalUnrealizedPnl,
        totalRealizedPnl: data.totalRealizedPnl,
        marginRatio: data.marginRatio,
        healthFactor: data.healthFactor,
        totalCollateral: data.totalCollateral,
        usedMargin: data.usedMargin,
        availableMargin: data.availableMargin,
        timestamp: data.timestamp
      };

      set({
        portfolioData: data,
        positions,
        summary,
        lastUpdate: Date.now(),
        error: null
      });
    },

    updatePosition: (id, updates) => {
      set((state) => ({
        positions: state.positions.map((pos) =>
          pos.id === id ? { ...pos, ...updates } : pos
        ),
      }));
    },

    removePosition: (id) => {
      set((state) => ({
        positions: state.positions.filter((pos) => pos.id !== id),
      }));
    },

    setConnected: (connected) => {
      set({ isConnected: connected });
    },

    setConnecting: (connecting) => {
      set({ isConnecting: connecting });
    },

    setError: (error) => {
      set({ error });
    },

    setConnectionError: (error) => {
      set({ connectionError: error });
    },

    clearError: () => {
      set({ error: null, connectionError: null });
    },

    // WebSocket actions
    connect: async (authToken, userId) => {
      const state = get();
      if (state.isConnecting || state.isConnected) {
        return;
      }

      set({ isConnecting: true, connectionError: null });

      try {
        await portfolioWebSocketService.connect(authToken, userId);
        set({ isConnected: true, isConnecting: false });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Connection failed';
        set({ 
          isConnected: false, 
          isConnecting: false, 
          connectionError: errorMessage 
        });
        throw error;
      }
    },

    disconnect: () => {
      portfolioWebSocketService.disconnect();
      set({ isConnected: false, isConnecting: false });
    },

    forceReconnect: async () => {
      const state = get();
      if (!state.isConnected) {
        throw new Error('Not connected to portfolio WebSocket');
      }

      try {
        await portfolioWebSocketService.forceReconnect();
        set({ connectionError: null });
      } catch (error) {
        const errorMessage = error instanceof Error ? error.message : 'Reconnection failed';
        set({ connectionError: errorMessage });
        throw error;
      }
    },

    // Computed values
    getTotalValue: () => {
      const state = get();
      return state.summary?.totalValue || 0;
    },

    getTotalPnl: () => {
      const state = get();
      return state.summary?.totalUnrealizedPnl || 0;
    },

    getPnlPercentage: () => {
      const state = get();
      const totalValue = state.summary?.totalValue || 0;
      const totalPnl = state.summary?.totalUnrealizedPnl || 0;
      
      if (totalValue === 0) return 0;
      return (totalPnl / (totalValue - totalPnl)) * 100;
    },

    getHealthStatus: () => {
      const state = get();
      const healthFactor = state.summary?.healthFactor || 100;
      
      if (healthFactor >= 80) return 'healthy';
      if (healthFactor >= 50) return 'warning';
      return 'danger';
    },

    getPositionById: (id) => {
      const state = get();
      return state.positions.find(pos => pos.id === id);
    },

    getPositionsBySymbol: (symbol) => {
      const state = get();
      return state.positions.filter(pos => pos.symbol === symbol);
    },
  }))
);

// Subscribe to WebSocket updates and update store
let unsubscribeWebSocket: (() => void) | null = null;

// Function to start WebSocket subscription
export const startPortfolioWebSocketSubscription = () => {
  if (unsubscribeWebSocket) {
    return; // Already subscribed
  }

  unsubscribeWebSocket = portfolioWebSocketService.subscribe((data) => {
    usePortfolioStore.getState().setPortfolioData(data);
  });
};

// Function to stop WebSocket subscription
export const stopPortfolioWebSocketSubscription = () => {
  if (unsubscribeWebSocket) {
    unsubscribeWebSocket();
    unsubscribeWebSocket = null;
  }
};

// Auto-subscribe when store is created
startPortfolioWebSocketSubscription();

// Selectors for common use cases
export const usePortfolioSummary = () => usePortfolioStore((state) => state.summary);
export const usePortfolioPositions = () => usePortfolioStore((state) => state.positions);
export const usePortfolioConnection = () => usePortfolioStore((state) => ({
  isConnected: state.isConnected,
  isConnecting: state.isConnecting,
  error: state.connectionError
}));
export const usePortfolioError = () => usePortfolioStore((state) => state.error);

// Computed selectors
export const useTotalValue = () => usePortfolioStore((state) => state.getTotalValue());
export const useTotalPnl = () => usePortfolioStore((state) => state.getTotalPnl());
export const usePnlPercentage = () => usePortfolioStore((state) => state.getPnlPercentage());
export const useHealthStatus = () => usePortfolioStore((state) => state.getHealthStatus());
