import React, { useState, useEffect, useMemo } from 'react';
import axios from 'axios';

interface Market {
  id: string;
  symbol: string;
  baseAsset: string;
  quoteAsset: string;
  isActive: boolean;
  maxLeverage: number;
  currentPrice: number;
  priceChange24h: number;
  volume24h: number;
  category: string;
  description: string;
  logoUrl: string;
}

interface MarketContextType {
  markets: Market[];
  selectedMarket: Market | null;
  loading: boolean;
  error: string | null;
  selectMarket: (market: Market) => void;
  selectMarketBySymbol: (symbol: string) => void;
  searchMarkets: (query: string) => Market[];
  getMarketsByCategory: (category: string) => Market[];
  refreshMarkets: () => Promise<void>;
}

const MarketContext = React.createContext<MarketContextType | undefined>(undefined);

export const MarketProvider: React.FC<{ children: React.ReactNode }> = ({ children }) => {
  const [markets, setMarkets] = useState<Market[]>([]);
  const [selectedMarket, setSelectedMarket] = useState<Market | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const baseUrl = import.meta.env.VITE_API_BASE_URL || 'http://localhost:3002';

  // Fetch markets from Supabase API
  const fetchMarkets = async () => {
    try {
      setLoading(true);
      setError(null);
      
      const response = await axios.get(`${baseUrl}/api/markets`);
      
      if (response.data.success && response.data.markets) {
        setMarkets(response.data.markets);
        
        // Auto-select first market if none selected
        if (!selectedMarket && response.data.markets.length > 0) {
          setSelectedMarket(response.data.markets[0]);
        }
      } else {
        throw new Error('Invalid response format');
      }
    } catch (err) {
      console.error('Error fetching markets:', err);
      setError(err instanceof Error ? err.message : 'Failed to fetch markets');
    } finally {
      setLoading(false);
    }
  };

  // Initial load
  useEffect(() => {
    fetchMarkets();
  }, []);

  // Select market by object
  const selectMarket = (market: Market) => {
    setSelectedMarket(market);
  };

  // Select market by symbol
  const selectMarketBySymbol = (symbol: string) => {
    const market = markets.find(m => m.symbol === symbol);
    if (market) {
      setSelectedMarket(market);
    }
  };

  // Search markets
  const searchMarkets = (query: string): Market[] => {
    if (!query.trim()) return markets;
    
    const lowercaseQuery = query.toLowerCase();
    return markets.filter(market => 
      market.symbol.toLowerCase().includes(lowercaseQuery) ||
      market.baseAsset.toLowerCase().includes(lowercaseQuery) ||
      market.description.toLowerCase().includes(lowercaseQuery)
    );
  };

  // Get markets by category
  const getMarketsByCategory = (category: string): Market[] => {
    return markets.filter(market => market.category === category);
  };

  // Refresh markets
  const refreshMarkets = async () => {
    await fetchMarkets();
  };

  const contextValue: MarketContextType = useMemo(() => ({
    markets,
    selectedMarket,
    loading,
    error,
    selectMarket,
    selectMarketBySymbol,
    searchMarkets,
    getMarketsByCategory,
    refreshMarkets
  }), [markets, selectedMarket, loading, error, selectMarket, selectMarketBySymbol, searchMarkets, getMarketsByCategory, refreshMarkets]);

  return (
    <MarketContext.Provider value={contextValue}>
      {children}
    </MarketContext.Provider>
  );
};

// Hook to use market context
export const useMarkets = (): MarketContextType => {
  const context = React.useContext(MarketContext);
  if (context === undefined) {
    throw new Error('useMarkets must be used within a MarketProvider');
  }
  return context;
};

// Hook to get selected market
export const useSelectedMarket = (): Market | null => {
  const { selectedMarket } = useMarkets();
  return selectedMarket;
};

// Hook to get markets by category
export const useMarketsByCategory = (category: string): Market[] => {
  const { getMarketsByCategory } = useMarkets();
  return getMarketsByCategory(category);
};

export default MarketContext;
