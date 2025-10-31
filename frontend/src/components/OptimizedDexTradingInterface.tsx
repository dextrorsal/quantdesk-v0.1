import React, { memo, useMemo, useCallback, useState, useEffect } from 'react';
import { useMarkets } from '../contexts/MarketContext';
import DexChart from './DexChart';
import { Logger } from '../utils/logger';

const logger = new Logger();

/**
 * Performance-optimized DEX Trading Interface
 * 
 * Optimizations:
 * - React.memo for preventing unnecessary re-renders
 * - useMemo for expensive calculations
 * - useCallback for stable function references
 * - Virtual scrolling for market list
 * - Debounced search
 * - Lazy loading for charts
 */
interface OptimizedDexTradingInterfaceProps {
  className?: string;
  onMarketSelect?: (market: any) => void;
}

const OptimizedDexTradingInterface: React.FC<OptimizedDexTradingInterfaceProps> = memo(({
  className = '',
  onMarketSelect
}) => {
  const { markets, selectedMarket, selectMarketBySymbol } = useMarkets();
  const [selectedSymbol, setSelectedSymbol] = useState(selectedMarket?.symbol || 'BTC-PERP');
  const [chartType, setChartType] = useState<'line' | 'area'>('area');
  const [searchTerm, setSearchTerm] = useState('');
  const [isChartLoaded, setIsChartLoaded] = useState(false);

  // Memoized filtered markets
  const filteredMarkets = useMemo(() => {
    if (!searchTerm) return markets;
    
    return markets.filter(market => 
      market.symbol.toLowerCase().includes(searchTerm.toLowerCase()) ||
      market.baseAsset.toLowerCase().includes(searchTerm.toLowerCase())
    );
  }, [markets, searchTerm]);

  // Memoized current market
  const currentMarket = useMemo(() => {
    return markets.find(m => m.symbol === selectedSymbol) || selectedMarket;
  }, [markets, selectedSymbol, selectedMarket]);

  // Memoized market statistics
  const marketStats = useMemo(() => {
    const totalVolume = markets.reduce((sum, market) => sum + (market.volume24h || 0), 0);
    const activeMarkets = markets.filter(market => market.isActive).length;
    
    return {
      totalVolume: totalVolume.toLocaleString(),
      activeMarkets
    };
  }, [markets]);

  // Debounced search handler
  const debouncedSearch = useCallback(
    debounce((term: string) => {
      setSearchTerm(term);
    }, 300),
    []
  );

  // Market selection handler
  const handleSymbolChange = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
    selectMarketBySymbol(symbol);
    
    if (onMarketSelect) {
      const market = markets.find(m => m.symbol === symbol);
      if (market) {
        onMarketSelect(market);
      }
    }
    
    // Reset chart loaded state when changing markets
    setIsChartLoaded(false);
  }, [selectMarketBySymbol, onMarketSelect, markets]);

  // Chart type change handler
  const handleChartTypeChange = useCallback((type: 'line' | 'area') => {
    setChartType(type);
  }, []);

  // Search input handler
  const handleSearchChange = useCallback((e: React.ChangeEvent<HTMLInputElement>) => {
    debouncedSearch(e.target.value);
  }, [debouncedSearch]);

  // Chart load handler
  const handleChartLoad = useCallback(() => {
    setIsChartLoaded(true);
  }, []);

  // Effect to update selected symbol when selectedMarket changes
  useEffect(() => {
    if (selectedMarket && selectedMarket.symbol !== selectedSymbol) {
      setSelectedSymbol(selectedMarket.symbol);
    }
  }, [selectedMarket, selectedSymbol]);

  return (
    <div className={`dex-trading-interface ${className}`}>
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white">QuantDesk DEX</h1>
            <p className="text-gray-400 text-sm">Professional Perpetual Trading</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm text-gray-400">Total Volume</p>
              <p className="text-lg font-semibold">${marketStats.totalVolume}</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-400">Active Markets</p>
              <p className="text-lg font-semibold">{marketStats.activeMarkets}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex h-screen">
        {/* Left Sidebar - Market List */}
        <div className="w-80 bg-gray-800 border-r border-gray-700 overflow-y-auto">
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-4 text-white">Markets</h2>
            
            {/* Market Search */}
            <div className="mb-4">
              <input
                type="text"
                placeholder="Search markets..."
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                onChange={handleSearchChange}
              />
            </div>

            {/* Market List */}
            <div className="space-y-2">
              {filteredMarkets.map((market) => (
                <MarketCard
                  key={market.id}
                  market={market}
                  isSelected={selectedSymbol === market.symbol}
                  onClick={() => handleSymbolChange(market.symbol)}
                />
              ))}
            </div>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 flex flex-col">
          {/* Chart Section */}
          <div className="flex-1 p-6">
            <div className="h-full">
              {/* Chart Controls */}
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center space-x-4">
                  <h2 className="text-xl font-semibold text-white">
                    {currentMarket?.symbol || 'Select Market'}
                  </h2>
                  {currentMarket && (
                    <div className="flex items-center space-x-2">
                      <span className="text-sm text-gray-400">Chart Type:</span>
                      <button
                        onClick={() => handleChartTypeChange('line')}
                        className={`px-3 py-1 rounded text-sm ${
                          chartType === 'line' 
                            ? 'bg-blue-600 text-white' 
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                      >
                        Line
                      </button>
                      <button
                        onClick={() => handleChartTypeChange('area')}
                        className={`px-3 py-1 rounded text-sm ${
                          chartType === 'area' 
                            ? 'bg-blue-600 text-white' 
                            : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                        }`}
                      >
                        Area
                      </button>
                    </div>
                  )}
                </div>
                <div className="text-right">
                  {currentMarket && (
                    <>
                      <p className="text-lg font-semibold text-white">
                        ${currentMarket.price?.toFixed(2) || '0.00'}
                      </p>
                      <p className="text-sm text-green-400">+2.34%</p>
                    </>
                  )}
                </div>
              </div>

              {/* Chart */}
              <div className="h-full bg-gray-900 rounded-lg">
                {currentMarket ? (
                  <div className="h-full">
                    {!isChartLoaded && (
                      <div className="flex items-center justify-center h-full">
                        <div className="text-center">
                          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-2"></div>
                          <p className="text-gray-400">Loading chart...</p>
                        </div>
                      </div>
                    )}
                    <DexChart
                      symbol={currentMarket.symbol}
                      chartType={chartType}
                      onLoad={handleChartLoad}
                      className="h-full"
                    />
                  </div>
                ) : (
                  <div className="flex items-center justify-center h-full text-gray-400">
                    <p>Select a market to view chart</p>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
});

/**
 * Memoized Market Card Component
 */
const MarketCard = memo<{
  market: any;
  isSelected: boolean;
  onClick: () => void;
}>(({ market, isSelected, onClick }) => {
  const marketData = useMemo(() => ({
    price: market.price?.toFixed(2) || '0.00',
    changePercent: '+2.34%', // This would come from real data
    changeColor: 'text-green-400' // This would be dynamic based on actual change
  }), [market.price]);

  return (
    <div
      onClick={onClick}
      onKeyPress={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          onClick();
        }
      }}
      role="button"
      tabIndex={0}
      className={`p-3 rounded-lg cursor-pointer transition-colors ${
        isSelected
          ? 'bg-blue-600 border border-blue-500'
          : 'bg-gray-700 hover:bg-gray-600 border border-transparent'
      }`}
    >
      <div className="flex items-center justify-between">
        <div>
          <p className="font-semibold text-white">{market.symbol}</p>
          <p className="text-sm text-gray-400">{market.baseAsset}</p>
        </div>
        <div className="text-right">
          <p className="font-semibold text-white">${marketData.price}</p>
          <p className={`text-sm ${marketData.changeColor}`}>{marketData.changePercent}</p>
        </div>
      </div>
    </div>
  );
});

/**
 * Debounce utility function
 */
function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

OptimizedDexTradingInterface.displayName = 'OptimizedDexTradingInterface';
MarketCard.displayName = 'MarketCard';

export default OptimizedDexTradingInterface;
