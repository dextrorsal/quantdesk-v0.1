import React from 'react';
import { useMarkets } from '../contexts/MarketContext';

/**
 * DEX-style crypto heatmap
 * Clean, lightweight alternative to TradingView heatmap
 * Uses our real market data
 */
const DexHeatmap: React.FC = () => {
  const { markets } = useMarkets();

  // Group markets by category for better organization
  const marketsByCategory = markets.reduce((acc, market) => {
    const category = market.category || 'other';
    if (!acc[category]) acc[category] = [];
    acc[category].push(market);
    return acc;
  }, {} as Record<string, typeof markets>);

  // Get color based on price change
  const getChangeColor = (change: number) => {
    if (change > 0) return 'bg-green-500';
    if (change < 0) return 'bg-red-500';
    return 'bg-gray-500';
  };

  // Get size based on volume (normalized)
  const getSizeClass = (volume: number) => {
    if (volume > 1000000) return 'w-16 h-16'; // Large
    if (volume > 500000) return 'w-12 h-12';  // Medium
    if (volume > 100000) return 'w-10 h-10';  // Small
    return 'w-8 h-8'; // Tiny
  };

  return (
    <div className="bg-gray-900 rounded-lg p-6">
      <h3 className="text-xl font-bold text-white mb-4">📊 Market Heatmap</h3>
      
      {/* Categories */}
      {Object.entries(marketsByCategory).map(([category, categoryMarkets]) => (
        <div key={category} className="mb-6">
          <h4 className="text-lg font-semibold text-gray-300 mb-3 capitalize">
            {category.replace('-', ' ')}
          </h4>
          
          {/* Heatmap Grid */}
          <div className="grid grid-cols-8 gap-2">
            {categoryMarkets.slice(0, 16).map((market) => (
              <div
                key={market.id}
                className={`
                  ${getSizeClass(market.volume24h || 0)}
                  ${getChangeColor(market.change24h || 0)}
                  rounded-lg flex items-center justify-center cursor-pointer
                  hover:opacity-80 transition-opacity
                  relative group
                `}
                title={`${market.symbol}: $${market.price?.toFixed(2) || '0.00'} (${((market.change24h || 0) / (market.price || 1) * 100).toFixed(2)}%)`}
              >
                {/* Symbol */}
                <span className="text-white text-xs font-bold">
                  {market.symbol.replace('-PERP', '')}
                </span>
                
                {/* Tooltip */}
                <div className="absolute bottom-full left-1/2 transform -translate-x-1/2 mb-2 px-2 py-1 bg-gray-800 text-white text-xs rounded opacity-0 group-hover:opacity-100 transition-opacity whitespace-nowrap z-10">
                  <div className="font-semibold">{market.symbol}</div>
                  <div>${market.price?.toFixed(2) || '0.00'}</div>
                  <div className={market.change24h && market.change24h > 0 ? 'text-green-400' : 'text-red-400'}>
                    {market.change24h && market.change24h > 0 ? '+' : ''}{((market.change24h || 0) / (market.price || 1) * 100).toFixed(2)}%
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      ))}
      
      {/* Legend */}
      <div className="mt-6 flex items-center space-x-6 text-sm">
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-green-500 rounded"></div>
          <span className="text-gray-300">Positive</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-red-500 rounded"></div>
          <span className="text-gray-300">Negative</span>
        </div>
        <div className="flex items-center space-x-2">
          <div className="w-4 h-4 bg-gray-500 rounded"></div>
          <span className="text-gray-300">Neutral</span>
        </div>
        <div className="text-gray-400">
          Size = Volume
        </div>
      </div>
    </div>
  );
};

export default DexHeatmap;
