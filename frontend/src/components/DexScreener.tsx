import React, { useState } from 'react';
import { useMarkets } from '../contexts/MarketContext';
import { useTickerClick } from '../hooks/useTickerClick';

/**
 * DEX-style market screener
 * Clean, lightweight alternative to TradingView screener
 * Uses our real market data with sorting and filtering
 */
const DexScreener: React.FC = () => {
  const { markets } = useMarkets();
  const { handleTickerClick } = useTickerClick();
  const [sortBy, setSortBy] = useState<'volume' | 'change' | 'price' | 'name'>('volume');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');
  const [filterCategory, setFilterCategory] = useState<string>('all');

  // Get unique categories
  const categories = ['all', ...Array.from(new Set(markets.map(m => m.category).filter(Boolean)))];

  // Filter and sort markets
  const filteredMarkets = markets
    .filter(market => filterCategory === 'all' || market.category === filterCategory)
    .sort((a, b) => {
      let aValue: number, bValue: number;
      
      switch (sortBy) {
        case 'volume':
          aValue = a.volume24h || 0;
          bValue = b.volume24h || 0;
          break;
        case 'change':
          aValue = a.change24h || 0;
          bValue = b.change24h || 0;
          break;
        case 'price':
          aValue = a.price || 0;
          bValue = b.price || 0;
          break;
        case 'name':
          return sortOrder === 'asc' 
            ? a.symbol.localeCompare(b.symbol)
            : b.symbol.localeCompare(a.symbol);
        default:
          return 0;
      }
      
      return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
    });

  return (
    <div className="bg-gray-900 rounded-lg p-6">
      <h3 className="text-xl font-bold text-white mb-4">🔍 Market Screener</h3>
      
      {/* Controls */}
      <div className="flex flex-wrap gap-4 mb-6">
        {/* Sort By */}
        <div className="flex items-center space-x-2">
          <label className="text-gray-300 text-sm">Sort by:</label>
          <select
            value={sortBy}
            onChange={(e) => setSortBy(e.target.value as any)}
            className="bg-gray-800 text-white px-3 py-1 rounded text-sm border border-gray-600"
            aria-label="Sort by"
          >
            <option value="volume">Volume</option>
            <option value="change">24h Change</option>
            <option value="price">Price</option>
            <option value="name">Name</option>
          </select>
        </div>

        {/* Sort Order */}
        <div className="flex items-center space-x-2">
          <label className="text-gray-300 text-sm">Order:</label>
          <select
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value as any)}
            className="bg-gray-800 text-white px-3 py-1 rounded text-sm border border-gray-600"
            aria-label="Sort order"
          >
            <option value="desc">Descending</option>
            <option value="asc">Ascending</option>
          </select>
        </div>

        {/* Category Filter */}
        <div className="flex items-center space-x-2">
          <label className="text-gray-300 text-sm">Category:</label>
          <select
            value={filterCategory}
            onChange={(e) => setFilterCategory(e.target.value)}
            className="bg-gray-800 text-white px-3 py-1 rounded text-sm border border-gray-600"
            aria-label="Filter by category"
          >
            {categories.map(category => (
              <option key={category} value={category}>
                {category === 'all' ? 'All' : category.replace('-', ' ')}
              </option>
            ))}
          </select>
        </div>
      </div>

      {/* Market Table */}
      <div className="overflow-x-auto">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left py-2 text-gray-300">Symbol</th>
              <th className="text-right py-2 text-gray-300">Price</th>
              <th className="text-right py-2 text-gray-300">24h Change</th>
              <th className="text-right py-2 text-gray-300">Volume</th>
              <th className="text-right py-2 text-gray-300">Open Interest</th>
              <th className="text-left py-2 text-gray-300">Category</th>
            </tr>
          </thead>
          <tbody>
            {filteredMarkets.map((market) => (
              <tr 
                key={market.id} 
                className="border-b border-gray-800 hover:bg-gray-800 transition-colors cursor-pointer"
                onClick={() => handleTickerClick(market.symbol)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter' || e.key === ' ') {
                    handleTickerClick(market.symbol);
                  }
                }}
                role="button"
                tabIndex={0}
              >
                <td className="py-2">
                  <span className="text-white font-medium">
                    {market.symbol}
                  </span>
                </td>
                <td className="py-2 text-right">
                  <span className="text-white">
                    ${market.price?.toFixed(2) || '0.00'}
                  </span>
                </td>
                <td className="py-2 text-right">
                  <span className={market.change24h && market.change24h > 0 ? 'text-green-400' : 'text-red-400'}>
                    {market.change24h && market.change24h > 0 ? '+' : ''}
                    {market.change24h ? (market.change24h / (market.price || 1) * 100).toFixed(2) : '0.00'}%
                  </span>
                </td>
                <td className="py-2 text-right">
                  <span className="text-gray-300">
                    ${(market.volume24h || 0).toLocaleString()}
                  </span>
                </td>
                <td className="py-2 text-right">
                  <span className="text-gray-300">
                    ${(market.openInterest || 0).toLocaleString()}
                  </span>
                </td>
                <td className="py-2">
                  <span className="text-gray-400 text-xs capitalize">
                    {market.category?.replace('-', ' ') || 'other'}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Summary */}
      <div className="mt-4 text-sm text-gray-400">
        Showing {filteredMarkets.length} of {markets.length} markets
      </div>
    </div>
  );
};

export default DexScreener;
