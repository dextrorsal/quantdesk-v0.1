import React from 'react';
import { useMarkets } from '../contexts/MarketContext';
import { useTickerClick } from '../hooks/useTickerClick';

/**
 * Simple DEX-style ticker tape
 * Clean, basic design like Uniswap/DexScreener
 */
const DexTickerTape: React.FC = () => {
  const { markets } = useMarkets();
  const { handleTickerClick } = useTickerClick();

  // Get first 8 markets for ticker tape
  const topMarkets = markets.slice(0, 8);

  return (
    <div className="bg-gray-900 border-b border-gray-700 overflow-hidden">
      <div className="flex animate-scroll">
        {/* Duplicate for seamless loop */}
        {[...topMarkets, ...topMarkets].map((market, index) => (
          <div
            key={`${market.id}-${index}`}
            onClick={() => handleTickerClick(market.symbol)}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                handleTickerClick(market.symbol);
              }
            }}
            role="button"
            tabIndex={0}
            className="flex items-center space-x-3 px-4 py-2 cursor-pointer hover:bg-gray-800 transition-colors min-w-max"
          >
            {/* Symbol */}
            <span className="text-white font-medium text-sm">
              {market.symbol}
            </span>
            
            {/* Price */}
            <span className="text-gray-300 font-mono text-sm">
              ${market.price?.toFixed(2) || '0.00'}
            </span>
            
            {/* Change */}
            <span className="text-green-400 text-sm">
              +1.23%
            </span>
            
            {/* Separator */}
            <div className="w-px h-4 bg-gray-600"></div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default DexTickerTape;