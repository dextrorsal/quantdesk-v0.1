import React, { useState } from 'react';
import { useMarkets } from '../contexts/MarketContext';
import DexChart from './DexChart';

/**
 * DEX-style trading interface
 * Clean, minimal design inspired by Uniswap, DexScreener, etc.
 */
const DexTradingInterface: React.FC = () => {
  const { markets, selectedMarket, selectMarketBySymbol } = useMarkets();
  const [selectedSymbol, setSelectedSymbol] = useState(selectedMarket?.symbol || 'BTC-PERP');
  const [chartType, setChartType] = useState<'line' | 'area'>('area');

  const handleSymbolChange = (symbol: string) => {
    setSelectedSymbol(symbol);
    selectMarketBySymbol(symbol);
  };

  const currentMarket = markets.find(m => m.symbol === selectedSymbol) || selectedMarket;

  return (
    <div className="min-h-screen bg-gray-900 text-white">
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold">QuantDesk DEX</h1>
            <p className="text-gray-400 text-sm">Professional Perpetual Trading</p>
          </div>
          <div className="flex items-center space-x-4">
            <div className="text-right">
              <p className="text-sm text-gray-400">Total Volume</p>
              <p className="text-lg font-semibold">$2.4M</p>
            </div>
            <div className="text-right">
              <p className="text-sm text-gray-400">Active Markets</p>
              <p className="text-lg font-semibold">{markets.length}</p>
            </div>
          </div>
        </div>
      </div>

      <div className="flex h-screen">
        {/* Left Sidebar - Market List */}
        <div className="w-80 bg-gray-800 border-r border-gray-700 overflow-y-auto">
          <div className="p-4">
            <h2 className="text-lg font-semibold mb-4">Markets</h2>
            
            {/* Market Search */}
            <div className="mb-4">
              <input
                type="text"
                placeholder="Search markets..."
                className="w-full px-3 py-2 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
              />
            </div>

            {/* Market List */}
            <div className="space-y-2">
              {markets.map((market) => (
                <div
                  key={market.id}
                  onClick={() => handleSymbolChange(market.symbol)}
                  onKeyPress={(e) => {
                    if (e.key === 'Enter' || e.key === ' ') {
                      handleSymbolChange(market.symbol);
                    }
                  }}
                  role="button"
                  tabIndex={0}
                  className={`p-3 rounded-lg cursor-pointer transition-colors ${
                    selectedSymbol === market.symbol
                      ? 'bg-blue-600 border border-blue-500'
                      : 'bg-gray-700 hover:bg-gray-600 border border-transparent'
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <div>
                      <p className="font-semibold">{market.symbol}</p>
                      <p className="text-sm text-gray-400">{market.baseAsset}</p>
                    </div>
                    <div className="text-right">
                      <p className="font-semibold">${market.price?.toFixed(2) || '0.00'}</p>
                      <p className="text-sm text-green-400">+2.34%</p>
                    </div>
                  </div>
                </div>
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
                  <h2 className="text-xl font-semibold">
                    {currentMarket?.symbol || selectedSymbol}
                  </h2>
                  <div className="flex space-x-2">
                    <button
                      onClick={() => setChartType('area')}
                      className={`px-3 py-1 rounded text-sm ${
                        chartType === 'area'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      Area
                    </button>
                    <button
                      onClick={() => setChartType('line')}
                      className={`px-3 py-1 rounded text-sm ${
                        chartType === 'line'
                          ? 'bg-blue-600 text-white'
                          : 'bg-gray-700 text-gray-300 hover:bg-gray-600'
                      }`}
                    >
                      Line
                    </button>
                  </div>
                </div>
                
                <div className="flex items-center space-x-4">
                  <div className="text-right">
                    <p className="text-sm text-gray-400">24h Change</p>
                    <p className="text-lg font-semibold text-green-400">+$1,234.56</p>
                  </div>
                  <div className="text-right">
                    <p className="text-sm text-gray-400">24h Volume</p>
                    <p className="text-lg font-semibold">$456.7K</p>
                  </div>
                </div>
              </div>

              {/* Chart */}
              <DexChart 
                symbol={selectedSymbol}
                height={400}
                chartType={chartType}
                showVolume={false}
              />
            </div>
          </div>

          {/* Trading Panel */}
          <div className="bg-gray-800 border-t border-gray-700 p-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Order Form */}
              <div className="lg:col-span-2">
                <h3 className="text-lg font-semibold mb-4">Place Order</h3>
                <div className="grid grid-cols-2 gap-4">
                  <div>
                    <label className="block text-sm font-medium mb-2">Side</label>
                    <div className="flex space-x-2">
                      <button className="flex-1 py-2 px-4 bg-green-600 text-white rounded-lg font-semibold">
                        Long
                      </button>
                      <button className="flex-1 py-2 px-4 bg-gray-700 text-gray-300 rounded-lg font-semibold hover:bg-gray-600">
                        Short
                      </button>
                    </div>
                  </div>
                  <div>
                    <label htmlFor="leverage" className="block text-sm font-medium mb-2">Leverage</label>
                    <select 
                      id="leverage"
                      className="w-full py-2 px-3 bg-gray-700 text-white rounded-lg border border-gray-600"
                      aria-label="Select leverage"
                    >
                      <option>1x</option>
                      <option>2x</option>
                      <option>5x</option>
                      <option>10x</option>
                      <option>20x</option>
                    </select>
                  </div>
                  <div>
                    <label htmlFor="size" className="block text-sm font-medium mb-2">Size</label>
                    <input
                      id="size"
                      type="number"
                      placeholder="0.00"
                      className="w-full py-2 px-3 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                  <div>
                    <label htmlFor="price" className="block text-sm font-medium mb-2">Price</label>
                    <input
                      id="price"
                      type="number"
                      placeholder="Market"
                      className="w-full py-2 px-3 bg-gray-700 text-white rounded-lg border border-gray-600 focus:border-blue-500 focus:outline-none"
                    />
                  </div>
                </div>
                <button className="w-full mt-4 py-3 bg-blue-600 text-white rounded-lg font-semibold hover:bg-blue-700 transition-colors">
                  Place Order
                </button>
              </div>

              {/* Market Info */}
              <div>
                <h3 className="text-lg font-semibold mb-4">Market Info</h3>
                <div className="space-y-3">
                  <div className="flex justify-between">
                    <span className="text-gray-400">Symbol</span>
                    <span className="font-semibold">{currentMarket?.symbol}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Base Asset</span>
                    <span className="font-semibold">{currentMarket?.baseAsset}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Quote Asset</span>
                    <span className="font-semibold">{currentMarket?.quoteAsset}</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Max Leverage</span>
                    <span className="font-semibold">{currentMarket?.maxLeverage}x</span>
                  </div>
                  <div className="flex justify-between">
                    <span className="text-gray-400">Category</span>
                    <span className="font-semibold">{currentMarket?.category}</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default DexTradingInterface;
