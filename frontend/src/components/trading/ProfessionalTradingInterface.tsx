import React, { useState, useCallback, useMemo } from 'react';
import { useWallet } from '@solana/wallet-adapter-react';
import { useMarkets } from '../../contexts/MarketContext';
import { useAccount } from '../../contexts/AccountContext';
import ProfessionalTradingPanel from './ProfessionalTradingPanel';
import ProfessionalPositionsTable from './ProfessionalPositionsTable';
import ProfessionalOrderBook from './ProfessionalOrderBook';
import DexChart from '../DexChart';
import { BarChart3, List } from 'lucide-react';
import ConditionalOrderForm from './ConditionalOrderForm';
import OrdersStatusPanel from './OrdersStatusPanel';

/**
 * Professional Trading Interface inspired by Drift.trade
 * Features:
 * - Multi-panel layout
 * - Real-time data updates
 * - Professional styling
 * - Responsive design
 * - Advanced trading features
 */

interface ProfessionalTradingInterfaceProps {
  className?: string;
}

const ProfessionalTradingInterface: React.FC<ProfessionalTradingInterfaceProps> = ({
  className = ''
}) => {
  const { connected } = useWallet();
  const { markets, selectedMarket, selectMarketBySymbol } = useMarkets();
  const { accountState } = useAccount();
  
  const [selectedSymbol, setSelectedSymbol] = useState(selectedMarket?.symbol || 'BTC-PERP');
  const [activeTab, setActiveTab] = useState<'trading' | 'positions' | 'orders'>('trading');
  const [showOrderBook, setShowOrderBook] = useState(true);
  const [showChart, setShowChart] = useState(true);

  // Get current market data
  const currentMarket = useMemo(() => 
    markets.find(m => m.symbol === selectedSymbol) || selectedMarket, 
    [markets, selectedSymbol, selectedMarket]
  );

  const currentPrice = currentMarket?.price || 50000;

  // Handle symbol change
  const handleSymbolChange = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
    selectMarketBySymbol(symbol);
  }, [selectMarketBySymbol]);

  // Handle order placement
  const handleOrderPlaced = useCallback((order: any) => {
    console.log('Order placed:', order);
    // TODO: Implement order placement logic
  }, []);

  // Handle price click from order book
  const handlePriceClick = useCallback((price: number, side: 'buy' | 'sell') => {
    console.log('Price clicked:', price, side);
    // TODO: Set price in trading panel
  }, []);

  // Calculate total PnL
  const totalPnL = useMemo(() => {
    // TODO: Calculate from actual positions
    return 1250; // Mock value
  }, []);

  return (
    <div className={`min-h-screen bg-gray-900 text-white ${className}`}>
      {/* Header */}
      <div className="bg-gray-800 border-b border-gray-700 px-6 py-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-6">
            <div>
              <h1 className="text-2xl font-bold">QuantDesk Pro</h1>
              <p className="text-gray-400 text-sm">Professional Perpetual Trading</p>
            </div>
            
            {/* Market Selector */}
            <div className="flex items-center space-x-2">
              <select
                value={selectedSymbol}
                onChange={(e) => handleSymbolChange(e.target.value)}
                className="bg-gray-700 border border-gray-600 px-3 py-2 text-white focus:border-blue-500 focus:outline-none font-mono transition-all"
              >
                {markets.map((market) => (
                  <option key={market.id} value={market.symbol}>
                    {market.symbol}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="flex items-center space-x-6">
            {/* Account Summary */}
            {accountState && (
              <div className="text-right">
                <p className="text-sm text-gray-400">Account Health</p>
                <div className="flex items-center space-x-2">
                  <div className={`w-2 h-2 rounded-full ${
                    accountHealth > 80 ? 'bg-green-500' : 
                    accountHealth > 50 ? 'bg-yellow-500' : 'bg-red-500'
                  }`} />
                  <p className="text-lg font-semibold">{accountHealth.toFixed(1)}%</p>
                </div>
              </div>
            )}
            
            <div className="text-right">
              <p className="text-sm text-gray-400">Total PnL</p>
              <p className={`text-lg font-semibold ${
                totalPnL >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                ${totalPnL.toFixed(2)}
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main Content */}
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

        {/* Main Trading Area */}
        <div className="flex-1 flex flex-col">
          {/* Chart Section */}
          {showChart && (
            <div className="h-96 border-b border-gray-700">
              <div className="h-full">
                <DexChart symbol={selectedSymbol} />
              </div>
            </div>
          )}

          {/* Trading Panels */}
          <div className="flex-1 flex">
            {/* Left Panel - Order Book */}
            {showOrderBook && (
              <div className="w-80 border-r border-gray-700">
                <ProfessionalOrderBook
                  symbol={selectedSymbol}
                  currentPrice={currentPrice}
                  onPriceClick={handlePriceClick}
                  className="h-full"
                />
              </div>
            )}

            {/* Center Panel - Trading Interface */}
            <div className="flex-1 flex flex-col">
              {/* Tab Navigation */}
              <div className="flex border-b border-gray-700">
                <button
                  onClick={() => setActiveTab('trading')}
                  className={`px-6 py-3 font-semibold transition-colors ${
                    activeTab === 'trading'
                      ? 'text-blue-400 border-b-2 border-blue-400'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <BarChart3 className="w-4 h-4 inline mr-2" />
                  Trading
                </button>
                <button
                  onClick={() => setActiveTab('positions')}
                  className={`px-6 py-3 font-semibold transition-colors ${
                    activeTab === 'positions'
                      ? 'text-blue-400 border-b-2 border-blue-400'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <TrendingUp className="w-4 h-4 inline mr-2" />
                  Positions
                </button>
                <button
                  onClick={() => setActiveTab('orders')}
                  className={`px-6 py-3 font-semibold transition-colors ${
                    activeTab === 'orders'
                      ? 'text-blue-400 border-b-2 border-blue-400'
                      : 'text-gray-400 hover:text-white'
                  }`}
                >
                  <List className="w-4 h-4 inline mr-2" />
                  Orders
                </button>
              </div>

              {/* Tab Content */}
              <div className="flex-1 overflow-hidden">
                {activeTab === 'trading' && (
                  <div className="h-full flex">
                    <div className="flex-1 p-6">
                      <ProfessionalTradingPanel
                        symbol={selectedSymbol}
                        currentPrice={currentPrice}
                        onOrderPlaced={handleOrderPlaced}
                        className="h-full"
                      />
                    </div>
                  </div>
                )}
                
                {activeTab === 'positions' && (
                  <div className="h-full p-6">
                    <ProfessionalPositionsTable className="h-full" />
                  </div>
                )}
                
                {activeTab === 'orders' && (
                  <div className="h-full p-6">
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 h-full">
                      <ConditionalOrderForm symbol={selectedSymbol} />
                      <OrdersStatusPanel />
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      </div>

      {/* Connection Status */}
      {!connected && (
        <div className="fixed bottom-4 right-4 bg-red-600 text-white px-4 py-2 rounded-lg shadow-lg">
          <p className="text-sm">Please connect your wallet to start trading</p>
        </div>
      )}
    </div>
  );
};

export default ProfessionalTradingInterface;
