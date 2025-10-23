import React from 'react';
import { useMarkets } from '../contexts/MarketContext';
import { useTickerClick } from '../hooks/useTickerClick';
import ClickableTicker, { TickerText } from '../components/ClickableTicker';

/**
 * Test component to verify end-to-end ticker click flow
 * This component demonstrates the seamless integration
 */
const TickerClickTest: React.FC = () => {
  const { markets, selectedMarket, selectMarketBySymbol } = useMarkets();
  const { handleTickerClick, parseTickerFromText } = useTickerClick();

  const testMessage = "BTC is pumping! Check out SOL and ETH too! FARTCOIN is also trending.";

  return (
    <div className="p-6 bg-gray-900 text-white">
      <h2 className="text-2xl font-bold mb-4">🧪 Ticker Click Integration Test</h2>
      
      {/* Current Market Status */}
      <div className="mb-6 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">📊 Current Market Status</h3>
        <p><strong>Selected Market:</strong> {selectedMarket?.symbol || 'None'}</p>
        <p><strong>Total Markets:</strong> {markets.length}</p>
        <p><strong>Available Symbols:</strong> {markets.slice(0, 5).map(m => m.baseAsset).join(', ')}...</p>
      </div>

      {/* Test Clickable Tickers */}
      <div className="mb-6 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">🖱️ Clickable Tickers</h3>
        <div className="space-x-4">
          <ClickableTicker symbol="BTC">BTC</ClickableTicker>
          <ClickableTicker symbol="ETH">ETH</ClickableTicker>
          <ClickableTicker symbol="SOL">SOL</ClickableTicker>
          <ClickableTicker symbol="FARTCOIN">FARTCOIN</ClickableTicker>
        </div>
        <p className="text-sm text-gray-400 mt-2">Click any ticker above to test navigation</p>
      </div>

      {/* Test Ticker Text Parsing */}
      <div className="mb-6 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">💬 Chat Message with Tickers</h3>
        <div className="bg-gray-700 p-3 rounded">
          <TickerText text={testMessage} />
        </div>
        <p className="text-sm text-gray-400 mt-2">Tickers in the message above should be clickable</p>
      </div>

      {/* Market List */}
      <div className="mb-6 p-4 bg-gray-800 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">📈 Available Markets</h3>
        <div className="grid grid-cols-2 gap-2 max-h-40 overflow-y-auto">
          {markets.slice(0, 10).map((market) => (
            <div 
              key={market.id}
              className={`p-2 rounded cursor-pointer transition-colors ${
                selectedMarket?.id === market.id 
                  ? 'bg-blue-600 text-white' 
                  : 'bg-gray-700 hover:bg-gray-600'
              }`}
              onClick={() => selectMarketBySymbol(market.symbol)}
            >
              <div className="font-semibold">{market.symbol}</div>
              <div className="text-sm text-gray-400">${market.currentPrice?.toFixed(2) || '0.00'}</div>
            </div>
          ))}
        </div>
      </div>

      {/* Test Results */}
      <div className="p-4 bg-gray-800 rounded-lg">
        <h3 className="text-lg font-semibold mb-2">✅ Test Results</h3>
        <div className="space-y-2">
          <div className="flex items-center space-x-2">
            <span className="text-green-400">✓</span>
            <span>MarketProvider: {markets.length > 0 ? 'Connected' : 'Failed'}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-green-400">✓</span>
            <span>Ticker Click: {typeof handleTickerClick === 'function' ? 'Ready' : 'Failed'}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-green-400">✓</span>
            <span>Market Selection: {selectedMarket ? 'Working' : 'No selection'}</span>
          </div>
          <div className="flex items-center space-x-2">
            <span className="text-green-400">✓</span>
            <span>Ticker Parsing: {parseTickerFromText(testMessage).length > 0 ? 'Working' : 'Failed'}</span>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TickerClickTest;
