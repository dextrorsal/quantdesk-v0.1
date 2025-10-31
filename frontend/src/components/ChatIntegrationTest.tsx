import React from 'react';
import ChatWindow from './ChatWindow';
import { useMarkets } from '../contexts/MarketContext';

/**
 * Chat Integration Test Component
 * Demonstrates the seamless integration between chat and trading
 */
const ChatIntegrationTest: React.FC = () => {
  const { markets, selectedMarket } = useMarkets();

  return (
    <div className="p-6 text-white">
      <h1 className="text-3xl font-bold mb-6">💬 Chat Integration Test</h1>
      
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chat Window */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-4">💬 Global Chat</h2>
          <div className="h-96">
            <ChatWindow channelId="global" />
          </div>
          <div className="mt-4 p-3 bg-gray-700 rounded">
            <p className="text-sm text-gray-300">
              <strong>Try typing:</strong> "BTC is pumping! Check out $ETH and SOL-PERP. Also, don't forget about FARTCOIN!"
            </p>
            <p className="text-xs text-gray-400 mt-2">
              Tickers will be clickable and navigate to charts automatically.
            </p>
          </div>
        </div>

        {/* Market Info */}
        <div className="bg-gray-800 rounded-lg p-4">
          <h2 className="text-xl font-semibold mb-4">📊 Market Integration</h2>
          
          <div className="mb-4">
            <h3 className="text-lg font-medium mb-2">Selected Market:</h3>
            {selectedMarket ? (
              <div className="panel-blue p-4 rounded-lg border border-primary-500">
                <p className="text-lg font-semibold">{selectedMarket.symbol}</p>
                <p>Base Asset: {selectedMarket.baseAsset}</p>
                <p>Category: {selectedMarket.category}</p>
                <p>Current Price: ${selectedMarket.price?.toFixed(2) || 'N/A'}</p>
              </div>
            ) : (
              <p className="text-gray-400">No market selected. Click a ticker in chat!</p>
            )}
          </div>

          <div className="mb-4">
            <h3 className="text-lg font-medium mb-2">Available Markets:</h3>
            <div className="grid grid-cols-2 gap-2">
              {markets.slice(0, 8).map((market) => (
                <div key={market.id} className="text-sm p-2 bg-gray-700 rounded">
                  <span className="font-semibold">{market.symbol}</span>
                  <span className="text-gray-400 ml-2">{market.baseAsset}</span>
                </div>
              ))}
            </div>
          </div>

          <div className="p-3 bg-green-900 rounded border border-green-500">
            <h4 className="font-semibold text-green-300 mb-2">✅ Integration Features:</h4>
            <ul className="text-sm text-green-200 space-y-1">
              <li>• Clickable tickers in chat messages</li>
              <li>• Automatic navigation to trading interface</li>
              <li>• Real-time market selection</li>
              <li>• Seamless user experience</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Demo Instructions */}
      <div className="mt-6 p-4 bg-blue-900 rounded-lg border border-blue-500">
        <h3 className="text-lg font-semibold text-blue-300 mb-2">🎬 Demo Instructions:</h3>
        <ol className="text-blue-200 space-y-2">
          <li>1. <strong>Type a message</strong> with ticker symbols like "$BTC", "ETH-PERP", or "SOL"</li>
          <li>2. <strong>Click on any ticker</strong> in the chat message</li>
          <li>3. <strong>Watch the magic!</strong> You'll be navigated to the trading interface</li>
          <li>4. <strong>See the selected market</strong> update in real-time</li>
        </ol>
      </div>
    </div>
  );
};

export default ChatIntegrationTest;
