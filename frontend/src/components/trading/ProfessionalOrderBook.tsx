import React, { useState, useEffect, useMemo, useRef } from 'react';
import { useWebSocket } from '../../hooks/useWebSocket';

/**
 * Professional Order Book inspired by Drift.trade
 * Features:
 * - Real-time order book updates
 * - Depth visualization
 * - Click-to-trade functionality
 * - Professional styling
 * - Responsive design
 */

interface OrderBookEntry {
  price: number;
  size: number;
  total: number;
}

// Terminal styling for order book

interface ProfessionalOrderBookProps {
  symbol: string;
  currentPrice: number;
  onPriceClick?: (price: number, side: 'buy' | 'sell') => void;
  className?: string;
}

const ProfessionalOrderBook: React.FC<ProfessionalOrderBookProps> = ({
  symbol,
  currentPrice,
  onPriceClick,
  className = ''
}) => {
  const [bids, setBids] = useState<OrderBookEntry[]>([]);
  const [asks, setAsks] = useState<OrderBookEntry[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const wsService = useWebSocket();
  const timerRef = useRef<NodeJS.Timeout>();

  // Generate mock order book data
  const generateMockOrderBook = () => {
    const basePrice = currentPrice || 50000;
    const newBids: OrderBookEntry[] = [];
    const newAsks: OrderBookEntry[] = [];
    
    let bidTotal = 0;
    let askTotal = 0;
    
    // Generate bids (decreasing prices)
    for (let i = 0; i < 15; i++) {
      const price = basePrice - (i + 1) * 0.5;
      const size = Math.random() * 5 + 0.1;
      bidTotal += size;
      newBids.push({ price, size, total: bidTotal });
    }
    
    // Generate asks (increasing prices)
    for (let i = 0; i < 15; i++) {
      const price = basePrice + (i + 1) * 0.5;
      const size = Math.random() * 5 + 0.1;
      askTotal += size;
      newAsks.push({ price, size, total: askTotal });
    }
    
    setBids(newBids);
    setAsks(newAsks);
    setIsLoading(false);
  };

  // Fetch real order book data
  const fetchOrderBook = async () => {
    try {
      const apiSymbol = symbol.includes('-PERP') ? symbol : `${symbol}-PERP`;
      const response = await fetch(`/api/markets/${apiSymbol}/orderbook`);
      
      if (response.status === 429) return; // Rate limited
      if (!response.ok) return;
      
      const data = await response.json();
      const orderbook = data?.orderbook;
      
      if (orderbook?.bids && orderbook?.asks) {
        let bidTotal = 0;
        let askTotal = 0;
        
        const processedBids = orderbook.bids.map(([price, size]: [number, number]) => {
          bidTotal += size;
          return { price, size, total: bidTotal };
        });
        
        const processedAsks = orderbook.asks.map(([price, size]: [number, number]) => {
          askTotal += size;
          return { price, size, total: askTotal };
        });
        
        setBids(processedBids);
        setAsks(processedAsks);
        setIsLoading(false);
      }
    } catch (error) {
      console.warn('Failed to fetch order book, using mock data:', error);
      generateMockOrderBook();
    }
  };

  // WebSocket subscription for real-time updates
  useEffect(() => {
    if (!wsService.isConnected) return;

    const subscribeToOrderBook = async () => {
      try {
        const unsubscribe = await wsService.subscribe(
          `orderbook:${symbol}`,
          (message) => {
            if (message.data?.bids && message.data?.asks) {
              let bidTotal = 0;
              let askTotal = 0;
              
              const processedBids = message.data.bids.map(([price, size]: [number, number]) => {
                bidTotal += size;
                return { price, size, total: bidTotal };
              });
              
              const processedAsks = message.data.asks.map(([price, size]: [number, number]) => {
                askTotal += size;
                return { price, size, total: askTotal };
              });
              
              setBids(processedBids);
              setAsks(processedAsks);
              setIsLoading(false);
            }
          }
        );
        
        return unsubscribe;
      } catch (error) {
        console.warn('WebSocket subscription failed, falling back to polling:', error);
        return null;
      }
    };

    const unsubscribe = subscribeToOrderBook();
    
    return () => {
      unsubscribe?.then(unsub => unsub?.());
    };
  }, [wsService.isConnected, symbol]);

  // Fallback polling
  useEffect(() => {
    if (isLoading) {
      fetchOrderBook();
    }
    
    const jitter = Math.random() * 200; // 0-200ms jitter
    timerRef.current = setTimeout(fetchOrderBook, 1000 + jitter);
    
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
    };
  }, [symbol, currentPrice, isLoading]);

  // Calculate spread
  const spread = useMemo(() => {
    if (bids.length === 0 || asks.length === 0) return 0;
    return asks[0].price - bids[0].price;
  }, [bids, asks]);

  const spreadPercent = useMemo(() => {
    if (spread === 0 || !currentPrice) return 0;
    return (spread / currentPrice) * 100;
  }, [spread, currentPrice]);

  // Calculate max size for depth visualization
  const maxBidSize = useMemo(() => 
    bids.length > 0 ? Math.max(...bids.map(b => b.total)) : 1, [bids]
  );
  
  const maxAskSize = useMemo(() => 
    asks.length > 0 ? Math.max(...asks.map(a => a.total)) : 1, [asks]
  );

  // Handle price click
  const handlePriceClick = (price: number, side: 'buy' | 'sell') => {
    onPriceClick?.(price, side);
  };

  // Format number
  const formatNumber = (num: number, decimals: number = 2) => {
    return num.toFixed(decimals);
  };

  // Format size
  const formatSize = (size: number) => {
    if (size >= 1000) {
      return `${(size / 1000).toFixed(1)}k`;
    }
    return size.toFixed(3);
  };

  return (
    <div className={`bg-gray-900 border border-gray-700 ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <h3 className="text-lg font-semibold text-white">Order Book</h3>
          <div className="text-right">
            <p className="text-sm text-gray-400">Spread</p>
            <p className="text-sm font-semibold text-white">
              ${formatNumber(spread)} ({formatNumber(spreadPercent, 3)}%)
            </p>
          </div>
        </div>
      </div>

      {/* Order Book Table */}
      <div className="overflow-hidden">
        {/* Header */}
        <div className="grid grid-cols-3 gap-4 p-3 text-xs font-medium text-gray-400 border-b border-gray-800">
          <div className="text-right">Price</div>
          <div className="text-right">Size</div>
          <div className="text-right">Total</div>
        </div>

        {/* Asks (Sell Orders) */}
        <div className="max-h-64 overflow-y-auto">
          {asks.slice().reverse().map((ask) => (
            <div
              key={`ask-${ask.price}`}
              className="grid grid-cols-3 gap-4 p-2 hover:bg-gray-800/50 transition-colors cursor-pointer group orderbook-ask"
              onClick={() => handlePriceClick(ask.price, 'sell')}
            >
              <div className="text-right text-red-400 font-mono text-sm group-hover:text-red-300">
                ${formatNumber(ask.price)}
              </div>
              <div className="text-right text-gray-300 font-mono text-sm">
                {formatSize(ask.size)}
              </div>
              <div className="text-right text-gray-400 font-mono text-sm relative">
                {formatSize(ask.total)}
                {/* Depth visualization */}
                <div 
                  className="absolute inset-0 bg-red-500/10 pointer-events-none"
                  style={{ width: `${(ask.total / maxAskSize) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>

        {/* Current Price */}
        <div className="p-3 border-y border-gray-800 bg-gray-800/30">
          <div className="text-center">
            <p className="text-sm text-gray-400">Current Price</p>
            <p className="text-lg font-semibold text-white">
              ${formatNumber(currentPrice)}
            </p>
          </div>
        </div>

        {/* Bids (Buy Orders) */}
        <div className="max-h-64 overflow-y-auto">
          {bids.map((bid) => (
            <div
              key={`bid-${bid.price}`}
              className="grid grid-cols-3 gap-4 p-2 hover:bg-gray-800/50 transition-colors cursor-pointer group orderbook-bid"
              onClick={() => handlePriceClick(bid.price, 'buy')}
            >
              <div className="text-right text-green-400 font-mono text-sm group-hover:text-green-300">
                ${formatNumber(bid.price)}
              </div>
              <div className="text-right text-gray-300 font-mono text-sm">
                {formatSize(bid.size)}
              </div>
              <div className="text-right text-gray-400 font-mono text-sm relative">
                {formatSize(bid.total)}
                {/* Depth visualization */}
                <div 
                  className="absolute inset-0 bg-green-500/10 pointer-events-none"
                  style={{ width: `${(bid.total / maxBidSize) * 100}%` }}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Loading State */}
      {isLoading && (
        <div className="p-8 text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4" />
          <p className="text-gray-400 text-sm">Loading order book...</p>
        </div>
      )}

      {/* Empty State */}
      {!isLoading && bids.length === 0 && asks.length === 0 && (
        <div className="p-8 text-center">
          <p className="text-gray-400 text-sm">No order book data available</p>
        </div>
      )}

      {/* Footer */}
      <div className="p-3 border-t border-gray-700 text-xs text-gray-400">
        <p>Click on prices to set order price</p>
      </div>
    </div>
  );
};

export default ProfessionalOrderBook;
