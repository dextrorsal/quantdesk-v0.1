import React, { useState, useEffect } from 'react'
import { usePrice } from '../contexts/PriceContext'

interface Position {
  id: string;
  marketId: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  margin: number;
  leverage: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  liquidationPrice: number;
  healthFactor: number;
  marginRatio: number;
  isLiquidated: boolean;
  createdAt: string;
  updatedAt: string;
}

const Positions: React.FC = () => {
  const [positions, setPositions] = useState<Position[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const { getPrice } = usePrice();
  
  // Fetch positions from backend
  useEffect(() => {
    const fetchPositions = async () => {
      try {
        setError(null);
        const response = await fetch('/api/positions', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.positions) {
            setPositions(data.positions);
          } else {
            setError('Failed to load positions data');
            setPositions([]);
          }
        } else {
          const errorData = await response.json().catch(() => ({}));
          setError(errorData.error || `Failed to fetch positions: ${response.statusText}`);
          setPositions([]);
        }
      } catch (error) {
        console.error('Error fetching positions:', error);
        setError('Network error: Unable to fetch positions');
        setPositions([]);
      } finally {
        setLoading(false);
      }
    };

    fetchPositions();

    // 🚀 NEW: Listen for real-time position updates
    const handlePositionUpdate = (event: CustomEvent) => {
      const update = event.detail;
      console.log('📈 Received position update:', update);
      
      if (update.status === 'closed') {
        // Remove closed position from list
        setPositions(prev => prev.filter(p => p.id !== update.positionId));
      } else {
        // Refresh positions for other updates
        fetchPositions();
      }
    };

    window.addEventListener('positionStatusUpdate', handlePositionUpdate as EventListener);

    return () => {
      window.removeEventListener('positionStatusUpdate', handlePositionUpdate as EventListener);
    };
  }, []);

  const handleClosePosition = async (positionId: string) => {
    try {
      const response = await fetch(`/api/positions/${positionId}/close`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${localStorage.getItem('token')}`
        }
      });

      if (response.ok) {
        // Refresh positions after closing
        const updatedPositions = positions.filter(p => p.id !== positionId);
        setPositions(updatedPositions);
        console.log('Position closed successfully');
      } else {
        console.error('Failed to close position:', response.statusText);
      }
    } catch (error) {
      console.error('Error closing position:', error);
    }
  };

  if (loading) {
    return (
      <div className="h-full bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <div className="text-gray-400">Loading positions...</div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="h-full bg-gray-900 flex items-center justify-center">
        <div className="text-center">
          <div className="text-red-400 mb-2">⚠️ Error Loading Positions</div>
          <div className="text-gray-400 text-sm mb-4">{error}</div>
          <button 
            onClick={() => window.location.reload()}
            className="px-4 py-2 bg-blue-600 text-white rounded hover:bg-blue-700 transition-colors"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="h-full bg-gray-900 flex flex-col">
      <div className="flex-1 flex items-center justify-center">
        {positions.length > 0 ? (
          <div className="overflow-x-auto w-full">
            <table className="min-w-full text-xs terminal-table">
              <thead className="text-gray-400">
                <tr className="text-left">
                  <th className="py-2 px-4 font-mono">Market</th>
                  <th className="py-2 px-4 font-mono">Side</th>
                  <th className="py-2 px-4 text-right font-mono">Size</th>
                  <th className="py-2 px-4 text-right font-mono">Entry</th>
                  <th className="py-2 px-4 text-right font-mono">Mark</th>
                  <th className="py-2 px-4 text-right font-mono">Unrealized PnL</th>
                  <th className="py-2 px-4 text-right">Margin</th>
                  <th className="py-2 px-4 text-right">Leverage</th>
                  <th className="py-2 px-4 text-right">Health</th>
                  <th className="py-2 px-4 text-right">Actions</th>
                </tr>
              </thead>
              <tbody>
                {positions.map((position) => (
                  <tr key={position.id} className="border-t border-gray-800 hover:bg-gray-800/50">
                    <td className="py-2 px-4 text-white font-medium">{position.symbol}</td>
                    <td className="py-2 px-4">
                      <span className={`px-2 py-1 rounded text-xs font-medium ${
                        position.side === 'long' 
                          ? 'bg-green-900/30 text-green-400 border border-green-500/30' 
                          : 'bg-red-900/30 text-red-400 border border-red-500/30'
                      }`}>
                        {position.side === 'long' ? 'Long' : 'Short'}
                      </span>
                    </td>
                    <td className="py-2 px-4 text-right text-white font-mono">{position.size.toFixed(3)}</td>
                    <td className="py-2 px-4 text-right text-white font-mono">${position.entryPrice.toFixed(2)}</td>
                    <td className="py-2 px-4 text-right text-white font-mono">${position.currentPrice.toFixed(2)}</td>
                    <td className="py-2 px-4 text-right">
                      <div className="flex flex-col items-end">
                        <span className={`font-mono font-medium ${
                          position.unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'
                        }`}>
                          {position.unrealizedPnl >= 0 ? '+' : ''}${position.unrealizedPnl.toFixed(2)}
                        </span>
                        <span className={`text-xs ${
                          position.unrealizedPnlPercent >= 0 ? 'text-green-500' : 'text-red-500'
                        }`}>
                          {position.unrealizedPnlPercent >= 0 ? '+' : ''}{position.unrealizedPnlPercent.toFixed(2)}%
                        </span>
                      </div>
                    </td>
                    <td className="py-2 px-4 text-right text-white font-mono">${position.margin.toFixed(2)}</td>
                    <td className="py-2 px-4 text-right">
                      <span className="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs font-medium">
                        {position.leverage}x
                      </span>
                    </td>
                    <td className="py-2 px-4 text-right">
                      <div className="flex flex-col items-end">
                        <span className={`text-xs font-medium ${
                          position.healthFactor > 0.8 ? 'text-green-400' :
                          position.healthFactor > 0.5 ? 'text-yellow-400' : 'text-red-400'
                        }`}>
                          {(position.healthFactor * 100).toFixed(1)}%
                        </span>
                        <span className="text-xs text-gray-500">
                          Liq: ${position.liquidationPrice.toFixed(2)}
                        </span>
                      </div>
                    </td>
                    <td className="py-2 px-4 text-right">
                      <div className="flex gap-1">
                        <button 
                          className="px-2 py-1 bg-gray-700 text-gray-300 rounded text-xs hover:bg-gray-600 transition-colors"
                          onClick={() => console.log('TP/SL not implemented yet')}
                          title="Take Profit / Stop Loss"
                        >
                          TP/SL
                        </button>
                        <button 
                          className="px-2 py-1 bg-red-600 text-white rounded text-xs hover:bg-red-700 transition-colors"
                          onClick={() => handleClosePosition(position.id)}
                          title="Close Position"
                        >
                          Close
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <div className="text-center text-gray-400">
            <div className="text-sm mb-2">No open positions yet</div>
            <div className="text-xs">Start trading to see your positions here</div>
          </div>
        )}
      </div>
    </div>
  )
}

export default Positions
