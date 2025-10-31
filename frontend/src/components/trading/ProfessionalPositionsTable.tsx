import React, { useState, useMemo } from 'react';
import { ChevronDown, ChevronUp, TrendingUp, TrendingDown, Settings } from 'lucide-react';

/**
 * Professional Positions Table inspired by Drift.trade
 * Features:
 * - Real-time PnL updates
 * - Position management actions
 * - Risk indicators
 * - Responsive design
 * - Professional styling
 */

interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  leverage: number;
  margin: number;
  unrealizedPnl: number;
  unrealizedPnlPercent: number;
  liquidationPrice: number;
  timestamp: number;
}

interface ProfessionalPositionsTableProps {
  className?: string;
}

const ProfessionalPositionsTable: React.FC<ProfessionalPositionsTableProps> = ({
  className = ''
}) => {
  const [expandedPosition, setExpandedPosition] = useState<string | null>(null);
  const [sortBy, setSortBy] = useState<keyof Position>('timestamp');
  const [sortOrder, setSortOrder] = useState<'asc' | 'desc'>('desc');

  // Mock positions data (replace with real data from context)
  const mockPositions: Position[] = [
    {
      id: '1',
      symbol: 'BTC-PERP',
      side: 'long',
      size: 0.5,
      entryPrice: 45000,
      currentPrice: 46500,
      leverage: 5,
      margin: 4500,
      unrealizedPnl: 750,
      unrealizedPnlPercent: 16.67,
      liquidationPrice: 36000,
      timestamp: Date.now() - 3600000 // 1 hour ago
    },
    {
      id: '2',
      symbol: 'ETH-PERP',
      side: 'short',
      size: 2.0,
      entryPrice: 3200,
      currentPrice: 3100,
      leverage: 3,
      margin: 2133.33,
      unrealizedPnl: 200,
      unrealizedPnlPercent: 9.38,
      liquidationPrice: 3840,
      timestamp: Date.now() - 7200000 // 2 hours ago
    }
  ];

  // Use mock data for now, replace with real positions
  const displayPositions = mockPositions;

  // Sort positions
  const sortedPositions = useMemo(() => {
    return [...displayPositions].sort((a, b) => {
      const aValue = a[sortBy];
      const bValue = b[sortBy];
      
      if (typeof aValue === 'number' && typeof bValue === 'number') {
        return sortOrder === 'asc' ? aValue - bValue : bValue - aValue;
      }
      
      if (typeof aValue === 'string' && typeof bValue === 'string') {
        return sortOrder === 'asc' 
          ? aValue.localeCompare(bValue)
          : bValue.localeCompare(aValue);
      }
      
      return 0;
    });
  }, [displayPositions, sortBy, sortOrder]);

  // Calculate total unrealized PnL
  const totalUnrealizedPnl = useMemo(() => {
    return sortedPositions.reduce((sum, position) => sum + position.unrealizedPnl, 0);
  }, [sortedPositions]);

  // Handle sort
  const handleSort = (column: keyof Position) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('desc');
    }
  };

  // Handle position close
  const handleClosePosition = (positionId: string) => {
    console.log('Closing position:', positionId);
    // TODO: Implement position closing
  };

  // Handle position partial close
  const handlePartialClose = (positionId: string, percentage: number) => {
    console.log('Partially closing position:', positionId, percentage);
    // TODO: Implement partial position closing
  };

  // Format timestamp
  const formatTimestamp = (timestamp: number) => {
    const now = Date.now();
    const diff = now - timestamp;
    const hours = Math.floor(diff / 3600000);
    const minutes = Math.floor((diff % 3600000) / 60000);
    
    if (hours > 0) {
      return `${hours}h ${minutes}m ago`;
    }
    return `${minutes}m ago`;
  };

  // Get risk color based on liquidation distance
  const getRiskColor = (liquidationPrice: number, currentPrice: number, side: 'long' | 'short') => {
    const distance = side === 'long' 
      ? (currentPrice - liquidationPrice) / currentPrice
      : (liquidationPrice - currentPrice) / currentPrice;
    
    if (distance < 0.1) return 'text-red-400';
    if (distance < 0.2) return 'text-yellow-400';
    return 'text-green-400';
  };

  return (
    <div className={`bg-gray-900 border border-gray-700 rounded-lg ${className}`}>
      {/* Header */}
      <div className="p-4 border-b border-gray-700">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-white">Positions</h3>
            <p className="text-sm text-gray-400">
              {sortedPositions.length} position{sortedPositions.length !== 1 ? 's' : ''}
            </p>
          </div>
          <div className="text-right">
            <p className="text-sm text-gray-400">Total Unrealized PnL</p>
            <p className={`text-lg font-semibold ${
              totalUnrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'
            }`}>
              ${totalUnrealizedPnl.toFixed(2)}
            </p>
          </div>
        </div>
      </div>

      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th 
                className="text-left p-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('symbol')}
              >
                <div className="flex items-center">
                  Market
                  {sortBy === 'symbol' && (
                    sortOrder === 'asc' ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />
                  )}
                </div>
              </th>
              <th 
                className="text-left p-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('side')}
              >
                <div className="flex items-center">
                  Side
                  {sortBy === 'side' && (
                    sortOrder === 'asc' ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />
                  )}
                </div>
              </th>
              <th 
                className="text-right p-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('size')}
              >
                <div className="flex items-center justify-end">
                  Size
                  {sortBy === 'size' && (
                    sortOrder === 'asc' ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />
                  )}
                </div>
              </th>
              <th 
                className="text-right p-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('entryPrice')}
              >
                <div className="flex items-center justify-end">
                  Entry Price
                  {sortBy === 'entryPrice' && (
                    sortOrder === 'asc' ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />
                  )}
                </div>
              </th>
              <th 
                className="text-right p-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white transition-colors"
                onClick={() => handleSort('unrealizedPnl')}
              >
                <div className="flex items-center justify-end">
                  Unrealized PnL
                  {sortBy === 'unrealizedPnl' && (
                    sortOrder === 'asc' ? <ChevronUp className="w-4 h-4 ml-1" /> : <ChevronDown className="w-4 h-4 ml-1" />
                  )}
                </div>
              </th>
              <th className="text-right p-4 text-sm font-medium text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortedPositions.map((position) => (
              <React.Fragment key={position.id}>
                <tr className="border-b border-gray-800 hover:bg-gray-800/50 transition-colors">
                  <td className="p-4">
                    <div className="flex items-center">
                      <span className="font-semibold text-white">{position.symbol}</span>
                      <span className="ml-2 text-xs text-gray-400">
                        {formatTimestamp(position.timestamp)}
                      </span>
                    </div>
                  </td>
                  <td className="p-4">
                    <div className={`flex items-center ${
                      position.side === 'long' ? 'text-green-400' : 'text-red-400'
                    }`}>
                      {position.side === 'long' ? (
                        <TrendingUp className="w-4 h-4 mr-1" />
                      ) : (
                        <TrendingDown className="w-4 h-4 mr-1" />
                      )}
                      <span className="font-semibold capitalize">{position.side}</span>
                    </div>
                  </td>
                  <td className="p-4 text-right">
                    <div>
                      <div className="text-white font-semibold">{position.size}</div>
                      <div className="text-xs text-gray-400">{position.leverage}x</div>
                    </div>
                  </td>
                  <td className="p-4 text-right">
                    <div>
                      <div className="text-white font-semibold">${position.entryPrice.toFixed(2)}</div>
                      <div className="text-xs text-gray-400">${position.currentPrice.toFixed(2)}</div>
                    </div>
                  </td>
                  <td className="p-4 text-right">
                    <div>
                      <div className={`font-semibold ${
                        position.unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        ${position.unrealizedPnl.toFixed(2)}
                      </div>
                      <div className={`text-xs ${
                        position.unrealizedPnlPercent >= 0 ? 'text-green-400' : 'text-red-400'
                      }`}>
                        {position.unrealizedPnlPercent >= 0 ? '+' : ''}{position.unrealizedPnlPercent.toFixed(2)}%
                      </div>
                    </div>
                  </td>
                  <td className="p-4 text-right">
                    <div className="flex items-center justify-end space-x-2">
                      <button
                        onClick={() => setExpandedPosition(
                          expandedPosition === position.id ? null : position.id
                        )}
                        className="p-1 text-gray-400 hover:text-white transition-colors"
                      >
                        <Settings className="w-4 h-4" />
                      </button>
                      <button
                        onClick={() => handleClosePosition(position.id)}
                        className="px-3 py-1 text-xs bg-red-600 hover:bg-red-700 text-white rounded transition-colors"
                      >
                        Close
                      </button>
                    </div>
                  </td>
                </tr>
                
                {/* Expanded Row */}
                {expandedPosition === position.id && (
                  <tr className="border-b border-gray-800 bg-gray-800/30">
                    <td colSpan={6} className="p-4">
                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <p className="text-gray-400">Margin</p>
                          <p className="text-white font-semibold">${position.margin.toFixed(2)}</p>
                        </div>
                        <div>
                          <p className="text-gray-400">Liquidation Price</p>
                          <p className={`font-semibold ${getRiskColor(
                            position.liquidationPrice, 
                            position.currentPrice, 
                            position.side
                          )}`}>
                            ${position.liquidationPrice.toFixed(2)}
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-400">Risk Level</p>
                          <p className={`font-semibold ${getRiskColor(
                            position.liquidationPrice, 
                            position.currentPrice, 
                            position.side
                          )}`}>
                            {position.liquidationPrice > position.currentPrice * 0.9 ? 'High' : 
                             position.liquidationPrice > position.currentPrice * 0.8 ? 'Medium' : 'Low'}
                          </p>
                        </div>
                        <div>
                          <p className="text-gray-400">Partial Close</p>
                          <div className="flex space-x-1">
                            {[25, 50, 75].map(percentage => (
                              <button
                                key={percentage}
                                onClick={() => handlePartialClose(position.id, percentage)}
                                className="px-2 py-1 text-xs bg-gray-600 hover:bg-gray-500 text-white rounded transition-colors"
                              >
                                {percentage}%
                              </button>
                            ))}
                          </div>
                        </div>
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>

      {/* Empty State */}
      {sortedPositions.length === 0 && (
        <div className="p-8 text-center">
          <div className="text-gray-400 mb-2">
            <TrendingUp className="w-12 h-12 mx-auto mb-4 opacity-50" />
            <p className="text-lg font-semibold">No Open Positions</p>
            <p className="text-sm">Start trading to see your positions here</p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProfessionalPositionsTable;
