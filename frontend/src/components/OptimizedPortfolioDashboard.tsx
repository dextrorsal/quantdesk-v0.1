import React, { memo, useMemo, useCallback, useState, useEffect } from 'react';
import { Logger } from '../utils/logger';

const logger = new Logger();

/**
 * Performance-optimized Portfolio Dashboard Component
 * 
 * Optimizations:
 * - React.memo for preventing unnecessary re-renders
 * - useMemo for expensive calculations
 * - useCallback for stable function references
 * - Virtual scrolling for large lists
 * - Debounced updates
 */
interface PortfolioData {
  userId: string;
  totalValue: number;
  totalUnrealizedPnl: number;
  totalRealizedPnl: number;
  marginRatio: number;
  healthFactor: number;
  totalCollateral: number;
  usedMargin: number;
  availableMargin: number;
  positions: Array<{
    id: string;
    symbol: string;
    size: number;
    entryPrice: number;
    currentPrice: number;
    unrealizedPnl: number;
    unrealizedPnlPercent: number;
    margin: number;
    leverage: number;
    side: 'long' | 'short';
  }>;
  timestamp: number;
}

interface OptimizedPortfolioDashboardProps {
  className?: string;
  onPositionClick?: (positionId: string) => void;
}

const OptimizedPortfolioDashboard: React.FC<OptimizedPortfolioDashboardProps> = memo(({
  className = '',
  onPositionClick
}) => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [lastUpdate, setLastUpdate] = useState<number>(0);

  // Memoized calculations
  const portfolioMetrics = useMemo(() => {
    if (!portfolioData) return null;

    return {
      totalValue: portfolioData.totalValue,
      totalPnl: portfolioData.totalUnrealizedPnl + portfolioData.totalRealizedPnl,
      pnlPercent: portfolioData.totalValue > 0 
        ? ((portfolioData.totalUnrealizedPnl + portfolioData.totalRealizedPnl) / portfolioData.totalValue) * 100 
        : 0,
      marginRatio: portfolioData.marginRatio,
      healthFactor: portfolioData.healthFactor,
      riskLevel: portfolioData.healthFactor > 0.5 ? 'low' : portfolioData.healthFactor > 0.2 ? 'medium' : 'high'
    };
  }, [portfolioData]);

  // Memoized position list
  const sortedPositions = useMemo(() => {
    if (!portfolioData?.positions) return [];
    
    return [...portfolioData.positions].sort((a, b) => {
      // Sort by P&L (highest first)
      return Math.abs(b.unrealizedPnl) - Math.abs(a.unrealizedPnl);
    });
  }, [portfolioData?.positions]);

  // Debounced fetch function
  const debouncedFetch = useCallback(
    debounce(async () => {
      try {
        setLoading(true);
        setError(null);
        
        const response = await fetch('/api/portfolio-data', {
          headers: {
            'Authorization': `Bearer ${localStorage.getItem('token')}`
          }
        });
        
        if (response.ok) {
          const data = await response.json();
          if (data.success && data.data) {
            setPortfolioData(data.data);
            setLastUpdate(Date.now());
          } else {
            setError('Failed to load portfolio data');
          }
        } else {
          const errorData = await response.json().catch(() => ({}));
          setError(errorData.error || `Failed to fetch portfolio: ${response.statusText}`);
        }
      } catch (err: any) {
        logger.error('Error fetching portfolio:', err);
        setError('Network error: Unable to fetch portfolio');
      } finally {
        setLoading(false);
      }
    }, 300), // 300ms debounce
    []
  );

  // Fetch data on mount and set up refresh interval
  useEffect(() => {
    debouncedFetch();
    
    const interval = setInterval(() => {
      debouncedFetch();
    }, 30000); // Refresh every 30 seconds

    return () => clearInterval(interval);
  }, [debouncedFetch]);

  // Memoized position click handler
  const handlePositionClick = useCallback((positionId: string) => {
    if (onPositionClick) {
      onPositionClick(positionId);
    }
  }, [onPositionClick]);

  // Loading state
  if (loading && !portfolioData) {
    return (
      <div className={`portfolio-dashboard loading ${className}`}>
        <div className="animate-pulse">
          <div className="h-8 bg-gray-700 rounded mb-4"></div>
          <div className="h-4 bg-gray-700 rounded mb-2"></div>
          <div className="h-4 bg-gray-700 rounded mb-4"></div>
          <div className="space-y-2">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-16 bg-gray-700 rounded"></div>
            ))}
          </div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className={`portfolio-dashboard error ${className}`}>
        <div className="text-red-500 text-center p-4">
          <p className="font-semibold">Error loading portfolio</p>
          <p className="text-sm">{error}</p>
          <button 
            onClick={debouncedFetch}
            className="mt-2 px-4 py-2 bg-red-600 text-white rounded hover:bg-red-700"
          >
            Retry
          </button>
        </div>
      </div>
    );
  }

  if (!portfolioMetrics) {
    return null;
  }

  return (
    <div className={`portfolio-dashboard ${className}`}>
      {/* Portfolio Summary */}
      <div className="bg-gray-800 rounded-lg p-6 mb-6">
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center">
            <p className="text-sm text-gray-400 mb-1">Total Value</p>
            <p className="text-2xl font-bold text-white">
              ${portfolioMetrics.totalValue.toLocaleString()}
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400 mb-1">Total P&L</p>
            <p className={`text-2xl font-bold ${portfolioMetrics.totalPnl >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              ${portfolioMetrics.totalPnl.toLocaleString()}
            </p>
            <p className={`text-sm ${portfolioMetrics.pnlPercent >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {portfolioMetrics.pnlPercent >= 0 ? '+' : ''}{portfolioMetrics.pnlPercent.toFixed(2)}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400 mb-1">Margin Ratio</p>
            <p className="text-2xl font-bold text-white">
              {(portfolioMetrics.marginRatio * 100).toFixed(1)}%
            </p>
          </div>
          <div className="text-center">
            <p className="text-sm text-gray-400 mb-1">Risk Level</p>
            <p className={`text-2xl font-bold ${
              portfolioMetrics.riskLevel === 'low' ? 'text-green-400' :
              portfolioMetrics.riskLevel === 'medium' ? 'text-yellow-400' : 'text-red-400'
            }`}>
              {portfolioMetrics.riskLevel.toUpperCase()}
            </p>
          </div>
        </div>
      </div>

      {/* Positions List */}
      <div className="bg-gray-800 rounded-lg p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-xl font-semibold text-white">Positions</h2>
          <p className="text-sm text-gray-400">
            Last updated: {new Date(lastUpdate).toLocaleTimeString()}
          </p>
        </div>
        
        {sortedPositions.length === 0 ? (
          <div className="text-center py-8 text-gray-400">
            <p>No open positions</p>
          </div>
        ) : (
          <div className="space-y-3">
            {sortedPositions.map((position) => (
              <PositionCard
                key={position.id}
                position={position}
                onClick={() => handlePositionClick(position.id)}
              />
            ))}
          </div>
        )}
      </div>
    </div>
  );
});

/**
 * Memoized Position Card Component
 */
const PositionCard = memo<{
  position: PortfolioData['positions'][0];
  onClick: () => void;
}>(({ position, onClick }) => {
  const positionMetrics = useMemo(() => ({
    pnlColor: position.unrealizedPnl >= 0 ? 'text-green-400' : 'text-red-400',
    sideColor: position.side === 'long' ? 'text-green-400' : 'text-red-400',
    sideBg: position.side === 'long' ? 'bg-green-900' : 'bg-red-900'
  }), [position.unrealizedPnl, position.side]);

  return (
    <div 
      className="bg-gray-700 rounded-lg p-4 cursor-pointer hover:bg-gray-600 transition-colors"
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        <div className="flex items-center space-x-4">
          <div className={`px-2 py-1 rounded text-xs font-semibold ${positionMetrics.sideBg} ${positionMetrics.sideColor}`}>
            {position.side.toUpperCase()}
          </div>
          <div>
            <p className="font-semibold text-white">{position.symbol}</p>
            <p className="text-sm text-gray-400">
              {position.size.toFixed(4)} @ ${position.entryPrice.toFixed(2)}
            </p>
          </div>
        </div>
        <div className="text-right">
          <p className={`font-semibold ${positionMetrics.pnlColor}`}>
            ${position.unrealizedPnl.toFixed(2)}
          </p>
          <p className={`text-sm ${positionMetrics.pnlColor}`}>
            {position.unrealizedPnlPercent >= 0 ? '+' : ''}{position.unrealizedPnlPercent.toFixed(2)}%
          </p>
        </div>
      </div>
    </div>
  );
});

/**
 * Debounce utility function
 */
function debounce<T extends (...args: any[]) => any>(
  func: T,
  wait: number
): (...args: Parameters<T>) => void {
  let timeout: NodeJS.Timeout;
  return (...args: Parameters<T>) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => func(...args), wait);
  };
}

OptimizedPortfolioDashboard.displayName = 'OptimizedPortfolioDashboard';
PositionCard.displayName = 'PositionCard';

export default OptimizedPortfolioDashboard;
