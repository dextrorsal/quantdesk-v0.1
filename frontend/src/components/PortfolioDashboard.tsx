// Portfolio Dashboard Component for QuantDesk
// Displays comprehensive portfolio overview with P&L analytics and risk metrics
// Updated to use real portfolio data from backend

import React, { useState, useEffect } from 'react'
import { useSpring, animated } from 'react-spring'
import { Logger } from '../utils/logger'

const logger = new Logger()

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

const PortfolioDashboard: React.FC = () => {
  const [portfolioData, setPortfolioData] = useState<PortfolioData | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // React Spring animations for smooth value changes
  const portfolioValueSpring = useSpring({
    number: portfolioData?.totalValue || 0,
    from: { number: 0 },
    config: { tension: 120, friction: 14 }
  })

  const unrealizedPnlSpring = useSpring({
    number: portfolioData?.totalUnrealizedPnl || 0,
    from: { number: 0 },
    config: { tension: 120, friction: 14 }
  })

  const realizedPnlSpring = useSpring({
    number: portfolioData?.totalRealizedPnl || 0,
    from: { number: 0 },
    config: { tension: 120, friction: 14 }
  })

  const marginRatioSpring = useSpring({
    number: portfolioData?.marginRatio || 0,
    from: { number: 0 },
    config: { tension: 120, friction: 14 }
  })

  // Fetch portfolio data
  useEffect(() => {
    const fetchPortfolio = async () => {
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
          } else {
            setError('Failed to load portfolio data');
          }
        } else {
          const errorData = await response.json().catch(() => ({}));
          setError(errorData.error || `Failed to fetch portfolio: ${response.statusText}`);
        }
      } catch (err: any) {
        console.error('Error fetching portfolio:', err);
        setError('Network error: Unable to fetch portfolio');
      } finally {
        setLoading(false);
      }
    };

    fetchPortfolio();

    // 🚀 NEW: Listen for real-time portfolio updates
    const handlePortfolioUpdate = (event: CustomEvent) => {
      const update = event.detail;
      console.log('💰 Received portfolio update:', update);
      
      // Update portfolio data with new values
      setPortfolioData(prev => prev ? { ...prev, ...update } : null);
    };

    window.addEventListener('portfolioStatusUpdate', handlePortfolioUpdate as EventListener);

    // Set up polling for real-time updates (fallback)
    const interval = setInterval(fetchPortfolio, 5000); // Update every 5 seconds

    return () => {
      window.removeEventListener('portfolioStatusUpdate', handlePortfolioUpdate as EventListener);
      clearInterval(interval);
    };
  }, []);

  const formatCurrency = (value: number) => {
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: 'USD',
      minimumFractionDigits: 2,
      maximumFractionDigits: 2,
    }).format(value)
  }

  const formatPercentage = (value: number) => {
    return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  }

  const getPnlColor = (value: number) => {
    if (value > 0) return 'text-green-400'
    if (value < 0) return 'text-red-400'
    return 'text-gray-400'
  }

  const getPnlBgColor = (value: number) => {
    if (value > 0) return 'bg-green-400/10 border-green-400/20'
    if (value < 0) return 'bg-red-400/10 border-red-400/20'
    return 'bg-gray-400/10 border-gray-400/20'
  }

  const getHealthColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'text-green-400'
      case 'warning': return 'text-yellow-400'
      case 'danger': return 'text-red-400'
      default: return 'text-gray-400'
    }
  }

  const getHealthBgColor = (status: string) => {
    switch (status) {
      case 'healthy': return 'bg-green-500/20'
      case 'warning': return 'bg-yellow-500/20'
      case 'danger': return 'bg-red-500/20'
      default: return 'bg-gray-500/20'
    }
  }

  // Loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-blue-500 mx-auto mb-4"></div>
          <div className="text-gray-400">Loading portfolio...</div>
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center">
          <div className="text-red-400 mb-2">⚠️ Error Loading Portfolio</div>
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

  // No portfolio data
  if (!portfolioData) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-center text-gray-400">
          <div className="text-lg mb-2">No Portfolio Data</div>
          <div className="text-sm">Connect your wallet to view your portfolio</div>
        </div>
      </div>
    );
  }

  const healthStatus = portfolioData.healthFactor > 0.8 ? 'healthy' : 
                     portfolioData.healthFactor > 0.5 ? 'warning' : 'danger';
  const pnlPercentage = portfolioData.totalValue > 0 
    ? (portfolioData.totalUnrealizedPnl / portfolioData.totalValue) * 100 
    : 0;

  return (
    <div className="space-y-6">
      {/* Connection Status */}
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-bold text-white">Portfolio Dashboard</h2>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-green-400"></div>
          <span className="text-sm text-gray-400">Live</span>
          {portfolioData.timestamp && (
            <span className="text-xs text-gray-500">
              Updated {new Date(portfolioData.timestamp).toLocaleTimeString()}
            </span>
          )}
        </div>
      </div>

      {/* Portfolio Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Value */}
        <div className="trading-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-400">Total Value</div>
              <animated.div className="text-2xl font-bold text-white">
                {portfolioValueSpring.number.to((n: number) => formatCurrency(n))}
              </animated.div>
            </div>
            <div className="w-12 h-12 bg-gray-500/20 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
              </svg>
            </div>
          </div>
        </div>

        {/* Unrealized P&L */}
        <div className={`trading-card ${getPnlBgColor(portfolioData.totalUnrealizedPnl)}`}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-400">Unrealized P&L</div>
              <animated.div className={`text-2xl font-bold ${getPnlColor(portfolioData.totalUnrealizedPnl)}`}>
                {unrealizedPnlSpring.number.to((n: number) => formatCurrency(n))}
              </animated.div>
              <div className={`text-sm ${getPnlColor(pnlPercentage)}`}>
                {formatPercentage(pnlPercentage)}
              </div>
            </div>
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
              portfolioData.totalUnrealizedPnl > 0 ? 'bg-green-500/20' : 
              portfolioData.totalUnrealizedPnl < 0 ? 'bg-red-500/20' : 'bg-gray-500/20'
            }`}>
              <svg className={`w-6 h-6 ${
                portfolioData.totalUnrealizedPnl > 0 ? 'text-green-400' : 
                portfolioData.totalUnrealizedPnl < 0 ? 'text-red-400' : 'text-gray-400'
              }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
          </div>
        </div>

        {/* Realized P&L */}
        <div className={`trading-card ${getPnlBgColor(portfolioData.totalRealizedPnl)}`}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-400">Realized P&L</div>
              <animated.div className={`text-2xl font-bold ${getPnlColor(portfolioData.totalRealizedPnl)}`}>
                {realizedPnlSpring.number.to((n: number) => formatCurrency(n))}
              </animated.div>
            </div>
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
              portfolioData.totalRealizedPnl > 0 ? 'bg-green-500/20' : 
              portfolioData.totalRealizedPnl < 0 ? 'bg-red-500/20' : 'bg-gray-500/20'
            }`}>
              <svg className={`w-6 h-6 ${
                portfolioData.totalRealizedPnl > 0 ? 'text-green-400' : 
                portfolioData.totalRealizedPnl < 0 ? 'text-red-400' : 'text-gray-400'
              }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Health Factor */}
        <div className="trading-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-400">Health Factor</div>
              <animated.div className={`text-2xl font-bold ${getHealthColor(healthStatus)}`}>
                {(portfolioData.healthFactor * 100).toFixed(1)}%
              </animated.div>
              <div className={`text-sm ${getHealthColor(healthStatus)}`}>
                {healthStatus.charAt(0).toUpperCase() + healthStatus.slice(1)}
              </div>
            </div>
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${getHealthBgColor(healthStatus)}`}>
              <svg className={`w-6 h-6 ${getHealthColor(healthStatus)}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Portfolio Details */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {/* Collateral */}
        <div className="trading-card">
          <h3 className="text-lg font-semibold text-white mb-4">Collateral</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Total</span>
              <span className="text-white">{formatCurrency(portfolioData.totalCollateral)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Used</span>
              <span className="text-white">{formatCurrency(portfolioData.usedMargin)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Available</span>
              <span className="text-white">{formatCurrency(portfolioData.availableMargin)}</span>
            </div>
          </div>
        </div>

        {/* Positions Summary */}
        <div className="trading-card">
          <h3 className="text-lg font-semibold text-white mb-4">Positions</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Total Positions</span>
              <span className="text-white">{portfolioData.positions.length}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Long Positions</span>
              <span className="text-green-400">
                {portfolioData.positions.filter(p => p.side === 'long').length}
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Short Positions</span>
              <span className="text-red-400">
                {portfolioData.positions.filter(p => p.side === 'short').length}
              </span>
            </div>
          </div>
        </div>

        {/* Risk Metrics */}
        <div className="trading-card">
          <h3 className="text-lg font-semibold text-white mb-4">Risk Metrics</h3>
          <div className="space-y-2">
            <div className="flex justify-between">
              <span className="text-gray-400">Margin Ratio</span>
              <span className={`${getHealthColor(healthStatus)}`}>
                {portfolioData.marginRatio.toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Health Factor</span>
              <span className={`${getHealthColor(healthStatus)}`}>
                {(portfolioData.healthFactor * 100).toFixed(1)}%
              </span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Status</span>
              <span className={`${getHealthColor(healthStatus)}`}>
                {healthStatus.charAt(0).toUpperCase() + healthStatus.slice(1)}
              </span>
            </div>
          </div>
        </div>
      </div>

      {/* Positions Table */}
      {portfolioData.positions.length > 0 && (
        <div className="trading-card">
          <h3 className="text-lg font-semibold text-white mb-4">Open Positions</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b border-gray-700">
                  <th className="text-left py-2 text-gray-400">Symbol</th>
                  <th className="text-left py-2 text-gray-400">Side</th>
                  <th className="text-left py-2 text-gray-400">Size</th>
                  <th className="text-left py-2 text-gray-400">Entry Price</th>
                  <th className="text-left py-2 text-gray-400">Current Price</th>
                  <th className="text-left py-2 text-gray-400">P&L</th>
                  <th className="text-left py-2 text-gray-400">Leverage</th>
                </tr>
              </thead>
              <tbody>
                {portfolioData.positions.map((position) => (
                  <tr key={position.id} className="border-b border-gray-800">
                    <td className="py-2 text-white">{position.symbol}</td>
                    <td className={`py-2 ${position.side === 'long' ? 'text-green-400' : 'text-red-400'}`}>
                      {position.side.toUpperCase()}
                    </td>
                    <td className="py-2 text-white">{position.size.toFixed(4)}</td>
                    <td className="py-2 text-white">{formatCurrency(position.entryPrice)}</td>
                    <td className="py-2 text-white">{formatCurrency(position.currentPrice)}</td>
                    <td className={`py-2 ${getPnlColor(position.unrealizedPnl)}`}>
                      {formatCurrency(position.unrealizedPnl)}
                    </td>
                    <td className="py-2 text-white">{position.leverage}x</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  )
}

export default PortfolioDashboard