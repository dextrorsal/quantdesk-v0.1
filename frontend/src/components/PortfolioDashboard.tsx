// Portfolio Dashboard Component for QuantDesk
// Displays comprehensive portfolio overview with P&L analytics and risk metrics

import React, { useState, useEffect } from 'react'
import { portfolioService, PortfolioSummary, RiskMetrics, PerformanceMetrics } from '../services/portfolioService'

const PortfolioDashboard: React.FC = () => {
  const [portfolioSummary, setPortfolioSummary] = useState<PortfolioSummary | null>(null)
  const [riskMetrics, setRiskMetrics] = useState<RiskMetrics | null>(null)
  const [performanceMetrics, setPerformanceMetrics] = useState<PerformanceMetrics | null>(null)
  // const [activeTab, setActiveTab] = useState<'overview' | 'positions' | 'trades' | 'orders' | 'analytics'>('overview') // For future use

  useEffect(() => {
    // Load initial data
    setPortfolioSummary(portfolioService.getPortfolioSummary())
    setRiskMetrics(portfolioService.getRiskMetrics())
    setPerformanceMetrics(portfolioService.getPerformanceMetrics())

    // Update data every 5 seconds
    const interval = setInterval(() => {
      setPortfolioSummary(portfolioService.getPortfolioSummary())
      setRiskMetrics(portfolioService.getRiskMetrics())
      setPerformanceMetrics(portfolioService.getPerformanceMetrics())
    }, 5000)

    return () => clearInterval(interval)
  }, [])

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

  if (!portfolioSummary || !riskMetrics || !performanceMetrics) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="text-gray-400">Loading portfolio data...</div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      {/* Portfolio Summary Cards */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Total Equity */}
        <div className="trading-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-400">Total Equity</div>
              <div className="text-2xl font-bold text-white">
                {formatCurrency(portfolioSummary.totalEquity)}
              </div>
            </div>
            <div className="w-12 h-12 bg-blue-500/20 rounded-lg flex items-center justify-center">
              <svg className="w-6 h-6 text-blue-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8c-1.657 0-3 .895-3 2s1.343 2 3 2 3 .895 3 2-1.343 2-3 2m0-8c1.11 0 2.08.402 2.599 1M12 8V7m0 1v8m0 0v1m0-1c-1.11 0-2.08-.402-2.599-1" />
              </svg>
            </div>
          </div>
        </div>

        {/* Unrealized P&L */}
        <div className={`trading-card ${getPnlBgColor(portfolioSummary.totalUnrealizedPnl)}`}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-400">Unrealized P&L</div>
              <div className={`text-2xl font-bold ${getPnlColor(portfolioSummary.totalUnrealizedPnl)}`}>
                {formatCurrency(portfolioSummary.totalUnrealizedPnl)}
              </div>
            </div>
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
              portfolioSummary.totalUnrealizedPnl > 0 ? 'bg-green-500/20' : 
              portfolioSummary.totalUnrealizedPnl < 0 ? 'bg-red-500/20' : 'bg-gray-500/20'
            }`}>
              <svg className={`w-6 h-6 ${
                portfolioSummary.totalUnrealizedPnl > 0 ? 'text-green-400' : 
                portfolioSummary.totalUnrealizedPnl < 0 ? 'text-red-400' : 'text-gray-400'
              }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7h8m0 0v8m0-8l-8 8-4-4-6 6" />
              </svg>
            </div>
          </div>
        </div>

        {/* Realized P&L */}
        <div className={`trading-card ${getPnlBgColor(portfolioSummary.totalRealizedPnl)}`}>
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-400">Realized P&L</div>
              <div className={`text-2xl font-bold ${getPnlColor(portfolioSummary.totalRealizedPnl)}`}>
                {formatCurrency(portfolioSummary.totalRealizedPnl)}
              </div>
            </div>
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
              portfolioSummary.totalRealizedPnl > 0 ? 'bg-green-500/20' : 
              portfolioSummary.totalRealizedPnl < 0 ? 'bg-red-500/20' : 'bg-gray-500/20'
            }`}>
              <svg className={`w-6 h-6 ${
                portfolioSummary.totalRealizedPnl > 0 ? 'text-green-400' : 
                portfolioSummary.totalRealizedPnl < 0 ? 'text-red-400' : 'text-gray-400'
              }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
              </svg>
            </div>
          </div>
        </div>

        {/* Margin Ratio */}
        <div className="trading-card">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm text-gray-400">Margin Ratio</div>
              <div className={`text-2xl font-bold ${
                portfolioSummary.marginRatio > 80 ? 'text-red-400' :
                portfolioSummary.marginRatio > 60 ? 'text-yellow-400' : 'text-green-400'
              }`}>
                {portfolioSummary.marginRatio.toFixed(1)}%
              </div>
            </div>
            <div className={`w-12 h-12 rounded-lg flex items-center justify-center ${
              portfolioSummary.marginRatio > 80 ? 'bg-red-500/20' :
              portfolioSummary.marginRatio > 60 ? 'bg-yellow-500/20' : 'bg-green-500/20'
            }`}>
              <svg className={`w-6 h-6 ${
                portfolioSummary.marginRatio > 80 ? 'text-red-400' :
                portfolioSummary.marginRatio > 60 ? 'text-yellow-400' : 'text-green-400'
              }`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
            </div>
          </div>
        </div>
      </div>

      {/* Risk Metrics */}
      <div className="trading-card">
        <h3 className="text-lg font-semibold text-white mb-4">Risk Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center">
            <div className="text-2xl font-bold text-white">{riskMetrics.totalTrades}</div>
            <div className="text-sm text-gray-400">Total Trades</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-green-400">{riskMetrics.winningTrades}</div>
            <div className="text-sm text-gray-400">Winning Trades</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-red-400">{riskMetrics.losingTrades}</div>
            <div className="text-sm text-gray-400">Losing Trades</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">{riskMetrics.winRate.toFixed(1)}%</div>
            <div className="text-sm text-gray-400">Win Rate</div>
          </div>
        </div>
      </div>

      {/* Performance Metrics */}
      <div className="trading-card">
        <h3 className="text-lg font-semibold text-white mb-4">Performance Metrics</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="text-center">
            <div className={`text-2xl font-bold ${getPnlColor(performanceMetrics.totalReturn)}`}>
              {formatPercentage(performanceMetrics.totalReturn)}
            </div>
            <div className="text-sm text-gray-400">Total Return</div>
          </div>
          <div className="text-center">
            <div className={`text-2xl font-bold ${getPnlColor(performanceMetrics.dailyReturn)}`}>
              {formatPercentage(performanceMetrics.dailyReturn)}
            </div>
            <div className="text-sm text-gray-400">Daily Return</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">{performanceMetrics.sharpeRatio.toFixed(2)}</div>
            <div className="text-sm text-gray-400">Sharpe Ratio</div>
          </div>
          <div className="text-center">
            <div className="text-2xl font-bold text-white">{performanceMetrics.maxDrawdown.toFixed(1)}%</div>
            <div className="text-sm text-gray-400">Max Drawdown</div>
          </div>
        </div>
      </div>

      {/* Additional Metrics */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <div className="trading-card">
          <h3 className="text-lg font-semibold text-white mb-4">Trading Statistics</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Profit Factor</span>
              <span className="text-white font-medium">{performanceMetrics.profitFactor.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Trade Duration</span>
              <span className="text-white font-medium">{performanceMetrics.avgTradeDuration}h</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Volatility</span>
              <span className="text-white font-medium">{performanceMetrics.volatility.toFixed(2)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Sortino Ratio</span>
              <span className="text-white font-medium">{performanceMetrics.sortinoRatio.toFixed(2)}</span>
            </div>
          </div>
        </div>

        <div className="trading-card">
          <h3 className="text-lg font-semibold text-white mb-4">Risk Analysis</h3>
          <div className="space-y-3">
            <div className="flex justify-between">
              <span className="text-gray-400">Portfolio Value</span>
              <span className="text-white font-medium">{formatCurrency(riskMetrics.portfolioValue)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Total Margin</span>
              <span className="text-white font-medium">{formatCurrency(riskMetrics.totalMargin)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Win</span>
              <span className="text-green-400 font-medium">{formatCurrency(riskMetrics.avgWin)}</span>
            </div>
            <div className="flex justify-between">
              <span className="text-gray-400">Avg Loss</span>
              <span className="text-red-400 font-medium">{formatCurrency(riskMetrics.avgLoss)}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}

export default PortfolioDashboard
