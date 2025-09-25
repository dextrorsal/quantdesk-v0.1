// Positions Table Component for QuantDesk
// Displays detailed position information with real-time updates

import React, { useState, useEffect } from 'react'
import { portfolioService, Position } from '../services/portfolioService'

const PositionsTable: React.FC = () => {
  const [positions, setPositions] = useState<Position[]>([])
  const [sortField, setSortField] = useState<keyof Position>('symbol')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')

  useEffect(() => {
    // Load initial positions
    setPositions(portfolioService.getPositions())

    // Update positions every 2 seconds
    const interval = setInterval(() => {
      setPositions(portfolioService.getPositions())
    }, 2000)

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

  // const formatPercentage = (value: number) => {
  //   return `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`
  // } // For future use

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp)
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    })
  }

  const getPnlColor = (value: number) => {
    if (value > 0) return 'text-green-400'
    if (value < 0) return 'text-red-400'
    return 'text-gray-400'
  }

  const getSideColor = (side: string) => {
    return side === 'long' ? 'text-green-400' : 'text-red-400'
  }

  const getSideBgColor = (side: string) => {
    return side === 'long' ? 'bg-green-400/10' : 'bg-red-400/10'
  }

  const handleSort = (field: keyof Position) => {
    if (sortField === field) {
      setSortDirection(sortDirection === 'asc' ? 'desc' : 'asc')
    } else {
      setSortField(field)
      setSortDirection('asc')
    }
  }

  const sortedPositions = [...positions].sort((a, b) => {
    const aValue = a[sortField]
    const bValue = b[sortField]
    
    if (typeof aValue === 'string' && typeof bValue === 'string') {
      return sortDirection === 'asc' 
        ? aValue.localeCompare(bValue)
        : bValue.localeCompare(aValue)
    }
    
    if (typeof aValue === 'number' && typeof bValue === 'number') {
      return sortDirection === 'asc' ? aValue - bValue : bValue - aValue
    }
    
    return 0
  })

  const SortIcon: React.FC<{ field: keyof Position }> = ({ field }) => {
    if (sortField !== field) {
      return <span className="text-gray-400">↕</span>
    }
    return <span className="text-white">{sortDirection === 'asc' ? '↑' : '↓'}</span>
  }

  if (positions.length === 0) {
    return (
      <div className="trading-card">
        <div className="text-center py-12">
          <div className="w-16 h-16 bg-gray-800 rounded-full flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 19v-6a2 2 0 00-2-2H5a2 2 0 00-2 2v6a2 2 0 002 2h2a2 2 0 002-2zm0 0V9a2 2 0 012-2h2a2 2 0 012 2v10m-6 0a2 2 0 002 2h2a2 2 0 002-2m0 0V5a2 2 0 012-2h2a2 2 0 012 2v14a2 2 0 01-2 2h-2a2 2 0 01-2-2z" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-white mb-2">No Open Positions</h3>
          <p className="text-gray-400">Start trading to see your positions here</p>
        </div>
      </div>
    )
  }

  return (
    <div className="trading-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Open Positions</h3>
        <div className="text-sm text-gray-400">
          {positions.length} position{positions.length !== 1 ? 's' : ''}
        </div>
      </div>

      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="border-b border-gray-700">
              <th 
                className="text-left py-3 px-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white"
                onClick={() => handleSort('symbol')}
              >
                <div className="flex items-center space-x-1">
                  <span>Symbol</span>
                  <SortIcon field="symbol" />
                </div>
              </th>
              <th 
                className="text-left py-3 px-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white"
                onClick={() => handleSort('side')}
              >
                <div className="flex items-center space-x-1">
                  <span>Side</span>
                  <SortIcon field="side" />
                </div>
              </th>
              <th 
                className="text-right py-3 px-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white"
                onClick={() => handleSort('size')}
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Size</span>
                  <SortIcon field="size" />
                </div>
              </th>
              <th 
                className="text-right py-3 px-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white"
                onClick={() => handleSort('entryPrice')}
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Entry Price</span>
                  <SortIcon field="entryPrice" />
                </div>
              </th>
              <th 
                className="text-right py-3 px-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white"
                onClick={() => handleSort('currentPrice')}
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Current Price</span>
                  <SortIcon field="currentPrice" />
                </div>
              </th>
              <th 
                className="text-right py-3 px-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white"
                onClick={() => handleSort('leverage')}
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Leverage</span>
                  <SortIcon field="leverage" />
                </div>
              </th>
              <th 
                className="text-right py-3 px-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white"
                onClick={() => handleSort('unrealizedPnl')}
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Unrealized P&L</span>
                  <SortIcon field="unrealizedPnl" />
                </div>
              </th>
              <th 
                className="text-right py-3 px-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white"
                onClick={() => handleSort('marginRatio')}
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Margin Ratio</span>
                  <SortIcon field="marginRatio" />
                </div>
              </th>
              <th 
                className="text-right py-3 px-4 text-sm font-medium text-gray-400 cursor-pointer hover:text-white"
                onClick={() => handleSort('liquidationPrice')}
              >
                <div className="flex items-center justify-end space-x-1">
                  <span>Liquidation Price</span>
                  <SortIcon field="liquidationPrice" />
                </div>
              </th>
              <th className="text-left py-3 px-4 text-sm font-medium text-gray-400">Opened</th>
              <th className="text-center py-3 px-4 text-sm font-medium text-gray-400">Actions</th>
            </tr>
          </thead>
          <tbody>
            {sortedPositions.map((position) => (
              <tr key={position.id} className="border-b border-gray-800 hover:bg-gray-800/50">
                <td className="py-3 px-4">
                  <div className="font-medium text-white">{position.symbol}</div>
                </td>
                <td className="py-3 px-4">
                  <span className={`px-2 py-1 rounded text-xs font-medium ${getSideBgColor(position.side)} ${getSideColor(position.side)}`}>
                    {position.side.toUpperCase()}
                  </span>
                </td>
                <td className="py-3 px-4 text-right font-mono text-white">
                  {position.size.toFixed(3)}
                </td>
                <td className="py-3 px-4 text-right font-mono text-white">
                  {formatCurrency(position.entryPrice)}
                </td>
                <td className="py-3 px-4 text-right font-mono text-white">
                  {formatCurrency(position.currentPrice)}
                </td>
                <td className="py-3 px-4 text-right">
                  <span className="px-2 py-1 bg-blue-500/20 text-blue-400 rounded text-xs font-medium">
                    {position.leverage}x
                  </span>
                </td>
                <td className="py-3 px-4 text-right">
                  <div className={`font-mono font-medium ${getPnlColor(position.unrealizedPnl)}`}>
                    {formatCurrency(position.unrealizedPnl)}
                  </div>
                </td>
                <td className="py-3 px-4 text-right">
                  <div className={`font-medium ${
                    position.marginRatio > 80 ? 'text-red-400' :
                    position.marginRatio > 60 ? 'text-yellow-400' : 'text-green-400'
                  }`}>
                    {position.marginRatio.toFixed(1)}%
                  </div>
                </td>
                <td className="py-3 px-4 text-right font-mono text-white">
                  {formatCurrency(position.liquidationPrice)}
                </td>
                <td className="py-3 px-4 text-sm text-gray-400">
                  {formatTime(position.openTime)}
                </td>
                <td className="py-3 px-4 text-center">
                  <div className="flex items-center justify-center space-x-2">
                    <button className="text-blue-400 hover:text-blue-300 text-sm">
                      Close
                    </button>
                    <button className="text-gray-400 hover:text-gray-300 text-sm">
                      Edit
                    </button>
                  </div>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

export default PositionsTable
