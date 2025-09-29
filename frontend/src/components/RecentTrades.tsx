// Recent Trades Component for QuantDesk
// Displays real-time trade updates from WebSocket

import React, { useState, useEffect } from 'react'
import { usePrice } from '../contexts/PriceContext'

interface RecentTradesProps {
  symbol?: string
  maxTrades?: number
}

const RecentTrades: React.FC<RecentTradesProps> = ({ 
  symbol = 'BTC/USDT', 
  maxTrades = 20 
}) => {
  const { connectionStatus } = usePrice()
  const [trades, setTrades] = useState<any[]>([])

  useEffect(() => {
    // For now, use mock trade data since we don't have real trade data yet
    const symbolTrades: any[] = []
    setTrades(symbolTrades.slice(0, maxTrades))
  }, [symbol, maxTrades])

  const formatPrice = (price: number) => {
    return price.toFixed(2)
  }

  const formatSize = (size: number) => {
    return size.toFixed(3)
  }

  const formatTime = (timestamp: number) => {
    const date = new Date(timestamp)
    return date.toLocaleTimeString('en-US', { 
      hour12: false, 
      hour: '2-digit', 
      minute: '2-digit', 
      second: '2-digit' 
    })
  }

  return (
    <div className="trading-card">
      <div className="flex items-center justify-between mb-4">
        <h3 className="text-lg font-semibold text-white">Recent Trades</h3>
        <div className="flex items-center space-x-2">
          <div className={`w-2 h-2 rounded-full ${connectionStatus === 'connected' ? 'bg-green-400' : 'bg-red-400'}`}></div>
          <span className="text-xs text-gray-400">{symbol}</span>
        </div>
      </div>

      <div className="space-y-1">
        {trades.length > 0 ? (
          trades.map((trade, index) => (
            <div
              key={`${trade.id}-${index}`}
              className="flex items-center justify-between py-1 px-2 rounded hover:bg-gray-800 transition-colors"
            >
              <div className="flex items-center space-x-3">
                <div className={`text-xs font-medium ${
                  trade.side === 'buy' ? 'text-green-400' : 'text-red-400'
                }`}>
                  {trade.side.toUpperCase()}
                </div>
                <div className="text-sm text-white font-mono">
                  ${formatPrice(trade.price)}
                </div>
                <div className="text-xs text-gray-400 font-mono">
                  {formatSize(trade.size)}
                </div>
              </div>
              <div className="text-xs text-gray-500 font-mono">
                {formatTime(trade.timestamp)}
              </div>
            </div>
          ))
        ) : (
          <div className="text-center py-8 text-gray-400">
            <div className="text-sm">No recent trades</div>
            <div className="text-xs mt-1">
              {isConnected ? 'Waiting for trades...' : 'WebSocket disconnected'}
            </div>
          </div>
        )}
      </div>

      {trades.length > 0 && (
        <div className="mt-4 pt-3 border-t border-gray-700">
          <div className="flex justify-between text-xs text-gray-400">
            <span>Total: {trades.length} trades</span>
            <span>Live updates: {isConnected ? 'ON' : 'OFF'}</span>
          </div>
        </div>
      )}
    </div>
  )
}

export default RecentTrades
