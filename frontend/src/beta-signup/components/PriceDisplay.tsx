/**
 * Reusable Price Display Components
 * Industry-standard price display components used throughout the app
 */

import React from 'react'
import { usePriceData, usePrice } from '../../contexts/PriceContext'

interface PriceDisplayProps {
  symbol: string
  showChange?: boolean
  showChangePercent?: boolean
  showTimestamp?: boolean
  className?: string
  priceClassName?: string
  changeClassName?: string
}

/**
 * Basic Price Display Component
 */
export const PriceDisplay: React.FC<PriceDisplayProps> = ({
  symbol,
  showChange = true,
  showChangePercent = true,
  showTimestamp = false,
  className = '',
  priceClassName = '',
  changeClassName = ''
}) => {
  const priceData = usePriceData(symbol)
  const { isConnected } = usePrice()

  if (!priceData) {
    return (
      <div className={`price-display ${className}`}>
        <span className="text-gray-400">Loading...</span>
      </div>
    )
  }

  const formatPrice = (price: number): string => {
    if (price === 0) return '$0.00'
    if (price >= 1000) return `$${price.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`
    if (price >= 1) return `$${price.toFixed(2)}`
    if (price >= 0.01) return `$${price.toFixed(4)}`
    return `$${price.toFixed(6)}`
  }

  const formatChange = (change: number): string => {
    const sign = change >= 0 ? '+' : ''
    return `${sign}${change.toFixed(2)}`
  }

  const formatChangePercent = (changePercent: number): string => {
    const sign = changePercent >= 0 ? '+' : ''
    return `${sign}${changePercent.toFixed(2)}%`
  }

  const getChangeColor = (change: number): string => {
    if (change > 0) return 'text-green-400'
    if (change < 0) return 'text-red-400'
    return 'text-gray-400'
  }

  return (
    <div className={`price-display ${className}`}>
      <div className="flex items-center gap-2">
        <span className={`price ${priceClassName}`}>
          {formatPrice(priceData.price)}
        </span>
        
        {showChange && (
          <span className={`change ${changeClassName} ${getChangeColor(priceData.change)}`}>
            {formatChange(priceData.change)}
          </span>
        )}
        
        {showChangePercent && (
          <span className={`change-percent ${changeClassName} ${getChangeColor(priceData.changePercent)}`}>
            {formatChangePercent(priceData.changePercent)}
          </span>
        )}
        
        {showTimestamp && (
          <span className="text-xs text-gray-500">
            {new Date(priceData.timestamp).toLocaleTimeString()}
          </span>
        )}
        
        {/* Connection indicator */}
        <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
      </div>
    </div>
  )
}

/**
 * Price Grid Component - Shows multiple prices in a grid
 */
interface PriceGridProps {
  symbols: string[]
  columns?: number
  className?: string
}

export const PriceGrid: React.FC<PriceGridProps> = ({
  symbols,
  columns = 3,
  className = ''
}) => {
  const { prices } = usePrice()

  return (
    <div className={`price-grid grid gap-4 ${className}`} style={{ gridTemplateColumns: `repeat(${columns}, 1fr)` }}>
      {symbols.map(symbol => {
        const priceData = prices.get(symbol)
        return (
          <div key={symbol} className="price-card bg-gray-800/50 rounded-lg p-4 border border-gray-700">
            <div className="flex justify-between items-center mb-2">
              <span className="text-sm font-medium text-gray-300">{symbol}</span>
              <div className={`w-2 h-2 rounded-full ${priceData ? 'bg-green-400 animate-pulse' : 'bg-red-400'}`} />
            </div>
            <PriceDisplay symbol={symbol} showChange={true} showChangePercent={true} />
          </div>
        )
      })}
    </div>
  )
}

/**
 * Price Ticker Component - Horizontal scrolling price ticker
 */
interface PriceTickerProps {
  symbols: string[]
  speed?: number
  className?: string
}

export const PriceTicker: React.FC<PriceTickerProps> = ({
  symbols,
  speed = 30,
  className = ''
}) => {
  const { prices } = usePrice()

  return (
    <div className={`price-ticker overflow-hidden whitespace-nowrap ${className}`}>
      <div 
        className="inline-block animate-scroll"
        style={{ 
          animationDuration: `${speed}s`,
          animationTimingFunction: 'linear',
          animationIterationCount: 'infinite'
        }}
      >
        {symbols.map((symbol, index) => {
          const priceData = prices.get(symbol)
          return (
            <span key={symbol} className="inline-block mr-8">
              <span className="text-gray-300 mr-2">{symbol}</span>
              <PriceDisplay symbol={symbol} showChange={false} showChangePercent={true} />
            </span>
          )
        })}
      </div>
    </div>
  )
}

/**
 * Price Change Indicator Component
 */
interface PriceChangeIndicatorProps {
  symbol: string
  className?: string
}

export const PriceChangeIndicator: React.FC<PriceChangeIndicatorProps> = ({
  symbol,
  className = ''
}) => {
  const priceData = usePriceData(symbol)

  if (!priceData) {
    return <div className={`price-change-indicator ${className}`}>-</div>
  }

  const getChangeIcon = (change: number) => {
    if (change > 0) return '↗'
    if (change < 0) return '↘'
    return '→'
  }

  const getChangeColor = (change: number) => {
    if (change > 0) return 'text-green-400 bg-green-400/10'
    if (change < 0) return 'text-red-400 bg-red-400/10'
    return 'text-gray-400 bg-gray-400/10'
  }

  return (
    <div className={`price-change-indicator flex items-center gap-1 px-2 py-1 rounded ${getChangeColor(priceData.change)} ${className}`}>
      <span className="text-sm">{getChangeIcon(priceData.change)}</span>
      <span className="text-sm font-medium">
        {priceData.changePercent >= 0 ? '+' : ''}{priceData.changePercent.toFixed(2)}%
      </span>
    </div>
  )
}

export default PriceDisplay
