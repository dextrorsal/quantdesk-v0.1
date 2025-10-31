import React from 'react'
import { TrendingUp, TrendingDown } from 'lucide-react'
import { usePrice } from '../contexts/PriceContext'

interface MarketData {
  symbol: string
  name: string
  price: number
  change24h: number
  volume24h: number
  high24h: number
  low24h: number
  openInterest: number
}

const marketSymbols = [
  { symbol: 'BTC/USDT', name: 'Bitcoin', volume24h: 2.94, high24h: 44100, low24h: 42800, openInterest: 1.2 },
  { symbol: 'ETH/USDT', name: 'Ethereum', volume24h: 1.87, high24h: 3250, low24h: 3150, openInterest: 0.8 },
  { symbol: 'SOL/USDT', name: 'Solana', volume24h: 0.89, high24h: 240, low24h: 215, openInterest: 0.5 },
  { symbol: 'AVAX/USDT', name: 'Avalanche', volume24h: 0.45, high24h: 1050, low24h: 980, openInterest: 0.3 },
  { symbol: 'MATIC/USDT', name: 'Polygon', volume24h: 0.67, high24h: 3.00, low24h: 2.80, openInterest: 0.2 },
  { symbol: 'ADA/USDT', name: 'Cardano', volume24h: 0.34, high24h: 0.90, low24h: 0.81, openInterest: 0.15 },
]

const MarketsPage: React.FC = () => {
  const { getPrice } = usePrice()
  
  // Create markets with real-time prices
  const markets: MarketData[] = marketSymbols.map(market => {
    const priceData = getPrice(market.symbol)
    return {
      ...market,
      price: priceData?.price || 0,
      change24h: priceData?.changePercent || 0
    }
  })

  return (
    <div className="p-6">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-bold text-white">Markets</h1>
        <div className="text-sm text-dark-400">
          {markets.length} markets available
        </div>
      </div>

      <div className="trading-card">
        <div className="overflow-x-auto">
          <table className="min-w-full">
            <thead>
              <tr className="text-left text-dark-400 text-sm">
                <th className="py-3 pr-4">Market</th>
                <th className="py-3 pr-4 text-right">Price</th>
                <th className="py-3 pr-4 text-right">24h Change</th>
                <th className="py-3 pr-4 text-right">24h Volume</th>
                <th className="py-3 pr-4 text-right">24h High</th>
                <th className="py-3 pr-4 text-right">24h Low</th>
                <th className="py-3 pr-4 text-right">Open Interest</th>
                <th className="py-3 pr-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody>
              {markets.map((market) => (
                <tr key={market.symbol} className="border-t border-dark-700 hover:bg-dark-700/50 transition-colors">
                  <td className="py-4 pr-4">
                    <div>
                      <div className="text-white font-semibold">{market.symbol}</div>
                      <div className="text-sm text-dark-400">{market.name}</div>
                    </div>
                  </td>
                  <td className="py-4 pr-4 text-right">
                    <div className="text-white font-semibold">${market.price.toLocaleString()}</div>
                  </td>
                  <td className="py-4 pr-4 text-right">
                    <div className={`flex items-center justify-end space-x-1 ${
                      market.change24h >= 0 ? 'text-success-400' : 'text-danger-400'
                    }`}>
                      {market.change24h >= 0 ? (
                        <TrendingUp className="h-4 w-4" />
                      ) : (
                        <TrendingDown className="h-4 w-4" />
                      )}
                      <span className="font-semibold">
                        {market.change24h >= 0 ? '+' : ''}{market.change24h.toFixed(2)}%
                      </span>
                    </div>
                  </td>
                  <td className="py-4 pr-4 text-right">
                    <div className="text-white">${market.volume24h}B</div>
                  </td>
                  <td className="py-4 pr-4 text-right">
                    <div className="text-dark-300">${market.high24h.toLocaleString()}</div>
                  </td>
                  <td className="py-4 pr-4 text-right">
                    <div className="text-dark-300">${market.low24h.toLocaleString()}</div>
                  </td>
                  <td className="py-4 pr-4 text-right">
                    <div className="text-white">${market.openInterest}B</div>
                  </td>
                  <td className="py-4 pr-4 text-right">
                    <button className="btn-primary px-4 py-2 text-sm">
                      Trade
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  )
}

export default MarketsPage
