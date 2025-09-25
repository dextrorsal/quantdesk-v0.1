import React, { useState, useEffect } from 'react'
import RecentTrades from './RecentTrades'

interface OrderBookEntry {
  price: number
  size: number
  total: number
}

const OrderBook: React.FC = () => {
  const [bids, setBids] = useState<OrderBookEntry[]>([])
  const [asks, setAsks] = useState<OrderBookEntry[]>([])
  const [activeTab, setActiveTab] = useState<'orderbook' | 'trades'>('orderbook')

  // Generate realistic order book data
  useEffect(() => {
    const generateOrderBook = () => {
      const basePrice = 43250
      const newBids: OrderBookEntry[] = []
      const newAsks: OrderBookEntry[] = []
      
      let bidTotal = 0
      let askTotal = 0

      // Generate bids (decreasing prices)
      for (let i = 0; i < 10; i++) {
        const price = basePrice - (i + 1) * 0.5
        const size = Math.random() * 5 + 0.1
        bidTotal += size
        newBids.push({
          price,
          size: Number(size.toFixed(3)),
          total: Number(bidTotal.toFixed(3))
        })
      }

      // Generate asks (increasing prices)
      for (let i = 0; i < 10; i++) {
        const price = basePrice + (i + 1) * 0.5
        const size = Math.random() * 5 + 0.1
        askTotal += size
        newAsks.push({
          price,
          size: Number(size.toFixed(3)),
          total: Number(askTotal.toFixed(3))
        })
      }

      setBids(newBids)
      setAsks(newAsks)
    }

    generateOrderBook()
    
    // Update every 2 seconds to simulate real-time data
    const interval = setInterval(generateOrderBook, 2000)
    return () => clearInterval(interval)
  }, [])

  const maxTotal = Math.max(...bids.map(b => b.total), ...asks.map(a => a.total))

  return (
    <div className="trading-card h-full flex flex-col">
      {/* Tab Navigation */}
      <div className="flex border-b border-gray-800 flex-shrink-0">
        <button
          onClick={() => setActiveTab('orderbook')}
          className={`px-4 py-3 text-sm font-medium transition-colors ${
            activeTab === 'orderbook'
              ? 'text-white border-b-2 border-blue-500 bg-gray-900'
              : 'text-gray-400 hover:text-white bg-gray-900'
          }`}
        >
          Order Book
        </button>
        <button
          onClick={() => setActiveTab('trades')}
          className={`px-4 py-3 text-sm font-medium transition-colors ${
            activeTab === 'trades'
              ? 'text-white border-b-2 border-blue-500 bg-gray-900'
              : 'text-gray-400 hover:text-white bg-gray-900'
          }`}
        >
          Recent Trades
        </button>
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-hidden">
        {activeTab === 'orderbook' ? (
          <div className="h-full flex flex-col">
            {/* Header */}
            <div className="text-xs text-gray-400 grid grid-cols-3 gap-2 px-4 py-2 border-b border-gray-800">
              <span>Price</span>
              <span className="text-right">Size</span>
              <span className="text-right">Total</span>
            </div>

            {/* Asks (Sell Orders) */}
            <div className="flex-1 space-y-0.5 px-4 py-2 overflow-y-auto custom-scrollbar">
              {asks.slice().reverse().map((ask, index) => {
                const width = (ask.total / maxTotal) * 100
                return (
                  <div key={`ask-${index}`} className="relative group cursor-pointer hover:bg-gray-800/30 rounded">
                    <div 
                      className="absolute left-0 top-0 bottom-0 bg-red-500/10 rounded-l" 
                      style={{ width: `${width}%` }} 
                    />
                    <div className="relative grid grid-cols-3 gap-2 text-xs py-1 px-2">
                      <span className="text-red-400 font-mono">${ask.price.toFixed(2)}</span>
                      <span className="text-right text-white font-mono">{ask.size.toFixed(3)}</span>
                      <span className="text-right text-gray-400 font-mono">{ask.total.toFixed(3)}</span>
                    </div>
                  </div>
                )
              })}
            </div>

            {/* Current Price */}
            <div className="border-t border-b border-gray-800 py-2 px-4 flex-shrink-0">
              <div className="text-center">
                <div className="text-sm font-bold text-white">$43,250.50</div>
                <div className="text-xs text-green-400">+2.45%</div>
              </div>
            </div>

            {/* Bids (Buy Orders) */}
            <div className="flex-1 space-y-0.5 px-4 py-2 overflow-y-auto custom-scrollbar">
              {bids.map((bid, index) => {
                const width = (bid.total / maxTotal) * 100
                return (
                  <div key={`bid-${index}`} className="relative group cursor-pointer hover:bg-gray-800/30 rounded">
                    <div 
                      className="absolute right-0 top-0 bottom-0 bg-green-500/10 rounded-r" 
                      style={{ width: `${width}%` }} 
                    />
                    <div className="relative grid grid-cols-3 gap-2 text-xs py-1 px-2">
                      <span className="text-green-400 font-mono">${bid.price.toFixed(2)}</span>
                      <span className="text-right text-white font-mono">{bid.size.toFixed(3)}</span>
                      <span className="text-right text-gray-400 font-mono">{bid.total.toFixed(3)}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>
        ) : (
          <RecentTrades />
        )}
      </div>
    </div>
  )
}

export default OrderBook