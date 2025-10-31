import React, { useState, useEffect, useMemo, useRef } from 'react'
import RecentTrades from './RecentTrades'
import { usePrice } from '../contexts/PriceContext'

interface OrderBookEntry {
  price: number
  size: number
  total: number
}

interface OrderBookProps {
  symbol?: string
  className?: string
}

const OrderBook: React.FC<OrderBookProps> = ({ symbol = 'BTC/USDT', className = '' }) => {
  const [bids, setBids] = useState<Array<[number, number]>>([])
  const [asks, setAsks] = useState<Array<[number, number]>>([])
  const [activeTab, setActiveTab] = useState<'orderbook' | 'trades'>('orderbook')
  const [priceIncrement, setPriceIncrement] = useState<'$1' | '$0.1' | '$0.01'>('$1')
  const timerRef = useRef<number | null>(null)
  
  // Get real-time price from PriceContext
  const { getPrice } = usePrice()
  const currentPrice = getPrice(symbol)?.price || 0

  // Poll live orderbook from backend with jitter (like backup)
  useEffect(() => {
    const fetchBook = async () => {
      try {
        // Convert symbol format (BTC -> BTC-PERP)
        const apiSymbol = symbol.includes('-PERP') ? symbol : `${symbol}-PERP`
        const res = await fetch(`/api/markets/${apiSymbol}/orderbook`)
        if (res.status === 429) return // rate limited, wait for next poll
        if (!res.ok) return
        const json = await res.json()
        const ob = json?.orderbook
        if (ob?.bids && ob?.asks) {
          setBids(ob.bids as Array<[number, number]>)
          setAsks(ob.asks as Array<[number, number]>)
        }
      } catch (_) {
        // ignore network errors, fallback to mock data
        generateMockOrderBook()
      }
    }

    const generateMockOrderBook = () => {
      const basePrice = currentPrice || 50000
      const newBids: Array<[number, number]> = []
      const newAsks: Array<[number, number]> = []
      
      // Generate bids (decreasing prices)
      for (let i = 0; i < 20; i++) {
        const price = basePrice - (i + 1) * 0.5
        const size = Math.random() * 5 + 0.1
        newBids.push([price, size])
      }

      // Generate asks (increasing prices)
      for (let i = 0; i < 20; i++) {
        const price = basePrice + (i + 1) * 0.5
        const size = Math.random() * 5 + 0.1
        newAsks.push([price, size])
      }

      setBids(newBids)
      setAsks(newAsks)
    }

    fetchBook()
    const jitter = Math.random() * 200 // 0-200ms jitter
    timerRef.current = window.setInterval(fetchBook, 1000 + jitter)
    
    return () => { 
      if (timerRef.current) window.clearInterval(timerRef.current) 
    }
  }, [symbol, currentPrice])

  // Calculate max sizes for heatmap visualization
  const maxAsk = useMemo(() => (asks.length ? Math.max(...asks.map(a => a[1])) : 1), [asks])
  const maxBid = useMemo(() => (bids.length ? Math.max(...bids.map(b => b[1])) : 1), [bids])

  // Calculate spread
  const spread = useMemo(() => {
    if (asks.length > 0 && bids.length > 0) {
      const bestAsk = asks[0][0]
      const bestBid = bids[0][0]
      const spreadValue = bestAsk - bestBid
      const spreadPercent = (spreadValue / currentPrice) * 100
      return { value: spreadValue, percent: spreadPercent }
    }
    return { value: 0, percent: 0 }
  }, [asks, bids, currentPrice])

  return (
    <div className={`bg-black border border-primary-500 overflow-hidden ${className}`} style={{ display: 'flex', flexDirection: 'column' }}>
      {/* Header with tabs */}
      <div style={{ padding: 12, borderBottom: '1px solid var(--primary-500)' }}>
        <div style={{ display: 'flex', gap: 16 }}>
          <button
            onClick={() => setActiveTab('orderbook')}
            style={{ 
              color: activeTab === 'orderbook' ? 'var(--text-primary)' : 'var(--text-secondary)', 
              fontWeight: 600,
              background: 'none',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            Order Book
          </button>
          <button
            onClick={() => setActiveTab('trades')}
            style={{ 
              color: activeTab === 'trades' ? 'var(--text-primary)' : 'var(--text-secondary)', 
              fontWeight: 600,
              background: 'none',
              border: 'none',
              cursor: 'pointer'
            }}
          >
            Recent Trades
          </button>
        </div>
        
        {activeTab === 'orderbook' && (
          <>
            <div style={{ display: 'flex', gap: 8, marginTop: 8 }}>
              {(['$1', '$0.1', '$0.01'] as const).map(inc => (
                <button
                  key={inc}
                  onClick={() => setPriceIncrement(inc)}
                  style={{ 
                    padding: '6px 10px', 
                    backgroundColor: priceIncrement === inc ? 'var(--background-secondary)' : 'transparent', 
                    border: '1px solid var(--primary-500)', 
                    borderRadius: 6, 
                    color: 'var(--text-primary)', 
                    fontSize: 12,
                    cursor: 'pointer'
                  }}
                >
                  {inc}
                </button>
              ))}
              <button style={{ padding: '6px 10px', backgroundColor: 'transparent', border: '1px solid var(--primary-500)', borderRadius: 6, color: 'var(--text-primary)', fontSize: 12 }}>USD</button>
            </div>
            <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: 10, fontWeight: 500, paddingTop: 8 }}>
              <span style={{ flex: 1, textAlign: 'left' }}>Price</span>
              <span style={{ flex: 1, textAlign: 'right' }}>Size</span>
            </div>
          </>
        )}
      </div>

      {/* Tab Content */}
      {activeTab === 'orderbook' ? (
        <>
          {/* Asks with heatmap bars (top) */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {asks.slice(0, 20).map(([pPrice, pSize], i) => {
              const pct = (pSize / maxAsk) * 100
              const p = pPrice || currentPrice * (1.001 + i * 0.0001)
              return (
                <div key={`ask-${i}`} style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center', padding: '0 12px', borderBottom: '1px solid var(--background-secondary)' }}>
                  <div style={{ position: 'absolute', right: 0, top: 0, height: '100%', width: `${pct}%`, backgroundColor: '#ef4444', opacity: 0.15 }} />
                  <span style={{ color: '#ef4444', fontSize: 12, zIndex: 1 }}>{p.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                  <span style={{ flex: 1 }} />
                  <span style={{ color: '#ffffff', fontSize: 12, zIndex: 1 }}>{pSize >= 1000 ? `${Math.round(pSize/100)/10}K` : pSize}</span>
                </div>
              )
            })}
          </div>

          {/* Mid price / spread */}
          <div style={{ height: 40, display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 12px', borderTop: '2px solid var(--primary-500)', borderBottom: '2px solid var(--primary-500)', backgroundColor: 'var(--background-secondary)' }}>
            <div style={{ fontSize: 16, fontWeight: 700, color: 'var(--text-primary)' }}>{currentPrice.toLocaleString()}</div>
            <div style={{ fontSize: 12, color: 'var(--text-secondary)' }}>Spread: ${spread.value.toFixed(2)} ({spread.percent.toFixed(3)}%)</div>
          </div>

          {/* Bids with heatmap bars (bottom) */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {bids.slice(0, 20).map(([pPrice, pSize], i) => {
              const pct = (pSize / maxBid) * 100
              const p = pPrice || currentPrice * (0.999 - i * 0.0001)
              return (
                <div key={`bid-${i}`} style={{ position: 'relative', height: 20, display: 'flex', alignItems: 'center', padding: '0 12px', borderBottom: '1px solid var(--background-secondary)' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${pct}%`, backgroundColor: '#10b981', opacity: 0.15 }} />
                  <span style={{ color: '#10b981', fontSize: 12, zIndex: 1 }}>{p.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                  <span style={{ flex: 1 }} />
                  <span style={{ color: '#ffffff', fontSize: 12, zIndex: 1 }}>{pSize >= 1000 ? `${Math.round(pSize/100)/10}K` : pSize}</span>
                </div>
              )
            })}
          </div>
        </>
      ) : (
        <RecentTrades symbol={symbol} />
      )}
    </div>
  )
}

export default OrderBook