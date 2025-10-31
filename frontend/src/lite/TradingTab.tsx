import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import ConditionalOrderForm from '../components/trading/ConditionalOrderForm';
import OrdersStatusPanel from '../components/trading/OrdersStatusPanel';
import { useParams } from 'react-router-dom';
import { QuantDeskTradingViewChart } from '../components/charts/BorrowedChart';
import { usePrice } from '../contexts/PriceContext';
import { useMarkets } from '../contexts/MarketContext';
import { useAccount } from '../contexts/AccountContext';
import { useResponsiveDesign } from '../hooks/useResponsiveDesign';
import { WebSocketService } from '../services/websocketService';
import { useTickerClick } from '../hooks/useTickerClick';

// Modern Error Boundary Component
class TradingErrorBoundary extends React.Component<
  { children: React.ReactNode },
  { hasError: boolean; error?: Error }
> {
  constructor(props: { children: React.ReactNode }) {
    super(props);
    this.state = { hasError: false };
  }

  static getDerivedStateFromError(error: Error) {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    console.error('TradingTab Error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div className="p-6 text-center text-red-400">
          <h2 className="text-xl font-bold mb-2">Trading Interface Error</h2>
          <p className="text-sm">Something went wrong with the trading interface.</p>
          <button 
            onClick={() => this.setState({ hasError: false })}
            className="mt-4 px-4 py-2 bg-primary-500 text-white hover:bg-primary-600"
          >
            Try Again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}

// Modern Order Book Component - Enhanced with backup functionality
const OrderBook: React.FC<{
  symbol: string;
  currentPrice: number;
  className?: string;
}> = ({ symbol, currentPrice, className = '' }) => {
  const [bids, setBids] = useState<Array<[number, number]>>([])
  const [asks, setAsks] = useState<Array<[number, number]>>([])
  const [activeTab, setActiveTab] = useState<'orderbook' | 'trades'>('orderbook')
  const [priceIncrement, setPriceIncrement] = useState<'$1' | '$0.1' | '$0.01'>('$1')
  const timerRef = useRef<number | null>(null)

  // WebSocket service for real-time data
  const wsService = useMemo(() => new WebSocketService('ws://localhost:3002'), [])

  useEffect(() => {
    // Connect to WebSocket for real-time order book updates
    const connectWebSocket = async () => {
      try {
        await wsService.connect()
        
        if (wsService.getConnectionStatus()) {
          // Subscribe to order book updates for this symbol
          const unsubscribeOrderBook = wsService.subscribe(`orderbook:${symbol}`, (data) => {
            if (data?.bids && data?.asks) {
              setBids(data.bids)
              setAsks(data.asks)
            }
          })
          
          // Cleanup subscription on unmount
          return unsubscribeOrderBook
        }
      } catch (error) {
        console.warn('WebSocket connection failed, using fallback:', error)
      }
    }

    const cleanup = connectWebSocket()
    
    return () => {
      cleanup?.then(unsubscribe => unsubscribe?.())
    }
  }, [symbol, wsService])

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
      {/* Header - Compact */}
      <div style={{ padding: 8, borderBottom: '1px solid var(--primary-500)' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', color: 'var(--text-secondary)', fontSize: 9, fontWeight: 500 }}>
          <span style={{ flex: 1, textAlign: 'left' }}>Price</span>
          <span style={{ flex: 1, textAlign: 'right' }}>Size</span>
        </div>
      </div>

      {/* Tab Content */}
      {activeTab === 'orderbook' ? (
        <>
          {/* Asks with heatmap bars (top) */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {asks.slice(0, 12).map(([pPrice, pSize], i) => {
              const pct = (pSize / maxAsk) * 100
              const p = pPrice || currentPrice * (1.001 + i * 0.0001)
              return (
                <div key={`ask-${i}`} style={{ position: 'relative', height: 14, display: 'flex', alignItems: 'center', padding: '0 6px', borderBottom: '1px solid var(--background-secondary)' }}>
                  <div style={{ position: 'absolute', right: 0, top: 0, height: '100%', width: `${pct}%`, backgroundColor: '#ef4444', opacity: 0.15 }} />
                  <span style={{ color: '#ef4444', fontSize: 9, zIndex: 1 }}>{p.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                  <span style={{ flex: 1 }} />
                  <span style={{ color: '#ffffff', fontSize: 8, zIndex: 1 }}>{pSize >= 1000 ? `${Math.round(pSize/100)/10}K` : pSize}</span>
                </div>
              )
            })}
          </div>

          {/* Mid price / spread */}
          <div style={{ height: 24, display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0 6px', borderTop: '1px solid var(--primary-500)', borderBottom: '1px solid var(--primary-500)', backgroundColor: 'var(--background-secondary)' }}>
            <div style={{ fontSize: 12, fontWeight: 700, color: 'var(--text-primary)' }}>{currentPrice.toLocaleString()}</div>
            <div style={{ fontSize: 8, color: 'var(--text-secondary)' }}>${spread.value.toFixed(2)}</div>
          </div>

          {/* Bids with heatmap bars (bottom) */}
          <div style={{ display: 'flex', flexDirection: 'column' }}>
            {bids.slice(0, 12).map(([pPrice, pSize], i) => {
              const pct = (pSize / maxBid) * 100
              const p = pPrice || currentPrice * (0.999 - i * 0.0001)
              return (
                <div key={`bid-${i}`} style={{ position: 'relative', height: 14, display: 'flex', alignItems: 'center', padding: '0 6px', borderBottom: '1px solid var(--background-secondary)' }}>
                  <div style={{ position: 'absolute', left: 0, top: 0, height: '100%', width: `${pct}%`, backgroundColor: '#10b981', opacity: 0.15 }} />
                  <span style={{ color: '#10b981', fontSize: 9, zIndex: 1 }}>{p.toLocaleString(undefined, { maximumFractionDigits: 2 })}</span>
                  <span style={{ flex: 1 }} />
                  <span style={{ color: '#ffffff', fontSize: 8, zIndex: 1 }}>{pSize >= 1000 ? `${Math.round(pSize/100)/10}K` : pSize}</span>
                </div>
              )
            })}
          </div>
        </>
      ) : (
        <div className="p-4 text-center text-gray-400">
          <p>Recent trades will be displayed here</p>
        </div>
      )}
    </div>
  );
};

// Modern Trading Panel Component - Enhanced with Hyperliquid/Drift-style functionality
const TradingPanel: React.FC<{
  symbol: string;
  currentPrice: number;
  className?: string;
}> = ({ symbol, currentPrice, className = '' }) => {
  const [orderType, setOrderType] = useState<'market' | 'limit'>('market');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderSize, setOrderSize] = useState('0');
  const [orderPrice, setOrderPrice] = useState('');
  const [leverage, setLeverage] = useState(1);
  const [reduceOnly, setReduceOnly] = useState(false);
  const [slippagePreset, setSlippagePreset] = useState<'dynamic'|'0.1'|'0.5'|'1'|'custom'>('dynamic');
  const [slippageCustom, setSlippageCustom] = useState('');
  const [marginMode, setMarginMode] = useState<'cross' | 'isolated'>('isolated');
  const [maxMode, setMaxMode] = useState(false);

  // Get real account data
  const { 
    accountState, 
    collateralAccounts, 
    totalBalance, 
    canTrade,
    loading: accountLoading 
  } = useAccount()
  
  // Calculate available balance for trading
  const availableBalance = totalBalance || 0

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault();
    console.log('Order submitted:', { orderType, side, orderSize, orderPrice, leverage, marginMode, maxMode });
  }, [orderType, side, orderSize, orderPrice, leverage, marginMode, maxMode]);

  const formatSize = (value: string) => {
    const num = parseFloat(value)
    return isNaN(num) ? '0' : num.toString()
  }

  const calculateOrderValue = () => {
    const sizeNum = parseFloat(orderSize)
    const priceNum = orderType === 'market' ? currentPrice : parseFloat(orderPrice)
    return sizeNum * priceNum
  }

  const getFeeRate = () => {
    return orderType === 'market' ? '0.0000%' : '0.0000%' // Taker vs Maker fees
  }

  const leverageOptions = [1, 2, 5, 10, 20, 25, 50, 75, 100];

  return (
    <div className={`bg-black border border-primary-500 overflow-hidden ${className}`}>
      {/* Header */}
      <div className="bg-black border-b border-primary-500 px-2 py-1 flex items-center justify-between">
        <div>
          <div className="text-white font-semibold text-xs">Place Order</div>
          <div className="text-gray-400 text-[10px]">{symbol}</div>
        </div>
        <div className="text-right">
          <div className="text-white font-bold text-xs">${typeof currentPrice === 'number' ? currentPrice.toLocaleString() : '0'}</div>
          <div className="text-gray-400 text-[10px]">{symbol}/USDT</div>
        </div>
      </div>

      {/* Order Type Tabs */}
      <div className="px-2 pt-2 flex items-center gap-2 text-xs">
        <button type="button" onClick={() => setOrderType('market')} className={`px-2 py-1 rounded ${orderType==='market'?'bg-primary-500 text-white':'bg-gray-800 text-gray-300'}`}>Market</button>
        <button type="button" onClick={() => setOrderType('limit')} className={`px-2 py-1 rounded ${orderType==='limit'?'bg-primary-500 text-white':'bg-gray-800 text-gray-300'}`}>Limit</button>
      </div>

      {/* Trading Form */}
      <form onSubmit={handleSubmit} className="p-2 space-y-2">
        {/* Margin Mode */}
        <div>
          <label className="block text-white text-[10px] font-medium mb-1">Margin Mode:</label>
          <div className="grid grid-cols-2 gap-1">
            <button 
              type="button" 
              onClick={() => setMarginMode('cross')} 
              className={`px-2 py-1 text-[10px] transition-colors ${
                marginMode === 'cross' ? 'bg-primary-500 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
              aria-label="Set cross margin mode"
            >
              Cross
            </button>
            <button 
              type="button" 
              onClick={() => setMarginMode('isolated')} 
              className={`px-2 py-1 text-[10px] transition-colors ${
                marginMode === 'isolated' ? 'bg-primary-500 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
              aria-label="Set isolated margin mode"
            >
              Isolated
            </button>
          </div>
        </div>

        {/* Order Type */}
        <div>
          <label className="block text-white text-[10px] font-medium mb-1">Order Type</label>
          <div className="grid grid-cols-4 gap-1">
            {(['market', 'limit', 'stop', 'post-only'] as const).map(t => (
              <button
                key={t}
                type="button"
                onClick={() => setOrderType(t)}
                className={`px-1 py-1 text-[9px] transition-colors ${
                  orderType === t ? 'bg-primary-500 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
                aria-label={`Set order type to ${t}`}
              >
                {t === 'post-only' ? 'Post' : t.charAt(0).toUpperCase() + t.slice(1)}
              </button>
            ))}
          </div>
        </div>

        {/* Leverage with presets + slider + textbox */}
        <div>
          <div className="flex items-center justify-between mb-1">
            <label className="text-white text-[10px] font-medium">Leverage MAX</label>
            <button
              type="button"
              onClick={() => setMaxMode(!maxMode)}
              className={`px-2 py-1 text-[9px] transition-colors ${
                maxMode ? 'bg-primary-500 text-white' : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
              }`}
            >
              MAX
            </button>
          </div>
          <div className="grid grid-cols-5 gap-1 mb-1">
            {[0,25,50,75,100].map(p => (
              <button key={p} type="button" onClick={() => setLeverage(Math.max(1, Math.round((p/100)*100)))} className={`py-1 text-[9px] ${leverage===p?'bg-primary-500 text-white':'bg-gray-800 text-gray-300 hover:bg-gray-700'}`}>{p}%</button>
            ))}
          </div>
          
          {/* Leverage Slider */}
          <div className="relative">
            <input
              type="range"
              min="1"
              max="100"
              value={leverage}
              onChange={(e) => setLeverage(Number(e.target.value))}
              className="w-full h-1 bg-gray-700 appearance-none cursor-pointer slider"
              style={{
                background: `linear-gradient(to right, #3b82f6 0%, #3b82f6 ${leverage}%, #374151 ${leverage}%, #374151 100%)`
              }}
            />
            <div className="flex justify-between text-[8px] text-gray-400 mt-1">
              <span>1x</span>
              <span>25x</span>
              <span>50x</span>
              <span>75x</span>
              <span>100x</span>
            </div>
            <div className="text-center mt-1">
              <span className="text-white font-bold text-sm">{leverage}x</span>
            </div>
          <input
            type="number"
            min={1}
            max={100}
            value={leverage}
            onChange={(e)=> setLeverage(Math.max(1, Math.min(100, Number(e.target.value||'1'))))}
            className="mt-1 w-full px-2 py-1 bg-gray-800 border border-gray-600 text-white text-xs"
          />
          </div>
        </div>

        {/* Size */}
        <div>
          <label htmlFor="order-size" className="block text-white text-[10px] font-medium mb-1">Size</label>
          <input
            id="order-size"
            type="number"
            value={orderSize}
            onChange={(e) => setOrderSize(formatSize(e.target.value))}
            className="w-full px-2 py-1 bg-gray-800 border border-gray-600 text-white text-xs focus:border-primary-500 focus:outline-none"
            placeholder="0.00"
            min="0"
            step="0.01"
            aria-describedby="order-size-help"
          />
          <div className="mt-1 grid grid-cols-4 gap-1">
            {[25, 50, 75, 100].map(p => (
              <button 
                key={p} 
                type="button" 
                onClick={() => setOrderSize(String(p))} 
                className="py-1 text-[9px] bg-gray-800 text-gray-300 border border-gray-600 hover:bg-gray-700 transition-colors"
              >
                {p}%
              </button>
            ))}
          </div>
          <p id="order-size-help" className="text-gray-400 text-[8px] mt-1">
            Amount in {symbol}
          </p>
        </div>

        {/* Price (Limit only) */}
        {orderType === 'limit' && (
          <div>
            <label htmlFor="order-price" className="block text-white text-[10px] font-medium mb-1">Price</label>
            <input
              id="order-price"
              type="number"
              value={orderPrice}
              onChange={(e) => setOrderPrice(e.target.value)}
              className="w-full px-2 py-1 bg-gray-800 border border-gray-600 text-white text-xs focus:border-primary-500 focus:outline-none"
              placeholder={typeof currentPrice === 'number' ? currentPrice.toFixed(2) : ''}
              min="0"
              step="0.01"
            />
          </div>
        )}

        {/* Reduce Only + Slippage */}
        <div className="grid grid-cols-2 gap-2">
          <label className="flex items-center gap-2 text-[10px] text-gray-300">
            <input type="checkbox" checked={reduceOnly} onChange={(e)=> setReduceOnly(e.target.checked)} />
            Reduce Only
          </label>
          <div>
            <div className="text-[10px] text-gray-300 mb-1">Slippage</div>
            <div className="flex items-center gap-1 text-[10px]">
              {(['dynamic','0.1','0.5','1'] as const).map(s => (
                <button key={s} type="button" onClick={()=> setSlippagePreset(s)} className={`px-2 py-1 rounded ${slippagePreset===s?'bg-primary-500 text-white':'bg-gray-800 text-gray-300 border border-gray-600 hover:bg-gray-700'}`}>{s==='dynamic'?'Dynamic':`${s}%`}</button>
              ))}
              <input value={slippageCustom} onChange={(e)=> { setSlippagePreset('custom'); setSlippageCustom(e.target.value); }} placeholder="custom %" className="w-16 px-2 py-1 bg-gray-800 border border-gray-600 text-white" />
            </div>
          </div>
        </div>

        {/* Order Summary */}
        <div className="bg-gray-800 p-2 space-y-1">
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Available Balance:</span>
            <span className="text-white">
              {accountLoading ? 'Loading...' : `$${availableBalance.toFixed(2)}`}
            </span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Oracle (est):</span>
            <span className="text-white">{typeof currentPrice === 'number' ? `$${currentPrice.toFixed(2)}` : 'N/A'}</span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Order Value:</span>
            <span className="text-white">
              {calculateOrderValue() > 0 ? `$${calculateOrderValue().toFixed(2)}` : 'N/A'}
            </span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Fee:</span>
            <span className="text-white">Taker {getFeeRate()}</span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Est. Liq Price:</span>
            <span className="text-white">—</span>
          </div>
          {(parseFloat(orderSize) || 0) <= 0 && (
            <div className="text-[9px] text-yellow-400">Enter a size to enable TP/SL and precise estimates.</div>
          )}
        </div>

        {/* Buy/Sell Buttons */}
        <div className="grid grid-cols-2 gap-1">
          <button
            type="submit"
            className={`py-2 px-2 font-semibold text-xs transition-colors ${
              side === 'buy'
                ? 'bg-green-600 hover:bg-green-700 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-600'
            }`}
            onClick={() => setSide('buy')}
          >
            Buy {symbol}
          </button>
          <button
            type="submit"
            className={`py-2 px-2 font-semibold text-xs transition-colors ${
              side === 'sell'
                ? 'bg-red-600 hover:bg-red-700 text-white'
                : 'bg-gray-800 text-gray-300 hover:bg-gray-700 border border-gray-600'
            }`}
            onClick={() => setSide('sell')}
          >
            Sell {symbol}
          </button>
        </div>
      </form>

      {/* TP/SL inside Limit (Drift-style) */}
      {orderType === 'limit' && (
        <div className="p-2 border-t border-primary-500">
          <div className="text-[10px] text-gray-300 mb-2">TP/SL</div>
          {(parseFloat(orderSize) || 0) > 0 ? (
            <ConditionalOrderForm symbol={`${symbol}-PERP`} />
          ) : (
            <div className="text-[10px] text-gray-400">Enter a valid size to configure TP/SL.</div>
          )}
        </div>
      )}
    </div>
  );
};

// Perp Overview Component - Margin and Position Summary
const PerpOverview: React.FC = () => {
  const { getPrice } = usePrice()
  
  // Get real account data
  const { 
    accountState, 
    collateralAccounts, 
    positions, 
    totalBalance, 
    accountHealth,
    loading: accountLoading 
  } = useAccount()
  
  // Calculate real perp data from account state
  const totalMargin = totalBalance || 0
  const usedMargin = positions.reduce((sum, pos) => sum + (pos.margin || 0), 0)
  const availableMargin = totalMargin - usedMargin
  const marginRatio = totalMargin > 0 ? (usedMargin / totalMargin) * 100 : 0
  const unrealizedPnL = positions.reduce((sum, pos) => sum + (pos.unrealizedPnL || 0), 0)
  const totalEquity = totalMargin + unrealizedPnL
  const openInterest = positions.reduce((sum, pos) => sum + ((pos.size || 0) * (pos.entryPrice || 0)), 0)
  
  // Mock funding data (this would come from backend in production)
  const fundingRate = -0.0125
  const nextFunding = '06:46:34'

  return (
    <div className="bg-black border border-primary-500 overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="px-2 py-1 border-b border-primary-500 bg-gray-800">
        <h3 className="text-white text-xs font-semibold">Perp Overview</h3>
      </div>

      {/* Content */}
      <div className="p-2 space-y-2 flex-1">
        {/* Margin Info */}
        <div className="space-y-1">
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Total Margin:</span>
            <span className="text-white">
              {accountLoading ? 'Loading...' : `$${totalMargin.toLocaleString()}`}
            </span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Used Margin:</span>
            <span className="text-white">
              {accountLoading ? 'Loading...' : `$${usedMargin.toLocaleString()}`}
            </span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Available:</span>
            <span className="text-green-400">
              {accountLoading ? 'Loading...' : `$${availableMargin.toLocaleString()}`}
            </span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Margin Ratio:</span>
            <span className={`font-semibold ${marginRatio > 80 ? 'text-red-400' : marginRatio > 60 ? 'text-yellow-400' : 'text-green-400'}`}>
              {accountLoading ? 'Loading...' : `${marginRatio.toFixed(1)}%`}
            </span>
          </div>
        </div>

        {/* PnL Info */}
        <div className="border-t border-primary-500 pt-2 space-y-1">
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Unrealized PnL:</span>
            <span className={`font-semibold ${unrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {accountLoading ? 'Loading...' : `${unrealizedPnL >= 0 ? '+' : ''}$${unrealizedPnL.toLocaleString()}`}
            </span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Total Equity:</span>
            <span className="text-white font-semibold">
              {accountLoading ? 'Loading...' : `$${totalEquity.toLocaleString()}`}
            </span>
          </div>
        </div>

        {/* Market Info */}
        <div className="border-t border-primary-500 pt-2 space-y-1">
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Open Interest:</span>
            <span className="text-white">
              {accountLoading ? 'Loading...' : `$${openInterest.toLocaleString()}`}
            </span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Funding Rate:</span>
            <span className={`${fundingRate >= 0 ? 'text-red-400' : 'text-green-400'}`}>
              {fundingRate >= 0 ? '+' : ''}{(fundingRate * 100).toFixed(4)}%
            </span>
          </div>
          <div className="flex justify-between text-[9px]">
            <span className="text-gray-400">Next Funding:</span>
            <span className="text-white">{nextFunding}</span>
          </div>
        </div>
      </div>
    </div>
  )
}

// Account Overview Component - Integrated with real account data
const AccountOverview: React.FC = () => {
  const { getPrice } = usePrice()
  const { 
    accountState, 
    collateralAccounts, 
    positions, 
    totalBalance, 
    accountHealth,
    loading,
    error 
  } = useAccount()
  
  const [activeTab, setActiveTab] = useState(0)

  const tabs = ['Balances', 'Positions', 'Open Orders', 'TWAP', 'Trade History', 'Funding History', 'Order History']
  
  // Calculate real totals from account data
  const totalUnrealizedPnL = positions.reduce((sum, p) => sum + (p.unrealizedPnL || 0), 0)
  const totalMargin = positions.reduce((sum, p) => sum + (p.margin || 0), 0)

  const renderTabContent = () => {
    switch (activeTab) {
      case 0: // Balances
        return (
          <div className="p-2">
            {loading ? (
              <div className="text-center text-gray-400 text-[9px]">Loading balances...</div>
            ) : error ? (
              <div className="text-center text-red-400 text-[9px]">Error loading balances</div>
            ) : (
              <div className="grid grid-cols-3 gap-2 text-[9px]">
                {/* SOL Balance */}
                <div className="text-center">
                  <div className="text-gray-400 mb-1">SOL</div>
                  <div className="text-white font-semibold">
                    {collateralAccounts.find(acc => acc.assetType === 'SOL')?.amount.toFixed(4) || '0.0000'}
                  </div>
                </div>
                {/* USDC Balance */}
                <div className="text-center">
                  <div className="text-gray-400 mb-1">USDC</div>
                  <div className="text-white font-semibold">
                    {collateralAccounts.find(acc => acc.assetType === 'USDC')?.amount.toFixed(2) || '0.00'}
                  </div>
                </div>
                {/* Total USD Value */}
                <div className="text-center">
                  <div className="text-gray-400 mb-1">USD</div>
                  <div className="text-white font-semibold">
                    ${totalBalance.toFixed(2)}
                  </div>
                </div>
              </div>
            )}
          </div>
        )
      
      case 1: // Positions
        return (
          <div className="p-2">
            {/* Positions Table Header */}
            <div className="grid grid-cols-5 gap-1 pb-1 mb-2 border-b border-primary-500">
              <div className="text-gray-400 text-[8px] font-medium">Symbol</div>
              <div className="text-gray-400 text-[8px] font-medium">Size</div>
              <div className="text-gray-400 text-[8px] font-medium">Entry Price</div>
              <div className="text-gray-400 text-[8px] font-medium">Mark Price</div>
              <div className="text-gray-400 text-[8px] font-medium">PnL</div>
            </div>

            {positions.length > 0 ? (
              <div className="space-y-1">
                {positions.map((position, index) => {
                  const currentPrice = getPrice(`${position.market}/USDT`)?.price || position.markPrice || 0
                  const isProfit = (position.unrealizedPnL || 0) > 0
                  
                  return (
                    <div key={index} className="grid grid-cols-5 gap-1 py-1 text-[8px]">
                      <div className="text-white font-medium">{position.market || 'N/A'}</div>
                      <div className="text-white">{position.size?.toFixed(4) || '0.0000'}</div>
                      <div className="text-white">${position.entryPrice?.toLocaleString() || '0'}</div>
                      <div className="text-white">${currentPrice.toLocaleString()}</div>
                      <div className={`font-medium ${isProfit ? 'text-green-400' : 'text-red-400'}`}>
                        {isProfit ? '+' : ''}${(position.unrealizedPnL || 0).toFixed(2)}
                      </div>
                    </div>
                  )
                })}
              </div>
            ) : (
              <div className="text-center py-4 text-gray-400">
                <div className="text-sm mb-1">📊</div>
                <div className="text-[8px]">No open positions</div>
              </div>
            )}
          </div>
        )
      
      default:
        return (
          <div className="p-2 text-center text-gray-400">
            <div className="text-[8px]">No data available</div>
          </div>
        )
    }
  }

  return (
    <div className="bg-black border border-primary-500 overflow-hidden h-full flex flex-col">
      {/* Header */}
      <div className="px-2 py-1 border-b border-primary-500 bg-gray-800">
        <div className="flex justify-between items-center mb-1">
          <h3 className="text-white text-xs font-semibold">Account Overview</h3>
          <div className="flex gap-3 text-[9px]">
            <div className="text-center">
              <div className="text-gray-400 text-[8px]">Account Equity</div>
              <div className="text-white font-semibold">${(1000 + totalUnrealizedPnL).toFixed(2)}</div>
            </div>
            <div className="text-center">
              <div className="text-gray-400 text-[8px]">Unrealized PNL</div>
              <div className={`font-semibold ${totalUnrealizedPnL >= 0 ? 'text-green-400' : 'text-red-400'}`}>
                {totalUnrealizedPnL >= 0 ? '+' : ''}${totalUnrealizedPnL.toFixed(2)}
              </div>
            </div>
            <div className="text-center">
              <div className="text-gray-400 text-[8px]">Margin Ratio</div>
              <div className="text-white font-semibold">{(totalMargin / (1000 + totalUnrealizedPnL) * 100).toFixed(2)}%</div>
            </div>
          </div>
        </div>
      </div>

      {/* Tab Navigation */}
      <div className="flex border-b border-primary-500 bg-black overflow-x-auto">
        {tabs.map((tab, index) => (
          <button
            key={tab}
            onClick={() => setActiveTab(index)}
            className={`px-2 py-1 text-[9px] font-medium border-b-2 transition-colors whitespace-nowrap ${
              activeTab === index
                ? 'text-white border-primary-500'
                : 'text-gray-400 border-transparent hover:text-white'
            }`}
          >
            {tab}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="flex-1 overflow-auto">
        {renderTabContent()}
      </div>
    </div>
  )
}

// Modern Chart Component
const ChartSection: React.FC<{
  symbol: string;
  interval: string;
  height: number | string;
  dropdownRef: React.RefObject<HTMLDivElement>;
  showPairDropdown: boolean;
  setShowPairDropdown: (show: boolean) => void;
  handleSymbolSelect: (symbol: string) => void;
  symbols: string[];
  className?: string;
}> = ({ symbol, interval, height, dropdownRef, showPairDropdown, setShowPairDropdown, handleSymbolSelect, symbols, className = '' }) => {
  const handleIntervalChange = (newInterval: string) => {
    // Handle interval change
    console.log('Interval changed to:', newInterval);
  };

  const handleSymbolChange = (newSymbol: string) => {
    // Handle symbol change
    console.log('Symbol changed to:', newSymbol);
  };

  // Mock data for the header - in real app this would come from price context
  const mockPriceData = {
    price: 109376.50,
    change: 322.00,
    changePercent: 0.27,
    markPrice: 109350.25,
    indexPrice: 109380.75,
    high24h: 111200.00,
    low24h: 108500.00,
    volume24h: 7.95,
    volume24hUsd: 1.72,
    fundingRate: 0.0100,
    nextFunding: '06:46:34'
  };

  return (
    <div className={`bg-black border border-primary-500 overflow-hidden flex flex-col h-full ${className}`}>
      {/* Professional Chart Header - Like Quanto/Drift */}
      <div className="bg-black border-b border-primary-500 px-3 py-2 flex-shrink-0">
        {/* Top Row: Symbol, Price, Change, Timeframes */}
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center space-x-3">
            {/* Market Selection Dropdown - Like Quanto */}
            <div ref={dropdownRef} className="relative">
              <button 
                onClick={() => setShowPairDropdown(!showPairDropdown)}
                className="flex items-center gap-2 px-3 py-1 bg-gray-800 border border-gray-600 text-white font-semibold hover:bg-gray-700 transition-colors"
              >
                <span className="text-lg font-semibold">{symbol}</span>
                <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M6 9l6 6 6-6"/>
                </svg>
              </button>
              
              {showPairDropdown && (
                <div className="absolute top-full left-0 mt-1 w-48 bg-gray-800 border border-gray-600 shadow-lg z-10" role="listbox" aria-label="Select trading pair">
                  <div className="p-2">
                    {symbols.map((symbol) => (
                      <button
                        key={symbol}
                        onClick={() => handleSymbolSelect(symbol)}
                        className={`w-full text-left px-3 py-2 text-sm transition-colors ${
                          selectedSymbol === symbol
                            ? 'bg-primary-500 text-white'
                            : 'text-gray-300 hover:bg-gray-700'
                        }`}
                        role="option"
                        aria-selected={selectedSymbol === symbol}
                      >
                        {symbol}
                      </button>
                    ))}
                  </div>
                </div>
              )}
            </div>
            
            <div className="flex items-center space-x-2">
              <span className="text-lg font-mono text-white">
                ${mockPriceData.price.toLocaleString()}
              </span>
              <span className={`text-sm font-medium ${
                mockPriceData.change >= 0 ? 'text-green-400' : 'text-red-400'
              }`}>
                {mockPriceData.change >= 0 ? '+' : ''}{mockPriceData.change.toFixed(2)} 
                ({mockPriceData.changePercent >= 0 ? '+' : ''}{mockPriceData.changePercent.toFixed(2)}%)
              </span>
            </div>
          </div>
          <div className="flex items-center space-x-1">
            {['1m', '5m', '15m', '1h', '4h', '1d'].map((int) => (
              <button
                key={int}
                className={`px-2 py-1 text-xs transition-colors ${
                  interval === int
                    ? 'bg-primary-500 text-white'
                    : 'bg-gray-800 text-gray-300 hover:bg-gray-700'
                }`}
                onClick={() => handleIntervalChange(int)}
                aria-pressed={interval === int}
              >
                {int}
              </button>
            ))}
          </div>
        </div>

        {/* Bottom Row: Market Data */}
        <div className="grid grid-cols-4 gap-4 text-xs">
          <div className="flex justify-between">
            <span className="text-gray-400">Mark Price:</span>
            <span className="text-white font-mono">${mockPriceData.markPrice.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Index Price:</span>
            <span className="text-white font-mono">${mockPriceData.indexPrice.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">24h High:</span>
            <span className="text-white font-mono">${mockPriceData.high24h.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">24h Low:</span>
            <span className="text-white font-mono">${mockPriceData.low24h.toLocaleString()}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">24h Volume:</span>
            <span className="text-white font-mono">{mockPriceData.volume24h}M {symbol}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">24h Total:</span>
            <span className="text-white font-mono">${mockPriceData.volume24hUsd}B</span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Funding Rate:</span>
            <span className={`font-mono ${mockPriceData.fundingRate >= 0 ? 'text-green-400' : 'text-red-400'}`}>
              {mockPriceData.fundingRate >= 0 ? '+' : ''}{mockPriceData.fundingRate.toFixed(4)}%
            </span>
          </div>
          <div className="flex justify-between">
            <span className="text-gray-400">Next Funding:</span>
            <span className="text-white font-mono">{mockPriceData.nextFunding}</span>
          </div>
        </div>
      </div>

      {/* Chart Container */}
      <div className="relative flex-1">
        <QuantDeskTradingViewChart
          symbol={symbol}
          interval={interval}
          height={height}
          onSymbolChange={handleSymbolChange}
        />
      </div>
    </div>
  );
};

// Main TradingTab Component
const TradingTab: React.FC = () => {
  const { symbol: urlSymbol } = useParams<{ symbol?: string }>();
  const [selectedSymbol, setSelectedSymbol] = useState('BTC');
  const [selectedInterval, setSelectedInterval] = useState('1h');
  const [showPairDropdown, setShowPairDropdown] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Hooks
  const { getPrice, connectionStatus } = usePrice();
  const { markets, selectMarketBySymbol } = useMarkets();
  const screenSize = useResponsiveDesign();
  const { handleTickerClick } = useTickerClick();

  // Get current price
  const currentPrice = useMemo(() => {
    const price = getPrice(`${selectedSymbol}/USDT`)?.price;
    return typeof price === 'number' ? price : 0;
  }, [getPrice, selectedSymbol]);

  // Available symbols
  const symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'ARB', 'OP', 'DOGE', 'ADA', 'DOT', 'LINK'];

  // Handle symbol selection
  const handleSymbolSelect = useCallback((symbol: string) => {
    setSelectedSymbol(symbol);
    setShowPairDropdown(false);
    selectMarketBySymbol(symbol);
  }, [selectMarketBySymbol]);

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowPairDropdown(false);
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  return (
    <TradingErrorBoundary>
      <div className="flex flex-col h-screen bg-black text-white">
        {/* Main Content */}
        <div className="flex-1 p-2 overflow-hidden">
          {/* Desktop Layout - Chart Dominant */}
          <div className="hidden lg:grid lg:h-full lg:grid-rows-[60%_40%] lg:gap-2">
            {/* Top Row - Chart, Order Book, Trading Panel */}
            <div className="grid grid-cols-[68%_10.5%_21.5%] min-h-0">
              {/* Chart - Primary Focus (68% width - even more dominant) */}
              <div className="min-h-0">
                <ChartSection
                  symbol={selectedSymbol}
                  interval={selectedInterval}
                  height="100%"
                  dropdownRef={dropdownRef}
                  showPairDropdown={showPairDropdown}
                  setShowPairDropdown={setShowPairDropdown}
                  handleSymbolSelect={handleSymbolSelect}
                  symbols={symbols}
                />
              </div>
              
              {/* Order Book - Compact (10.5% width) */}
              <div className="min-h-0">
                <OrderBook
                  symbol={selectedSymbol}
                  currentPrice={currentPrice}
                  className="h-full"
                />
              </div>
              
              {/* Trading Panel (21.5% width) */}
              <div className="min-h-0">
                <TradingPanel
                  symbol={selectedSymbol}
                  currentPrice={currentPrice}
                  className="h-full"
                />
                <div className="mt-2 space-y-2">
                  <ConditionalOrderForm symbol={`${selectedSymbol}-PERP`} currentPrice={currentPrice} />
                  <OrdersStatusPanel />
                </div>
              </div>
            </div>
            
            {/* Bottom Row - Account Overview + Perp Overview (symmetrical) */}
            <div className="grid grid-cols-[1fr_32%] min-h-0">
              {/* Account Overview */}
              <div className="min-h-0">
                <AccountOverview />
                <div className="mt-2">
                  <ConditionalOrderForm symbol={`${selectedSymbol}-PERP`} currentPrice={currentPrice} />
                </div>
                {/* Orders Tab (Lite-style) */}
                <div className="mt-2 bg-black border border-primary-500">
                  <div className="px-2 py-1 border-b border-primary-500 text-xs text-white">Orders</div>
                  <div className="p-2">
                    <OrdersStatusPanel />
                  </div>
                </div>
              </div>
              
              {/* Perp Overview - Same width as Order Book + Trading Panel (10.5% + 21.5% = 32%) */}
              <div className="min-h-0">
                <PerpOverview />
              </div>
            </div>
          </div>

          {/* Mobile Layout */}
          <div className="lg:hidden space-y-4">
            {/* Mobile Chart */}
            <ChartSection
              symbol={selectedSymbol}
              interval={selectedInterval}
              height={400}
            />

            {/* Mobile Order Book */}
            <OrderBook
              symbol={selectedSymbol}
              currentPrice={currentPrice}
              className="h-96"
            />

            {/* Mobile Trading Panel */}
            <TradingPanel
              symbol={selectedSymbol}
              currentPrice={currentPrice}
            />
          </div>
        </div>

      </div>
    </TradingErrorBoundary>
  );
};

export default TradingTab;
