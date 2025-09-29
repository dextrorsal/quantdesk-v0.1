import React, { useState, useEffect, useRef, useCallback } from 'react';
import SimpleChart from '../components/charts/SimpleChart'
import TradingViewWidgetNew from '../components/TradingViewWidgetNew';
import TradingViewTickerTape from '../components/TradingViewTickerTape';
import { usePrice } from '../contexts/PriceContext';

// Drift-inspired TradingTab component with QuantDesk theme
const TradingTabDrift: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC');
  const [selectedInterval, setSelectedInterval] = useState('1h');
  const [chartType, setChartType] = useState<'widget' | 'lightweight'>('widget');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('market');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderSize, setOrderSize] = useState('0');
  const [orderPrice, setOrderPrice] = useState('');
  const [leverage, setLeverage] = useState('1x');
  const [showPairDropdown, setShowPairDropdown] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  // Get real-time prices
  const { getPrice } = usePrice();

  const symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'ARB', 'OP', 'DOGE', 'ADA', 'DOT', 'LINK'];

  // Close dropdown when clicking outside
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(event.target as Node)) {
        setShowPairDropdown(false);
      }
    };

    if (showPairDropdown) {
      document.addEventListener('mousedown', handleClickOutside);
    }

    return () => {
      document.removeEventListener('mousedown', handleClickOutside);
    };
  }, [showPairDropdown]);

  const currentPrice = getPrice(`${selectedSymbol}/USDT`)?.price || 0;
  const priceChange = getPrice(`${selectedSymbol}/USDT`)?.change || 0;
  const priceChangePercent = getPrice(`${selectedSymbol}/USDT`)?.changePercent || 0;

  const formatCurrency = (value: number) => new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2
  }).format(value);

  const formatPercent = (value: number) => `${value >= 0 ? '+' : ''}${value.toFixed(2)}%`;

  const getChangeColor = (value: number) => {
    if (value > 0) return 'text-green-500';
    if (value < 0) return 'text-red-500';
    return 'text-gray-400';
  };

  const filteredSymbols = symbols.filter(symbol =>
    symbol.toLowerCase().includes(searchQuery.toLowerCase())
  );

  return (
    <div style={{
      display: 'flex',
      flexDirection: 'column',
      height: '100%',
      backgroundColor: '#000000',
      overflow: 'hidden'
    }}>

      {/* TradingView Ticker Tape */}
      <div style={{ backgroundColor: 'transparent', borderBottom: '1px solid #333333' }}>
        <TradingViewTickerTape 
          height={50}
          colorTheme="dark"
          isTransparent={true}
          displayMode="adaptive"
        />
      </div>

      {/* Main Content Area - Drift Layout */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        
        {/* Left Panel - Chart Area */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          
          {/* Chart Header - Drift Style */}
          <div style={{
            height: '60px',
            backgroundColor: '#000000',
            borderBottom: '1px solid #333333',
            display: 'flex',
            alignItems: 'center',
            padding: '0 20px',
            justifyContent: 'space-between',
            flexShrink: 0
          }}>
            {/* Left - Symbol Info */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              {/* Chart Type Toggle */}
              <div style={{ display: 'flex', backgroundColor: '#1a1a1a', borderRadius: '8px', padding: '2px' }}>
                <button
                  onClick={() => setChartType('widget')}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: chartType === 'widget' ? 'var(--primary-500)' : 'transparent',
                    color: chartType === 'widget' ? '#ffffff' : '#9ca3af',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    border: 'none',
                    outline: 'none',
                    transition: 'background-color 0.2s ease, color 0.2s ease'
                  }}
                >
                  Widget
                </button>
                <button
                  onClick={() => setChartType('lightweight')}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: chartType === 'lightweight' ? 'var(--primary-500)' : 'transparent',
                    color: chartType === 'lightweight' ? '#ffffff' : '#9ca3af',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    border: 'none',
                    outline: 'none',
                    transition: 'background-color 0.2s ease, color 0.2s ease'
                  }}
                >
                  Lightweight
                </button>
              </div>

              {/* Symbol Dropdown */}
              <div className="relative" ref={dropdownRef}>
                <button
                  onClick={() => setShowPairDropdown(!showPairDropdown)}
                  className="flex items-center gap-2 bg-gray-800 hover:bg-gray-700 text-white px-4 py-2 rounded-lg transition-colors"
                >
                  <span className="text-lg font-bold">{selectedSymbol}/USDT</span>
                  <svg className="w-4 h-4 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
                  </svg>
                </button>
                {showPairDropdown && (
                  <div className="absolute z-20 bg-gray-900 border border-gray-700 rounded-lg shadow-lg mt-2 w-48 max-h-60 overflow-y-auto">
                    <input
                      type="text"
                      placeholder="Search symbols..."
                      className="w-full p-2 bg-gray-800 border-b border-gray-700 text-white text-sm focus:outline-none"
                      value={searchQuery}
                      onChange={(e) => setSearchQuery(e.target.value)}
                    />
                    {filteredSymbols.map(symbol => (
                      <div
                        key={symbol}
                        className="px-4 py-2 text-sm text-gray-300 hover:bg-gray-700 hover:text-white cursor-pointer"
                        onClick={() => {
                          setSelectedSymbol(symbol);
                          setShowPairDropdown(false);
                          setSearchQuery('');
                        }}
                      >
                        {symbol}/USDT
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Live Price */}
              <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#ffffff' }}>
                {formatCurrency(currentPrice)}
                <span className={`ml-2 text-sm ${getChangeColor(priceChangePercent)}`}>
                  {formatPercent(priceChangePercent)}
                </span>
              </div>

              {/* Timeframe Selector */}
              <div style={{ display: 'flex', gap: '8px' }}>
                {['1m', '5m', '15m', '1h', '4h', '1d'].map(interval => (
                  <button
                    key={interval}
                    onClick={() => setSelectedInterval(interval)}
                    style={{
                      padding: '6px 10px',
                      backgroundColor: selectedInterval === interval ? 'var(--primary-500)' : '#1a1a1a',
                      color: selectedInterval === interval ? '#ffffff' : '#9ca3af',
                      borderRadius: '6px',
                      fontSize: '12px',
                      fontWeight: '500',
                      cursor: 'pointer',
                      border: 'none',
                      outline: 'none',
                      transition: 'background-color 0.2s ease, color 0.2s ease'
                    }}
                  >
                    {interval}
                  </button>
                ))}
              </div>
            </div>

            {/* Right - Market Data */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Mark:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>{formatCurrency(currentPrice)}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Index:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>{formatCurrency(currentPrice)}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>24h High:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>{formatCurrency(getPrice(`${selectedSymbol}/USDT`)?.high24h || 0)}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>24h Low:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>{formatCurrency(getPrice(`${selectedSymbol}/USDT`)?.low24h || 0)}</span>
              </div>
            </div>
          </div>

          {/* Chart Area - Drift Style */}
          <div style={{ 
            flex: 1,
            position: 'relative', 
            backgroundColor: '#0a0a0a',
            border: '1px solid #1a1a1a',
            borderRadius: '8px',
            overflow: 'hidden',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
            margin: '16px',
            minHeight: '400px',
            display: 'flex',
            flexDirection: 'column'
          }}>
            {chartType === 'widget' ? (
              <div style={{ flex: 1, width: '100%', height: '100%' }}>
                <TradingViewWidgetNew />
              </div>
            ) : (
              <SimpleChart 
                symbol={selectedSymbol} 
                height={400}
                timeframe={selectedInterval}
              />
            )}
          </div>

          {/* Bottom Info Bar - Drift Style */}
          <div style={{
            height: '40px',
            backgroundColor: '#000000',
            borderTop: '1px solid #333333',
            display: 'flex',
            alignItems: 'center',
            padding: '0 20px',
            justifyContent: 'space-between',
            flexShrink: 0
          }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>24h Volume:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>{formatCurrency(getPrice(`${selectedSymbol}/USDT`)?.volume24h || 0)}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Funding Rate:</span>
                <span style={{ fontSize: '12px', color: '#10b981', fontWeight: '600' }}>+0.0100%</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Next Funding:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>06:46:34</span>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Order Book & Order Entry - Drift Style */}
        <div style={{ 
          width: '400px', 
          backgroundColor: '#000000', 
          borderLeft: '1px solid #333333', 
          display: 'flex', 
          flexDirection: 'column', 
          flexShrink: 0 
        }}>
          
          {/* Order Book - Drift Style */}
          <div style={{ flex: 1, borderBottom: '1px solid #333333', display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid #333333' }}>
              <h3 style={{ fontSize: '14px', fontWeight: '600', color: '#ffffff' }}>Order Book</h3>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '10px 20px' }}>
              {/* Mock Order Book Data */}
              <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#9ca3af', marginBottom: '8px' }}>
                <span>Price (USDT)</span>
                <span>Size ({selectedSymbol})</span>
              </div>
              {/* Asks */}
              {[109630.01, 109608.09, 109586.17, 109564.25, 109542.33].map((price, index) => (
                <div key={`ask-${index}`} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', color: '#ef5350', padding: '2px 0' }}>
                  <span>{formatCurrency(price)}</span>
                  <span>{(Math.random() * 0.5 + 0.1).toFixed(3)}</span>
                </div>
              ))}
              <div style={{ textAlign: 'center', padding: '8px 0', fontSize: '12px', color: '#9ca3af', borderTop: '1px solid #333333', borderBottom: '1px solid #333333', margin: '8px 0' }}>
                Spread: <span style={{ color: '#ffffff' }}>0.15</span>
              </div>
              {/* Bids */}
              {[109520.42, 109498.50, 109476.58, 109454.66, 109432.74].map((price, index) => (
                <div key={`bid-${index}`} style={{ display: 'flex', justifyContent: 'space-between', fontSize: '13px', color: '#26a69a', padding: '2px 0' }}>
                  <span>{formatCurrency(price)}</span>
                  <span>{(Math.random() * 0.5 + 0.1).toFixed(3)}</span>
                </div>
              ))}
            </div>
          </div>

          {/* Order Entry - Drift Style */}
          <div style={{ padding: '20px', borderBottom: '1px solid #333333' }}>
            <h3 style={{ fontSize: '14px', fontWeight: '600', color: '#ffffff', marginBottom: '16px' }}>Place Order</h3>
            
            {/* Buy/Sell Toggle */}
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '16px' }}>
              <div style={{ display: 'flex', gap: '5px', backgroundColor: '#1a1a1a', borderRadius: '8px', padding: '2px' }}>
                <button
                  onClick={() => setSide('buy')}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: side === 'buy' ? '#26a69a' : 'transparent',
                    color: side === 'buy' ? '#ffffff' : '#9ca3af',
                    borderRadius: '6px',
                    fontSize: '14px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    border: 'none',
                    outline: 'none',
                    transition: 'background-color 0.2s ease, color 0.2s ease'
                  }}
                >
                  Buy
                </button>
                <button
                  onClick={() => setSide('sell')}
                  style={{
                    padding: '8px 16px',
                    backgroundColor: side === 'sell' ? '#ef5350' : 'transparent',
                    color: side === 'sell' ? '#ffffff' : '#9ca3af',
                    borderRadius: '6px',
                    fontSize: '14px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    border: 'none',
                    outline: 'none',
                    transition: 'background-color 0.2s ease, color 0.2s ease'
                  }}
                >
                  Sell
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Leverage:</span>
                <select
                  value={leverage}
                  onChange={(e) => setLeverage(e.target.value)}
                  style={{
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333333',
                    borderRadius: '6px',
                    padding: '6px 8px',
                    color: '#ffffff',
                    fontSize: '12px',
                    outline: 'none'
                  }}
                >
                  {['1x', '5x', '10x', '20x', '50x', '100x'].map(lvl => (
                    <option key={lvl} value={lvl}>{lvl}</option>
                  ))}
                </select>
              </div>
            </div>

            {/* Order Type */}
            <div style={{ display: 'flex', gap: '5px', backgroundColor: '#1a1a1a', borderRadius: '8px', padding: '2px', marginBottom: '16px' }}>
              {['market', 'limit', 'stop'].map(type => (
                <button
                  key={type}
                  onClick={() => setOrderType(type as 'market' | 'limit' | 'stop')}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: orderType === type ? 'var(--primary-500)' : 'transparent',
                    color: orderType === type ? '#ffffff' : '#9ca3af',
                    borderRadius: '6px',
                    fontSize: '13px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    border: 'none',
                    outline: 'none',
                    transition: 'background-color 0.2s ease, color 0.2s ease',
                    textTransform: 'capitalize'
                  }}
                >
                  {type}
                </button>
              ))}
            </div>

            {/* Price Input */}
            {orderType === 'limit' && (
              <div style={{ marginBottom: '16px' }}>
                <label style={{ display: 'block', fontSize: '12px', color: '#9ca3af', marginBottom: '5px' }}>Price (USDT)</label>
                <input
                  type="number"
                  value={orderPrice}
                  onChange={(e) => setOrderPrice(e.target.value)}
                  placeholder="Enter price"
                  style={{
                    width: '100%',
                    padding: '10px 12px',
                    backgroundColor: '#1a1a1a',
                    border: '1px solid #333333',
                    borderRadius: '6px',
                    color: '#ffffff',
                    fontSize: '14px',
                    outline: 'none'
                  }}
                />
              </div>
            )}

            {/* Size Input */}
            <div style={{ marginBottom: '16px' }}>
              <label style={{ display: 'block', fontSize: '12px', color: '#9ca3af', marginBottom: '5px' }}>Size ({selectedSymbol})</label>
              <input
                type="number"
                value={orderSize}
                onChange={(e) => setOrderSize(e.target.value)}
                placeholder="Enter amount"
                style={{
                  width: '100%',
                  padding: '10px 12px',
                  backgroundColor: '#1a1a1a',
                  border: '1px solid #333333',
                  borderRadius: '6px',
                  color: '#ffffff',
                  fontSize: '14px',
                  outline: 'none'
                }}
              />
            </div>

            {/* Order Summary */}
            <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '12px', color: '#9ca3af', marginBottom: '16px' }}>
              <span>Available Balance: <span style={{ color: '#ffffff' }}>0.00 USDT</span></span>
              <span>Order Value: <span style={{ color: '#ffffff' }}>N/A</span></span>
            </div>

            {/* Submit Button */}
            <button
              style={{
                width: '100%',
                padding: '12px',
                backgroundColor: side === 'buy' ? '#26a69a' : '#ef5350',
                color: '#ffffff',
                borderRadius: '6px',
                fontSize: '16px',
                fontWeight: '600',
                cursor: 'pointer',
                border: 'none',
                outline: 'none',
                transition: 'background-color 0.2s ease'
              }}
            >
              {side === 'buy' ? `Buy ${selectedSymbol}` : `Sell ${selectedSymbol}`}
            </button>
          </div>

          {/* Recent Trades - Drift Style */}
          <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
            <div style={{ padding: '16px 20px', borderBottom: '1px solid #333333' }}>
              <h3 style={{ fontSize: '14px', fontWeight: '600', color: '#ffffff' }}>Recent Trades</h3>
            </div>
            <div style={{ flex: 1, overflowY: 'auto', padding: '10px 20px' }}>
              <div style={{ fontSize: '12px', color: '#9ca3af', textAlign: 'center', padding: '20px 0' }}>
                No recent trades
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingTabDrift;
