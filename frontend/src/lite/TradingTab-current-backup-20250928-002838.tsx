import React, { useState, useEffect, useRef, useCallback } from 'react';
import QuantDeskChart from '../components/charts/QuantDeskChart'
import TradingViewWidgetTest from '../components/TradingViewWidgetTest';
import TradingViewTickerTape from '../components/TradingViewTickerTape';
import { usePrice } from '../contexts/PriceContext';

// TradingTab component - updated for black backgrounds
const TradingTab: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC');
  const [selectedInterval, setSelectedInterval] = useState('1h');
  const [chartType, setChartType] = useState<'widget' | 'lightweight'>('lightweight');
  const [chartHeight] = useState(500); // Fixed height for better layout
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
  const intervals = ['1m', '5m', '15m', '1h', '4h', '1d'];
  const orderTypes = [
    { value: 'market', label: 'Market' },
    { value: 'limit', label: 'Limit' },
    { value: 'stop', label: 'Stop' }
  ];

  return (
    <div style={{
      width: '100%',
      height: '100%', // Use full available height from parent container
      backgroundColor: '#000000',
      color: '#ffffff',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>

      {/* TradingView Ticker Tape */}
      <div style={{ backgroundColor: 'transparent', borderBottom: '1px solid #333333' }}>
        <TradingViewTickerTape />
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        {/* Center Panel - Chart */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          {/* Chart Header */}
          <div style={{
            height: '60px',
            backgroundColor: '#000000',
            borderBottom: '1px solid var(--primary-500)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 24px',
            flexShrink: 0
          }}>
            {/* Left - Symbol Info with Sleek Dropdown */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              {/* Chart Type Toggle */}
              <div style={{ display: 'flex', backgroundColor: '#1a1a1a', borderRadius: '8px', padding: '2px' }}>
                <button
                  onClick={() => setChartType('widget')}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: chartType === 'widget' ? 'var(--primary-500)' : 'transparent',
                    border: 'none',
                    borderRadius: '6px',
                    color: '#ffffff',
                    fontSize: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  Widget
                </button>
                <button
                  onClick={() => setChartType('lightweight')}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: chartType === 'lightweight' ? 'var(--primary-500)' : 'transparent',
                    border: 'none',
                    borderRadius: '6px',
                    color: '#ffffff',
                    fontSize: '12px',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  Lightweight
                </button>
              </div>
              {/* Sleek Pair Selector Dropdown */}
              <div ref={dropdownRef} style={{ position: 'relative' }}>
                <button 
                  onClick={() => setShowPairDropdown(!showPairDropdown)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 12px',
                    backgroundColor: '#000000',
                    border: '1px solid var(--primary-500)',
                    borderRadius: '8px',
                    color: '#ffffff',
                    fontSize: '14px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    minWidth: '140px'
                  }}
                >
                  <span>{selectedSymbol}USDT</span>
                  <svg 
                    width="12" 
                    height="12" 
                    viewBox="0 0 24 24" 
                    fill="none" 
                    stroke="currentColor" 
                    strokeWidth="2"
                    style={{ transform: showPairDropdown ? 'rotate(180deg)' : 'rotate(0deg)', transition: 'transform 0.2s ease' }}
                  >
                    <path d="M6 9l6 6 6-6"/>
                  </svg>
                </button>
                
                {/* Dropdown Menu */}
                {showPairDropdown && (
                  <div style={{
                    position: 'absolute',
                    top: '100%',
                    left: '0',
                    marginTop: '4px',
                    backgroundColor: '#000000',
                    border: '1px solid var(--primary-500)',
                    borderRadius: '8px',
                    boxShadow: '0 10px 25px rgba(0, 0, 0, 0.3)',
                    zIndex: 1000,
                    minWidth: '280px',
                    maxHeight: '400px',
                    overflowY: 'auto'
                  }}>
                    {/* Search */}
                    <div style={{ padding: '12px', borderBottom: '1px solid var(--primary-500)' }}>
                      <input
                        type="text"
                        placeholder="Search pairs..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          backgroundColor: '#000000',
                          border: '1px solid var(--primary-500)',
                          borderRadius: '6px',
                          color: '#ffffff',
                          fontSize: '12px',
                          outline: 'none'
                        }}
                      />
                    </div>
                  
                  {/* Categories */}
                  <div style={{ padding: '8px 12px', borderBottom: '1px solid var(--primary-500)' }}>
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
                      <button style={{
                        padding: '4px 8px',
                        background: 'linear-gradient(135deg, var(--primary-500), var(--primary-600))',
                        border: 'none',
                        borderRadius: '4px',
                        color: '#ffffff',
                        fontSize: '10px',
                        cursor: 'pointer',
                        transition: 'all 0.3s ease',
                        boxShadow: '0 2px 4px rgba(0, 0, 0, 0.1)'
                      }}
                      onMouseEnter={(e) => {
                        e.currentTarget.style.transform = 'translateY(-1px)';
                        e.currentTarget.style.boxShadow = '0 4px 8px rgba(0, 0, 0, 0.2)';
                      }}
                      onMouseLeave={(e) => {
                        e.currentTarget.style.transform = 'translateY(0)';
                        e.currentTarget.style.boxShadow = '0 2px 4px rgba(0, 0, 0, 0.1)';
                      }}>
                        USDT-M
                      </button>
                      <button style={{
                        padding: '4px 8px',
                        backgroundColor: 'transparent',
                        border: '1px solid var(--primary-500)',
                        borderRadius: '4px',
                        color: '#9ca3af',
                        fontSize: '10px',
                        cursor: 'pointer'
                      }}>
                        Coin-M
                      </button>
                    </div>
                    <div style={{ display: 'flex', gap: '6px' }}>
                      <button style={{
                        padding: '4px 8px',
                        backgroundColor: 'var(--primary-500)',
                        border: 'none',
                        borderRadius: '4px',
                        color: '#ffffff',
                        fontSize: '10px',
                        cursor: 'pointer'
                      }}>
                        All
                      </button>
                      <button style={{
                        padding: '4px 8px',
                        backgroundColor: 'transparent',
                        border: '1px solid var(--primary-500)',
                        borderRadius: '4px',
                        color: '#9ca3af',
                        fontSize: '10px',
                        cursor: 'pointer'
                      }}>
                        New
                      </button>
                      <button style={{
                        padding: '4px 8px',
                        backgroundColor: 'transparent',
                        border: '1px solid var(--primary-500)',
                        borderRadius: '4px',
                        color: '#9ca3af',
                        fontSize: '10px',
                        cursor: 'pointer'
                      }}>
                        AI
                      </button>
                      <button style={{
                        padding: '4px 8px',
                        backgroundColor: 'transparent',
                        border: '1px solid var(--primary-500)',
                        borderRadius: '4px',
                        color: '#9ca3af',
                        fontSize: '10px',
                        cursor: 'pointer'
                      }}>
                        Meme
                      </button>
                    </div>
                  </div>
                  
                    {/* Pairs List */}
                    <div style={{ padding: '8px 0' }}>
                      {symbols
                        .filter(symbol => 
                          symbol.toLowerCase().includes(searchQuery.toLowerCase()) ||
                          `${symbol}USDT`.toLowerCase().includes(searchQuery.toLowerCase())
                        )
                        .map((symbol, index) => (
                          <button
                            key={symbol}
                            onClick={() => {
                              setSelectedSymbol(symbol);
                              setShowPairDropdown(false);
                              setSearchQuery('');
                            }}
                            style={{
                              width: '100%',
                              display: 'flex',
                              alignItems: 'center',
                              justifyContent: 'space-between',
                              padding: '8px 12px',
                              backgroundColor: symbol === selectedSymbol ? 'var(--primary-500)' : 'transparent',
                              color: '#ffffff',
                              fontSize: '12px',
                              cursor: 'pointer',
                              border: 'none',
                              textAlign: 'left',
                              transition: 'background-color 0.2s ease'
                            }}
                          >
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <span style={{ fontSize: '12px', color: '#9ca3af' }}>⭐</span>
                              <span style={{ fontWeight: '600' }}>{symbol}USDT</span>
                            </div>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                              <span style={{ color: '#ffffff' }}>{getPrice(`${symbol}/USDT`)?.price?.toFixed(3) || '0.000'}</span>
                              <span style={{ color: getPrice(`${symbol}/USDT`)?.changePercent >= 0 ? '#10b981' : '#ef4444' }}>
                                {getPrice(`${symbol}/USDT`)?.changePercent >= 0 ? '+' : ''}{getPrice(`${symbol}/USDT`)?.changePercent?.toFixed(2) || '0.00'}%
                              </span>
                            </div>
                          </button>
                        ))}
                    </div>
                  </div>
                )}
              </div>
              
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#10b981' }}>
                {getPrice(`${selectedSymbol}/USDT`)?.price?.toFixed(3) || '0.000'}
              </div>
              <div style={{ fontSize: '14px', color: getPrice(`${selectedSymbol}/USDT`)?.changePercent >= 0 ? '#10b981' : '#ef4444', fontWeight: '600' }}>
                {getPrice(`${selectedSymbol}/USDT`)?.changePercent >= 0 ? '+' : ''}{getPrice(`${selectedSymbol}/USDT`)?.changePercent?.toFixed(2) || '0.00'}%
              </div>
            </div>

            {/* Center - Timeframe Buttons */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              {intervals.map(interval => (
                <button
                  key={interval}
                  onClick={() => setSelectedInterval(interval)}
                  style={{
                    padding: '6px 12px',
                    backgroundColor: selectedInterval === interval ? '#000000' : 'transparent',
                    border: '1px solid var(--primary-500)',
                    borderRadius: '4px',
                    color: selectedInterval === interval ? '#64748b' : '#ffffff',
                    fontSize: '12px',
                    fontWeight: '600',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease'
                  }}
                >
                  {interval}
                </button>
              ))}
            </div>

            {/* Right - Chart Controls */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '24px' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Mark price:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>
                  ${getPrice(`${selectedSymbol}/USDT`)?.price?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Index price:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>
                  ${getPrice(`${selectedSymbol}/USDT`)?.price?.toFixed(2) || '0.00'}
                </span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>24h high:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>{(getPrice(`${selectedSymbol}/USDT`)?.price * 1.02 || 221.667).toFixed(3)}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>24h low:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>{(getPrice(`${selectedSymbol}/USDT`)?.price * 0.98 || 212.237).toFixed(3)}</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>24h quantity ({selectedSymbol}):</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>7.95M</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>24h total (USDT):</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>1.72B</span>
              </div>
            </div>
            
            <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
              <span style={{ fontSize: '12px', color: '#9ca3af' }}>Funding rate/Countdown:</span>
              <span style={{ fontSize: '12px', color: '#10b981', fontWeight: '600' }}>+0.0100%</span>
              <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>/</span>
              <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>06:46:34</span>
            </div>
          </div>

          {/* Chart Container */}
          <div style={{ 
            height: `${chartHeight}px`, 
            position: 'relative', 
            backgroundColor: '#0a0a0a',
            border: '1px solid #1a1a1a',
            borderRadius: '8px',
            overflow: 'hidden',
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
            flexShrink: 0,
            transition: 'all 0.3s cubic-bezier(0.4, 0, 0.2, 1)'
          }}>
            {chartType === 'widget' ? (
              <TradingViewWidgetTest />
            ) : (
              <QuantDeskChart 
                key={`${selectedSymbol}-${selectedInterval}`}
                symbol={selectedSymbol} 
                height={chartHeight}
                timeframe={selectedInterval}
              />
            )}
          </div>

          {/* Bottom Panel - Positions & Orders */}
          <div style={{
            flex: 1, // Expand to fill remaining space
            backgroundColor: '#0a0a0a',
            border: '1px solid #1a1a1a',
            borderRadius: '8px',
            display: 'flex',
            minHeight: '200px', // Minimum height to ensure readability
            boxShadow: '0 4px 12px rgba(0, 0, 0, 0.3)',
            overflow: 'hidden',
            marginTop: '16px'
          }}>
            {/* Positions Tab */}
            <div style={{ flex: 1, padding: '16px', display: 'flex', flexDirection: 'column' }}>
              <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#ffffff' }}>
                Positions
              </div>
              
              {/* P&L Summary */}
              <div style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>Total P&L:</span>
                  <span style={{ fontSize: '12px', color: '#10b981', fontWeight: '600' }}>+$0.00</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>Unrealized P&L:</span>
                  <span style={{ fontSize: '12px', color: '#10b981', fontWeight: '600' }}>+$0.00</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>Realized P&L:</span>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>$0.00</span>
                </div>
              </div>

              {/* Position Details */}
              <div style={{ flex: 1, overflowY: 'auto' }}>
                <div style={{ fontSize: '12px', color: '#9ca3af', textAlign: 'center', padding: '20px 0' }}>
                  No positions yet
                </div>
              </div>
            </div>

            {/* Open Orders Tab */}
            <div style={{ flex: 1, padding: '16px', borderLeft: '1px solid var(--primary-500)', display: 'flex', flexDirection: 'column' }}>
              <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#ffffff' }}>
                Open Orders
              </div>
              
              {/* Order Summary */}
              <div style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>Active Orders:</span>
                  <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>0</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>Total Value:</span>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>$0.00</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>Margin Used:</span>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>$0.00</span>
                </div>
              </div>

              {/* Order List */}
              <div style={{ flex: 1, overflowY: 'auto' }}>
                <div style={{ fontSize: '12px', color: '#9ca3af', textAlign: 'center', padding: '20px 0' }}>
                  No open orders
                </div>
              </div>
            </div>

            {/* Trade History Tab */}
            <div style={{ flex: 1, padding: '16px', borderLeft: '1px solid var(--primary-500)', display: 'flex', flexDirection: 'column' }}>
              <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#ffffff' }}>
                Trade History
              </div>
              
              {/* Trade Summary */}
              <div style={{ marginBottom: '16px' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>Today's Trades:</span>
                  <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>0</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>Total Volume:</span>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>$0.00</span>
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '8px' }}>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>Avg. Price:</span>
                  <span style={{ fontSize: '12px', color: '#9ca3af' }}>$0.00</span>
                </div>
              </div>

              {/* Trade List */}
              <div style={{ flex: 1, overflowY: 'auto' }}>
                <div style={{ fontSize: '12px', color: '#9ca3af', textAlign: 'center', padding: '20px 0' }}>
                  No trades yet
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Right Panel - Order Book & Trading */}
        <div style={{ width: '350px', display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--primary-500)' }}>
          {/* Order Book */}
          <div style={{ flex: 1, padding: '16px' }}>
            <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#ffffff' }}>
              Order Book
            </div>
            
            {/* Sell Orders */}
            <div style={{ marginBottom: '8px' }}>
              <div style={{ fontSize: '11px', color: '#9ca3af', marginBottom: '4px' }}>Price (USDT)</div>
              <div style={{ fontSize: '12px', color: '#ef4444' }}>{(getPrice(`${selectedSymbol}/USDT`)?.price * 1.0003 || 214.25).toFixed(2)}</div>
              <div style={{ fontSize: '12px', color: '#ef4444' }}>{(getPrice(`${selectedSymbol}/USDT`)?.price * 1.0001 || 214.20).toFixed(2)}</div>
              <div style={{ fontSize: '12px', color: '#ef4444' }}>{(getPrice(`${selectedSymbol}/USDT`)?.price * 0.9999 || 214.15).toFixed(2)}</div>
            </div>

            {/* Spread */}
            <div style={{ 
              height: '1px', 
              backgroundColor: 'var(--primary-500)', 
              margin: '8px 0',
              position: 'relative'
            }}>
              <div style={{
                position: 'absolute',
                top: '-8px',
                left: '50%',
                transform: 'translateX(-50%)',
                fontSize: '10px',
                color: '#9ca3af',
                backgroundColor: '#000000',
                padding: '0 8px'
              }}>
                Spread: 0.15
              </div>
            </div>

            {/* Buy Orders */}
            <div style={{ marginTop: '8px' }}>
              <div style={{ fontSize: '12px', color: '#10b981' }}>{(getPrice(`${selectedSymbol}/USDT`)?.price * 0.9997 || 214.10).toFixed(2)}</div>
              <div style={{ fontSize: '12px', color: '#10b981' }}>{(getPrice(`${selectedSymbol}/USDT`)?.price * 0.9995 || 214.05).toFixed(2)}</div>
              <div style={{ fontSize: '12px', color: '#10b981' }}>{(getPrice(`${selectedSymbol}/USDT`)?.price * 0.9993 || 214.00).toFixed(2)}</div>
            </div>
          </div>

          {/* Trading Panel */}
          <div style={{ 
            height: '400px', 
            padding: '16px', 
            borderTop: '1px solid var(--primary-500)',
            backgroundColor: '#000000'
          }}>
            {/* Margin Mode & Leverage */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Margin Mode:</span>
                <button style={{
                  padding: '4px 8px',
                  backgroundColor: 'var(--primary-500)',
                  border: 'none',
                  borderRadius: '4px',
                  color: '#ffffff',
                  fontSize: '11px',
                  cursor: 'pointer'
                }}>
                  Isolated
                </button>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Leverage:</span>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <button style={{
                    padding: '4px 8px',
                    backgroundColor: 'var(--primary-500)',
                    border: 'none',
                    borderRadius: '4px',
                    color: '#ffffff',
                    fontSize: '11px',
                    cursor: 'pointer'
                  }}>
                    20x
                  </button>
                  <button style={{
                    padding: '4px 8px',
                    backgroundColor: '#000000',
                    border: '1px solid var(--primary-500)',
                    borderRadius: '4px',
                    color: '#9ca3af',
                    fontSize: '11px',
                    cursor: 'pointer'
                  }}>
                    20x
                  </button>
                </div>
              </div>
            </div>

            {/* Order Type Tabs */}
            <div style={{ display: 'flex', marginBottom: '16px' }}>
              {orderTypes.map(type => (
                <button
                  key={type.value}
                  onClick={() => setOrderType(type.value as any)}
                  style={{
                    flex: 1,
                    padding: '8px',
                    backgroundColor: orderType === type.value ? '#000000' : 'transparent',
                    border: '1px solid var(--primary-500)',
                    borderBottom: 'none',
                    color: orderType === type.value ? '#64748b' : '#ffffff',
                    fontSize: '12px',
                    fontWeight: '600',
                    cursor: 'pointer'
                  }}
                >
                  {type.label}
                </button>
              ))}
              <button style={{
                flex: 1,
                padding: '8px',
                backgroundColor: 'transparent',
                border: '1px solid var(--primary-500)',
                borderBottom: 'none',
                color: '#9ca3af',
                fontSize: '12px',
                fontWeight: '600',
                cursor: 'pointer'
              }}>
                Post only
              </button>
            </div>

            {/* Leverage */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '8px' }}>Leverage</div>
              <select
                value={leverage}
                onChange={(e) => setLeverage(e.target.value)}
                style={{
                  width: '100%',
                  padding: '8px',
                  backgroundColor: '#000000',
                  border: '1px solid var(--primary-500)',
                  borderRadius: '4px',
                  color: '#ffffff',
                  fontSize: '12px'
                }}
              >
                <option value="1x">1x</option>
                <option value="2x">2x</option>
                <option value="5x">5x</option>
                <option value="10x">10x</option>
                <option value="25x">25x</option>
              </select>
            </div>

            {/* Order Size */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '8px' }}>Size</div>
              <input
                type="text"
                value={orderSize}
                onChange={(e) => setOrderSize(e.target.value)}
                placeholder="0.0"
                style={{
                  width: '100%',
                  padding: '8px',
                  backgroundColor: '#000000',
                  border: '1px solid var(--primary-500)',
                  borderRadius: '4px',
                  color: '#ffffff',
                  fontSize: '14px'
                }}
              />
            </div>

            {/* Order Price (for Limit orders) */}
            {orderType === 'limit' && (
              <div style={{ marginBottom: '16px' }}>
                <div style={{ fontSize: '12px', color: '#9ca3af', marginBottom: '8px' }}>Price</div>
                <input
                  type="text"
                  value={orderPrice}
                  onChange={(e) => setOrderPrice(e.target.value)}
                  placeholder="0.0"
                  style={{
                    width: '100%',
                    padding: '8px',
                    backgroundColor: '#000000',
                    border: '1px solid var(--primary-500)',
                    borderRadius: '4px',
                    color: '#ffffff',
                    fontSize: '14px'
                  }}
                />
              </div>
            )}

            {/* Buy/Sell Buttons */}
            <div style={{ display: 'flex', gap: '8px', marginBottom: '16px' }}>
              <button
                onClick={() => setSide('buy')}
                style={{
                  flex: 1,
                  padding: '12px',
                  background: side === 'buy' ? 'linear-gradient(135deg, var(--primary-500), var(--primary-600))' : 'transparent',
                  border: '1px solid var(--primary-500)',
                  borderRadius: '6px',
                  color: '#ffffff',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  boxShadow: side === 'buy' ? '0 4px 6px rgba(0, 0, 0, 0.1)' : 'none'
                }}
                onMouseEnter={(e) => {
                  if (side !== 'buy') {
                    e.currentTarget.style.background = 'linear-gradient(135deg, var(--primary-500), var(--primary-600))';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 8px 12px rgba(0, 0, 0, 0.2)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (side !== 'buy') {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }
                }}
              >
                Buy {selectedSymbol}
              </button>
              <button
                onClick={() => setSide('sell')}
                style={{
                  flex: 1,
                  padding: '12px',
                  background: side === 'sell' ? 'linear-gradient(135deg, #ef4444, #dc2626)' : 'transparent',
                  border: '1px solid #ef4444',
                  borderRadius: '6px',
                  color: '#ffffff',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer',
                  transition: 'all 0.3s ease',
                  boxShadow: side === 'sell' ? '0 4px 6px rgba(0, 0, 0, 0.1)' : 'none'
                }}
                onMouseEnter={(e) => {
                  if (side !== 'sell') {
                    e.currentTarget.style.background = 'linear-gradient(135deg, #ef4444, #dc2626)';
                    e.currentTarget.style.transform = 'translateY(-2px)';
                    e.currentTarget.style.boxShadow = '0 8px 12px rgba(0, 0, 0, 0.2)';
                  }
                }}
                onMouseLeave={(e) => {
                  if (side !== 'sell') {
                    e.currentTarget.style.background = 'transparent';
                    e.currentTarget.style.transform = 'translateY(0)';
                    e.currentTarget.style.boxShadow = 'none';
                  }
                }}
              >
                Sell {selectedSymbol}
              </button>
            </div>

            {/* Order Summary */}
            <div style={{ fontSize: '11px', color: '#9ca3af' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span>Available Balance:</span>
                <span>0.00 USDT</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span>Order Value:</span>
                <span>N/A</span>
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
                <span>Fee:</span>
                <span>Taker 0.0000%</span>
              </div>
            </div>
          </div>

          {/* Market Depth Chart */}
          <div style={{
            height: '200px',
            backgroundColor: '#000000',
            borderTop: '1px solid var(--primary-500)',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#ffffff' }}>
              Market Depth
            </div>
            
            {/* Depth Chart Placeholder */}
            <div style={{ 
              flex: 1, 
              backgroundColor: '#0f172a', 
              border: '1px solid var(--primary-500)', 
              borderRadius: '4px',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              position: 'relative',
              overflow: 'hidden'
            }}>
              {/* Buy Side Depth Bars */}
              <div style={{ position: 'absolute', left: '0', top: '0', width: '50%', height: '100%' }}>
                {[0.8, 0.6, 0.4, 0.2].map((height, i) => (
                  <div key={i} style={{
                    position: 'absolute',
                    right: '0',
                    bottom: `${i * 25}%`,
                    width: `${height * 100}%`,
                    height: '20%',
                    backgroundColor: '#10b981',
                    opacity: 0.3,
                    borderRadius: '2px'
                  }} />
                ))}
              </div>
              
              {/* Sell Side Depth Bars */}
              <div style={{ position: 'absolute', right: '0', top: '0', width: '50%', height: '100%' }}>
                {[0.2, 0.4, 0.6, 0.8].map((height, i) => (
                  <div key={i} style={{
                    position: 'absolute',
                    left: '0',
                    bottom: `${i * 25}%`,
                    width: `${height * 100}%`,
                    height: '20%',
                    backgroundColor: '#ef4444',
                    opacity: 0.3,
                    borderRadius: '2px'
                  }} />
                ))}
              </div>
              
              {/* Center Line */}
              <div style={{
                position: 'absolute',
                left: '50%',
                top: '0',
                width: '1px',
                height: '100%',
                backgroundColor: 'var(--primary-500)',
                transform: 'translateX(-50%)'
              }} />
              
              <div style={{ fontSize: '12px', color: '#9ca3af', zIndex: 10 }}>
                Market Depth Chart
              </div>
            </div>
          </div>

          {/* Recent Trades */}
          <div style={{
            height: '200px',
            backgroundColor: '#000000',
            borderTop: '1px solid var(--primary-500)',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#ffffff' }}>
              Recent Trades
            </div>
            
            {/* Trade List */}
            <div style={{ flex: 1, overflowY: 'auto' }}>
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

export default TradingTab;