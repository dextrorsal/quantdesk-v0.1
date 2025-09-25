import React, { useState, useEffect, useRef } from 'react';
import TradingViewChart from '../components/TradingViewChart';

// TradingTab component - updated for black backgrounds
const TradingTab: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC');
  const [selectedInterval, setSelectedInterval] = useState('1h');
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('market');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderSize, setOrderSize] = useState('0');
  const [orderPrice, setOrderPrice] = useState('');
  const [leverage, setLeverage] = useState('1x');
  const [showPairDropdown, setShowPairDropdown] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const dropdownRef = useRef<HTMLDivElement>(null);

  const symbols = ['BTC', 'ETH', 'SOL', 'ADA', 'DOGE', 'MATIC'];

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

      {/* Market Ticker Bar */}
      <div style={{
        height: '40px',
        backgroundColor: '#111827',
        borderBottom: '1px solid #374151',
        display: 'flex',
        alignItems: 'center',
        overflowX: 'auto',
        padding: '0 16px',
        flexShrink: 0
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '24px', whiteSpace: 'nowrap' }}>
          <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>NEW</span>
          <span style={{ fontSize: '12px', color: '#ffffff' }}>MSFTUSDT</span>
          <span style={{ fontSize: '12px', color: '#ef4444' }}>-0.78%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>HOT</span>
          <span style={{ fontSize: '12px', color: '#ffffff' }}>ASTERUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+37.94%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>BARDUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+50.55%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>OGUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+4.38%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>PORTALSUSDT</span>
          <span style={{ fontSize: '12px', color: '#ef4444' }}>-5.93%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>ZKCUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+13.13%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>UBUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+6.69%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>UXLINKUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+16.55%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>AVAXUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+2.94%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>HEMIUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+47.86%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>AVNTUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+3.33%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>BTCUSDT</span>
          <span style={{ fontSize: '12px', color: '#ef4444' }}>-0.02%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>SOLUSDT</span>
          <span style={{ fontSize: '12px', color: '#ef4444' }}>-1.90%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>XPLUSDT</span>
          <span style={{ fontSize: '12px', color: '#ef4444' }}>-9.45%</span>
          
          <span style={{ fontSize: '12px', color: '#ffffff' }}>ETHUSDT</span>
          <span style={{ fontSize: '12px', color: '#10b981' }}>+0.11%</span>
        </div>
      </div>

      {/* Main Content Area */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0 }}>
        {/* Center Panel - Chart */}
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
          {/* Chart Header */}
          <div style={{
            height: '60px',
            backgroundColor: '#111827',
            borderBottom: '1px solid #374151',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            padding: '0 24px',
            flexShrink: 0
          }}>
            {/* Left - Symbol Info with Sleek Dropdown */}
            <div style={{ display: 'flex', alignItems: 'center', gap: '16px' }}>
              {/* Sleek Pair Selector Dropdown */}
              <div ref={dropdownRef} style={{ position: 'relative' }}>
                <button 
                  onClick={() => setShowPairDropdown(!showPairDropdown)}
                  style={{
                    display: 'flex',
                    alignItems: 'center',
                    gap: '8px',
                    padding: '8px 12px',
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
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
                    backgroundColor: '#1f2937',
                    border: '1px solid #374151',
                    borderRadius: '8px',
                    boxShadow: '0 10px 25px rgba(0, 0, 0, 0.3)',
                    zIndex: 1000,
                    minWidth: '280px',
                    maxHeight: '400px',
                    overflowY: 'auto'
                  }}>
                    {/* Search */}
                    <div style={{ padding: '12px', borderBottom: '1px solid #374151' }}>
                      <input
                        type="text"
                        placeholder="Search pairs..."
                        value={searchQuery}
                        onChange={(e) => setSearchQuery(e.target.value)}
                        style={{
                          width: '100%',
                          padding: '8px 12px',
                          backgroundColor: '#111827',
                          border: '1px solid #374151',
                          borderRadius: '6px',
                          color: '#ffffff',
                          fontSize: '12px',
                          outline: 'none'
                        }}
                      />
                    </div>
                  
                  {/* Categories */}
                  <div style={{ padding: '8px 12px', borderBottom: '1px solid #374151' }}>
                    <div style={{ display: 'flex', gap: '6px', marginBottom: '8px' }}>
                      <button style={{
                        padding: '4px 8px',
                        backgroundColor: '#3b82f6',
                        border: 'none',
                        borderRadius: '4px',
                        color: '#ffffff',
                        fontSize: '10px',
                        cursor: 'pointer'
                      }}>
                        USDT-M
                      </button>
                      <button style={{
                        padding: '4px 8px',
                        backgroundColor: 'transparent',
                        border: '1px solid #374151',
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
                        backgroundColor: '#3b82f6',
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
                        border: '1px solid #374151',
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
                        border: '1px solid #374151',
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
                        border: '1px solid #374151',
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
                              backgroundColor: symbol === selectedSymbol ? '#1e3a8a' : 'transparent',
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
                              <span style={{ color: '#ffffff' }}>214.191</span>
                              <span style={{ color: '#10b981' }}>+0.44%</span>
                            </div>
                          </button>
                        ))}
                    </div>
                  </div>
                )}
              </div>
              
              <div style={{ fontSize: '24px', fontWeight: 'bold', color: '#10b981' }}>
                214.191
              </div>
              <div style={{ fontSize: '14px', color: '#ef4444', fontWeight: '600' }}>
                -4.145 (-1.90%)
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
                    border: '1px solid #374151',
                    borderRadius: '4px',
                    color: selectedInterval === interval ? '#3b82f6' : '#ffffff',
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
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>214.191</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Index price:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>214.263</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>24h high:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>221.667</span>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>24h low:</span>
                <span style={{ fontSize: '12px', color: '#ffffff', fontWeight: '600' }}>212.237</span>
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

          {/* Chart Area */}
          <div style={{ height: '400px', position: 'relative', backgroundColor: '#000000', flexShrink: 0 }}>
            <TradingViewChart 
              symbol={selectedSymbol} 
              timeframe={selectedInterval} 
              height={400} 
            />
          </div>

          {/* Bottom Panel - Positions & Orders */}
          <div style={{
            flex: 1, // Expand to fill remaining space
            backgroundColor: '#111827',
            borderTop: '1px solid #374151',
            display: 'flex',
            minHeight: '200px' // Minimum height to ensure readability
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
            <div style={{ flex: 1, padding: '16px', borderLeft: '1px solid #374151', display: 'flex', flexDirection: 'column' }}>
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
            <div style={{ flex: 1, padding: '16px', borderLeft: '1px solid #374151', display: 'flex', flexDirection: 'column' }}>
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
        <div style={{ width: '350px', display: 'flex', flexDirection: 'column', borderLeft: '1px solid #374151' }}>
          {/* Order Book */}
          <div style={{ flex: 1, padding: '16px' }}>
            <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#ffffff' }}>
              Order Book
            </div>
            
            {/* Sell Orders */}
            <div style={{ marginBottom: '8px' }}>
              <div style={{ fontSize: '11px', color: '#9ca3af', marginBottom: '4px' }}>Price (USDT)</div>
              <div style={{ fontSize: '12px', color: '#ef4444' }}>214.25</div>
              <div style={{ fontSize: '12px', color: '#ef4444' }}>214.20</div>
              <div style={{ fontSize: '12px', color: '#ef4444' }}>214.15</div>
            </div>

            {/* Spread */}
            <div style={{ 
              height: '1px', 
              backgroundColor: '#374151', 
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
              <div style={{ fontSize: '12px', color: '#10b981' }}>214.10</div>
              <div style={{ fontSize: '12px', color: '#10b981' }}>214.05</div>
              <div style={{ fontSize: '12px', color: '#10b981' }}>214.00</div>
            </div>
          </div>

          {/* Trading Panel */}
          <div style={{ 
            height: '400px', 
            padding: '16px', 
            borderTop: '1px solid #374151',
            backgroundColor: '#111827'
          }}>
            {/* Margin Mode & Leverage */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Margin Mode:</span>
                <button style={{
                  padding: '4px 8px',
                  backgroundColor: '#3b82f6',
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
                    backgroundColor: '#3b82f6',
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
                    backgroundColor: '#111827',
                    border: '1px solid #374151',
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
                    border: '1px solid #374151',
                    borderBottom: 'none',
                    color: orderType === type.value ? '#3b82f6' : '#ffffff',
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
                border: '1px solid #374151',
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
                  border: '1px solid #374151',
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
                  border: '1px solid #374151',
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
                    border: '1px solid #374151',
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
                  backgroundColor: side === 'buy' ? '#000000' : '#111827',
                  border: '1px solid #3b82f6',
                  borderRadius: '6px',
                  color: side === 'buy' ? '#3b82f6' : '#3b82f6',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer'
                }}
              >
                Buy {selectedSymbol}
              </button>
              <button
                onClick={() => setSide('sell')}
                style={{
                  flex: 1,
                  padding: '12px',
                  backgroundColor: side === 'sell' ? '#000000' : '#111827',
                  border: '1px solid #ef4444',
                  borderRadius: '6px',
                  color: side === 'sell' ? '#ef4444' : '#ef4444',
                  fontSize: '14px',
                  fontWeight: '600',
                  cursor: 'pointer'
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
            backgroundColor: '#111827',
            borderTop: '1px solid #374151',
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
              border: '1px solid #374151', 
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
                backgroundColor: '#6b7280',
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
            backgroundColor: '#111827',
            borderTop: '1px solid #374151',
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

      {/* Bottom Ticker - AsterDex Style */}
      <div style={{
        height: '40px',
        backgroundColor: '#111827',
        borderTop: '1px solid #374151',
        display: 'flex',
        alignItems: 'center',
        gap: '24px',
        padding: '0 24px',
        overflowX: 'auto',
        flexShrink: 0
      }}>
        {symbols.map(symbol => {
          const prices = {
            'BTC': '214.19',
            'ETH': '3200.45',
            'SOL': '200.15',
            'BNB': '600.30',
            'ADA': '0.45',
            'DOT': '6.20'
          }
          const price = prices[symbol as keyof typeof prices] || '100.00'
          return (
            <div key={symbol} style={{ display: 'flex', alignItems: 'center', gap: '8px', minWidth: '120px' }}>
              <span style={{ fontSize: '12px', fontWeight: '600', color: '#ffffff' }}>{symbol}USDT</span>
              <span style={{ fontSize: '12px', color: '#ffffff' }}>{price}</span>
              <span style={{ fontSize: '12px', color: '#10b981' }}>+0.438%</span>
            </div>
          )
        })}
      </div>
    </div>
  );
};

export default TradingTab;