import React, { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import QuantDeskChart from '../components/charts/QuantDeskChart'
import TradingViewWidget from '../components/TradingViewWidget';
import TradingViewTickerTape from '../components/TradingViewTickerTape';
import { usePrice } from '../contexts/PriceContext';

// TradingTab component - updated for black backgrounds with mobile responsiveness
const TradingTab: React.FC = () => {
  const [selectedSymbol, setSelectedSymbol] = useState('BTC');
  const [selectedInterval, setSelectedInterval] = useState('1h');
  const [chartType, setChartType] = useState<'widget' | 'lightweight'>('widget');
  const [chartHeight] = useState(550); // Balanced height for single monitor
  const [orderType, setOrderType] = useState<'market' | 'limit' | 'stop'>('market');
  const [side, setSide] = useState<'buy' | 'sell'>('buy');
  const [orderSize, setOrderSize] = useState('0');
  const [orderPrice, setOrderPrice] = useState('');
  const [leverage, setLeverage] = useState(1);
  const [maxMode, setMaxMode] = useState(false);
  const [marginMode, setMarginMode] = useState<'cross' | 'isolated'>('isolated');
  const [showPairDropdown, setShowPairDropdown] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [activeMobileTab, setActiveMobileTab] = useState<'chart' | 'orderbook' | 'trading'>('chart');
  const [isMobile, setIsMobile] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);
  
  // Get real-time prices
  const { getPrice } = usePrice();

  const symbols = ['BTC', 'ETH', 'SOL', 'AVAX', 'MATIC', 'ARB', 'OP', 'DOGE', 'ADA', 'DOT', 'LINK'];

  // Mobile detection and responsive logic
  useEffect(() => {
    const checkMobile = () => {
      const mobile = window.innerWidth <= 768;
      console.log('📱 Mobile detection:', { mobile, currentChartType: chartType });
      setIsMobile(mobile);
      
      // Always use lightweight chart on mobile to avoid TradingView errors
      if (mobile) {
        console.log('📱 Setting chart to lightweight for mobile');
        setChartType('lightweight');
      } else {
        // On desktop, allow user to choose between widget and lightweight
        console.log('🖥️ Setting chart to widget for desktop');
        setChartType('widget');
      }
    };
    
    checkMobile();
    window.addEventListener('resize', checkMobile);
    
    return () => window.removeEventListener('resize', checkMobile);
  }, []); // Remove chartType dependency to prevent re-renders

  // Prevent chartType changes on mobile
  useEffect(() => {
    if (isMobile && chartType !== 'lightweight') {
      console.log('🔄 Mobile detected - forcing lightweight chart');
      setChartType('lightweight');
    }
  }, [isMobile, chartType]);

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

  // Memoized mobile chart - Always uses lightweight chart (never TradingView widget)
  const MemoizedMobileChart = useMemo(() => {
    return (
      <QuantDeskChart 
        symbol={selectedSymbol} 
        height={300}
        timeframe={selectedInterval}
        isMobile={true} // Optimize for mobile - fewer candles, faster loading
        key={`mobile-chart-${selectedSymbol}-${selectedInterval}`}
      />
    );
  }, [selectedSymbol, selectedInterval]);

  // Mobile Tab Navigation Component
  const MobileTabNavigation = () => (
    <div style={{
      position: 'fixed',
      bottom: 0,
      left: 0,
      right: 0,
      height: '60px',
      backgroundColor: '#000000',
      borderTop: '1px solid var(--primary-500)',
      display: 'flex',
      zIndex: 1000
    }}>
      {[
        { key: 'chart', label: 'Chart', icon: '📊' },
        { key: 'orderbook', label: 'Orders', icon: '📋' },
        { key: 'trading', label: 'Trade', icon: '💰' }
      ].map(tab => (
        <button
          key={tab.key}
          onClick={() => setActiveMobileTab(tab.key as any)}
          style={{
            flex: 1,
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            backgroundColor: activeMobileTab === tab.key ? 'var(--primary-500)' : 'transparent',
            border: 'none',
            color: '#ffffff',
            fontSize: '10px',
            cursor: 'pointer',
            transition: 'all 0.2s ease',
            minHeight: '44px'
          }}
        >
          <span style={{ fontSize: '16px', marginBottom: '2px' }}>{tab.icon}</span>
          <span>{tab.label}</span>
        </button>
      ))}
    </div>
  );

  // Mobile Chart View Component
  const MobileChartView = () => (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
      {/* Mobile Chart Header */}
      <div style={{
        height: '50px',
        backgroundColor: '#000000',
        borderBottom: '1px solid var(--primary-500)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
        padding: '0 16px',
        flexShrink: 0
      }}>
        {/* Symbol Selector */}
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
              minWidth: '120px'
            }}
          >
            <span>{selectedSymbol}USDT</span>
            <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M6 9l6 6 6-6"/>
            </svg>
          </button>
          
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
              minWidth: '200px',
              maxHeight: '300px',
              overflowY: 'auto'
            }}>
              <div style={{ padding: '8px 0' }}>
                {symbols.map((symbol) => (
                  <button
                    key={symbol}
                    onClick={() => {
                      setSelectedSymbol(symbol);
                      setShowPairDropdown(false);
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
                      textAlign: 'left'
                    }}
                  >
                    <span>{symbol}USDT</span>
                    <span style={{ color: getPrice(`${symbol}/USDT`)?.changePercent >= 0 ? '#10b981' : '#ef4444' }}>
                      {getPrice(`${symbol}/USDT`)?.changePercent >= 0 ? '+' : ''}{getPrice(`${symbol}/USDT`)?.changePercent?.toFixed(2) || '0.00'}%
                    </span>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Price Display */}
        <div style={{ textAlign: 'right' }}>
          <div style={{ fontSize: '18px', fontWeight: 'bold', color: '#10b981' }}>
            {getPrice(`${selectedSymbol}/USDT`)?.price?.toFixed(3) || '0.000'}
          </div>
          <div style={{ fontSize: '12px', color: getPrice(`${selectedSymbol}/USDT`)?.changePercent >= 0 ? '#10b981' : '#ef4444' }}>
            {getPrice(`${selectedSymbol}/USDT`)?.changePercent >= 0 ? '+' : ''}{getPrice(`${selectedSymbol}/USDT`)?.changePercent?.toFixed(2) || '0.00'}%
          </div>
        </div>
      </div>

      {/* Mobile Chart - Beautiful lightweight chart */}
      <div style={{ 
        flex: 1,
        backgroundColor: '#000000',
        border: '2px solid #1e40af',
        borderRadius: '16px',
        margin: '16px',
        overflow: 'hidden',
        boxShadow: '0 8px 32px rgba(30, 64, 175, 0.3), 0 0 0 1px rgba(59, 130, 246, 0.1)',
        position: 'relative',
        background: 'linear-gradient(135deg, #000000 0%, #0f172a 100%)'
      }}>
        {/* Chart Header */}
        <div style={{
          padding: '16px 20px',
          borderBottom: '2px solid #1e40af',
          backgroundColor: 'linear-gradient(90deg, #0f172a 0%, #1e293b 100%)',
          background: 'linear-gradient(90deg, #0f172a 0%, #1e293b 100%)',
          position: 'relative'
        }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <div style={{ 
              fontSize: '16px', 
              fontWeight: '700', 
              color: '#ffffff',
              textShadow: '0 2px 4px rgba(0, 0, 0, 0.5)'
            }}>
              📈 {selectedSymbol}USDT Chart
              <span style={{ 
                fontSize: '10px', 
                color: '#60a5fa', 
                marginLeft: '8px',
                backgroundColor: 'rgba(30, 64, 175, 0.2)',
                padding: '2px 6px',
                borderRadius: '4px',
                border: '1px solid rgba(59, 130, 246, 0.3)'
              }}>
                Lightweight
              </span>
            </div>
            <div style={{ 
              fontSize: '13px', 
              color: '#60a5fa',
              fontWeight: '600',
              backgroundColor: 'rgba(30, 64, 175, 0.2)',
              padding: '4px 8px',
              borderRadius: '6px',
              border: '1px solid rgba(59, 130, 246, 0.3)'
            }}>
              {selectedInterval} • 🔴 Live
            </div>
          </div>
        </div>
        
        {/* Chart Container */}
        <div style={{ 
          height: '300px',
          backgroundColor: '#000000',
          position: 'relative',
          background: 'radial-gradient(circle at center, #0f172a 0%, #000000 100%)'
        }}>
          {MemoizedMobileChart}
        </div>
        
        {/* Chart Footer */}
        <div style={{
          padding: '12px 20px',
          borderTop: '2px solid #1e40af',
          backgroundColor: 'linear-gradient(90deg, #0f172a 0%, #1e293b 100%)',
          background: 'linear-gradient(90deg, #0f172a 0%, #1e293b 100%)',
          fontSize: '11px',
          color: '#94a3b8',
          textAlign: 'center',
          fontWeight: '500'
        }}>
          ⚡ Powered by QuantDesk • Real-time data • Professional Trading
        </div>
      </div>

      {/* Mobile Timeframe Buttons */}
      <div style={{ 
        display: 'flex', 
        gap: '8px', 
        padding: '16px 20px',
        backgroundColor: 'linear-gradient(90deg, #0f172a 0%, #1e293b 100%)',
        background: 'linear-gradient(90deg, #0f172a 0%, #1e293b 100%)',
        borderTop: '2px solid #1e40af',
        justifyContent: 'center',
        boxShadow: '0 -4px 16px rgba(0, 0, 0, 0.3)'
      }}>
        {intervals.map(interval => (
          <button
            key={interval}
            onClick={() => setSelectedInterval(interval)}
            style={{
              padding: '10px 16px',
              backgroundColor: selectedInterval === interval 
                ? 'linear-gradient(135deg, #3b82f6 0%, #1e40af 100%)'
                : 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
              background: selectedInterval === interval 
                ? 'linear-gradient(135deg, #3b82f6 0%, #1e40af 100%)'
                : 'linear-gradient(135deg, #1e293b 0%, #0f172a 100%)',
              border: selectedInterval === interval 
                ? '2px solid #60a5fa' 
                : '2px solid #374151',
              borderRadius: '12px',
              color: selectedInterval === interval ? '#ffffff' : '#94a3b8',
              fontSize: '13px',
              fontWeight: '700',
              cursor: 'pointer',
              minHeight: '44px',
              minWidth: '44px',
              transition: 'all 0.3s ease',
              boxShadow: selectedInterval === interval 
                ? '0 4px 16px rgba(59, 130, 246, 0.4), 0 0 0 1px rgba(96, 165, 250, 0.2)' 
                : '0 2px 8px rgba(0, 0, 0, 0.2)',
              textShadow: selectedInterval === interval ? '0 1px 2px rgba(0, 0, 0, 0.5)' : 'none'
            }}
          >
            {interval}
          </button>
        ))}
      </div>
    </div>
  );

  // Mobile Order Book View Component
  const MobileOrderBookView = () => (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '16px' }}>
      <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px', color: '#ffffff' }}>
        Order Book
      </div>
      
      {/* Order Book Content */}
      <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
        {/* Buy Orders */}
        <div style={{ flex: 1, marginBottom: '16px' }}>
          <div style={{ fontSize: '14px', color: '#10b981', marginBottom: '8px' }}>Buy Orders</div>
          {[
            { price: 109351, size: '77.6K' },
            { price: 109352, size: '109' },
            { price: 109353, size: '77.6K' },
            { price: 109354, size: '77.8K' },
            { price: 109355, size: '995' }
          ].map((order, index) => (
            <div key={index} style={{
              display: 'flex',
              justifyContent: 'space-between',
              padding: '8px 0',
              borderBottom: '1px solid #1a1a1a',
              fontSize: '12px'
            }}>
              <span style={{ color: '#10b981' }}>{order.price.toLocaleString()}</span>
              <span style={{ color: '#ffffff' }}>{order.size}</span>
            </div>
          ))}
        </div>

        {/* Current Price */}
        <div style={{
          padding: '12px',
          backgroundColor: '#1a1a1a',
          borderRadius: '8px',
          textAlign: 'center',
          marginBottom: '16px'
        }}>
          <div style={{ fontSize: '20px', fontWeight: '700', color: '#ffffff' }}>109,376</div>
          <div style={{ fontSize: '12px', color: '#9ca3af' }}>Spread: $26 (0.023%)</div>
        </div>

        {/* Sell Orders */}
        <div style={{ flex: 1 }}>
          <div style={{ fontSize: '14px', color: '#ef4444', marginBottom: '8px' }}>Sell Orders</div>
          {[
            { price: 109409, size: '109' },
            { price: 109408, size: '110K' },
            { price: 109405, size: '120K' },
            { price: 109403, size: '110K' },
            { price: 109401, size: '110K' }
          ].map((order, index) => (
            <div key={index} style={{
              display: 'flex',
              justifyContent: 'space-between',
              padding: '8px 0',
              borderBottom: '1px solid #1a1a1a',
              fontSize: '12px'
            }}>
              <span style={{ color: '#ef4444' }}>{order.price.toLocaleString()}</span>
              <span style={{ color: '#ffffff' }}>{order.size}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );

  // Mobile Trading View Component
  const MobileTradingView = () => (
    <div style={{ height: '100%', display: 'flex', flexDirection: 'column', padding: '16px' }}>
      <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '16px', color: '#ffffff' }}>
        Trading
      </div>

      {/* Order Type Selection */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ fontSize: '14px', color: '#9ca3af', marginBottom: '8px' }}>Order Type</div>
        <div style={{ display: 'flex', gap: '8px' }}>
          {orderTypes.map(type => (
            <button
              key={type.value}
              onClick={() => setOrderType(type.value as any)}
              style={{
                flex: 1,
                padding: '12px',
                backgroundColor: orderType === type.value ? 'var(--primary-500)' : 'transparent',
                border: '1px solid var(--primary-500)',
                borderRadius: '8px',
                color: '#ffffff',
                fontSize: '14px',
                fontWeight: '600',
                cursor: 'pointer',
                minHeight: '44px'
              }}
            >
              {type.label}
            </button>
          ))}
        </div>
      </div>

      {/* Leverage */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '8px' }}>
          <span style={{ fontSize: '14px', color: '#9ca3af' }}>Leverage</span>
          <span style={{ fontSize: '16px', fontWeight: '600', color: '#ffffff' }}>{leverage}x</span>
        </div>
        <input
          type="range"
          min="1"
          max="100"
          step="1"
          value={leverage}
          onChange={(e) => setLeverage(parseInt(e.target.value))}
          style={{
            width: '100%',
            height: '6px',
            background: '#1a1a1a',
            outline: 'none',
            borderRadius: '3px'
          }}
        />
      </div>

      {/* Order Size */}
      <div style={{ marginBottom: '16px' }}>
        <div style={{ fontSize: '14px', color: '#9ca3af', marginBottom: '8px' }}>Size</div>
        <input
          type="text"
          value={orderSize}
          onChange={(e) => setOrderSize(e.target.value)}
          placeholder="0.0"
          style={{
            width: '100%',
            padding: '12px',
            backgroundColor: '#000000',
            border: '1px solid var(--primary-500)',
            borderRadius: '8px',
            color: '#ffffff',
            fontSize: '16px',
            minHeight: '44px'
          }}
        />
      </div>

      {/* Order Price (for Limit orders) */}
      {orderType === 'limit' && (
        <div style={{ marginBottom: '16px' }}>
          <div style={{ fontSize: '14px', color: '#9ca3af', marginBottom: '8px' }}>Price</div>
          <input
            type="text"
            value={orderPrice}
            onChange={(e) => setOrderPrice(e.target.value)}
            placeholder="0.0"
            style={{
              width: '100%',
              padding: '12px',
              backgroundColor: '#000000',
              border: '1px solid var(--primary-500)',
              borderRadius: '8px',
              color: '#ffffff',
              fontSize: '16px',
              minHeight: '44px'
            }}
          />
        </div>
      )}

      {/* Buy/Sell Buttons */}
      <div style={{ display: 'flex', gap: '12px', marginBottom: '16px' }}>
        <button
          onClick={() => setSide('buy')}
          style={{
            flex: 1,
            padding: '16px',
            background: side === 'buy' ? 'linear-gradient(135deg, var(--primary-500), var(--primary-600))' : 'transparent',
            border: '1px solid var(--primary-500)',
            borderRadius: '8px',
            color: '#ffffff',
            fontSize: '16px',
            fontWeight: '600',
            cursor: 'pointer',
            minHeight: '44px'
          }}
        >
          Buy {selectedSymbol}
        </button>
        <button
          onClick={() => setSide('sell')}
          style={{
            flex: 1,
            padding: '16px',
            background: side === 'sell' ? 'linear-gradient(135deg, #ef4444, #dc2626)' : 'transparent',
            border: '1px solid #ef4444',
            borderRadius: '8px',
            color: '#ffffff',
            fontSize: '16px',
            fontWeight: '600',
            cursor: 'pointer',
            minHeight: '44px'
          }}
        >
          Sell {selectedSymbol}
        </button>
      </div>

      {/* Order Summary */}
      <div style={{ fontSize: '12px', color: '#9ca3af' }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span>Available Balance:</span>
          <span>0.00 USDT</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '4px' }}>
          <span>Order Value:</span>
          <span>N/A</span>
        </div>
        <div style={{ display: 'flex', justifyContent: 'space-between' }}>
          <span>Fee:</span>
          <span>Taker 0.0000%</span>
        </div>
      </div>
    </div>
  );

  // Return mobile layout if on mobile device
  if (isMobile) {
    return (
      <div style={{
        width: '100%',
        height: '100vh',
        backgroundColor: '#000000',
        color: '#ffffff',
        fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
        display: 'flex',
        flexDirection: 'column',
        overflow: 'hidden',
        paddingBottom: '60px' // Space for bottom navigation
      }}>
        {/* Mobile Content */}
        <div style={{ flex: 1, overflow: 'hidden' }}>
          {activeMobileTab === 'chart' && <MobileChartView />}
          {activeMobileTab === 'orderbook' && <MobileOrderBookView />}
          {activeMobileTab === 'trading' && <MobileTradingView />}
        </div>

        {/* Mobile Tab Navigation */}
        <MobileTabNavigation />
      </div>
    );
  }

  return (
    <div style={{
      width: '100%',
      height: '100vh', // Use full viewport height
      backgroundColor: '#000000',
      color: '#ffffff',
      fontFamily: 'Inter, -apple-system, BlinkMacSystemFont, sans-serif',
      display: 'flex',
      flexDirection: 'column',
      overflow: 'hidden'
    }}>

      {/* Main Content Area - Fixed Height Issues */}
      <div style={{ flex: 1, display: 'flex', minHeight: 0, height: 'calc(100vh - 120px)' }}>
        {/* Center Panel - Chart (Much Larger) */}
        <div style={{ flex: '0 0 76%', display: 'flex', flexDirection: 'column', minHeight: 0 }}>
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
              {/* Chart Type Toggle - Hidden on mobile */}
              {!isMobile && (
                <div style={{ display: 'flex', backgroundColor: '#1a1a1a', borderRadius: '8px', padding: '2px' }}>
                  <button
                    onClick={() => setChartType('widget')}
                    style={{
                      padding: '6px 12px',
                      backgroundColor: chartType === 'widget' ? 'var(--primary-500)' : 'transparent',
                      border: 'none',
                      borderRadius: '6px',
                      color: '#ffffff',
                      fontSize: '10px',
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
                      fontSize: '10px',
                      cursor: 'pointer',
                      transition: 'all 0.2s ease'
                    }}
                  >
                    Lightweight
                  </button>
                </div>
              )}
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
                          fontSize: '10px',
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
                              fontSize: '10px',
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
                    fontSize: '10px',
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
              <TradingViewWidget />
            ) : (
              <QuantDeskChart 
                key={`${selectedSymbol}-${selectedInterval}`}
                symbol={selectedSymbol} 
                height={chartHeight}
                timeframe={selectedInterval}
                isMobile={false} // Desktop version - use full data
              />
            )}
          </div>

          {/* TradingView Ticker Tape - Above Terminal Panel */}
          <div style={{ backgroundColor: 'transparent', borderBottom: '1px solid #333333', marginBottom: '8px' }}>
            <TradingViewTickerTape />
          </div>

          {/* Bottom Panel - Professional Trading Dashboard */}
          <div style={{
            flex: 1,
            backgroundColor: 'var(--background-primary)',
            border: '1px solid var(--primary-500)',
            borderRadius: '12px',
            display: 'grid',
            gridTemplateRows: 'auto auto 1fr',
            minHeight: '300px',
            boxShadow: '0 8px 24px rgba(0, 0, 0, 0.4)',
            overflow: 'hidden',
            marginTop: '2px'
          }}>
            {/* Account Equity Summary */}
            <div style={{
              padding: '8px 12px',
              borderBottom: '1px solid var(--primary-500)',
              backgroundColor: 'var(--background-secondary)'
            }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
                <h3 style={{ fontSize: '12px', fontWeight: '600', color: 'var(--text-primary)', margin: 0 }}>
                  Account Overview
                </h3>
                <div style={{ display: 'flex', gap: '10px', fontSize: '11px' }}>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '9px' }}>Account Equity</div>
                    <div style={{ color: 'var(--text-primary)', fontWeight: '600' }}>$0.00</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '9px' }}>Unrealized PNL</div>
                    <div style={{ color: 'var(--success-500)', fontWeight: '600' }}>+$0.00</div>
                  </div>
                  <div style={{ textAlign: 'center' }}>
                    <div style={{ color: 'var(--text-secondary)', fontSize: '9px' }}>Margin Ratio</div>
                    <div style={{ color: 'var(--text-primary)', fontWeight: '600' }}>0.00%</div>
                  </div>
                </div>
              </div>
            </div>

            {/* Tab Navigation - Hyperliquid Style */}
            <div style={{
              display: 'flex',
              borderBottom: '1px solid var(--primary-500)',
              backgroundColor: 'var(--background-primary)',
              overflowX: 'auto'
            }}>
              {['Balances', 'Positions', 'Open Orders', 'TWAP', 'Trade History', 'Funding History', 'Order History'].map((tab, index) => (
                <button
                  key={tab}
                  style={{
                    padding: '8px 10px',
                    backgroundColor: 'transparent',
                    border: 'none',
                    borderBottom: index === 0 ? '2px solid var(--primary-500)' : '2px solid transparent',
                    color: index === 0 ? 'var(--text-primary)' : 'var(--text-secondary)',
                    fontSize: '11px',
                    fontWeight: '500',
                    cursor: 'pointer',
                    transition: 'all 0.2s ease',
                    whiteSpace: 'nowrap',
                    minWidth: 'fit-content'
                  }}
                  onMouseEnter={(e) => {
                    if (index !== 0) {
                      e.currentTarget.style.color = 'var(--text-primary)';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (index !== 0) {
                      e.currentTarget.style.color = 'var(--text-secondary)';
                    }
                  }}
                >
                  {tab}
                </button>
              ))}
            </div>

            {/* Tab Content */}
            <div style={{ flex: 1, overflow: 'hidden' }}>
              {/* Positions Tab */}
              <div style={{ padding: '8px 12px', height: '100%', overflow: 'auto' }}>
                {/* Positions Table Header */}
                <div style={{ 
                  display: 'grid', 
                  gridTemplateColumns: '1fr 1fr 1fr 1fr 1fr', 
                  gap: '6px',
                  padding: '4px 0',
                  borderBottom: '1px solid var(--primary-500)',
                  marginBottom: '8px'
                }}>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '10px', fontWeight: '500' }}>Symbol</div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '10px', fontWeight: '500' }}>Size</div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '10px', fontWeight: '500' }}>Entry Price</div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '10px', fontWeight: '500' }}>Mark Price</div>
                  <div style={{ color: 'var(--text-secondary)', fontSize: '10px', fontWeight: '500' }}>PnL</div>
                </div>

                {/* No Positions Message */}
                <div style={{ 
                  textAlign: 'center', 
                  padding: '12px 0',
                  color: 'var(--text-secondary)',
                  fontSize: '10px'
                }}>
                  <div style={{ marginBottom: '2px' }}>📊</div>
                  <div>No open positions</div>
                </div>
              </div>
            </div>
          </div>
        </div>

        {/* Middle Panel - Order Book (Much Narrower) */}
        <div style={{ 
          flex: '0 0 8%', 
          display: 'flex', 
          flexDirection: 'column',
          backgroundColor: 'var(--background-primary)',
          border: '1px solid var(--primary-500)',
          borderRadius: '8px',
          overflow: 'hidden',
          position: 'relative',
          maxHeight: '47vh',
          margin: '0 2px'
        }}>
          {/* Header with Tabs */}
          <div style={{ 
            display: 'flex', 
            flexDirection: 'column', 
            gap: '8px', 
            backgroundColor: 'var(--background-primary)', 
            padding: '12px'
          }}>
            {/* Tab Navigation */}
            <div style={{ display: 'flex', position: 'relative' }}>
              <div style={{
                padding: '8px 12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'var(--background-primary)',
                color: 'var(--text-primary)',
                fontWeight: '600',
                fontSize: '14px',
                flex: 1,
                cursor: 'pointer',
                borderBottom: '2px solid var(--primary-500)'
              }}>
              Order Book
              </div>
              <div style={{
                padding: '8px 12px',
                display: 'flex',
                alignItems: 'center',
                justifyContent: 'center',
                backgroundColor: 'var(--background-primary)',
                color: 'var(--text-secondary)',
                fontWeight: '600',
                fontSize: '14px',
                flex: 1,
                cursor: 'pointer'
              }}>
                Recent Trades
              </div>
            </div>
            
            {/* Controls */}
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '4px' }}>
                <button style={{
                  padding: '6px 10px',
                  backgroundColor: 'var(--background-secondary)',
                  border: '1px solid var(--primary-500)',
                  borderRadius: '6px',
                  color: 'var(--text-primary)',
                  fontSize: '12px',
                  cursor: 'pointer',
                  display: 'flex',
                  alignItems: 'center',
                  gap: '4px',
                  transition: 'all 0.2s ease'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--primary-500)';
                  e.currentTarget.style.color = 'var(--background-primary)';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--background-secondary)';
                  e.currentTarget.style.color = 'var(--text-primary)';
                }}>
                  $1
                  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M6 9l6 6 6-6"/>
                  </svg>
                </button>
              </div>
              <button style={{
                padding: '6px 10px',
                backgroundColor: 'transparent',
                border: '1px solid var(--primary-500)',
                borderRadius: '6px',
                color: 'var(--text-primary)',
                fontSize: '12px',
                cursor: 'pointer',
                display: 'flex',
                alignItems: 'center',
                gap: '4px',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--primary-500)';
                e.currentTarget.style.color = 'var(--background-primary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'transparent';
                e.currentTarget.style.color = 'var(--text-primary)';
              }}>
                USD
                <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <path d="M3 7l6-6 6 6M3 17l6 6 6-6"/>
                </svg>
              </button>
            </div>

            {/* Column Headers - Compact */}
            <div style={{ 
              display: 'flex', 
              justifyContent: 'space-between', 
              color: 'var(--text-secondary)', 
              fontSize: '10px',
              fontWeight: '500',
              padding: '2px 0'
            }}>
              <span style={{ flex: 1, textAlign: 'left' }}>Price</span>
              <span style={{ flex: 1, textAlign: 'right' }}>Size</span>
            </div>
          </div>

          {/* Order Book Content */}
          <div style={{ 
            flex: 1, 
            position: 'relative', 
            overflow: 'hidden',
            display: 'flex',
            flexDirection: 'column-reverse'
          }}>
            {/* Buy Orders (Green) */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              {[
                { price: 109351, size: '77.6K', total: '527K' },
                { price: 109352, size: '109', total: '449K' },
                { price: 109353, size: '77.6K', total: '449K' },
                { price: 109354, size: '77.8K', total: '371K' },
                { price: 109355, size: '995', total: '293K' },
                { price: 109356, size: '77.6K', total: '292K' },
                { price: 109358, size: '77.6K', total: '215K' },
                { price: 109359, size: '63.9K', total: '137K' },
                { price: 109360, size: '44.9K', total: '73.6K' },
                { price: 109361, size: '218', total: '28.6K' }
              ].map((order, index) => (
                <div key={index} style={{
                  position: 'relative',
                  width: '100%',
                  height: '20px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0 12px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  borderBottom: '1px solid var(--background-secondary)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--success-500)';
                  e.currentTarget.style.opacity = '0.1';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                  e.currentTarget.style.opacity = '1';
                }}>
                  {/* Background bars */}
              <div style={{
                position: 'absolute',
                    left: 0,
                    top: 0,
                    height: '100%',
                    width: `${(10 - index) * 10}%`,
                    backgroundColor: 'var(--success-500)',
                    opacity: '0.15',
                    zIndex: 0
                  }} />
                  
                  {/* Content - Compact */}
                  <span style={{ 
                    color: 'var(--success-500)', 
                    fontSize: '10px', 
                    fontWeight: '500',
                    flex: 1,
                    textAlign: 'left',
                    zIndex: 1
                  }}>
                    {order.price.toLocaleString()}
                  </span>
                  <span style={{ 
                    color: 'var(--text-primary)', 
                    fontSize: '10px', 
                    flex: 1,
                    textAlign: 'right',
                    zIndex: 1
                  }}>
                    {order.size}
                  </span>
              </div>
              ))}
            </div>

            {/* Current Price / Spread */}
            <div style={{
              height: '40px',
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'center',
              padding: '8px 12px',
              borderTop: '2px solid var(--primary-500)',
              borderBottom: '2px solid var(--primary-500)',
              backgroundColor: 'var(--background-secondary)'
            }}>
              <div style={{ fontSize: '16px', fontWeight: '700', color: 'var(--text-primary)' }}>
                109,376
              </div>
              <div style={{ fontSize: '12px', color: 'var(--text-secondary)' }}>
                Spread: $26 (0.023%)
            </div>
          </div>

            {/* Sell Orders (Red) */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              {[
                { price: 109409, size: '109', total: '982K' },
                { price: 109408, size: '110K', total: '982K' },
                { price: 109405, size: '120K', total: '872K' },
                { price: 109403, size: '110K', total: '752K' },
                { price: 109401, size: '110K', total: '641K' },
                { price: 109399, size: '110K', total: '531K' },
                { price: 109397, size: '110K', total: '421K' },
                { price: 109394, size: '115K', total: '310K' },
                { price: 109392, size: '110K', total: '195K' },
                { price: 109391, size: '50.0K', total: '85.0K' }
              ].map((order, index) => (
                <div key={index} style={{
                  position: 'relative',
                  width: '100%',
                  height: '20px',
                  display: 'flex',
                  justifyContent: 'space-between',
                  alignItems: 'center',
                  padding: '0 12px',
                  cursor: 'pointer',
                  transition: 'all 0.2s ease',
                  borderBottom: '1px solid var(--background-secondary)'
                }}
                onMouseEnter={(e) => {
                  e.currentTarget.style.backgroundColor = 'var(--danger-500)';
                  e.currentTarget.style.opacity = '0.1';
                }}
                onMouseLeave={(e) => {
                  e.currentTarget.style.backgroundColor = 'transparent';
                  e.currentTarget.style.opacity = '1';
                }}>
                  {/* Background bars - Sell orders start from left */}
                  <div style={{ 
                    position: 'absolute',
                    left: 0,
                    top: 0,
                    height: '100%',
                    width: `${(index + 1) * 10}%`,
                    backgroundColor: '#ff0000',
                    opacity: '0.5',
                    zIndex: 0
                  }} />
                  
                  {/* Content */}
                  <span style={{ 
                    color: 'var(--danger-500)', 
                    fontSize: '10px', 
                    fontWeight: '500',
                    flex: 1,
                    textAlign: 'left',
                    zIndex: 1
                  }}>
                    {order.price.toLocaleString()}
                  </span>
                  <span style={{ 
                    color: 'var(--text-primary)', 
                    fontSize: '10px', 
                    flex: 2,
                    textAlign: 'right',
                    zIndex: 1
                  }}>
                    {order.size}
                  </span>
                  <span style={{ 
                    color: 'var(--text-primary)', 
                    fontSize: '10px', 
                    flex: 2,
                    textAlign: 'right',
                    zIndex: 1
                  }}>
                    {order.total}
                  </span>
                </div>
              ))}
            </div>
          </div>

          {/* Footer Controls */}
          <div style={{
            padding: '12px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            borderTop: '1px solid var(--primary-500)',
            backgroundColor: 'var(--background-secondary)'
          }}>
            <div style={{ display: 'flex', gap: '8px' }}>
              <button style={{
                padding: '8px',
                backgroundColor: 'var(--background-primary)',
                border: '1px solid var(--primary-500)',
                borderRadius: '6px',
                color: 'var(--text-primary)',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--primary-500)';
                e.currentTarget.style.color = 'var(--background-primary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--background-primary)';
                e.currentTarget.style.color = 'var(--text-primary)';
              }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <rect x="3" y="18" width="18" height="4" rx="0.5" fill="var(--success-500)"/>
                  <rect x="3" y="13" width="11" height="4" rx="0.5" fill="var(--success-500)"/>
                  <rect x="3" y="7" width="11" height="4" rx="0.5" fill="var(--error-500)"/>
                  <rect x="3" y="2" width="18" height="4" rx="0.5" fill="var(--error-500)"/>
                </svg>
              </button>
              <button style={{
                padding: '8px',
                backgroundColor: 'var(--background-primary)',
                border: '1px solid var(--primary-500)',
                borderRadius: '6px',
                color: 'var(--text-secondary)',
                cursor: 'pointer',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--primary-500)';
                e.currentTarget.style.color = 'var(--background-primary)';
              }}
              onMouseLeave={(e) => {
                e.currentTarget.style.backgroundColor = 'var(--background-primary)';
                e.currentTarget.style.color = 'var(--text-secondary)';
              }}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none">
                  <rect x="2" y="17.79" width="9.474" height="4.211" rx="0.5" fill="var(--success-500)"/>
                  <rect x="12.526" y="17.79" width="9.474" height="4.211" rx="0.5" fill="var(--error-500)"/>
                  <rect x="5.158" y="12.526" width="6.316" height="4.211" rx="0.5" fill="var(--success-500)"/>
                  <rect x="12.526" y="12.526" width="6.316" height="4.211" rx="0.5" fill="var(--error-500)"/>
                  <rect x="7.263" y="7.263" width="4.211" height="4.211" rx="0.5" fill="var(--success-500)"/>
                  <rect x="12.526" y="7.263" width="4.211" height="4.211" rx="0.5" fill="var(--error-500)"/>
                  <rect x="9.368" y="2" width="2.105" height="4.211" rx="0.5" fill="var(--success-500)"/>
                  <rect x="12.526" y="2" width="2.105" height="4.211" rx="0.5" fill="var(--error-500)"/>
                </svg>
              </button>
            </div>
          </div>
        </div>

        {/* Right Panel - Trading (Much Smaller) */}
        <div style={{ flex: '0 0 16%', display: 'flex', flexDirection: 'column', borderLeft: '1px solid var(--primary-500)' }}>

          {/* Trading Panel - Compact */}
          <div style={{ 
            height: '300px', 
            padding: '12px', 
            borderTop: '1px solid var(--primary-500)',
            backgroundColor: '#000000'
          }}>
            {/* Margin Mode */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '8px' }}>
                <span style={{ fontSize: '12px', color: '#9ca3af' }}>Margin Mode:</span>
                <div style={{ display: 'flex', gap: '4px' }}>
                  <button 
                    onClick={() => setMarginMode('cross')}
                    style={{
                      padding: '4px 8px',
                      backgroundColor: marginMode === 'cross' ? 'var(--primary-500)' : '#000000',
                      border: marginMode === 'cross' ? 'none' : '1px solid var(--primary-500)',
                      borderRadius: '4px',
                      color: marginMode === 'cross' ? '#ffffff' : '#9ca3af',
                      fontSize: '11px',
                      cursor: 'pointer'
                    }}>
                    Cross
                  </button>
                  <button 
                    onClick={() => setMarginMode('isolated')}
                    style={{
                      padding: '4px 8px',
                      backgroundColor: marginMode === 'isolated' ? 'var(--primary-500)' : '#000000',
                      border: marginMode === 'isolated' ? 'none' : '1px solid var(--primary-500)',
                      borderRadius: '4px',
                      color: marginMode === 'isolated' ? '#ffffff' : '#9ca3af',
                      fontSize: '11px',
                      cursor: 'pointer'
                    }}>
                    Isolated
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
                    fontSize: '10px',
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

            {/* Leverage Slider */}
            <div style={{ marginBottom: '16px' }}>
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center', 
                marginBottom: '8px' 
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
                  <div style={{ fontSize: '12px', color: '#9ca3af' }}>Leverage</div>
                  <button 
                    onClick={() => setMaxMode(!maxMode)}
                    style={{
                      fontSize: '9px',
                      color: maxMode ? '#ffffff' : '#ffffff',
                      backgroundColor: maxMode ? 'var(--primary-500)' : 'transparent',
                      border: maxMode ? 'none' : '1px solid var(--primary-500)',
                      borderRadius: '2px',
                      padding: '2px 6px',
                      cursor: 'pointer',
                      fontWeight: maxMode ? '600' : '400',
                      minWidth: '30px'
                    }}
                  >
                    {maxMode ? 'EXIT MAX' : 'MAX'}
                  </button>
                </div>
                <div style={{ 
                  fontSize: '12px', 
                  color: '#ffffff', 
                  fontWeight: '600',
                  backgroundColor: 'var(--primary-500)',
                  padding: '2px 8px',
                  borderRadius: '4px'
                }}>
                  {maxMode ? '1000x' : `${leverage}x`}
                </div>
              </div>
              
              {/* Leverage Slider */}
              <div style={{ position: 'relative', marginBottom: '8px' }}>
                {/* Slider Track Background */}
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '0',
                  right: '0',
                  height: '6px',
                  background: 'var(--background-primary)',
                  borderRadius: '3px',
                  transform: 'translateY(-50%)',
                  pointerEvents: 'none'
                }} />
                
                {/* Slider Track Fill */}
                <div style={{
                  position: 'absolute',
                  top: '50%',
                  left: '0',
                  width: `${maxMode ? (Math.min(leverage * 10, 1000) / 1000) * 100 : (leverage / 100) * 100}%`,
                  height: '6px',
                  background: 'var(--primary-500)',
                  borderRadius: '3px',
                  transform: 'translateY(-50%)',
                  pointerEvents: 'none'
                }} />
                
                <input
                  type="range"
                  min="1"
                  max={maxMode ? "1000" : "100"}
                  step="1"
                  value={maxMode ? Math.min(leverage * 10, 1000) : leverage}
                  onChange={(e) => {
                    if (maxMode) {
                      setLeverage(Math.floor(parseInt(e.target.value) / 10));
                    } else {
                      setLeverage(parseInt(e.target.value));
                    }
                  }}
                  style={{
                    width: '100%',
                    height: '13px',
                    background: 'transparent',
                    outline: 'none',
                    appearance: 'none',
                    WebkitAppearance: 'none',
                    MozAppearance: 'none',
                    cursor: 'pointer',
                    position: 'relative',
                    zIndex: 1,
                    margin: '0',
                    padding: '0'
                  }}
                />
              </div>
              
              {/* Leverage Markers */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                fontSize: '10px', 
                color: '#6b7280',
                marginTop: '4px'
              }}>
                <span>1x</span>
                <span>{maxMode ? '250x' : '25x'}</span>
                <span>{maxMode ? '500x' : '50x'}</span>
                <span>{maxMode ? '750x' : '75x'}</span>
                <span>{maxMode ? '1000x' : '100x'}</span>
              </div>
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
            flexDirection: 'column',
            marginTop: '120px'
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

          {/* Market Information Panel */}
          <div style={{
            height: '200px',
            backgroundColor: '#000000',
            borderTop: '1px solid var(--primary-500)',
            padding: '16px',
            display: 'flex',
            flexDirection: 'column'
          }}>
            <div style={{ fontSize: '14px', fontWeight: '600', marginBottom: '12px', color: '#ffffff' }}>
              Market Info
            </div>
            
            {/* Market Data Grid */}
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '8px' }}>
              {/* Funding Rate */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '6px 8px',
                backgroundColor: 'var(--background-primary)',
                borderRadius: '4px'
              }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Funding Rate</span>
                <span style={{ fontSize: '11px', color: 'var(--danger-500)', fontWeight: '500' }}>-0.0125%</span>
              </div>

              {/* Open Interest */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '6px 8px',
                backgroundColor: 'var(--background-primary)',
                borderRadius: '4px'
              }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Open Interest</span>
                <span style={{ fontSize: '11px', color: 'var(--text-primary)', fontWeight: '500' }}>$2.1B</span>
              </div>

              {/* 24h Volume */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '6px 8px',
                backgroundColor: 'var(--background-primary)',
                borderRadius: '4px'
              }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>24h Volume</span>
                <span style={{ fontSize: '11px', color: 'var(--text-primary)', fontWeight: '500' }}>$45.2M</span>
              </div>

              {/* Trading Fees */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '6px 8px',
                backgroundColor: 'var(--background-primary)',
                borderRadius: '4px'
              }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Trading Fees</span>
                <span style={{ fontSize: '11px', color: 'var(--text-primary)', fontWeight: '500' }}>0.05% / 0.03%</span>
              </div>

              {/* Slippage Estimator */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '6px 8px',
                backgroundColor: 'var(--background-primary)',
                borderRadius: '4px'
              }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Est. Slippage</span>
                <span style={{ fontSize: '11px', color: 'var(--success-500)', fontWeight: '500' }}>0.02%</span>
              </div>

              {/* Liquidation Price */}
              <div style={{ 
                display: 'flex', 
                justifyContent: 'space-between', 
                alignItems: 'center',
                padding: '6px 8px',
                backgroundColor: 'var(--background-primary)',
                borderRadius: '4px'
              }}>
                <span style={{ fontSize: '11px', color: 'var(--text-secondary)' }}>Liq. Price</span>
                <span style={{ fontSize: '11px', color: 'var(--text-primary)', fontWeight: '500' }}>N/A</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default TradingTab;