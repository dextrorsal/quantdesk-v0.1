import React, { useEffect, useRef, memo } from 'react';

interface TradingViewWidgetProps {
  symbol?: string;
  interval?: string;
  theme?: 'light' | 'dark';
  height?: number;
  width?: string;
}

const TradingViewWidget: React.FC<TradingViewWidgetProps> = ({ 
  symbol = 'BINANCE:BTCUSDT.P',
  interval = '1h',
  theme = 'dark',
  height = 400,
  width = '100%'
}) => {
  const container = useRef<HTMLDivElement>(null);
  const [loading, setLoading] = React.useState(true);
  const [error, setError] = React.useState(false);

  useEffect(() => {
    if (!container.current) return;

    // Clear any existing content
    container.current.innerHTML = '';
    setLoading(true);
    setError(false);

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
    script.type = "text/javascript";
    script.async = true;
    
    // Add error handling
    script.onerror = () => {
      console.log('TradingView widget failed to load, using fallback');
      setError(true);
      setLoading(false);
    };
    
    script.onload = () => {
      setLoading(false);
    };
    script.innerHTML = `
      {
        "allow_symbol_change": true,
        "calendar": false,
        "details": false,
        "hide_side_toolbar": true,
        "hide_top_toolbar": false,
        "hide_legend": false,
        "hide_volume": false,
        "hotlist": false,
        "interval": "${interval}",
        "locale": "en",
        "save_image": true,
        "style": "1",
        "symbol": "${symbol}",
        "theme": "${theme}",
        "timezone": "Etc/UTC",
        "backgroundColor": "#0F0F0F",
        "gridColor": "rgba(242, 242, 242, 0.06)",
        "watchlist": [],
        "withdateranges": false,
        "compareSymbols": [],
        "studies": [],
        "autosize": true
      }`;
    
    container.current.appendChild(script);

    // Cleanup function
    return () => {
      if (container.current) {
        container.current.innerHTML = '';
      }
    };
  }, [symbol, interval, theme]);

  // Fallback component when TradingView fails to load
  const FallbackChart = () => (
    <div style={{
      height: `${height}px`,
      width: width,
      backgroundColor: '#0F0F0F',
      borderRadius: '8px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#ffffff',
      border: '1px solid #1a1a1a'
    }}>
      <div style={{ fontSize: '24px', marginBottom: '16px' }}>📊</div>
      <div style={{ fontSize: '18px', fontWeight: '600', marginBottom: '8px' }}>
        {symbol.split(':')[1]} Chart
      </div>
      <div style={{ fontSize: '14px', color: '#9ca3af', textAlign: 'center', marginBottom: '16px' }}>
        TradingView widget unavailable<br/>
        Using QuantDesk lightweight chart
      </div>
      <div style={{ fontSize: '12px', color: '#6b7280' }}>
        Interval: {interval} | Theme: {theme}
      </div>
    </div>
  );

  // Loading component
  const LoadingChart = () => (
    <div style={{
      height: `${height}px`,
      width: width,
      backgroundColor: '#0F0F0F',
      borderRadius: '8px',
      display: 'flex',
      flexDirection: 'column',
      alignItems: 'center',
      justifyContent: 'center',
      color: '#ffffff',
      border: '1px solid #1a1a1a'
    }}>
      <div style={{ 
        width: '40px', 
        height: '40px', 
        border: '3px solid #1a1a1a',
        borderTop: '3px solid var(--primary-500)',
        borderRadius: '50%',
        animation: 'spin 1s linear infinite',
        marginBottom: '16px'
      }} />
      <div style={{ fontSize: '14px', color: '#9ca3af' }}>
        Loading TradingView chart...
      </div>
    </div>
  );

  if (loading) {
    return <LoadingChart />;
  }

  if (error) {
    return <FallbackChart />;
  }

  return (
    <div 
      className="tradingview-widget-container" 
      ref={container} 
      style={{ 
        height: `${height}px`, 
        width: width,
        backgroundColor: '#0F0F0F'
      }}
    >
      <div 
        className="tradingview-widget-container__widget" 
        style={{ 
          height: `calc(100% - 32px)`, 
          width: "100%" 
        }}
      />
      <div className="tradingview-widget-copyright">
        <a 
          href={`https://www.tradingview.com/symbols/${symbol.split(':')[1]}/?exchange=${symbol.split(':')[0]}`} 
          rel="noopener nofollow" 
          target="_blank"
        >
          <span className="blue-text">{symbol.split(':')[1]} chart</span>
        </a>
        <span className="trademark"> by TradingView</span>
      </div>
    </div>
  );
};

export default memo(TradingViewWidget);