import React, { useEffect, useRef, memo } from 'react';

function TradingViewWidget() {
  const container = useRef();

  useEffect(
    () => {
      const script = document.createElement("script");
      script.src = "https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js";
      script.type = "text/javascript";
      script.async = true;
      script.innerHTML = `
        {
          "allow_symbol_change": true,
          "calendar": false,
          "details": false,
          "hide_side_toolbar": false,
          "hide_top_toolbar": false,
          "hide_legend": false,
          "hide_volume": false,
          "hotlist": false,
          "interval": "60",
          "locale": "en",
          "save_image": true,
          "style": "1",
          "symbol": "BITSTAMP:BTCUSD",
          "theme": "dark",
          "timezone": "Etc/UTC",
          "backgroundColor": "#0a0a0a",
          "gridColor": "rgba(242, 242, 242, 0.06)",
          "watchlist": [],
          "withdateranges": false,
          "compareSymbols": [],
          "studies": [],
          "autosize": true
        }`;
      container.current.appendChild(script);
    },
    []
  );

  return (
    <div 
      className="tradingview-widget-container" 
      ref={container} 
      style={{ 
        height: "100%", 
        width: "100%",
        position: "absolute",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0
      }}
    >
      <div 
        className="tradingview-widget-container__widget" 
        style={{ 
          height: "calc(100% - 32px)", 
          width: "100%" 
        }}
      />
      <div 
        className="tradingview-widget-copyright"
        style={{
          position: "absolute",
          bottom: "0",
          left: "0",
          right: "0",
          height: "32px",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          fontSize: "10px",
          color: "#6b7280",
          backgroundColor: "rgba(0, 0, 0, 0.1)"
        }}
      >
        <a 
          href="https://www.tradingview.com/symbols/BTCUSD/?exchange=BITSTAMP" 
          rel="noopener nofollow" 
          target="_blank"
          style={{ color: "var(--primary-500)", textDecoration: "none" }}
        >
          <span className="blue-text">BTCUSD chart</span>
        </a>
        <span className="trademark" style={{ marginLeft: "4px" }}> by TradingView</span>
      </div>
    </div>
  );
}

export default memo(TradingViewWidget);
