import React, { useEffect, useRef, memo } from 'react';

function TradingViewCryptoMarkets() {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current) return;

    // Clear any existing content
    container.current.innerHTML = '';

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-screener.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "defaultColumn": "overview",
        "screener_type": "crypto_mkt",
        "displayCurrency": "USD",
        "colorTheme": "dark",
        "isTransparent": true,
        "locale": "en",
        "width": "100%",
        "height": 600
      }`;
    
    container.current.appendChild(script);

    // Cleanup function
    return () => {
      if (container.current) {
        container.current.innerHTML = '';
      }
    };
  }, []);

  return (
    <div className="tradingview-widget-container" ref={container}>
      <div className="tradingview-widget-container__widget"></div>
      <div className="tradingview-widget-copyright" style={{ display: 'none' }}>
        <a 
          href="https://www.tradingview.com/markets/cryptocurrencies/prices-all/" 
          rel="noopener nofollow" 
          target="_blank"
        >
          <span className="blue-text">Crypto markets</span>
        </a>
        <span className="trademark"> by TradingView</span>
      </div>
    </div>
  );
}

export default memo(TradingViewCryptoMarkets);