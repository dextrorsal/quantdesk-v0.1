import React, { useEffect, useRef, memo } from 'react';

function TradingViewTickerTape() {
  const container = useRef();

  useEffect(() => {
    if (!container.current) return;

    // Clear any existing content first
    container.current.innerHTML = '';

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-ticker-tape.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "symbols": [
          {
            "proName": "BITSTAMP:BTCUSD",
            "title": "Bitcoin"
          },
          {
            "proName": "BITSTAMP:ETHUSD",
            "title": "Ethereum"
          },
          {
            "proName": "CRYPTO:SOLUSD",
            "title": "Solana"
          },
          {
            "proName": "BINANCE:XRPUSDT",
            "title": "XRP"
          },
          {
            "proName": "BINANCE:DOGEUSDT",
            "title": "DOGE"
          },
          {
            "proName": "BINANCE:PEPEUSDT",
            "title": "PEPE"
          },
          {
            "proName": "MEXC:SPXUSDT",
            "title": "SPX"
          },
          {
            "proName": "BITGET:FARTCOINUSDT",
            "title": "Fartcoin"
          }
        ],
        "colorTheme": "dark",
        "locale": "en",
        "largeChartUrl": "",
        "isTransparent": true,
        "showSymbolLogo": true,
        "displayMode": "adaptive"
      }`;
    
    container.current.appendChild(script);

    // Cleanup function to remove the script when component unmounts
    return () => {
      if (container.current) {
        container.current.innerHTML = '';
      }
    };
  }, []);

  return (
    <div ref={container}></div>
  );
}

export default memo(TradingViewTickerTape);