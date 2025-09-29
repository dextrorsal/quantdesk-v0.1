import React, { useEffect, useRef, memo } from 'react';

interface TradingViewHeatmapProps {
  dataSource?: 'Crypto' | 'Forex' | 'Stock';
  blockSize?: 'market_cap_calc' | 'volume' | '24h_vol_cmc';
  blockColor?: '24h_close_change|5' | '24h_close_change|10' | '24h_close_change|20';
  colorTheme?: 'light' | 'dark';
  hasTopBar?: boolean;
  isDataSetEnabled?: boolean;
  isZoomEnabled?: boolean;
  hasSymbolTooltip?: boolean;
  isMonoSize?: boolean;
  width?: string;
  height?: string;
  locale?: string;
}

function TradingViewHeatmap({
  dataSource = 'Crypto',
  blockSize = '24h_vol_cmc',
  blockColor = '24h_close_change|5',
  colorTheme = 'dark',
  hasTopBar = false,
  isDataSetEnabled = false,
  isZoomEnabled = true,
  hasSymbolTooltip = true,
  isMonoSize = false,
  width = '100%',
  height = '100%',
  locale = 'en'
}: TradingViewHeatmapProps) {
  const container = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!container.current) return;

    // Clear any existing content
    container.current.innerHTML = '';

    const script = document.createElement("script");
    script.src = "https://s3.tradingview.com/external-embedding/embed-widget-crypto-coins-heatmap.js";
    script.type = "text/javascript";
    script.async = true;
    script.innerHTML = `
      {
        "dataSource": "${dataSource}",
        "blockSize": "${blockSize}",
        "blockColor": "${blockColor}",
        "locale": "${locale}",
        "symbolUrl": "",
        "colorTheme": "${colorTheme}",
        "hasTopBar": ${hasTopBar},
        "isDataSetEnabled": ${isDataSetEnabled},
        "isZoomEnabled": ${isZoomEnabled},
        "hasSymbolTooltip": ${hasSymbolTooltip},
        "isMonoSize": ${isMonoSize},
        "width": "${width}",
        "height": "${height}"
      }`;
    
    container.current.appendChild(script);

    // Cleanup function
    return () => {
      if (container.current) {
        container.current.innerHTML = '';
      }
    };
  }, [dataSource, blockSize, blockColor, colorTheme, hasTopBar, isDataSetEnabled, isZoomEnabled, hasSymbolTooltip, isMonoSize, width, height, locale]);

  return (
    <div 
      className="tradingview-widget-container" 
      ref={container}
      style={{
        width: width,
        height: height,
        backgroundColor: 'transparent'
      }}
    >
      <div className="tradingview-widget-container__widget"></div>
      <div className="tradingview-widget-copyright" style={{ display: 'none' }}>
        <a 
          href="https://www.tradingview.com/heatmap/crypto/" 
          rel="noopener nofollow" 
          target="_blank"
        >
          <span className="blue-text">Crypto Heatmap</span>
        </a>
        <span className="trademark"> by TradingView</span>
      </div>
    </div>
  );
}

export default memo(TradingViewHeatmap);
