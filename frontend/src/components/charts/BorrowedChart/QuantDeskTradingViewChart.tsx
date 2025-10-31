import React, { useEffect, useRef, memo } from 'react';
import { useTheme } from '../../../contexts/ThemeContext';

interface QuantDeskTradingViewChartProps {
  symbol: string;
  interval?: string;
  height?: number | string;
  onSymbolChange?: (symbol: string) => void;
}

const QuantDeskTradingViewChart: React.FC<QuantDeskTradingViewChartProps> = ({
  symbol,
  interval = '60',
  height = 500,
  onSymbolChange
}) => {
  const container = useRef<HTMLDivElement>(null);
  const { theme } = useTheme();

  useEffect(() => {
    if (!container.current) return;

    // Clear container completely
    container.current.innerHTML = '';

    // Convert interval to TradingView format
    const intervalMap: { [key: string]: string } = {
      '1m': '1',
      '5m': '5',
      '15m': '15',
      '30m': '30',
      '60': '60',
      '1h': '60',
      '4h': '240',
      '1D': 'D',
      '1d': 'D',
      '1W': 'W',
      '1w': 'W'
    };

    const tvInterval = intervalMap[interval] || '60';

    // Create TradingView widget container
    const widgetContainer = document.createElement('div');
    widgetContainer.className = 'tradingview-widget-container';
    widgetContainer.style.width = '100%';
    widgetContainer.style.height = typeof height === 'string' ? height : `${height}px`;

    const widgetInner = document.createElement('div');
    widgetInner.className = 'tradingview-widget-container__widget';
    widgetInner.style.height = 'calc(100% - 32px)';
    widgetInner.style.width = '100%';

    widgetContainer.appendChild(widgetInner);
    container.current.appendChild(widgetContainer);

    // Get theme colors
    const computedStyle = getComputedStyle(document.documentElement);
    const bgPrimary = computedStyle.getPropertyValue('--bg-primary').trim() || '#000000';
    const bgTertiary = computedStyle.getPropertyValue('--bg-tertiary').trim() || '#1a1a1a';

    // Create script with TradingView's official dynamic symbol method
    const script = document.createElement('script');
    script.type = 'text/javascript';
    script.src = 'https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js';
    script.async = true;
    
    script.innerHTML = JSON.stringify({
      "autosize": true,
      "symbol": `BINANCE:${symbol}USDT`,
      "interval": tvInterval,
      "timezone": "Etc/UTC",
      "theme": "dark",
      "style": "1",
      "locale": "en",
      "toolbar_bg": bgPrimary,
      "enable_publishing": false,
      "hide_side_toolbar": false,
      "allow_symbol_change": false, // Disable symbol change to remove search field
      "save_image": false,
      "hide_top_toolbar": false,
      "hide_legend": false,
      "backgroundColor": bgPrimary,
      "gridColor": bgTertiary,
      "container_id": widgetContainer,
      // TradingView's official dynamic symbol support
      "tvwidgetsymbol": `BINANCE:${symbol}USDT`,
      // Disable search and symbol change features
      "hide_symbol_search": true,
      "hide_header_symbol_search": true
    });

    widgetContainer.appendChild(script);

    return () => {
      // Cleanup
      if (container.current) {
        container.current.innerHTML = '';
      }
    };
  }, [symbol, interval, theme, height]);

  return (
    <div 
      ref={container} 
      className="w-full h-full"
    />
  );
};

export default memo(QuantDeskTradingViewChart);