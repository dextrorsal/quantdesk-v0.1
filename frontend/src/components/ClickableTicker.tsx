import React from 'react';
import { useTickerClick } from '../hooks/useTickerClick';

interface ClickableTickerProps {
  symbol: string;
  className?: string;
  children?: React.ReactNode;
}

/**
 * Clickable ticker component for chat and text
 * Automatically detects ticker symbols and makes them clickable
 */
export const ClickableTicker: React.FC<ClickableTickerProps> = ({ 
  symbol, 
  className = '',
  children 
}) => {
  const { handleTickerClick } = useTickerClick();

  const handleClick = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    handleTickerClick(symbol);
  };

  return (
    <span
      className={`cursor-pointer text-blue-400 hover:text-blue-300 hover:underline transition-colors ${className}`}
      onClick={handleClick}
      title={`Click to view ${symbol} chart`}
    >
      {children || symbol}
    </span>
  );
};

/**
 * Component that automatically converts ticker mentions in text to clickable links
 */
interface TickerTextProps {
  text: string;
  className?: string;
}

export const TickerText: React.FC<TickerTextProps> = ({ text, className = '' }) => {
  const { parseTickerFromText } = useTickerClick();

  const renderTextWithTickers = () => {
    const tickers = parseTickerFromText(text);
    
    if (tickers.length === 0) {
      return <span>{text}</span>;
    }

    let processedText = text;
    
    // Replace ticker mentions with clickable components
    tickers.forEach(ticker => {
      const regex = new RegExp(`\\b${ticker}\\b`, 'gi');
      processedText = processedText.replace(regex, `__TICKER_${ticker}__`);
    });

    // Split by ticker placeholders and render
    const parts = processedText.split(/(__TICKER_\w+__)/);
    
    return (
      <span>
        {parts.map((part, index) => {
          if (part.startsWith('__TICKER_') && part.endsWith('__')) {
            const ticker = part.replace(/__TICKER_|__/g, '');
            return (
              <ClickableTicker key={index} symbol={ticker}>
                {ticker}
              </ClickableTicker>
            );
          }
          return <span key={index}>{part}</span>;
        })}
      </span>
    );
  };

  return (
    <span className={className}>
      {renderTextWithTickers()}
    </span>
  );
};

export default ClickableTicker;
