import { useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMarkets } from '../contexts/MarketContext';

interface TickerClickHook {
  handleTickerClick: (symbol: string) => void;
  parseTickerFromText: (text: string) => string[];
  isTickerMention: (text: string, symbol: string) => boolean;
}

/**
 * Hook for handling ticker clicks and mentions
 * Provides functionality to click tickers and navigate to charts
 */
export const useTickerClick = (): TickerClickHook => {
  const { selectMarketBySymbol, markets } = useMarkets();
  const navigate = useNavigate();

  // Handle ticker click - navigate to trading page with market loaded
  const handleTickerClick = useCallback((symbol: string) => {
    // Remove common suffixes for matching
    const cleanSymbol = symbol.replace(/[-/]/g, '').toUpperCase();
    
    // Try exact match first
    let market = markets.find(m => 
      m.symbol.toUpperCase() === symbol.toUpperCase() ||
      m.baseAsset.toUpperCase() === cleanSymbol
    );
    
    // Try partial match
    if (!market) {
      market = markets.find(m => 
        m.symbol.toUpperCase().includes(cleanSymbol) ||
        m.baseAsset.toUpperCase().includes(cleanSymbol)
      );
    }
    
    if (market) {
      // Select market in context
      selectMarketBySymbol(market.symbol);
      
      // Navigate to trading page with market symbol
      navigate(`/lite`);
      
      // Dispatch custom event for other components to listen
      window.dispatchEvent(new CustomEvent('tickerClicked', { 
        detail: { symbol: market.symbol, market } 
      }));
      
      console.log(`🎯 Navigated to trading page for ${market.symbol}`);
    } else {
      console.warn(`Market not found for symbol: ${symbol}`);
    }
  }, [selectMarketBySymbol, markets, navigate]);

  // Parse ticker symbols from text (e.g., "BTC is pumping!" -> ["BTC"])
  const parseTickerFromText = useCallback((text: string): string[] => {
    const tickers: string[] = [];
    
    // Common ticker patterns
    const patterns = [
      /\b[A-Z]{2,6}\b/g, // 2-6 uppercase letters
      /\$[A-Z]{2,6}\b/g, // $BTC, $ETH, etc.
      /\b[A-Z]{2,6}-PERP\b/g, // BTC-PERP, ETH-PERP
      /\b[A-Z]{2,6}\/USDT\b/g, // BTC/USDT, ETH/USDT
    ];
    
    patterns.forEach(pattern => {
      const matches = text.match(pattern);
      if (matches) {
        matches.forEach(match => {
          const cleanMatch = match.replace(/[$-\/]/g, '');
          if (cleanMatch.length >= 2 && cleanMatch.length <= 6) {
            tickers.push(cleanMatch);
          }
        });
      }
    });
    
    // Remove duplicates and filter against known markets
    const uniqueTickers = [...new Set(tickers)];
    return uniqueTickers.filter(ticker => 
      markets.some(market => 
        market.symbol.toUpperCase().includes(ticker.toUpperCase()) ||
        market.baseAsset.toUpperCase() === ticker.toUpperCase()
      )
    );
  }, [markets]);

  // Check if text contains a ticker mention
  const isTickerMention = useCallback((text: string, symbol: string): boolean => {
    const tickers = parseTickerFromText(text);
    return tickers.some(ticker => 
      ticker.toUpperCase() === symbol.toUpperCase()
    );
  }, [parseTickerFromText]);

  return {
    handleTickerClick,
    parseTickerFromText,
    isTickerMention
  };
};

export default useTickerClick;
