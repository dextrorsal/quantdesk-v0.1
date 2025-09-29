import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';

interface QuantDeskChartProps {
  symbol?: string;
  height?: number;
  timeframe?: string;
  isMobile?: boolean; // Add mobile flag to optimize data fetching
}

const QuantDeskChart: React.FC<QuantDeskChartProps> = ({ 
  symbol = 'BTC', 
  height = 500,
  timeframe = '1h',
  isMobile = false
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch extensive historical data from Binance API with multiple requests
  const fetchExtensiveBinanceData = async (symbol: string, interval: string): Promise<CandlestickData[]> => {
    try {
      setIsLoading(true);
      setError(null);
      
      const binanceSymbol = `${symbol}USDT`;
      // console.log(`📊 Fetching extensive ${binanceSymbol} data from Binance API (${interval})...`);
      
      // Calculate how many requests we need based on timeframe and device
      const requestsNeeded = (() => {
        if (isMobile) {
          // Mobile: Use less data for faster loading
          switch (interval) {
            case '1m': return 1; // 1 request = 1000 candles = ~16 hours
            case '5m': return 1; // 1 request = 1000 candles = ~3 days  
            case '15m': return 1; // 1 request = 1000 candles = ~10 days
            case '1h': return 1; // 1 request = 1000 candles = ~41 days
            case '4h': return 1; // 1 request = 1000 candles = ~166 days
            case '1d': return 1; // 1 request = 1000 candles = ~2.7 years
            default: return 1;
          }
        } else {
          // Desktop: Use more data for detailed analysis
          switch (interval) {
            case '1m': return 3; // 3 requests = 3000 candles = ~50 hours
            case '5m': return 3; // 3 requests = 3000 candles = ~10 days  
            case '15m': return 3; // 3 requests = 3000 candles = ~31 days
            case '1h': return 2; // 2 requests = 2000 candles = ~83 days
            case '4h': return 1; // 1 request = 1000 candles = ~166 days
            case '1d': return 1; // 1 request = 1000 candles = ~2.7 years
            default: return 1;
          }
        }
      })();
      
      let allData: CandlestickData[] = [];
      
      // Make multiple requests to get more historical data
      for (let i = 0; i < requestsNeeded; i++) {
        const endTime = i === 0 ? undefined : allData[0]?.time ? (allData[0].time as number) * 1000 - 1 : undefined;
        const url = endTime 
          ? `https://api.binance.com/api/v3/klines?symbol=${binanceSymbol}&interval=${interval}&limit=1000&endTime=${endTime}`
          : `https://api.binance.com/api/v3/klines?symbol=${binanceSymbol}&interval=${interval}&limit=1000`;
        
        const response = await fetch(url);
        
        if (!response.ok) {
          throw new Error(`Binance API error: ${response.status}`);
        }
        
        const klines = await response.json();
        
        const data = klines.map((kline: any[]) => ({
          time: Math.floor(kline[0] / 1000) as Time,
          open: parseFloat(kline[1]),
          high: parseFloat(kline[2]),
          low: parseFloat(kline[3]),
          close: parseFloat(kline[4]),
        }));
        
        // Prepend older data to the beginning
        allData = [...data, ...allData];
        
        // console.log(`📊 Request ${i + 1}/${requestsNeeded}: Loaded ${data.length} candles`);
        
        // Small delay between requests to be respectful to the API
        if (i < requestsNeeded - 1) {
          await new Promise(resolve => setTimeout(resolve, 100));
        }
      }
      
      // console.log(`✅ Successfully loaded ${allData.length} total candles for ${binanceSymbol} (${interval})`);
      setIsLoading(false);
      return allData;
    } catch (error) {
      console.error('Failed to fetch Binance data:', error);
      setError('Failed to load data from Binance API');
      setIsLoading(false);
      return [];
    }
  };

  // Fetch real data from Binance API (simplified version)
  const fetchBinanceData = async (symbol: string, interval: string, limit?: number): Promise<CandlestickData[]> => {
    try {
      setIsLoading(true);
      setError(null);
      
      const binanceSymbol = `${symbol}USDT`;
      // console.log(`📊 Fetching ${binanceSymbol} data from Binance API (${interval})...`);
      
      // Use fewer candles on mobile for faster loading
      const defaultLimit = isMobile ? 100 : 1000; // Mobile: 100 candles, Desktop: 1000 candles
      const actualLimit = limit || defaultLimit;
      
      const response = await fetch(
        `https://api.binance.com/api/v3/klines?symbol=${binanceSymbol}&interval=${interval}&limit=${actualLimit}`
      );
      
      if (!response.ok) {
        throw new Error(`Binance API error: ${response.status}`);
      }
      
      const klines = await response.json();
      
      const data = klines.map((kline: any[]) => ({
        time: Math.floor(kline[0] / 1000) as Time, // Convert to seconds and ensure integer
        open: parseFloat(kline[1]),
        high: parseFloat(kline[2]),
        low: parseFloat(kline[3]),
        close: parseFloat(kline[4]),
      }));
      
      // console.log(`✅ Successfully loaded ${data.length} candles for ${binanceSymbol}`);
      setIsLoading(false);
      return data;
    } catch (error) {
      console.error('Failed to fetch Binance data:', error);
      setError('Failed to load data from Binance API');
      setIsLoading(false);
      return [];
    }
  };

  useEffect(() => {
    if (!chartContainerRef.current) return;

    const initializeChart = async () => {
      // Create chart with professional styling
      const chart = createChart(chartContainerRef.current!, {
        layout: {
          background: { type: ColorType.Solid, color: '#0a0a0a' },
          textColor: '#ffffff',
        },
        width: chartContainerRef.current!.clientWidth,
        height: height,
        grid: {
          vertLines: { color: '#333333' },
          horzLines: { color: '#333333' },
        },
        rightPriceScale: {
          borderColor: '#333333',
          textColor: '#ffffff',
        },
        timeScale: {
          borderColor: '#333333',
          textColor: '#ffffff',
          timeVisible: true,
          secondsVisible: false,
          barSpacing: 8,
        },
        crosshair: {
          mode: 1,
          vertLine: {
            color: '#666666',
            width: 1,
            style: 2,
          },
          horzLine: {
            color: '#666666',
            width: 1,
            style: 2,
          },
        },
      });

      chartRef.current = chart;

      // Add candlestick series
      const candlestickSeries = chart.addCandlestickSeries({
        upColor: '#26a69a',
        downColor: '#ef5350',
        borderVisible: false,
        wickUpColor: '#26a69a',
        wickDownColor: '#ef5350',
      });

      candlestickSeriesRef.current = candlestickSeries;

      // Convert timeframe to Binance interval
      const binanceInterval = (() => {
        switch (timeframe) {
          case '1m': return '1m';
          case '5m': return '5m';
          case '15m': return '15m';
          case '1h': return '1h';
          case '4h': return '4h';
          case '1d': return '1d';
          default: return '1h';
        }
      })();

      // Fetch real data from Binance
      const initialData = await fetchExtensiveBinanceData(symbol, binanceInterval);
      candlestickSeries.setData(initialData);

      // Fit content to show all data
      chart.timeScale().fitContent();

      // Set up real-time updates (disabled on mobile for performance)
      let updateInterval: NodeJS.Timeout | null = null;
      if (!isMobile) {
        updateInterval = setInterval(() => {
          if (candlestickSeriesRef.current) {
            try {
              // Generate new candle data with proper time format
              const now = Date.now();
              const intervalMs = (() => {
                switch (timeframe) {
                  case '1m': return 60 * 1000;
                  case '5m': return 5 * 60 * 1000;
                  case '15m': return 15 * 60 * 1000;
                  case '1h': return 60 * 60 * 1000;
                  case '4h': return 4 * 60 * 60 * 1000;
                  case '1d': return 24 * 60 * 60 * 1000;
                  default: return 60 * 60 * 1000;
                }
              })();
              
              const time = Math.floor(now / intervalMs) * intervalMs / 1000 as Time;
              const volatility = 0.01;
              const change = (Math.random() - 0.5) * volatility;
              const open = 50000; // Base price
              const close = open * (1 + change);
              const high = Math.max(open, close) * (1 + Math.random() * 0.005);
              const low = Math.min(open, close) * (1 - Math.random() * 0.005);
              
              const newCandle = {
                time: time,
                open: open,
                high: high,
                low: low,
                close: close
              };
              
              // Update the series with new data
              candlestickSeriesRef.current.update(newCandle);
            } catch (error) {
              console.warn('Failed to update real-time data:', error);
            }
          }
        }, 5000); // Update every 5 seconds to reduce errors
      }

      // Handle resize with debouncing for mobile
      let resizeTimeout: NodeJS.Timeout | null = null;
      const handleResize = () => {
        if (resizeTimeout) {
          clearTimeout(resizeTimeout);
        }
        resizeTimeout = setTimeout(() => {
          if (chartContainerRef.current && chartRef.current) {
            chartRef.current.applyOptions({
              width: chartContainerRef.current.clientWidth,
              height: height,
            });
          }
        }, isMobile ? 300 : 100); // Longer debounce on mobile
      };

      window.addEventListener('resize', handleResize);

      // Cleanup
      return () => {
        if (updateInterval) {
          clearInterval(updateInterval);
        }
        window.removeEventListener('resize', handleResize);
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
        }
      };
    };

    initializeChart();
  }, [height, timeframe, isMobile]); // Add isMobile dependency

  // Update data when symbol or timeframe changes
  useEffect(() => {
    const updateData = async () => {
      if (candlestickSeriesRef.current && chartRef.current) {
        // console.log(`🔄 Updating chart data for ${symbol} (${timeframe})...`);
        
        const binanceInterval = (() => {
          switch (timeframe) {
            case '1m': return '1m';
            case '5m': return '5m';
            case '15m': return '15m';
            case '1h': return '1h';
            case '4h': return '4h';
            case '1d': return '1d';
            default: return '1h';
          }
        })();

        try {
          const newData = await fetchBinanceData(symbol, binanceInterval, 100);
          
          // Clear existing data and set new data
          candlestickSeriesRef.current.setData([]);
          candlestickSeriesRef.current.setData(newData);
          
          // Fit content to show all data
          chartRef.current.timeScale().fitContent();
          
          // console.log(`✅ Chart updated with new data for ${symbol}`, newData.length, 'candles');
        } catch (error) {
          console.error('Failed to update chart data:', error);
        }
      }
    };

    updateData();
  }, [symbol, timeframe, isMobile]); // Add isMobile dependency

  return (
    <div style={{ position: 'relative', width: '100%', height: `${height}px` }}>
      {isLoading && (
        <div style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          backgroundColor: '#0a0a0a',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 10,
          borderRadius: '8px'
        }}>
          <div style={{ color: '#ffffff', fontSize: '14px' }}>
            Loading {symbol} data from Binance...
          </div>
        </div>
      )}
      
      {error && (
        <div style={{
          position: 'absolute',
          top: '10px',
          right: '10px',
          backgroundColor: '#ef5350',
          color: '#ffffff',
          padding: '4px 8px',
          borderRadius: '4px',
          fontSize: '12px',
          zIndex: 10
        }}>
          {error}
        </div>
      )}
      
      <div 
        ref={chartContainerRef} 
        style={{ 
          width: '100%', 
          height: '100%',
          backgroundColor: '#0a0a0a',
          borderRadius: '8px',
          overflow: 'hidden'
        }} 
      />
    </div>
  );
};

export default QuantDeskChart;
