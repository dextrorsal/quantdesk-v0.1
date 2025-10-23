import React, { useEffect, useRef, useState } from 'react';
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, Time } from 'lightweight-charts';

interface QuantDeskChartProps {
  symbol?: string;
  height?: number;
  timeframe?: string;
}

const QuantDeskChart: React.FC<QuantDeskChartProps> = ({ 
  symbol = 'BTC', 
  height = 500,
  timeframe = '1h'
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Fetch real data from Binance API
  const fetchBinanceData = async (symbol: string, interval: string, limit: number = 100): Promise<CandlestickData[]> => {
    try {
      setIsLoading(true);
      setError(null);
      
      const binanceSymbol = `${symbol}USDT`;
      console.log(`📊 Fetching ${binanceSymbol} data from Binance API (${interval})...`);
      
      const response = await fetch(
        `https://api.binance.com/api/v3/klines?symbol=${binanceSymbol}&interval=${interval}&limit=${limit}`
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
      
      console.log(`✅ Successfully loaded ${data.length} candles for ${binanceSymbol}`);
      setIsLoading(false);
      return data;
    } catch (error) {
      console.warn('Failed to fetch Binance data, using mock data:', error);
      setError('Using mock data - Binance API unavailable');
      setIsLoading(false);
      return generateMockData(limit);
    }
  };

  // Generate realistic mock data as fallback
  const generateMockData = (count: number = 100): CandlestickData[] => {
    const data: CandlestickData[] = [];
    const now = Date.now();
    
    // Base price based on symbol
    let basePrice = 50000; // BTC
    if (symbol === 'ETH') basePrice = 3000;
    else if (symbol === 'SOL') basePrice = 100;
    else if (symbol === 'AVAX') basePrice = 25;
    
    let currentPrice = basePrice;
    
    // Time interval based on timeframe
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

    for (let i = count; i >= 0; i--) {
      const time = Math.floor((now - i * intervalMs) / 1000) as Time;
      
      // Generate realistic price movement
      const volatility = 0.02; // 2% volatility
      const change = (Math.random() - 0.5) * volatility;
      const open = currentPrice;
      const close = open * (1 + change);
      
      // Generate high and low
      const high = Math.max(open, close) * (1 + Math.random() * 0.01);
      const low = Math.min(open, close) * (1 - Math.random() * 0.01);
      
      data.push({
        time: time,
        open: open,
        high: high,
        low: low,
        close: close
      });
      
      currentPrice = close;
    }
    
    return data;
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
      const initialData = await fetchBinanceData(symbol, binanceInterval, 100);
      candlestickSeries.setData(initialData);

      // Fit content to show all data
      chart.timeScale().fitContent();

      // Set up real-time updates (using mock data for now)
      const updateInterval = setInterval(() => {
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

      // Handle resize
      const handleResize = () => {
        if (chartContainerRef.current && chartRef.current) {
          chartRef.current.applyOptions({
            width: chartContainerRef.current.clientWidth,
            height: height,
          });
        }
      };

      window.addEventListener('resize', handleResize);

      // Cleanup
      return () => {
        clearInterval(updateInterval);
        window.removeEventListener('resize', handleResize);
        if (chartRef.current) {
          chartRef.current.remove();
          chartRef.current = null;
        }
      };
    };

    initializeChart();
  }, [height, timeframe]);

  // Update data when symbol or timeframe changes
  useEffect(() => {
    const updateData = async () => {
      if (candlestickSeriesRef.current && chartRef.current) {
        console.log(`🔄 Updating chart data for ${symbol} (${timeframe})...`);
        
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
          
          console.log(`✅ Chart updated with new data for ${symbol}`, newData.length, 'candles');
        } catch (error) {
          console.error('Failed to update chart data:', error);
        }
      }
    };

    updateData();
  }, [symbol, timeframe]);

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
