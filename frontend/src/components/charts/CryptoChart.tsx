// Professional Crypto Chart with WebSocket real-time updates
// Based on https://github.com/Hamz-06/crypto-position-calculator
// Adapted for QuantDesk terminal aesthetic

import React, { useState, useEffect, useRef } from 'react';
import { createChart, IChartApi, ISeriesApi, ColorType, CrosshairMode } from 'lightweight-charts';

interface CandleData {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

interface Props {
  symbol: string;
  interval?: string;
  height?: number;
  onPriceClick?: (price: number, time: number) => void;
  selectedEntryPrice?: number;
  selectedStopLoss?: number;
  selectedTakeProfit?: number;
}

const CryptoChart: React.FC<Props> = ({ 
  symbol = 'BTCUSDT', 
  interval = '1h',
  height = 400,
  onPriceClick,
  selectedEntryPrice,
  selectedStopLoss,
  selectedTakeProfit
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null);
  const chartRef = useRef<IChartApi | null>(null);
  const seriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null);
  const entryPriceLineRef = useRef<any>(null);
  const stopLossLineRef = useRef<any>(null);
  const takeProfitLineRef = useRef<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const pollIntervalRef = useRef<NodeJS.Timeout | null>(null);

  // Chart initialization - Terminal styling (RUNS ONCE)
  useEffect(() => {
    if (!chartContainerRef.current) {
      console.log('❌ No chart container');
      return;
    }
    
    if (chartRef.current) {
      console.log('⚠️ Chart already initialized, skipping');
      return;
    }
    
    console.log('🎨 Initializing chart...');

    const chartOptions = {
      layout: {
        background: { type: ColorType.Solid, color: '#000' }, // Terminal black
        textColor: '#fff',
        fontFamily: 'JetBrains Mono, monospace',
      },
      grid: {
        vertLines: { color: '#333' }, // Terminal grid
        horzLines: { color: '#333' },
      },
      crosshair: {
        mode: CrosshairMode.Normal,
      },
      rightPriceScale: {
        borderColor: '#333',
      },
      timeScale: {
        borderColor: '#333',
        timeVisible: true,
        secondsVisible: false,
      },
      width: chartContainerRef.current.clientWidth,
      height: height, // Will be updated by ResizeObserver to fill container
      autoSize: true, // Let chart auto-resize to container
    };

    // Create chart
    chartRef.current = createChart(chartContainerRef.current, chartOptions);
    chartRef.current.timeScale().fitContent();
    
    // Add candlestick series - theme-aware colors
    const root = document.documentElement;
    const styles = getComputedStyle(root);
    const up = styles.getPropertyValue('--qd-candle-up').trim() || styles.getPropertyValue('--success-500').trim() || '#52c41a';
    const down = styles.getPropertyValue('--qd-candle-down').trim() || styles.getPropertyValue('--danger-500').trim() || '#ff4d4f';
    seriesRef.current = chartRef.current.addCandlestickSeries({
      upColor: up,
      downColor: down,
      borderUpColor: up,
      borderDownColor: down,
      wickUpColor: up,
      wickDownColor: down,
    });
    console.log('✅ Chart initialized with series:', !!seriesRef.current);

    // Handle resize
    const resizeObserver = new ResizeObserver((entries) => {
      if (chartContainerRef.current && chartRef.current) {
        try {
          const { width, height } = entries[0].contentRect;
          console.log('📐 Resizing chart:', width, 'x', height);
          chartRef.current.applyOptions({
            width: width,
            height: height,
          });
        } catch (e) {
          // Ignore errors after disposal
        }
      }
    });

    resizeObserver.observe(chartContainerRef.current);

    return () => {
      resizeObserver.disconnect();
      if (chartRef.current) {
        try {
          chartRef.current.remove();
          chartRef.current = null;
          seriesRef.current = null;
        } catch (e) {
          // Ignore errors during cleanup
        }
      }
    };
  }, []); // RUN ONCE - only create chart once

  // Update price lines (entry, stop loss, take profit)
  useEffect(() => {
    if (!seriesRef.current) return;

    // Remove old lines
    if (entryPriceLineRef.current) seriesRef.current.removePriceLine(entryPriceLineRef.current);
    if (stopLossLineRef.current) seriesRef.current.removePriceLine(stopLossLineRef.current);
    if (takeProfitLineRef.current) seriesRef.current.removePriceLine(takeProfitLineRef.current);

    // Add new lines if values provided
    if (selectedEntryPrice) {
      entryPriceLineRef.current = seriesRef.current.createPriceLine({
        price: selectedEntryPrice,
        color: '#3b82f6', // Blue for entry
        lineWidth: 2,
        lineStyle: 0, // Solid
        axisLabelVisible: true,
        title: 'Entry'
      });
    }

    if (selectedStopLoss) {
      stopLossLineRef.current = seriesRef.current.createPriceLine({
        price: selectedStopLoss,
        color: '#ff4d4f', // Red for stop loss
        lineWidth: 2,
        lineStyle: 0,
        axisLabelVisible: true,
        title: 'Stop Loss'
      });
    }

    if (selectedTakeProfit) {
      takeProfitLineRef.current = seriesRef.current.createPriceLine({
        price: selectedTakeProfit,
        color: '#52c41a', // Green for take profit
        lineWidth: 2,
        lineStyle: 0,
        axisLabelVisible: true,
        title: 'Take Profit'
      });
    }
  }, [selectedEntryPrice, selectedStopLoss, selectedTakeProfit]);

  // Fetch initial historical data (RUNS ONCE when symbol/interval changes)
  useEffect(() => {
    if (!seriesRef.current) {
      console.log('⏳ Waiting for chart to initialize...');
      return;
    }

    const binanceSymbol = symbol.endsWith('USDT') ? symbol : `${symbol}USDT`;
    
    const fetchInitialData = async () => {
      try {
        setLoading(true);
        console.log('📊 Fetching initial data for', binanceSymbol);
        
        const response = await fetch(
          `/api/oracle/binance/${binanceSymbol}?interval=${interval}&limit=500`
        );
        
        if (!response.ok) throw new Error('Failed to fetch candles');
        
        const data = await response.json();
        
        const candles: CandleData[] = data.map((d: any[]) => ({
          time: Math.floor(d[0] / 1000),
          open: parseFloat(d[1]),
          high: parseFloat(d[2]),
          low: parseFloat(d[3]),
          close: parseFloat(d[4]),
        }));

        // ✅ Use setData() ONLY for initial load
        if (seriesRef.current) {
          seriesRef.current.setData(candles);
          console.log('✅ Initial', candles.length, 'candles loaded');
        }
        setLoading(false);
      } catch (err) {
        console.error('❌ Initial data fetch error:', err);
        setError(`Failed to load ${symbol} data`);
        setLoading(false);
      }
    };

    fetchInitialData();
  }, [symbol, interval]);

  // WebSocket for real-time updates (NO FLICKER!)
  useEffect(() => {
    if (!seriesRef.current) return;

    const binanceSymbol = symbol.endsWith('USDT') ? symbol : `${symbol}USDT`;
    
    console.log('🔌 Connecting WebSocket for', binanceSymbol, interval);
    
    try {
      const ws = new WebSocket(`wss://stream.binance.com:9443/ws/${binanceSymbol.toLowerCase()}@kline_${interval}`);
      
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data);
          const k = data.k;
          
          if (k && k.x) { // x=true means candle is closed
            // ✅ Use update() for incremental updates - NO FLICKER!
            const candle: CandleData = {
              time: k.t / 1000, // Convert to seconds
              open: parseFloat(k.o),
              high: parseFloat(k.h),
              low: parseFloat(k.l),
              close: parseFloat(k.c),
            };
            
            if (seriesRef.current) {
              seriesRef.current.update(candle);
            }
          }
        } catch (err) {
          console.warn('WebSocket parse error:', err);
        }
      };
      
      ws.onerror = () => {
        console.warn('⚠️ WebSocket error for', binanceSymbol, '- chart will use initial data only');
      };
      
      wsRef.current = ws;
    } catch (err) {
      console.warn('WebSocket connection failed:', err);
    }

    return () => {
      if (wsRef.current) {
        wsRef.current.close();
        wsRef.current = null;
      }
    };
  }, [symbol, interval]);

  return (
    <div style={{ width: '100%', height: '100%', backgroundColor: '#000' }}>
      {error && (
        <div style={{
          padding: '4px 8px',
          fontSize: '11px',
          fontFamily: 'JetBrains Mono, monospace',
          color: '#faad14',
          backgroundColor: '#1a1a1a',
          borderBottom: '1px solid #333',
          textAlign: 'center'
        }}>
          ⚠️ {error}
        </div>
      )}
      {loading && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100%',
          color: '#999',
          fontSize: '12px',
          fontFamily: 'JetBrains Mono, monospace',
          gap: '8px'
        }}>
          <div style={{
            width: '24px',
            height: '24px',
            border: '2px solid #333',
            borderTop: '2px solid #3b82f6',
            borderRadius: '50%',
            animation: 'spin 1s linear infinite'
          }} />
          <span>Loading {symbol} chart...</span>
        </div>
      )}
      <div 
        ref={chartContainerRef} 
        style={{ 
          width: '100%', 
          height: '100%', // Fill parent container
          backgroundColor: '#000'
        }}
      />
      <style>{`
        @keyframes spin {
          0% { transform: rotate(0deg); }
          100% { transform: rotate(360deg); }
        }
      `}</style>
    </div>
  );
};

export default React.memo(CryptoChart);
