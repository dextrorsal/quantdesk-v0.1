import React, { useEffect, useRef, useState } from 'react'
import { createChart, ColorType, IChartApi, ISeriesApi, CandlestickData, Time, LineData } from 'lightweight-charts'
import { useWebSocket } from '../providers/WebSocketProvider'

interface PriceChartProps {
  symbol?: string
  height?: number
}

const PriceChart: React.FC<PriceChartProps> = ({ symbol = 'BTC/USDT', height = 400 }) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartRef = useRef<IChartApi | null>(null)
  const candlestickSeriesRef = useRef<ISeriesApi<'Candlestick'> | null>(null)
  const volumeSeriesRef = useRef<ISeriesApi<'Histogram'> | null>(null)
  const maSeriesRef = useRef<ISeriesApi<'Line'> | null>(null)
  const [timeframe, setTimeframe] = useState('1h')
  const [showVolume, setShowVolume] = useState(true)
  const [showMA, setShowMA] = useState(true)
  const { marketData: wsMarketData, isConnected } = useWebSocket()
  const [marketData, setMarketData] = useState(wsMarketData.get(symbol))

  // Mock data generation functions (kept for fallback)
  // const generateMockData = (): CandlestickData[] => { ... }
  // const generateVolumeData = (): HistogramData[] => { ... }

  // Calculate moving average
  const calculateMA = (data: CandlestickData[], period: number): LineData[] => {
    const maData: LineData[] = []
    
    for (let i = period - 1; i < data.length; i++) {
      let sum = 0
      for (let j = 0; j < period; j++) {
        sum += data[i - j].close
      }
      const ma = sum / period
      
      maData.push({
        time: data[i].time,
        value: Number(ma.toFixed(2)),
      })
    }
    
    return maData
  }

  useEffect(() => {
    if (!chartContainerRef.current) return

    // Create chart
    const chart = createChart(chartContainerRef.current, {
      layout: {
        background: { type: ColorType.Solid, color: '#0f172a' },
        textColor: '#f8fafc',
      },
      width: chartContainerRef.current.clientWidth,
      height: height,
      grid: {
        vertLines: { color: '#334155' },
        horzLines: { color: '#334155' },
      },
      crosshair: {
        mode: 1,
      },
      rightPriceScale: {
        borderColor: '#334155',
        // textColor: '#f8fafc', // Not supported in this version
      },
      timeScale: {
        borderColor: '#334155',
        // textColor: '#f8fafc', // Not supported in this version
        timeVisible: true,
        secondsVisible: false,
      },
    })

    // Create candlestick series
    const candlestickSeries = chart.addCandlestickSeries({
      upColor: '#22c55e',
      downColor: '#ef4444',
      borderUpColor: '#22c55e',
      borderDownColor: '#ef4444',
      wickUpColor: '#22c55e',
      wickDownColor: '#ef4444',
    })

    // Create volume series
    const volumeSeries = chart.addHistogramSeries({
      color: '#3b82f6',
      priceFormat: {
        type: 'volume',
      },
      priceScaleId: 'volume',
    })

    // Create moving average series
    const maSeries = chart.addLineSeries({
      color: '#f59e0b',
      lineWidth: 2,
      title: 'MA(20)',
    })

    // Generate mock candlestick data for chart (WebSocket provides real-time price updates)
    const generateChartData = (): CandlestickData[] => {
      const data: CandlestickData[] = []
      let basePrice = marketData?.price || 43250
      const now = Math.floor(Date.now() / 1000)
      
      for (let i = 100; i >= 0; i--) {
        const time = (now - i * 3600) as Time // 1 hour intervals
        const open = basePrice + (Math.random() - 0.5) * (basePrice * 0.02)
        const close = open + (Math.random() - 0.5) * (basePrice * 0.03)
        const high = Math.max(open, close) + Math.random() * (basePrice * 0.01)
        const low = Math.min(open, close) - Math.random() * (basePrice * 0.01)
        
        data.push({
          time,
          open: Number(open.toFixed(2)),
          high: Number(high.toFixed(2)),
          low: Number(low.toFixed(2)),
          close: Number(close.toFixed(2)),
        })
        
        basePrice = close
      }
      
      return data
    }

    const chartData = generateChartData()
    const volumeData = chartData.map(item => ({
      time: item.time,
      value: Math.random() * 1000 + 100,
      color: item.close >= item.open ? '#22c55e' : '#ef4444',
    }))
    
    const maData = calculateMA(chartData, 20)
    
    candlestickSeries.setData(chartData)
    volumeSeries.setData(volumeData)
    maSeries.setData(maData)

    // Store references
    chartRef.current = chart
    candlestickSeriesRef.current = candlestickSeries
    volumeSeriesRef.current = volumeSeries
    maSeriesRef.current = maSeries

    // Handle resize
    const handleResize = () => {
      if (chartContainerRef.current && chartRef.current) {
        chartRef.current.applyOptions({
          width: chartContainerRef.current.clientWidth,
        })
      }
    }

    window.addEventListener('resize', handleResize)

    return () => {
      window.removeEventListener('resize', handleResize)
      if (chartRef.current) {
        chartRef.current.remove()
      }
    }
  }, [height])

  // Toggle indicators visibility
  useEffect(() => {
    if (volumeSeriesRef.current) {
      volumeSeriesRef.current.applyOptions({
        visible: showVolume,
      })
    }
  }, [showVolume])

  useEffect(() => {
    if (maSeriesRef.current) {
      maSeriesRef.current.applyOptions({
        visible: showMA,
      })
    }
  }, [showMA])

  // Subscribe to real-time market data updates via WebSocket
  useEffect(() => {
    const currentData = wsMarketData.get(symbol)
    if (currentData) {
      setMarketData(currentData)
    }
  }, [wsMarketData, symbol])

  // Update market data when WebSocket data changes
  useEffect(() => {
    const currentData = wsMarketData.get(symbol)
    if (currentData) {
      setMarketData(currentData)
    }
  }, [wsMarketData, symbol])

  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

  return (
    <div className="trading-card">
      {/* Chart Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-6">
          <div>
            <div className="flex items-center space-x-2">
              <h3 className="text-lg font-semibold text-white">{symbol}</h3>
              <div className={`w-2 h-2 rounded-full ${isConnected ? 'bg-green-400' : 'bg-red-400'}`} title={isConnected ? 'Connected' : 'Disconnected'}></div>
            </div>
            <div className="text-xs text-gray-400">Perpetual</div>
          </div>
          <div className="text-2xl font-bold text-white">
            ${marketData?.price.toFixed(2) || '43,250.50'}
          </div>
          <div className={`text-sm font-medium ${marketData?.change24h && marketData.change24h >= 0 ? 'text-green-400' : 'text-red-400'}`}>
            {marketData?.change24h && marketData.change24h >= 0 ? '+' : ''}{marketData?.change24h.toFixed(2) || '2.45'}%
          </div>
          <div className="flex space-x-4 text-xs text-gray-400">
            <div>O: ${marketData?.open24h.toFixed(2) || '43,200.00'}</div>
            <div>H: ${marketData?.high24h.toFixed(2) || '43,300.00'}</div>
            <div>L: ${marketData?.low24h.toFixed(2) || '43,100.00'}</div>
            <div>C: ${marketData?.price.toFixed(2) || '43,250.50'}</div>
          </div>
        </div>
        
        {/* Chart Controls */}
        <div className="flex items-center space-x-4">
          {/* Timeframe Selector */}
          <div className="flex space-x-1 bg-gray-800 rounded-lg p-1">
            {timeframes.map((tf) => (
              <button
                key={tf}
                onClick={() => setTimeframe(tf)}
                className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                  timeframe === tf
                    ? 'bg-blue-600 text-white'
                    : 'text-gray-400 hover:text-white hover:bg-gray-700'
                }`}
              >
                {tf}
              </button>
            ))}
          </div>

          {/* Indicators Toggle */}
          <div className="flex space-x-2">
            <button
              onClick={() => setShowVolume(!showVolume)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                showVolume
                  ? 'bg-green-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              Volume
            </button>
            <button
              onClick={() => setShowMA(!showMA)}
              className={`px-3 py-1 rounded text-sm font-medium transition-colors ${
                showMA
                  ? 'bg-yellow-600 text-white'
                  : 'bg-gray-800 text-gray-400 hover:text-white'
              }`}
            >
              MA(20)
            </button>
          </div>
        </div>
      </div>

      {/* Chart Container */}
      <div 
        ref={chartContainerRef} 
        className="w-full"
        style={{ height: `${height}px` }}
      />
    </div>
  )
}

export default PriceChart
