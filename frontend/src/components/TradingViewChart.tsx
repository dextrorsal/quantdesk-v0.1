import React, { useEffect, useRef, useState, useCallback } from 'react'
import { ChartManager, SeriesData } from '../utils/ChartManager'
import { IndicatorManager, IndicatorInfo } from '../utils/IndicatorManager'
import { SeriesMarkerPosition, SeriesMarkerShape } from 'lightweight-charts'
import { 
  candlestickToIndicatorData,
  calculateSMA,
  calculateEMA,
  calculateRSI,
  calculateMACD,
  calculateBollingerBands,
  calculateVWAP,
  calculateStochastic
} from '../utils/indicators'
import { marketDataService } from '../services/marketDataService'

interface TradingViewChartProps {
  symbol: string
  timeframe: string
  height?: number
  width?: number
  onTimeframeChange?: (timeframe: string) => void
}

interface CandleData {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

const TradingViewChart: React.FC<TradingViewChartProps> = ({
  symbol,
  timeframe = '1h',
  height = 400,
  width = 800,
  onTimeframeChange
}) => {
  const chartContainerRef = useRef<HTMLDivElement>(null)
  const chartManagerRef = useRef<ChartManager | null>(null)
  const indicatorManagerRef = useRef<IndicatorManager | null>(null)
  const [isLoading, setIsLoading] = useState(true)
  const [activeIndicators, setActiveIndicators] = useState<string[]>([])
  const [showVolume, setShowVolume] = useState(true)
  const [, setCandleData] = useState<CandleData[]>([])
  const [showWatermark, setShowWatermark] = useState(true)
  const [showTooltips, setShowTooltips] = useState(true)
  const [showMarkers, setShowMarkers] = useState(true)

  // Load real market data from Pyth Network
  const loadRealMarketData = useCallback(async (symbol: string): Promise<CandleData[]> => {
    try {
      // Get price history from our backend (which gets it from Pyth Network)
      const priceHistory = await marketDataService.instance.getPriceHistory(symbol, 24)
      
      if (priceHistory && priceHistory.data.length > 0) {
        // Convert to TradingView format
        const candles = marketDataService.instance.convertToTradingViewFormat(priceHistory)
        
        // Convert to CandleData format
        return candles.map(candle => ({
          time: candle.time,
          open: candle.open,
          high: candle.high,
          low: candle.low,
          close: candle.close,
          volume: candle.volume
        }))
      }
      
      // Fallback to mock data if no real data available
      return generateMockData(symbol, 200)
    } catch (error) {
      console.error('Error loading real market data:', error)
      // Fallback to mock data
      return generateMockData(symbol, 200)
    }
  }, [])

  // Generate realistic mock data (fallback)
  const generateMockData = useCallback((symbol: string, count: number = 200): CandleData[] => {
    const basePrice = symbol === 'BTC' ? 214 : symbol === 'ETH' ? 3200 : symbol === 'SOL' ? 200 : 100
    const data: CandleData[] = []
    let currentPrice = basePrice
    
    for (let i = 0; i < count; i++) {
      const change = (Math.random() - 0.5) * 0.02 // ±1% change
      const open = currentPrice
      const close = open * (1 + change)
      const high = Math.max(open, close) * (1 + Math.random() * 0.01)
      const low = Math.min(open, close) * (1 - Math.random() * 0.01)
      const volume = Math.random() * 1000 + 100
      
      data.push({
        time: Math.floor(Date.now() / 1000) - (count - i) * 3600, // 1 hour intervals
        open,
        high,
        low,
        close,
        volume
      })
      
      currentPrice = close
    }
    
    return data
  }, [])






  // Initialize chart
  useEffect(() => {
    if (!chartContainerRef.current) return

    // Initialize chart manager
    chartManagerRef.current = new ChartManager({
      width: width,
      height: height,
    })

    // Initialize indicator manager
    indicatorManagerRef.current = new IndicatorManager(chartManagerRef.current)

    // Initialize chart
    const success = chartManagerRef.current.init(chartContainerRef.current)
    if (!success) {
      console.error('Failed to initialize chart')
      return
    }

    // Add watermark if enabled
    if (showWatermark) {
      addWatermark()
    }

    // Add tooltip if enabled
    if (showTooltips) {
      addTooltip()
    }

    // Add markers if enabled
    if (showMarkers) {
      setTimeout(() => addSeriesMarkers(), 1000) // Delay to ensure data is loaded
    }

    return () => {
      if (indicatorManagerRef.current) {
        indicatorManagerRef.current.destroy()
      }
      if (chartManagerRef.current) {
        chartManagerRef.current.destroy()
      }
    }
  }, [width, height])

  // Load and update data
  useEffect(() => {
    if (!chartManagerRef.current || !indicatorManagerRef.current) return

    setIsLoading(true)
    
    // Load real market data
    loadRealMarketData(symbol).then(data => {
      setCandleData(data)
      
      // Convert to SeriesData format
      const candlestickData: SeriesData[] = data.map(candle => ({
        time: candle.time,
        open: candle.open,
        high: candle.high,
        low: candle.low,
        close: candle.close,
      }))

      const volumeData: SeriesData[] = data.map(candle => ({
        time: candle.time,
        value: candle.volume,
        color: candle.close >= candle.open ? '#10b981' : '#ef4444',
      }))

      // Add candlestick series
      chartManagerRef.current!.addSeries('candlestick', 'candlestick', candlestickData, {
        upColor: '#10b981',
        downColor: '#ef4444',
        borderDownColor: '#ef4444',
        borderUpColor: '#10b981',
        wickDownColor: '#ef4444',
        wickUpColor: '#10b981',
      })

      // Add volume series
      if (showVolume) {
        chartManagerRef.current!.addSeries('volume', 'histogram', volumeData, {
          color: '#374151',
          priceFormat: {
            type: 'volume',
          },
          priceScaleId: 'volume',
          scaleMargins: {
            top: 0.8,
            bottom: 0,
          },
        })
      }

      // Add indicators
      updateIndicators(data)
      
      setIsLoading(false)
    }, 500)
  }, [symbol, timeframe, loadRealMarketData, showVolume])

  // Update indicators
  const updateIndicators = (data: CandleData[]) => {
    if (!chartManagerRef.current || !indicatorManagerRef.current) return

    // Remove existing indicators
    indicatorManagerRef.current.destroy()
    indicatorManagerRef.current = new IndicatorManager(chartManagerRef.current)

    // Convert to indicator data format
    const indicatorData = candlestickToIndicatorData(data)

    // Add active indicators
    activeIndicators.forEach(indicator => {
      switch (indicator) {
        case 'SMA20': {
          const sma20Data = calculateSMA(indicatorData, 20)
          const sma20Info: IndicatorInfo = {
            name: 'SMA 20',
            type: 'line',
            outputs: {
              sma: {
                type: 'line',
                plotOptions: {
                  color: '#3b82f6',
                  lineWidth: 2,
                  title: 'SMA 20',
                }
              }
            },
            parameters: { period: 20 }
          }
          indicatorManagerRef.current!.addIndicator(sma20Info, sma20Data)
          break
        }

        case 'EMA12': {
          const ema12Data = calculateEMA(indicatorData, 12)
          const ema12Info: IndicatorInfo = {
            name: 'EMA 12',
            type: 'line',
            outputs: {
              ema: {
                type: 'line',
                plotOptions: {
                  color: '#f59e0b',
                  lineWidth: 2,
                  title: 'EMA 12',
                }
              }
            },
            parameters: { period: 12 }
          }
          indicatorManagerRef.current!.addIndicator(ema12Info, ema12Data)
          break
        }

        case 'EMA26': {
          const ema26Data = calculateEMA(indicatorData, 26)
          const ema26Info: IndicatorInfo = {
            name: 'EMA 26',
            type: 'line',
            outputs: {
              ema: {
                type: 'line',
                plotOptions: {
                  color: '#8b5cf6',
                  lineWidth: 2,
                  title: 'EMA 26',
                }
              }
            },
            parameters: { period: 26 }
          }
          indicatorManagerRef.current!.addIndicator(ema26Info, ema26Data)
          break
        }

        case 'RSI': {
          const rsiData = calculateRSI(indicatorData)
          const rsiInfo: IndicatorInfo = {
            name: 'RSI',
            type: 'line',
            outputs: {
              rsi: {
                type: 'line',
                plotOptions: {
                  color: '#ef4444',
                  lineWidth: 2,
                  title: 'RSI',
                  priceScaleId: 'rsi',
                }
              }
            },
            parameters: { period: 14 }
          }
          indicatorManagerRef.current!.addIndicator(rsiInfo, rsiData, 1)
          break
        }

        case 'MACD': {
          const macdData = calculateMACD(indicatorData)
          
          const macdInfo: IndicatorInfo = {
            name: 'MACD',
            type: 'macd',
            outputs: {
              macd: {
                type: 'line',
                plotOptions: {
                  color: '#3b82f6',
                  lineWidth: 2,
                  title: 'MACD',
                }
              },
              signal: {
                type: 'line',
                plotOptions: {
                  color: '#ef4444',
                  lineWidth: 2,
                  title: 'Signal',
                }
              },
              histogram: {
                type: 'histogram',
                plotOptions: {
                  color: '#374151',
                  title: 'Histogram',
                }
              }
            },
            parameters: { fastPeriod: 12, slowPeriod: 26, signalPeriod: 9 }
          }
          indicatorManagerRef.current!.addIndicator(macdInfo, macdData, 1)
          break
        }

        case 'BB': {
          const bbData = calculateBollingerBands(indicatorData)
          const bbInfo: IndicatorInfo = {
            name: 'Bollinger Bands',
            type: 'bands',
            outputs: {
              upper: {
                type: 'line',
                plotOptions: {
                  color: '#6b7280',
                  lineWidth: 1,
                  title: 'BB Upper',
                }
              },
              middle: {
                type: 'line',
                plotOptions: {
                  color: '#9ca3af',
                  lineWidth: 1,
                  title: 'BB Middle',
                }
              },
              lower: {
                type: 'line',
                plotOptions: {
                  color: '#6b7280',
                  lineWidth: 1,
                  title: 'BB Lower',
                }
              }
            },
            parameters: { period: 20, stdDev: 2 }
          }
          indicatorManagerRef.current!.addIndicator(bbInfo, bbData)
          break
        }

        case 'VWAP': {
          const vwapData = calculateVWAP(indicatorData)
          const vwapInfo: IndicatorInfo = {
            name: 'VWAP',
            type: 'line',
            outputs: {
              vwap: {
                type: 'line',
                plotOptions: {
                  color: '#10b981',
                  lineWidth: 2,
                  title: 'VWAP',
                }
              }
            },
            parameters: {}
          }
          indicatorManagerRef.current!.addIndicator(vwapInfo, vwapData)
          break
        }

        case 'Stochastic': {
          const stochData = calculateStochastic(indicatorData)
          const stochInfo: IndicatorInfo = {
            name: 'Stochastic',
            type: 'stochastic',
            outputs: {
              k: {
                type: 'line',
                plotOptions: {
                  color: '#3b82f6',
                  lineWidth: 2,
                  title: '%K',
                }
              },
              d: {
                type: 'line',
                plotOptions: {
                  color: '#ef4444',
                  lineWidth: 2,
                  title: '%D',
                }
              }
            },
            parameters: { kPeriod: 14, dPeriod: 3 }
          }
          indicatorManagerRef.current!.addIndicator(stochInfo, stochData, 1)
          break
        }
      }
    })
  }

  const toggleIndicator = (indicator: string) => {
    setActiveIndicators(prev => 
      prev.includes(indicator) 
        ? prev.filter(i => i !== indicator)
        : [...prev, indicator]
    )
  }

  // Add watermark to chart
  const addWatermark = () => {
    if (!chartManagerRef.current || !chartContainerRef.current) return
    
    const chart = chartManagerRef.current.getChart()
    if (!chart) return

    // Create watermark element
    const watermark = document.createElement('div')
    watermark.style.cssText = `
      position: absolute;
      top: 50%;
      left: 50%;
      transform: translate(-50%, -50%);
      color: rgba(255, 255, 255, 0.1);
      font-size: 24px;
      font-weight: bold;
      pointer-events: none;
      z-index: 1;
      user-select: none;
    `
    watermark.textContent = 'QuantDesk DEX'
    
    chartContainerRef.current.appendChild(watermark)
  }

  // Add floating tooltip
  const addTooltip = () => {
    if (!chartManagerRef.current || !chartContainerRef.current) return
    
    const chart = chartManagerRef.current.getChart()
    if (!chart) return

    // Create tooltip element
    const tooltip = document.createElement('div')
    tooltip.id = 'chart-tooltip'
    tooltip.style.cssText = `
      position: absolute;
      background: rgba(0, 0, 0, 0.8);
      color: white;
      padding: 8px 12px;
      border-radius: 4px;
      font-size: 12px;
      pointer-events: none;
      z-index: 1000;
      display: none;
      border: 1px solid rgba(255, 255, 255, 0.2);
    `
    chartContainerRef.current.appendChild(tooltip)

    // Subscribe to crosshair move for tooltip updates
    chart.subscribeCrosshairMove((param) => {
      if (!param.point || !param.time) {
        tooltip.style.display = 'none'
        return
      }

      const seriesData = Array.from(param.seriesData.values())[0]
      if (!seriesData) {
        tooltip.style.display = 'none'
        return
      }

      const candle = seriesData as any
      tooltip.innerHTML = `
        <div><strong>${symbol}/USDT</strong></div>
        <div>O: ${candle.open?.toFixed(2) || 'N/A'}</div>
        <div>H: ${candle.high?.toFixed(2) || 'N/A'}</div>
        <div>L: ${candle.low?.toFixed(2) || 'N/A'}</div>
        <div>C: ${candle.close?.toFixed(2) || 'N/A'}</div>
        <div>V: ${candle.volume?.toFixed(0) || 'N/A'}</div>
      `
      
      tooltip.style.left = param.point.x + 'px'
      tooltip.style.top = param.point.y + 'px'
      tooltip.style.display = 'block'
    })
  }

  // Add series markers for trade signals
  const addSeriesMarkers = () => {
    if (!chartManagerRef.current) return
    
    const chart = chartManagerRef.current.getChart()
    if (!chart) return

    // Get the candlestick series
    const series = chartManagerRef.current.getSeries().get('candlestick')
    if (!series) return

    // Generate mock trade signals
    const markers = []
    const data = generateMockData(symbol, 200)
    
    for (let i = 20; i < data.length; i += 30) {
      const candle = data[i]
      const isBuy = Math.random() > 0.5
      
      markers.push({
        time: candle.time as any, // Convert to Time type
        position: (isBuy ? 'belowBar' : 'aboveBar') as SeriesMarkerPosition,
        color: isBuy ? '#10b981' : '#ef4444',
        shape: (isBuy ? 'arrowUp' : 'arrowDown') as SeriesMarkerShape,
        text: isBuy ? 'BUY' : 'SELL',
        size: 1
      })
    }

    series.series.setMarkers(markers)
  }

  const availableIndicators = [
    { id: 'SMA20', name: 'SMA 20', description: 'Simple Moving Average (20 periods)' },
    { id: 'EMA12', name: 'EMA 12', description: 'Exponential Moving Average (12 periods)' },
    { id: 'EMA26', name: 'EMA 26', description: 'Exponential Moving Average (26 periods)' },
    { id: 'RSI', name: 'RSI', description: 'Relative Strength Index' },
    { id: 'MACD', name: 'MACD', description: 'Moving Average Convergence Divergence' },
    { id: 'BB', name: 'Bollinger Bands', description: 'Bollinger Bands (20 periods, 2 std dev)' },
    { id: 'VWAP', name: 'VWAP', description: 'Volume Weighted Average Price' },
    { id: 'Stochastic', name: 'Stochastic', description: 'Stochastic Oscillator' },
  ]

  const timeframes = ['1m', '5m', '15m', '1h', '4h', '1d']

  if (isLoading) {
    return (
      <div style={{ 
        height: `${height}px`, 
        display: 'flex', 
        alignItems: 'center', 
        justifyContent: 'center',
        backgroundColor: '#000000',
        color: '#ffffff'
      }}>
        <div style={{ textAlign: 'center' }}>
          <div style={{ fontSize: '24px', marginBottom: '8px' }}>⏳</div>
          <div style={{ fontSize: '14px', color: '#9ca3af' }}>Loading TradingView chart...</div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ position: 'relative', backgroundColor: '#000000' }}>
      {/* Chart Controls */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        zIndex: 10,
        display: 'flex',
        gap: '8px',
        flexWrap: 'wrap'
      }}>
        {/* Timeframe Buttons */}
        <div style={{ display: 'flex', gap: '4px' }}>
          {timeframes.map((tf) => (
            <button
              key={tf}
              onClick={() => onTimeframeChange?.(tf)}
              style={{
                padding: '4px 8px',
                fontSize: '12px',
                backgroundColor: tf === timeframe ? '#10b981' : '#1f2937',
                color: '#ffffff',
                border: 'none',
                borderRadius: '4px',
                cursor: 'pointer',
                transition: 'background-color 0.2s'
              }}
            >
              {tf}
            </button>
          ))}
        </div>

        {/* Volume Toggle */}
        <button
          onClick={() => setShowVolume(!showVolume)}
          style={{
            padding: '4px 8px',
            fontSize: '12px',
            backgroundColor: showVolume ? '#10b981' : '#1f2937',
            color: '#ffffff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            transition: 'background-color 0.2s'
          }}
        >
          Volume
        </button>

        {/* Watermark Toggle */}
        <button
          onClick={() => setShowWatermark(!showWatermark)}
          style={{
            padding: '4px 8px',
            fontSize: '12px',
            backgroundColor: showWatermark ? '#3b82f6' : '#1f2937',
            color: '#ffffff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            transition: 'background-color 0.2s'
          }}
        >
          Watermark
        </button>

        {/* Tooltip Toggle */}
        <button
          onClick={() => setShowTooltips(!showTooltips)}
          style={{
            padding: '4px 8px',
            fontSize: '12px',
            backgroundColor: showTooltips ? '#8b5cf6' : '#1f2937',
            color: '#ffffff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            transition: 'background-color 0.2s'
          }}
        >
          Tooltips
        </button>

        {/* Markers Toggle */}
        <button
          onClick={() => setShowMarkers(!showMarkers)}
          style={{
            padding: '4px 8px',
            fontSize: '12px',
            backgroundColor: showMarkers ? '#f59e0b' : '#1f2937',
            color: '#ffffff',
            border: 'none',
            borderRadius: '4px',
            cursor: 'pointer',
            transition: 'background-color 0.2s'
          }}
        >
          Signals
        </button>
      </div>

      {/* Indicator Controls */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        zIndex: 10,
        display: 'flex',
        gap: '4px',
        flexWrap: 'wrap'
      }}>
        {availableIndicators.map((indicator) => (
          <button
            key={indicator.id}
            onClick={() => toggleIndicator(indicator.id)}
            style={{
              padding: '4px 8px',
              fontSize: '12px',
              backgroundColor: activeIndicators.includes(indicator.id) ? '#3b82f6' : '#1f2937',
              color: '#ffffff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer',
              transition: 'background-color 0.2s'
            }}
            title={indicator.description}
          >
            {indicator.name}
          </button>
        ))}
      </div>

      {/* Chart Info */}
      <div style={{
        position: 'absolute',
        bottom: '10px',
        left: '10px',
        color: '#9ca3af',
        fontSize: '12px',
        zIndex: 10
      }}>
        {symbol}/USDT • {timeframe} • TradingView Lightweight Charts
      </div>

      {/* Chart Container */}
      <div
        ref={chartContainerRef}
        style={{
          width: '100%',
          height: `${height}px`,
          backgroundColor: '#000000'
        }}
      />
    </div>
  )
}

export default TradingViewChart
