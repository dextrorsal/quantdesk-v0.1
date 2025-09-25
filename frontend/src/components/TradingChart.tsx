import React, { useEffect, useRef, useState } from 'react'

interface TradingChartProps {
  symbol: string
  timeframe: string
  height?: number
}

interface CandleData {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume: number
}

const TradingChart: React.FC<TradingChartProps> = ({ 
  symbol, 
  timeframe = '1h', 
  height = 400 
}) => {
  const canvasRef = useRef<HTMLCanvasElement>(null)
  const [candleData, setCandleData] = useState<CandleData[]>([])
  const [isLoading, setIsLoading] = useState(true)

  // Generate mock candle data with realistic prices
  const generateMockData = (symbol: string, count: number = 100): CandleData[] => {
    // Use realistic current prices
    const basePrice = symbol === 'BTC' ? 214 : symbol === 'ETH' ? 3200 : symbol === 'SOL' ? 200 : 100
    const data: CandleData[] = []
    let currentPrice = basePrice
    
    for (let i = 0; i < count; i++) {
      const change = (Math.random() - 0.5) * 0.02 // ±1% change
      const open = currentPrice
      const close = open * (1 + change)
      const high = Math.max(open, close) * (1 + Math.random() * 0.01)
      const low = Math.min(open, close) * (1 - Math.random() * 0.01)
      const volume = Math.random() * 1000
      
      data.push({
        time: Date.now() - (count - i) * 60 * 60 * 1000, // 1 hour intervals
        open,
        high,
        low,
        close,
        volume
      })
      
      currentPrice = close
    }
    
    return data
  }

  useEffect(() => {
    setIsLoading(true)
    // Simulate data loading
    setTimeout(() => {
      const data = generateMockData(symbol)
      setCandleData(data)
      setIsLoading(false)
    }, 500)
  }, [symbol, timeframe])

  useEffect(() => {
    if (!canvasRef.current || candleData.length === 0) return

    const canvas = canvasRef.current
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const width = canvas.width
    const height = canvas.height

    // Clear canvas
    ctx.fillStyle = '#000000'
    ctx.fillRect(0, 0, width, height)

    // Calculate price range
    const prices = candleData.flatMap(candle => [candle.high, candle.low])
    const minPrice = Math.min(...prices)
    const maxPrice = Math.max(...prices)
    const priceRange = maxPrice - minPrice
    const padding = priceRange * 0.1

    // Calculate candle dimensions
    const candleWidth = Math.max(1, width / candleData.length - 2)
    const candleSpacing = width / candleData.length

    // Draw grid lines
    ctx.strokeStyle = '#1f2937'
    ctx.lineWidth = 1
    for (let i = 0; i <= 5; i++) {
      const y = (height / 5) * i
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
    }

    // Draw candles
    candleData.forEach((candle, index) => {
      const x = index * candleSpacing + candleSpacing / 2
      const openY = height - ((candle.open - minPrice + padding) / (priceRange + padding * 2)) * height
      const closeY = height - ((candle.close - minPrice + padding) / (priceRange + padding * 2)) * height
      const highY = height - ((candle.high - minPrice + padding) / (priceRange + padding * 2)) * height
      const lowY = height - ((candle.low - minPrice + padding) / (priceRange + padding * 2)) * height

      // Determine candle color
      const isGreen = candle.close > candle.open
      ctx.fillStyle = isGreen ? '#10b981' : '#ef4444'
      ctx.strokeStyle = isGreen ? '#10b981' : '#ef4444'

      // Draw wick
      ctx.beginPath()
      ctx.moveTo(x, highY)
      ctx.lineTo(x, lowY)
      ctx.stroke()

      // Draw body
      const bodyTop = Math.min(openY, closeY)
      const bodyHeight = Math.abs(closeY - openY)
      ctx.fillRect(x - candleWidth / 2, bodyTop, candleWidth, Math.max(bodyHeight, 1))
    })

    // Draw price labels
    ctx.fillStyle = '#9ca3af'
    ctx.font = '12px monospace'
    ctx.textAlign = 'right'
    for (let i = 0; i <= 5; i++) {
      const price = maxPrice - (priceRange / 5) * i
      const y = (height / 5) * i + 4
      // Format price based on magnitude
      const formattedPrice = price > 1000 ? price.toFixed(0) : price.toFixed(2)
      ctx.fillText(formattedPrice, width - 10, y)
    }

    // Draw current price line
    if (candleData.length > 0) {
      const lastPrice = candleData[candleData.length - 1].close
      const y = height - ((lastPrice - minPrice + padding) / (priceRange + padding * 2)) * height
      
      ctx.strokeStyle = '#ffffff'
      ctx.lineWidth = 2
      ctx.setLineDash([5, 5])
      ctx.beginPath()
      ctx.moveTo(0, y)
      ctx.lineTo(width, y)
      ctx.stroke()
      ctx.setLineDash([])

      // Price label
      ctx.fillStyle = '#ffffff'
      ctx.font = 'bold 14px monospace'
      ctx.textAlign = 'left'
      const formattedCurrentPrice = lastPrice > 1000 ? lastPrice.toFixed(0) : lastPrice.toFixed(2)
      ctx.fillText(`$${formattedCurrentPrice}`, 10, y - 5)
    }

  }, [candleData])

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
          <div style={{ fontSize: '14px', color: '#9ca3af' }}>Loading chart...</div>
        </div>
      </div>
    )
  }

  return (
    <div style={{ position: 'relative', height: `${height}px`, backgroundColor: '#000000' }}>
      <canvas
        ref={canvasRef}
        width={800}
        height={height}
        style={{ 
          width: '100%', 
          height: '100%',
          backgroundColor: '#000000'
        }}
      />
      
      {/* Chart Controls */}
      <div style={{
        position: 'absolute',
        top: '10px',
        left: '10px',
        display: 'flex',
        gap: '8px'
      }}>
        {['1m', '5m', '15m', '1h', '4h', '1d'].map((tf) => (
          <button
            key={tf}
            onClick={() => {/* Handle timeframe change */}}
            style={{
              padding: '4px 8px',
              fontSize: '12px',
              backgroundColor: tf === timeframe ? '#10b981' : '#1f2937',
              color: '#ffffff',
              border: 'none',
              borderRadius: '4px',
              cursor: 'pointer'
            }}
          >
            {tf}
          </button>
        ))}
      </div>

      {/* Chart Info */}
      <div style={{
        position: 'absolute',
        top: '10px',
        right: '10px',
        color: '#9ca3af',
        fontSize: '12px'
      }}>
        {symbol}/USDT • {timeframe}
      </div>
    </div>
  )
}

export default TradingChart
