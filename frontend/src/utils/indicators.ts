import { SeriesData } from './ChartManager'
import { IndicatorData } from './IndicatorManager'

// Convert candlestick data to indicator format
export const candlestickToIndicatorData = (data: SeriesData[]): IndicatorData[] => {
  return data.map(candle => ({
    time: candle.time,
    open: candle.open || 0,
    high: candle.high || 0,
    low: candle.low || 0,
    close: candle.close || 0,
    volume: candle.value || 0
  }))
}

// Simple Moving Average
export const calculateSMA = (data: IndicatorData[], period: number): IndicatorData[] => {
  const result: IndicatorData[] = []
  
  for (let i = period - 1; i < data.length; i++) {
    const sum = data.slice(i - period + 1, i + 1).reduce((acc, candle) => acc + candle.close, 0)
    result.push({
      time: data[i].time,
      sma: sum / period
    })
  }
  
  return result
}

// Exponential Moving Average
export const calculateEMA = (data: IndicatorData[], period: number): IndicatorData[] => {
  const result: IndicatorData[] = []
  const multiplier = 2 / (period + 1)
  
  if (data.length === 0) return result
  
  result.push({ time: data[0].time, ema: data[0].close })
  
  for (let i = 1; i < data.length; i++) {
    const ema = (data[i].close - result[i - 1].ema) * multiplier + result[i - 1].ema
    result.push({ time: data[i].time, ema })
  }
  
  return result
}

// Relative Strength Index
export const calculateRSI = (data: IndicatorData[], period: number = 14): IndicatorData[] => {
  const result: IndicatorData[] = []
  const gains: number[] = []
  const losses: number[] = []
  
  for (let i = 1; i < data.length; i++) {
    const change = data[i].close - data[i - 1].close
    gains.push(change > 0 ? change : 0)
    losses.push(change < 0 ? -change : 0)
  }
  
  for (let i = period - 1; i < gains.length; i++) {
    const avgGain = gains.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period
    const avgLoss = losses.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period
    
    const rs = avgGain / avgLoss
    const rsi = 100 - (100 / (1 + rs))
    
    result.push({
      time: data[i + 1].time,
      rsi
    })
  }
  
  return result
}

// MACD (Moving Average Convergence Divergence)
export const calculateMACD = (data: IndicatorData[], fastPeriod: number = 12, slowPeriod: number = 26, signalPeriod: number = 9): IndicatorData[] => {
  const ema12 = calculateEMA(data, fastPeriod)
  const ema26 = calculateEMA(data, slowPeriod)
  
  const macd: IndicatorData[] = []
  
  // Calculate MACD line
  for (let i = 0; i < Math.min(ema12.length, ema26.length); i++) {
    macd.push({
      time: ema12[i].time,
      macd: ema12[i].ema - ema26[i].ema
    })
  }
  
  // Calculate Signal line (9-period EMA of MACD)
  const signalData = calculateEMA(macd.map(m => ({
    time: m.time,
    open: m.macd,
    high: m.macd,
    low: m.macd,
    close: m.macd
  })), signalPeriod)
  
  // Calculate Histogram
  const result: IndicatorData[] = []
  for (let i = 0; i < Math.min(macd.length, signalData.length); i++) {
    result.push({
      time: macd[i].time,
      macd: macd[i].macd,
      signal: signalData[i].ema,
      histogram: macd[i].macd - signalData[i].ema
    })
  }
  
  return result
}

// Bollinger Bands
export const calculateBollingerBands = (data: IndicatorData[], period: number = 20, stdDev: number = 2): IndicatorData[] => {
  const result: IndicatorData[] = []
  
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1)
    const mean = slice.reduce((sum, candle) => sum + candle.close, 0) / period
    const variance = slice.reduce((sum, candle) => sum + Math.pow(candle.close - mean, 2), 0) / period
    const standardDeviation = Math.sqrt(variance)
    
    const time = data[i].time
    result.push({
      time,
      upper: mean + (stdDev * standardDeviation),
      middle: mean,
      lower: mean - (stdDev * standardDeviation)
    })
  }
  
  return result
}

// Volume Weighted Average Price
export const calculateVWAP = (data: IndicatorData[]): IndicatorData[] => {
  const result: IndicatorData[] = []
  let cumulativeVolume = 0
  let cumulativeVolumePrice = 0
  
  for (const candle of data) {
    const typicalPrice = (candle.high + candle.low + candle.close) / 3
    const volumePrice = typicalPrice * candle.volume
    
    cumulativeVolume += candle.volume
    cumulativeVolumePrice += volumePrice
    
    result.push({
      time: candle.time,
      vwap: cumulativeVolumePrice / cumulativeVolume
    })
  }
  
  return result
}

// Stochastic Oscillator
export const calculateStochastic = (data: IndicatorData[], kPeriod: number = 14, dPeriod: number = 3): IndicatorData[] => {
  const result: IndicatorData[] = []
  
  for (let i = kPeriod - 1; i < data.length; i++) {
    const slice = data.slice(i - kPeriod + 1, i + 1)
    const highestHigh = Math.max(...slice.map(c => c.high))
    const lowestLow = Math.min(...slice.map(c => c.low))
    const currentClose = data[i].close
    
    const k = ((currentClose - lowestLow) / (highestHigh - lowestLow)) * 100
    
    result.push({
      time: data[i].time,
      k
    })
  }
  
  // Calculate %D (3-period SMA of %K)
  const dData = calculateSMA(result.map(r => ({
    time: r.time,
    open: r.k,
    high: r.k,
    low: r.k,
    close: r.k
  })), dPeriod)
  
  // Combine K and D
  const finalResult: IndicatorData[] = []
  for (let i = 0; i < Math.min(result.length, dData.length); i++) {
    finalResult.push({
      time: result[i].time,
      k: result[i].k,
      d: dData[i].sma
    })
  }
  
  return finalResult
}

// Average True Range
export const calculateATR = (data: IndicatorData[], period: number = 14): IndicatorData[] => {
  const result: IndicatorData[] = []
  const trueRanges: number[] = []
  
  for (let i = 1; i < data.length; i++) {
    const high = data[i].high
    const low = data[i].low
    const prevClose = data[i - 1].close
    
    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    )
    
    trueRanges.push(tr)
  }
  
  for (let i = period - 1; i < trueRanges.length; i++) {
    const atr = trueRanges.slice(i - period + 1, i + 1).reduce((a, b) => a + b) / period
    result.push({
      time: data[i + 1].time,
      atr
    })
  }
  
  return result
}

// Commodity Channel Index
export const calculateCCI = (data: IndicatorData[], period: number = 20): IndicatorData[] => {
  const result: IndicatorData[] = []
  
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1)
    const typicalPrices = slice.map(c => (c.high + c.low + c.close) / 3)
    const sma = typicalPrices.reduce((a, b) => a + b) / period
    const meanDeviation = typicalPrices.reduce((sum, tp) => sum + Math.abs(tp - sma), 0) / period
    
    const currentTP = (data[i].high + data[i].low + data[i].close) / 3
    const cci = (currentTP - sma) / (0.015 * meanDeviation)
    
    result.push({
      time: data[i].time,
      cci
    })
  }
  
  return result
}

// Williams %R
export const calculateWilliamsR = (data: IndicatorData[], period: number = 14): IndicatorData[] => {
  const result: IndicatorData[] = []
  
  for (let i = period - 1; i < data.length; i++) {
    const slice = data.slice(i - period + 1, i + 1)
    const highestHigh = Math.max(...slice.map(c => c.high))
    const lowestLow = Math.min(...slice.map(c => c.low))
    const currentClose = data[i].close
    
    const williamsR = ((highestHigh - currentClose) / (highestHigh - lowestLow)) * -100
    
    result.push({
      time: data[i].time,
      williamsR
    })
  }
  
  return result
}
