// Minimal adapter scaffolding for Lite Market and Charts
// Bridges QuantDesk state/services to vendored Quantify components.

export type ChartInterval = '1m' | '5m' | '15m' | '1h' | '4h' | '1d'

export interface ChartPoint {
  time: number
  open: number
  high: number
  low: number
  close: number
  volume?: number
}

// Temporary mock generator; replace with websocket-backed source
export function generateMockChartData(symbol: string, interval: string): ChartPoint[] {
  const now = Math.floor(Date.now() / 1000)
  const num = 200
  let price = 100 + Math.random() * 50
  const data: ChartPoint[] = []
  for (let i = num; i > 0; i--) {
    const t = now - i * 60
    const o = price
    const c = o + (Math.random() - 0.5) * 2
    const h = Math.max(o, c) + Math.random() * 1
    const l = Math.min(o, c) - Math.random() * 1
    price = c
    data.push({ time: t, open: o, high: h, low: l, close: c, volume: 100 + Math.random() * 500 })
  }
  return data
}

export function defaultSymbol(): string {
  return 'BTC'
}

export function defaultInterval(): ChartInterval {
  return '1h'
}


