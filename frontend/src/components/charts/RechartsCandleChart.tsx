import React, { useEffect, useMemo, useState } from 'react'
import { ComposedChart, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Customized } from 'recharts'

interface PricePoint { price: number; timestamp: string }

interface Candle { time: string; open: number; high: number; low: number; close: number }

interface Props { symbol: string; hours?: number; intervalMinutes?: number; height?: number }

const formatTime = (iso: string) => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

function bucketize(data: PricePoint[], intervalMinutes: number): Candle[] {
  if (!data.length) return []
  const ms = intervalMinutes * 60 * 1000
  const buckets = new Map<number, PricePoint[]>()
  for (const p of data) {
    const t = new Date(p.timestamp).getTime()
    const bucket = Math.floor(t / ms) * ms
    if (!buckets.has(bucket)) buckets.set(bucket, [])
    buckets.get(bucket)!.push(p)
  }
  const candles: Candle[] = []
  const sorted = [...buckets.entries()].sort((a, b) => a[0] - b[0])
  for (const [bucket, points] of sorted) {
    const sortedPts = points.sort((a, b) => new Date(a.timestamp).getTime() - new Date(b.timestamp).getTime())
    const open = sortedPts[0].price
    const close = sortedPts[sortedPts.length - 1].price
    let high = -Infinity, low = Infinity
    for (const pt of sortedPts) { if (pt.price > high) high = pt.price; if (pt.price < low) low = pt.price }
    candles.push({ time: new Date(bucket).toISOString(), open, high, low, close })
  }
  return candles
}

const CandleRenderer: React.FC<{ xAxisMap: any; yAxisMap: any; offset: any; data: Candle[] } > = ({ xAxisMap, yAxisMap, offset, data }) => {
  const x = Object.values(xAxisMap)[0] as any
  const y = Object.values(yAxisMap)[0] as any
  const scaleX = x.scale, scaleY = y.scale
  const candleWidth = Math.max(4, (scaleX.range()[1] - scaleX.range()[0]) / Math.max(30, data.length) * 0.7)
  return (
    <g>
      {data.map((d, i) => {
        const cx = scaleX(d.time)
        const yHigh = scaleY(d.high)
        const yLow = scaleY(d.low)
        const yOpen = scaleY(d.open)
        const yClose = scaleY(d.close)
        const up = d.close >= d.open
        const bodyTop = up ? yClose : yOpen
        const bodyBottom = up ? yOpen : yClose
        const color = up ? '#22c55e' : '#ef4444'
        return (
          <g key={i}>
            {/* wick */}
            <line x1={cx} x2={cx} y1={yHigh} y2={yLow} stroke={color} strokeWidth={1} />
            {/* body */}
            <rect x={cx - candleWidth / 2} y={bodyTop} width={candleWidth} height={Math.max(1, bodyBottom - bodyTop)} fill={color} />
          </g>
        )
      })}
    </g>
  )
}

const RechartsCandleChart: React.FC<Props> = ({ symbol, hours = 24, intervalMinutes = 15, height = 400 }) => {
  const [data, setData] = useState<PricePoint[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let mounted = true
    const fetchHistory = async () => {
      try {
        setLoading(true)
        setError(null)
        const res = await fetch(`/api/markets/${symbol}-PERP/price-history?hours=${hours}`)
        const json = await res.json()
        if (!mounted) return
        const arr: PricePoint[] = json?.success && Array.isArray(json.data) ? json.data : Array.isArray(json) ? json : []
        setData(arr)
      } catch (e) {
        if (mounted) setError(e instanceof Error ? e.message : 'Failed to load')
      } finally {
        if (mounted) setLoading(false)
      }
    }
    fetchHistory()
    return () => { mounted = false }
  }, [symbol, hours])

  const candles = useMemo(() => bucketize(data, intervalMinutes), [data, intervalMinutes])

  if (loading) return <div className="flex items-center justify-center h-full text-gray-400">Loading chart…</div>
  if (error) return <div className="flex items-center justify-center h-full text-red-400">{error}</div>

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height={height}>
        <ComposedChart data={candles} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
          <XAxis dataKey="time" tickFormatter={formatTime} stroke="#9ca3af" tick={{ fontSize: 12 }} />
          <YAxis domain={['auto', 'auto']} stroke="#9ca3af" tick={{ fontSize: 12 }} width={70} />
          <Tooltip
            contentStyle={{ backgroundColor: '#0b0f19', border: '1px solid #1f2937' }}
            labelStyle={{ color: '#9ca3af' }}
            formatter={(value: any, name: any) => [Number(value).toLocaleString(), name]}
            labelFormatter={(label: string) => new Date(label).toLocaleString()}
          />
          <Customized component={<CandleRenderer data={candles} />} />
        </ComposedChart>
      </ResponsiveContainer>
    </div>
  )
}

export default RechartsCandleChart
