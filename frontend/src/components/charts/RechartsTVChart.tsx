import React, { useEffect, useMemo, useState } from 'react'
import { Area, AreaChart, CartesianGrid, Tooltip, XAxis, YAxis, ResponsiveContainer } from 'recharts'

interface PricePoint {
  price: number
  timestamp: string
}

interface Props {
  symbol: string
  hours?: number
  height?: number
}

const formatTime = (iso: string) => new Date(iso).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })

const RechartsTVChart: React.FC<Props> = ({ symbol, hours = 24, height = 400 }) => {
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
        if (json?.success && Array.isArray(json.data)) {
          setData(json.data as PricePoint[])
        } else if (Array.isArray(json)) {
          setData(json as PricePoint[])
        } else {
          setError('Invalid response')
        }
      } catch (e) {
        if (mounted) setError(e instanceof Error ? e.message : 'Failed to load')
      } finally {
        if (mounted) setLoading(false)
      }
    }
    fetchHistory()
    return () => { mounted = false }
  }, [symbol, hours])

  const series = useMemo(() => data.map(d => ({ time: d.timestamp, price: d.price })), [data])

  if (loading) {
    return (
      <div className="flex items-center justify-center h-full text-gray-400">Loading chart…</div>
    )
  }
  if (error) {
    return (
      <div className="flex items-center justify-center h-full text-red-400">{error}</div>
    )
  }

  return (
    <div className="w-full h-full">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={series} margin={{ top: 10, right: 20, left: 0, bottom: 0 }}>
          <defs>
            <linearGradient id="priceGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.6} />
              <stop offset="100%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <CartesianGrid stroke="#1f2937" strokeDasharray="3 3" />
          <XAxis dataKey="time" tickFormatter={formatTime} stroke="#9ca3af" tick={{ fontSize: 12 }} />
          <YAxis domain={['auto', 'auto']} stroke="#9ca3af" tick={{ fontSize: 12 }} width={70} />
          <Tooltip
            contentStyle={{ backgroundColor: '#0b0f19', border: '1px solid #1f2937' }}
            labelStyle={{ color: '#9ca3af' }}
            formatter={(value: any) => [Number(value).toLocaleString(), 'Price']}
            labelFormatter={(label: string) => new Date(label).toLocaleString()}
          />
          <Area type="monotone" dataKey="price" stroke="#3b82f6" fill="url(#priceGradient)" strokeWidth={2} />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}

export default RechartsTVChart
