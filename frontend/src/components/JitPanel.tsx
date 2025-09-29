import React, { useEffect, useMemo, useState } from 'react'
import { createAuction, submitQuote, settleAuction } from '@/services/liquidity'

type Props = { token?: string }

export default function JitPanel({ token }: Props) {
  const [symbol, setSymbol] = useState('BTC-PERP')
  const [side, setSide] = useState<'buy'|'sell'>('buy')
  const [size, setSize] = useState(0.1)
  const [durationMs, setDurationMs] = useState(5000)
  const [auctionId, setAuctionId] = useState<string | null>(null)
  const [makerPrice, setMakerPrice] = useState<number>(0)
  const [expiresAt, setExpiresAt] = useState<number | null>(null)
  const [now, setNow] = useState(Date.now())
  const remaining = useMemo(() => expiresAt ? Math.max(0, expiresAt - now) : 0, [expiresAt, now])

  useEffect(() => {
    const t = setInterval(() => setNow(Date.now()), 200)
    return () => clearInterval(t)
  }, [])

  const startAuction = async () => {
    const res = await createAuction({ symbol, side, size, durationMs }, token)
    setAuctionId(res.auction.id)
    setExpiresAt(res.auction.expiresAt)
  }

  const quote = async () => {
    if (!auctionId) return
    await submitQuote(auctionId, 'maker-local', makerPrice, token)
  }

  const settle = async () => {
    if (!auctionId) return
    const res = await settleAuction(auctionId, token)
    alert(JSON.stringify(res.result))
  }

  return (
    <div className="rounded-md border p-3 space-y-2">
      <div className="font-semibold">JIT Liquidity (Prototype)</div>
      <div className="grid grid-cols-2 gap-2">
        <label>Symbol
          <input className="w-full border p-1" value={symbol} onChange={e=>setSymbol(e.target.value)} />
        </label>
        <label>Side
          <select className="w-full border p-1" value={side} onChange={e=>setSide(e.target.value as any)}>
            <option value="buy">buy</option>
            <option value="sell">sell</option>
          </select>
        </label>
        <label>Size
          <input className="w-full border p-1" type="number" step="0.01" value={size} onChange={e=>setSize(parseFloat(e.target.value))} />
        </label>
        <label>Duration (ms)
          <input className="w-full border p-1" type="number" value={durationMs} onChange={e=>setDurationMs(parseInt(e.target.value))} />
        </label>
      </div>
      <div className="flex items-center gap-2">
        <button className="px-3 py-1 bg-blue-600 text-white rounded" onClick={startAuction}>Start</button>
        {expiresAt && <span>Time left: {Math.ceil(remaining/100)/10}s</span>}
      </div>

      <div className="grid grid-cols-2 gap-2 items-end">
        <label>Maker Price
          <input className="w-full border p-1" type="number" step="0.01" value={makerPrice} onChange={e=>setMakerPrice(parseFloat(e.target.value))} />
        </label>
        <div className="flex gap-2">
          <button className="px-3 py-1 bg-emerald-600 text-white rounded" onClick={quote} disabled={!auctionId}>Quote</button>
          <button className="px-3 py-1 bg-slate-700 text-white rounded" onClick={settle} disabled={!auctionId}>Settle</button>
        </div>
      </div>
    </div>
  )
}


