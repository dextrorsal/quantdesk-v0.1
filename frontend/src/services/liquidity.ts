export type Auction = {
  id: string
  symbol: string
  side: 'buy' | 'sell'
  size: number
  referencePrice: number
  maxSlippageBps: number
  createdAt: number
  expiresAt: number
  settled: boolean
}

export async function createAuction(input: { symbol: string; side: 'buy'|'sell'; size: number; durationMs?: number; maxSlippageBps?: number }, token?: string) {
  const res = await fetch('/api/liquidity/auctions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: JSON.stringify(input)
  })
  if (!res.ok) throw new Error(`createAuction failed: ${res.status}`)
  return res.json()
}

export async function submitQuote(auctionId: string, makerId: string, price: number, token?: string) {
  const res = await fetch(`/api/liquidity/auctions/${auctionId}/quotes`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    },
    body: JSON.stringify({ makerId, price })
  })
  if (!res.ok) throw new Error(`submitQuote failed: ${res.status}`)
  return res.json()
}

export async function settleAuction(auctionId: string, token?: string) {
  const res = await fetch(`/api/liquidity/auctions/${auctionId}/settle`, {
    method: 'POST',
    headers: {
      ...(token ? { Authorization: `Bearer ${token}` } : {})
    }
  })
  if (!res.ok) throw new Error(`settleAuction failed: ${res.status}`)
  return res.json()
}


