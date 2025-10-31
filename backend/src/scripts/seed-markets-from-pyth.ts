import { getSupabaseService } from '../services/supabaseService'

async function main() {
  const db = getSupabaseService()
  const env = process.env as Record<string, string | undefined>
  const prefix = 'PYTH_PRICE_FEED_'

  const symbols: string[] = []
  for (const k of Object.keys(env)) {
    if (!k.startsWith(prefix)) continue
    const sym = k.substring(prefix.length).toUpperCase()
    const val = (env[k] || '').trim()
    if (val.startsWith('0x') && val.length === 66) symbols.push(sym)
  }

  if (symbols.length === 0) {
    console.log('No PYTH_PRICE_FEED_* entries found. Nothing to seed.')
    process.exit(0)
  }

  // Idempotent upsert into markets table by symbol (e.g., BTC-PERP)
  const nowIso = new Date().toISOString()
  let created = 0, updated = 0
  for (const base of symbols) {
    for (const kind of ['PERP', 'USD']) {
      const symbol = kind === 'PERP' ? `${base}-PERP` : `${base}/USD`
      const quote = kind === 'PERP' ? 'USDT' : 'USD'
      const payload = {
        symbol,
        baseAsset: base,
        quoteAsset: quote,
        isActive: true,
        maxLeverage: kind === 'PERP' ? 20 : 1,
        category: kind === 'PERP' ? 'perp' : 'spot',
        description: '',
        logoUrl: '',
        price: 0,
        change24h: 0,
        volume24h: 0,
        openInterest: 0,
        fundingRate: 0,
        timestamp: nowIso,
        confidence: 0,
      }

      const upd = await db.getClient()
        .from('markets')
        .update(payload)
        .eq('symbol', symbol)
        .select('id')
      if (upd?.data && upd.data.length > 0) { updated++; continue }

      await db.getClient().from('markets').insert(payload)
      created++
    }
  }

  console.log(`Seed complete. created=${created}, updated=${updated}`)
}

main().catch((e) => {
  console.error('Seed failed:', e)
  process.exit(1)
})


