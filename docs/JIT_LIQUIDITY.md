# JIT Liquidity (Prototype)

This document describes the minimal Just-in-Time (JIT) auction flow implemented in the backend for development/testing.

- Reference price: pulled from oracle service (Pyth with CoinGecko fallback)
- Auction window: short-lived (default 5s), quotes must arrive before expiry
- Settlement: best eligible quote wins (buyer prefers lowest, seller prefers highest) within slippage guard
- Persistence: in-memory (prototype), not durable across restarts

## Endpoints

- POST `/api/liquidity/auctions`
  - body: { "symbol": "BTC-PERP", "side": "buy"|"sell", "size": number, "durationMs": number, "maxSlippageBps": number }
  - returns: { success, auction }

- POST `/api/liquidity/auctions/:id/quotes`
  - body: { "makerId": string, "price": number }
  - returns: { success, auction }

- POST `/api/liquidity/auctions/:id/settle`
  - returns: { success, result: { filled, fillPrice?, makerId?, reason? } }

## Auth & Rate Limits

- All `/api/liquidity/*` routes require auth and are rate limited (120 req/min per user/IP).

## Notes & Next Steps

- Add persistence (tables: auctions, quotes, settlements)
- Emit WS events for auction created/quote/settled
- Integrate with on-chain settlement/AMM backstop
- Unit tests and keeper robustness
