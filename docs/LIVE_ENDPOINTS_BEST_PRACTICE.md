## QuantDesk Live Endpoints and Frontend Integration Best Practices

This guide documents the recommended, production-ready way to expose live market data from the backend and consume it from the frontend (Modern Trading tab). It consolidates what exists today with standard perp DEX practices (Drift/Flash/Hyperliquid style) and MCP-backed guidance.

### Core Objectives
- Low-latency, resilient price and depth streams with graceful fallbacks.
- Canonical market naming and symbol normalization.
- Security (auth, rate limiting) and cache design (Redis) that scales.
- Predictable message/REST schemas that are future-proof.

---

## 1) Backend: Canonical Routes (what exists today)

- GET `/api/markets`
  - Returns active markets with latest oracle price.
- GET `/api/markets/:symbol/price`
  - Returns latest Pyth price for a given symbol.
- GET `/api/markets/:symbol/orderbook`
  - Returns top-of-book (bids/asks) aggregated from pending orders in Postgres.
- WebSocket (internal service)
  - Broadcasts snapshots (order book, trades) at ~1s cadence.

Symbols: Prefer canonical perp names `BASE-PERP` (e.g., `SOL-PERP`). Accept aliases (`SOL`, `SOL/USDT`) and normalize server-side.

---

## 2) Price Best Practices (Pyth)

- Always expose (and consume) price with confidence. Include: `{ price, confidence, slot, ts }`.
- Volatility handling (Flash Trade pattern):
  - Compare reported price to EMA; if |price-ema| exceeds threshold → set High Volatility flag.
  - If confidence/price > 1% (wide CI), switch UI to Close-Only cues and widen spreads.
- Frontend consumption order:
  1) WebSocket price stream (if connected)
  2) REST polling fallback: `GET /api/markets/:symbol/price` (2–5s)
  3) Last cached value from Redis (stale-while-revalidate; mark UI as “delayed” when >3s old)

---

## 3) Order Book Best Practices

- REST: `GET /api/markets/:symbol/orderbook` returns
  ```json
  {
    "success": true,
    "orderbook": {
      "symbol": "SOL-PERP",
      "bids": [[price, size], ...],
      "asks": [[price, size], ...],
      "spread": number,
      "timestamp": number
    }
  }
  ```
- WS: Mirror the same shape inside a compact envelope (see §6).
- Depth size: top 20 levels per side by default; support `?levels=n` (<=100) with sane rate limits.
- Heatmap/UI: scale bar width to max(size) in current viewport; clamp extremes.
- Latency: target 1s cadence; WS preferred, REST polling fallback 1–2s with jitter.

---

## 4) Caching and Redis

- Keys (namespace): `qd:{ENV}:price:{SYMBOL}` and `qd:{ENV}:orderbook:{SYMBOL}`.
- TTL: 1–3s for price/orderbook snapshots; update on each tick.
- Pub/Sub channels: `qd:{ENV}:price` and `qd:{ENV}:orderbook` for broadcasting cache updates to WS layer.
- Health: `/health/redis` endpoint; on cache miss, hit source and repopulate.

---

## 5) WebSocket Message Envelope (forward-compatible)

```json
{
  "v": 1,
  "type": "orderbook|price|trades",
  "symbol": "SOL-PERP",
  "ts": 1730930000000,
  "data": { /* shape matches REST payload for that resource */ }
}
```

- Include `v` for versioning.
- For price: `data = { price, confidence, slot }`.
- For orderbook: `data = { bids, asks, spread }` (arrays of `[price,size]`).
- Consider gzip/deflate if payload growth becomes an issue.

---

## 6) Frontend Consumption Pattern (Modern Trading Tab)

- Symbol routing: `/trading/:symbol` → normalize to `BASE-PERP` internally; store canonical in context.
- Price source: PriceContext
  - Primary: WS stream (optional mode; no-console-error on failure),
  - Fallback: `GET /api/markets/:symbol/price` (2–5s),
  - Show volatility flag and widen displayed spread if `confidence/price > 1%`.
- Order book source: DepthService (new)
  - Poll `GET /api/markets/:symbol/orderbook` every 1s with jitter and SWR cache (keep last snapshot for smooth UI).
  - When WS is stable, switch to WS updates and reduce REST frequency.
- Resilience:
  - Backoff with jitter; stop after N attempts; show “disconnected” UI badge.
  - Treat snapshots older than 3s as stale (badge + dim heatmap).

---

## 7) Security and Rate Limiting

- Public routes (`/api/markets`, `/api/markets/:symbol/price`, `/api/markets/:symbol/orderbook`) → public rate limiter; 429 + `Retry-After`.
- Authed routes (`/api/orders`, `/api/positions`, etc.) → SIWS session cookie; never trust wallet address from client without server verification.
- Input validation: whitelist `:symbol` to known markets; reject unknowns.
- CORS: allow local dev origins; restrict in production.

---

## 8) Naming and Normalization

- Canonical: `BASE-PERP` uppercase (e.g., `SOL-PERP`).
- Accept: `SOL`, `SOL/USDT`, `SOLUSDT` and map to canonical internally.
- Include `marketId` in responses for stable joins; UI stores both `symbol` and `marketId`.

---

## 9) Monitoring and Metrics

- Log: endpoint latencies, Redis hit ratio, WS deliver success/fail, reconnect counts.
- Alert on: price staleness (>5s), orderbook staleness (>3s), WS disconnect loops, Redis error rate.

---

## 10) Implementation Checklist

- [ ] Frontend DepthService: poll `/api/markets/:symbol/orderbook` + SWR cache.
- [ ] PriceContext: expose `{ price, confidence, ts }`, with WS optional mode.
- [ ] Symbol utils: normalize to `BASE-PERP`; map aliases.
- [ ] Redis cache + Pub/Sub wired in backend services.
- [ ] WS envelope v1; fallback REST parity.
- [ ] Volatility flag + UI badges based on confidence/EMA.

---

## References (MCP-backed)
- Drift v2 WS orderbook structure and cadence.
- Flash Trade pricing engine (volatility/confidence handling, close-only rules).
- Helius Yellowstone gRPC resilience patterns (ping/pong, reconnect).


