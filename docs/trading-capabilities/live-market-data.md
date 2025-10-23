# Live Market Data Delivery

QuantDesk’s trading terminal relies on fast, predictable market data. This page summarizes how the platform serves price, order book, and trade information to the UI and partner integrations.

## REST Endpoints

- `GET /api/markets` – List of active markets with latest oracle price and metadata.
- `GET /api/markets/:symbol/price` – Latest mark price payload `{ price, confidence, slot, ts }`.
- `GET /api/markets/:symbol/orderbook` – Top-of-book bids/asks (default 20 levels per side) plus spread and timestamp.

Symbols are normalized server-side to canonical `BASE-PERP` format (e.g., `SOL-PERP`). Aliases such as `SOL`, `SOL/USDT`, or `SOLUSDT` are accepted and mapped internally. Responses include both `symbol` and `marketId` so frontends can join data reliably.

## WebSocket Envelope

Real-time updates mirror the REST shapes using a versioned envelope:

```json
{
  "v": 1,
  "type": "price | orderbook | trades",
  "symbol": "SOL-PERP",
  "ts": 1730930000000,
  "data": { /* same structure as the REST endpoint */ }
}
```

- Price messages carry `{ price, confidence, slot }` and volatility flags.
- Order book messages contain arrays of `[price, size]` pairs.
- Cadence targets ~1 second; payload compression (gzip/deflate) is available if throughput increases.

Clients subscribe via the WebSocket layer first, then fall back to REST polling (2–5 seconds with jitter) if the stream disconnects. UI badges call out staleness when snapshots are older than ~3 seconds.

## Caching & Fallbacks

- Redis caches fresh snapshots under `qd:{ENV}:price:{SYMBOL}` and `qd:{ENV}:orderbook:{SYMBOL}` with 1–3 second TTLs.
- Pub/Sub channels (`qd:{ENV}:price`, `qd:{ENV}:orderbook`) fan out updates to the WebSocket broadcaster.
- A stale-while-revalidate approach lets the terminal display the last good value immediately while it re-fetches in the background.

## Frontend Consumption Pattern

- **PriceContext** – Attempts WebSocket first, then falls back to REST polling, exposing `price`, `confidence`, and `lastUpdated` to UI components.
- **DepthService** – Polls REST with jitter until the WebSocket stabilizes, then switches to real-time updates; flags stale depth so heatmaps dim gracefully.
- **Volatility Handling** – If `confidence / price > 1%` or price deviates sharply from a local EMA, the UI widens spreads and labels the market “high volatility.”

## Security & Limits

- Public endpoints enforce rate limiting and CORS rules; unknown symbols are rejected.
- Authenticated trading endpoints still require wallet-verified sessions (Sign-In with Solana) even if a client knows the REST/WS URLs.
- Monitoring tracks endpoint latency, Redis hit ratios, WebSocket reconnects, and data freshness. Alerts fire if price snapshots are stale (>5s) or if the WS broadcaster loops.

These practices keep the terminal responsive under normal loads while giving enterprise users clear expectations for integration. For upstream ingestion pipelines, see the [Market Intelligence Pipeline](../ai-engine/market-intelligence-pipeline.md).
