# Story: Authoritative Price Hub (Pyth-Centric)

## Status
Draft

## Summary
Create a single authoritative price hub powering all Pro Terminal needs. Backend aggregates Pyth prices with 24h stats, exposes canonical endpoints and WebSocket topics; frontend consumes via a unified context and strict symbol normalization. Eliminates “N/A” and seeded placeholders, ensuring one truth for deposits, trading, quotes, and orderbooks.

## Context
- Current `/api/oracle/prices` returns a partial map; not all displayed assets are present.
- UI components normalize symbols differently and sometimes seed static prices.
- `PriceContext` WS often falls back; polling endpoint may not be canonical.
- Drift uses Pyth (push/pull) plus a robust subscription layer to maintain a central cache; all consumers read from it.

## Acceptance Criteria
1. Canonical Symbol Catalog (Backend)
   - AC1.1: A catalog exists mapping displayed assets and markets to stable keys and Pyth feed IDs.
   - AC1.2: A normalization function converts any of: BTC, BTC/USDT, BTCUSDT, BTC-PERP → `{ asset: BTC, market?: BTC-PERP }`.
2. Authoritative Price API
   - AC2.1: `GET /api/prices` (alias to current oracle service) returns a complete object for all catalog assets with: `price, confidence, updatedAt, change24h, volume24h, high24h, low24h, source`.
   - AC2.2: `GET /api/price/:asset` returns the same fields for a single asset.
   - AC2.3: Freshness guard: `updatedAt` reflects last ingest; endpoint never returns zero/undefined price for a catalogued asset (responds 503 with reason if truly unavailable). Max staleness 3s under steady-state ingest.
   - AC2.4: Performance: p50 < 250ms and p99 < 1s for `GET /api/prices` over 50 assets on dev hardware.
   - AC2.5: Error contract: per-asset record may include `{ status: 'fallback'|'unavailable', reason }` when using secondary sources or if missing.
   - AC2.6: Health: `GET /api/prices/health` returns `{ ok: boolean, lastIngestAt, assetCount }` for canary/CI.
   - AC2.7: Each asset includes `feedId` and `publishTime` when available, plus optional `ema` if exposed by the source.
   - AC2.8: Per-asset status classification `{ 'live'|'stale'|'high_conf'|'fallback'|'unavailable' }` using env thresholds `PRICE_STALE_MS` and `PRICE_MAX_CONF_RATIO`.
3. Orderbook/Candles Normalization
   - AC3.1: `GET /api/orderbook/:market` and `GET /api/candles/:market` accept normalized `market` symbols and respond consistently with the catalog.
4. WebSocket Topics
   - AC4.1: `ws prices:{asset}` broadcasts `{asset, price, confidence, updatedAt}` every change (or at most once per second if throttled); staleness never exceeds 3s relative to backend cache.
   - AC4.2: If WS isn’t available, server-side gracefully continues to publish updates to the cache and clients can poll.
5. Frontend Integration
   - AC5.1: `PriceContext` uses `/api/prices` for polling and subscribes to `prices:{asset}`; no static seeded values are shown.
   - AC5.2: QM/ORDER/OVERVIEW render values from the hub only; no per-component symbol mapping.
   - AC5.3: Symbol normalization is a shared util consumed by all UI.
   - AC5.4: Security: No direct Supabase oracle reads in frontend; all price data flows through backend endpoints.
   - AC5.5: UI badges for status (`LIVE`, `STALE`, `HIGH CONF`, `FALLBACK`) with tooltips showing `confidence`, `publishTime`, and `feedId`.
6. Validation
   - AC6.1: Manual test plan shows non-empty prices for all displayed assets in QM.
   - AC6.2: Deposits/withdrawals convert using `GET /api/price/:asset` and reflect current price within freshness window.
   - AC6.3: ORDER form default price matches live feed for selected market.

## Tasks/Subtasks
- [ ] Backend: Create catalog (assets + markets + Pyth feed IDs) and normalization util.
- [ ] Backend: Implement `/api/prices` (alias/upgrade) with Pyth primary + exchange fallback for 24h stats, include `feedId`, `publishTime`, `confidence`, optional `ema`.
- [ ] Backend: Implement WS `prices:{asset}` topic (throttled) and health/freshness.
- [ ] Backend: Normalize `orderbook/candles` endpoints with catalog.
- [ ] Frontend: Add `normalizeSymbol()` shared util and consume it in contexts and components.
- [ ] Frontend: Point `PriceContext` polling to `/api/prices`; subscribe to WS; remove seeded placeholders.
- [ ] Frontend: Update QM/ORDER/OVERVIEW to read from hub only; remove per-component fallbacks.
- [ ] Frontend: Render status badges + tooltips using `confidence`, `publishTime`, and `feedId`.
- [ ] Tests: API unit (freshness, mapping), integration (QM renders), e2e smoke.

## Dev Notes
- Start with BTC/ETH/SOL; extend to all Pro symbols once feed IDs are added.
- Use a small TTL (1–3s) in backend cache; include `source` and `updatedAt` for observability.
- Prefer server-side symbol normalization (frontend util is just a passthrough to backend rules).
 - Thresholds (env): `PRICE_STALE_MS` default 3000; `PRICE_MAX_CONF_RATIO` default 0.01 (1%).

## File List
- backend: `src/services/oracleHub.ts`, `src/routes/prices.ts`, `src/services/catalog.ts`
- frontend: `src/utils/symbols.ts`, `src/contexts/PriceContext.tsx`, `src/pro/index.tsx` (consumers)
- docs: `bmad/docs/price-hub-architecture.md`

## Change Log
- 2025-10-30: Draft story created.
