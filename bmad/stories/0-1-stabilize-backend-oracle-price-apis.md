# Story 0-1: Stabilize Backend Oracle and Price APIs

Status: ready-for-dev

## Story

As a trader, I need `/api/oracle/price/:asset` and `/api/prices` to return fresh non-zero prices so UI and tools display correct balances.

## Acceptance Criteria

1. GET `/api/oracle/price/SOL|BTC|ETH` returns JSON with `{ price:number, source:string, confidence:number, updatedAt }` and HTTP 200 within 250ms p50
2. GET `/api/prices` returns a map of assets with the same fields; stale feeds fall back to secondary provider
3. `/api/oracle/price/*` has health and freshness guardrails; returns 5xx only on irrecoverable errors; otherwise 200 + source=fallback
4. Structured logging added for all oracle operations
5. `/api/oracle/health` route exposes freshness metrics

## Tasks / Subtasks

- [ ] Enhance single-asset endpoint `/api/oracle/price/:asset` [AC1]
  - [ ] Verify existing implementation (already exists at line 134-219 in oracle.ts)
  - [ ] Ensure normalized asset symbol handling (BTC, ETH, SOL, etc.)
  - [ ] Verify uses `pythOracleService.getLatestPrice(asset)` or appropriate method
  - [ ] Enhance response to include full metadata: `{ price, source, confidence, updatedAt }`
  - [ ] Add health checks and freshness guardrails (30s staleness threshold)
  - [ ] Ensure fallback to CoinGecko on Pyth failure (200 + source=fallback)
- [ ] Enhance `/api/prices` endpoint [AC2]
  - [ ] **Current issue**: Returns `Record<string, number>` (prices only) - line 70: `prices = await pythOracleService.getAllPrices()`
  - [ ] **Fix**: Change to return `Record<string, { price, source, confidence, updatedAt }>` per asset
  - [ ] Use `pythOracleService.getPrice(symbol)` for each asset to get full `PythPriceData` with metadata
  - [ ] Verify stale feeds fall back to secondary provider automatically (already working in `fetchLatestPrices()`)
  - [ ] Ensure source attribution included (pyth-network vs coingecko-fallback vs cache)
- [ ] Add health and freshness guardrails [AC3]
  - [ ] Return 5xx only on irrecoverable errors
  - [ ] Return 200 + source=fallback on recoverable errors
  - [ ] Validate price freshness (<30s threshold)
- [ ] Add structured logging [AC4]
  - [ ] Log all oracle operations (price fetches, cache hits/misses)
  - [ ] Log fallback events and error conditions
  - [ ] Use consistent log format across all oracle operations
- [ ] Implement `/api/oracle/health` route [AC5]
  - [ ] Return freshness metrics for all assets
  - [ ] Include cache status (enabled, hits, misses)
  - [ ] Include last update times per asset

## Dev Notes

### Current Implementation Status ⚠️ IMPORTANT

**Pyth/Oracle integration is WORKING and does not need redesign/fixing.**

**Verified by actual code inspection:**

1. **Frontend QM Component** (`frontend/src/pro/index.tsx:2410-2424`):
   - Polls `/api/oracle/prices` every 5 seconds
   - Extracts asset from symbol (e.g., "BTC-PERP" → "BTC")
   - Uses `backendPrices[asset]` for price data
   - **Note**: Currently falls back to `pair.changePercent`, `pair.volume`, `pair.high24h` for metadata (from initial pairs data)

2. **Backend `/api/oracle/prices` endpoint** (`backend/src/routes/oracle.ts:53-100`):
   - Redis cache-aside pattern (1s TTL) if enabled
   - Calls `pythOracleService.getAllPrices()` 
   - Returns `{ success: true, data: Record<string, number>, timestamp, source }`
   - **Current limitation**: `data` is ONLY prices (numbers), NOT full metadata objects
   - Example response: `{ BTC: 12345.67, ETH: 2345.67, ... }` (no `changePercent`, `volume24h`, etc.)

3. **Pyth Service** (`backend/src/services/pythOracleService.ts`):
   - ✅ Uses `@pythnetwork/hermes-client` (HermesClient) - line 100: `new HermesClient('https://hermes.pyth.network')`
   - ✅ Fetches via `hermesClient.getLatestPriceUpdates(pythFeedIds)` - line 469
   - ✅ Normalizes prices: `actualPrice = price * Math.pow(10, exponent)` - line 487
   - ✅ In-memory cache: `priceCache` Map with 30s TTL - line 32, 536
   - ✅ `getAllPrices()` returns `Record<string, number>` (prices only) - line 1177-1194
   - ✅ `fetchLatestPrices()` returns `Map<string, PythPriceData>` with full metadata - line 456-548
   - ✅ CoinGecko fallback for Solana tokens (JUP, RAY, etc.) - lines 520-531
   - ✅ Database fallback via `getDatabaseFallback()` - queries `markets` and `oracle_prices` tables - lines 399-439

4. **Fallback Chain** (verified):
   1. Primary: Pyth Network via HermesClient (`getLatestPriceUpdates`)
   2. Secondary: CoinGecko API (for tokens marked 'COINGECKO' in PYTH_FEED_IDS)
   3. Tertiary: Supabase database (`markets` → `oracle_prices` table)

**This story is about ENHANCEMENT and STABILIZATION, not rebuilding:**
- **AC2**: `/api/prices` should return per-asset metadata (`{ price, source, confidence, updatedAt }`), not just numbers
- Adding health endpoints and metrics (AC5)
- Enhancing logging consistency (AC4)
- Improving error handling clarity (AC3)
- Adding freshness guardrails (AC3)

### Architecture Alignment

- Use `pythOracleService.getAllPrices()` and `pythOracleService.getLatestPrice()` from `backend/src/services/pythOracleService.ts`
- Route location: `backend/src/routes/oracle.ts`
- Error handling: Custom error classes from `backend/src/middleware/errorHandling.ts`
- Database: Use `databaseService` only (no direct Supabase calls)

### References

- [Source: bmad/docs/tech-spec-epic-0.md#Oracle Price API Stabilization]
- [Source: bmad/docs/architecture.md#Oracle Integration]
- [Source: backend/src/routes/oracle.ts - existing `/api/oracle/prices` implementation]

## Dev Agent Record

### Context Reference

- bmad/stories/0-1-stabilize-backend-oracle-price-apis.context.md

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List
