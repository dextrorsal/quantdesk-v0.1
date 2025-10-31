<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>0</epicId>
    <storyId>1</storyId>
    <title>stabilize-backend-oracle-price-apis</title>
    <status>ready-for-dev</status>
    <generatedAt>2025-10-30</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>bmad/stories/0-1-stabilize-backend-oracle-price-apis.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>trader</asA>
    <iWant>/api/oracle/price/:asset and /api/prices to return fresh non-zero prices</iWant>
    <soThat>UI and tools display correct balances</soThat>
    <tasks>
- [ ] Implement single-asset endpoint `/api/oracle/price/:asset` [AC1]
  - [ ] Normalize asset symbol (BTC, ETH, SOL, etc.)
  - [ ] Get price from `pythOracleService.getLatestPrice(asset)`
  - [ ] Return with metadata: `{ price, source, confidence, updatedAt }`
  - [ ] Health checks and freshness guardrails (30s staleness threshold)
  - [ ] Fallback to CoinGecko on Pyth failure (200 + source=fallback)
- [ ] Enhance `/api/prices` endpoint [AC2]
  - [ ] Verify existing implementation returns map with all fields
  - [ ] Ensure stale feeds fall back to secondary provider automatically
  - [ ] Add source attribution (pyth-network vs coingecko-fallback)
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
    </tasks>
  </story>

  <acceptanceCriteria>
1. GET `/api/oracle/price/SOL|BTC|ETH` returns JSON with `{ price:number, source:string, confidence:number, updatedAt }` and HTTP 200 within 250ms p50
2. GET `/api/prices` returns a map of assets with the same fields; stale feeds fall back to secondary provider
3. `/api/oracle/price/*` has health and freshness guardrails; returns 5xx only on irrecoverable errors; otherwise 200 + source=fallback
4. Structured logging added for all oracle operations
5. `/api/oracle/health` route exposes freshness metrics
  </acceptanceCriteria>

  <currentImplementationStatus>
    <status>WORKING - Verified by code inspection</status>
    <description>The Pyth/Oracle integration is currently working and successfully displaying live prices in the Quote Monitor (QM). This story focuses on ENHANCEMENT and STABILIZATION, not rebuilding.</description>
    <verification>
      <frontendQM>
        <file>frontend/src/pro/index.tsx</file>
        <lines>2410-2498</lines>
        <behavior>Polls /api/oracle/prices every 5 seconds (line 2422)</behavior>
        <dataExtraction>Extracts asset from symbol: "BTC-PERP" → "BTC" (lines 2395-2402)</behavior>
        <priceSource>Uses backendPrices[asset] for price (line 2477)</behavior>
        <metadataFallback>Falls back to pair.changePercent, pair.volume, pair.high24h for metadata (lines 2485, 2487, 2489)</metadataFallback>
        <note>QM works because it uses pair data for metadata, not backend response metadata</note>
      </frontendQM>
      <backendEndpoint>
        <file>backend/src/routes/oracle.ts</file>
        <lines>53-100</lines>
        <currentResponse>Returns { success: true, data: Record&lt;string, number&gt;, timestamp, source }</currentResponse>
        <limitation>data is ONLY prices (numbers), NOT full metadata objects with confidence, updatedAt, etc.</limitation>
        <method>Uses pythOracleService.getAllPrices() which returns Record&lt;string, number&gt; (line 70)</method>
        <caching>Redis cache-aside pattern with 1s TTL if enabled (lines 56-63, 82-84)</caching>
      </backendEndpoint>
      <pythService>
        <file>backend/src/services/pythOracleService.ts</file>
        <hermesClient>Uses @pythnetwork/hermes-client - initialized at line 100: new HermesClient('https://hermes.pyth.network')</hermesClient>
        <fetchMethod>hermesClient.getLatestPriceUpdates(pythFeedIds) - line 469</fetchMethod>
        <normalization>actualPrice = price * Math.pow(10, exponent) - line 487</normalization>
        <cache>In-memory priceCache Map with 30s TTL (line 32, updated at line 536)</cache>
        <getAllPrices>Returns Record&lt;string, number&gt; (prices only) - lines 1177-1194</getAllPrices>
        <fetchLatestPrices>Returns Map&lt;string, PythPriceData&gt; with full metadata - lines 456-548</fetchLatestPrices>
        <coinGecko>Fetches CoinGecko prices for Solana tokens marked 'COINGECKO' - lines 520-531</coinGecko>
        <databaseFallback>getDatabaseFallback() queries markets and oracle_prices tables - lines 399-439</databaseFallback>
      </pythService>
    </verification>
    <pythIntegration>
      <method>Uses @pythnetwork/hermes-client (HermesClient)</method>
      <fetch>hermesClient.getLatestPriceUpdates(pythFeedIds)</fetch>
      <normalization>Applies exponent: price * Math.pow(10, exponent)</normalization>
      <staleness>60s threshold check</staleness>
    </pythIntegration>
    <caching>
      <layer1>In-memory cache (priceCache Map) with 30s TTL</layer1>
      <layer2>Redis cache-aside pattern (1s TTL) in /api/oracle/prices route</layer2>
      <stats>Cache stats tracking: hits, misses, stale, errors</stats>
    </caching>
    <supabaseIntegration>
      <fallback>getDatabaseFallback(symbol) method</fallback>
      <tables>
        <table name="markets">Queries by symbol to get market_id</table>
        <table name="oracle_prices">Queries for last known price by market_id</table>
      </tables>
      <purpose>Tertiary fallback when Pyth and CoinGecko fail</purpose>
    </supabaseIntegration>
    <fallbackChain>
      <primary>Pyth Network (HermesClient)</primary>
      <secondary>CoinGecko (for Solana ecosystem tokens)</secondary>
      <tertiary>Database fallback (Supabase oracle_prices table)</tertiary>
    </fallbackChain>
    <frontendIntegration>
      <endpoint>/api/oracle/prices</endpoint>
      <polling>Frontend QM polls every 5s</polling>
      <status>Successfully displaying live prices in Quote Monitor</status>
      <sourceAttribution>Includes source: pyth-network, coingecko-fallback, cache</sourceAttribution>
    </frontendIntegration>
    <storyFocus>
      <criticalEnhancement>AC2: /api/prices must return per-asset metadata objects, not just numbers</criticalEnhancement>
      <enhancements>Adding health endpoints and metrics (AC5)</enhancements>
      <enhancements>Enhancing logging consistency (AC4)</enhancements>
      <enhancements>Improving error handling clarity (AC3)</enhancements>
      <enhancements>Adding freshness guardrails (AC3)</enhancements>
    </storyFocus>
  </currentImplementationStatus>

  <artifacts>
    <docs>
      <doc path="bmad/docs/tech-spec-epic-0.md" title="Epic 0 Technical Specification" section="Oracle Price API Stabilization">
        Defines current state: `/api/oracle/prices` exists and returns normalized prices via `pythOracleService.getAllPrices()`. Enhancements required: single-asset endpoint, health route, structured logging, freshness guardrails.
      </doc>
      <doc path="bmad/docs/architecture.md" title="QuantDesk Architecture Documentation" section="Oracle Integration">
        Backend-centric oracle: Pyth prices fetched by backend, normalized and cached. Oracle integration via `pythOracleService.getAllPrices()` returning `Record<string, number>`.
      </doc>
      <doc path="docs/LIVE_ENDPOINTS_BEST_PRACTICE.md" title="Live Endpoints Best Practices" section="Price Best Practices (Pyth)">
        Always expose price with confidence. Include: `{ price, confidence, slot, ts }`. Compare reported price to EMA for volatility handling. Frontend consumption order: WebSocket → REST polling → Redis cache.
      </doc>
    </docs>
    <code>
      <code path="backend/src/routes/oracle.ts" kind="controller" symbol="router.get('/price/:asset')" lines="134-219" reason="Existing single-asset endpoint implementation - needs enhancement for AC1 compliance">
        Current implementation exists but returns different format than required. Needs to return `{ price, source, confidence, updatedAt }` consistently.
      </code>
      <code path="backend/src/routes/oracle.ts" kind="controller" symbol="router.get('/prices')" lines="53-100" reason="Existing multi-asset endpoint - verify returns all required fields">
        Returns `{ success: true, data: prices, timestamp, source }`. May need enhancement to include confidence and updatedAt per asset.
      </code>
      <code path="backend/src/routes/oracle.ts" kind="controller" symbol="router.get('/health')" lines="249-279" reason="Health endpoint exists - verify exposes freshness metrics as required by AC5">
        Currently calls `pythOracleService.healthCheck()`. May need enhancement to include freshness metrics per asset.
      </code>
      <code path="backend/src/services/pythOracleService.ts" kind="service" symbol="getPrice(symbol)" lines="333-379" reason="Primary method to get price with metadata - returns PythPriceData with price, confidence, timestamp">
        Returns `PythPriceData | null` with price, confidence, exponent, publishTime, timestamp. Cache-first architecture with database fallback.
      </code>
      <code path="backend/src/services/pythOracleService.ts" kind="service" symbol="getLatestPrice(marketSymbol)" lines="885-963" reason="Alternative method for getting price - returns number only, uses getPrice internally">
        Returns `number | null`. Uses `getPrice()` internally then extracts price field. Has cache-first architecture.
      </code>
      <code path="backend/src/services/pythOracleService.ts" kind="service" symbol="getAllPrices()" lines="1177-1194" reason="Returns all prices as Record<string, number> - used by /api/prices endpoint">
        Returns normalized prices `Record<string, number>`. Used by `/api/prices` endpoint.
      </code>
      <code path="backend/src/services/pythOracleService.ts" kind="service" symbol="getPriceConfidence(asset)" lines="1219-1233" reason="Get confidence for specific asset">
        Returns confidence value for asset. Uses `getPrice()` internally.
      </code>
      <code path="backend/src/services/fallbackPriceService.ts" kind="service" symbol="getLatestPrices()" lines="24-50" reason="CoinGecko fallback service - used when Pyth fails">
        Returns `Record<string, PriceData>` with price, timestamp, source='CoinGecko'. Used as fallback when Pyth fails.
      </code>
      <code path="backend/src/utils/logger.ts" kind="utility" symbol="Logger class" lines="1-75" reason="Structured logging utility - use for AC4 structured logging requirement">
        Winston-based logger with JSON format. Provides debug, info, warn, error methods. Default meta: `{ service: 'quantdesk-backend' }`.
      </code>
      <code path="backend/src/middleware/errorHandling.ts" kind="middleware" symbol="ServiceUnavailableError" lines="73-78" reason="Custom error class for 503 errors - use for health check failures">
        Custom error class extending QuantDeskError. Status code 503, code 'SERVICE_UNAVAILABLE'.
      </code>
      <code path="backend/src/middleware/errorHandling.ts" kind="middleware" symbol="NotFoundError" lines="52-57" reason="Custom error class for 404 errors - use for invalid assets">
        Custom error class for 404 errors. Use for unsupported asset symbols.
      </code>
      <code path="backend/src/services/redisCache.ts" kind="service" symbol="get/set cache methods" reason="Redis cache service - used by /api/prices for caching">
        Cache service used by `/api/prices` endpoint with 1s TTL. Check if caching enabled via `CACHE_ENABLE` env var.
      </code>
    </code>
    <dependencies>
      <dependency ecosystem="node" package="@pythnetwork/hermes-client" version="^2.0.0" reason="Hermes client for fetching Pyth price feeds"/>
      <dependency ecosystem="node" package="axios" version="^1.12.2" reason="HTTP client for CoinGecko fallback API calls"/>
      <dependency ecosystem="node" package="winston" version="^3.11.0" reason="Structured logging library - use for AC4 requirement"/>
      <dependency ecosystem="node" package="redis" version="^4.6.10" reason="Redis client for caching - optional via CACHE_ENABLE env"/>
      <dependency ecosystem="node" package="express" version="^4.18.2" reason="Express router for API endpoints"/>
      <dependency ecosystem="node" package="@supabase/supabase-js" version="^2.58.0" reason="Supabase client - use via databaseService only"/>
    </dependencies>
  </artifacts>

  <constraints>
    <constraint>
      **Database Access**: MUST use `databaseService` from `backend/src/services/supabaseDatabase.ts` - NEVER direct Supabase calls
    </constraint>
    <constraint>
      **Error Handling**: MUST use custom error classes from `backend/src/middleware/errorHandling.ts` (NotFoundError, ServiceUnavailableError, etc.)
    </constraint>
    <constraint>
      **Oracle Service**: MUST use `pythOracleService.getAllPrices()` or `pythOracleService.getPrice()` - prices already normalized with exponent
    </constraint>
    <constraint>
      **Logging**: MUST use Logger class from `backend/src/utils/logger.ts` for structured logging (AC4)
    </constraint>
    <constraint>
      **Performance**: p50 latency target <250ms for `/api/oracle/price/:asset` (AC1)
    </constraint>
    <constraint>
      **Freshness**: 30s staleness threshold for prices (AC3) - validate in guardrails
    </constraint>
    <constraint>
      **Error Response**: Return 5xx only on irrecoverable errors; otherwise 200 + source=fallback (AC3)
    </constraint>
    <constraint>
      **Route Location**: All oracle routes in `backend/src/routes/oracle.ts` - maintain existing structure
    </constraint>
  </constraints>

  <interfaces>
    <interface name="GET /api/oracle/price/:asset" kind="REST endpoint" signature="router.get('/price/:asset', async (req, res) => {...})" path="backend/src/routes/oracle.ts">
      **Current State**: Exists at lines 134-219, but returns different format
      **Required Format**: `{ success: true, data: { price, source, confidence, updatedAt } }`
      **Enhancements**: Normalize response to match AC1, add freshness guardrails, improve error handling
    </interface>
    <interface name="GET /api/oracle/prices" kind="REST endpoint" signature="router.get('/prices', async (req, res) => {...})" path="backend/src/routes/oracle.ts">
      **Current State**: Exists at lines 53-100, returns `{ success: true, data: Record<string, number>, timestamp, source }`
      **Enhancements**: Verify returns map with all fields per AC2, ensure per-asset metadata (confidence, updatedAt)
    </interface>
    <interface name="GET /api/oracle/health" kind="REST endpoint" signature="router.get('/health', async (req, res) => {...})" path="backend/src/routes/oracle.ts">
      **Current State**: Exists at lines 249-279, calls `pythOracleService.healthCheck()`
      **Enhancements**: Add freshness metrics per asset, cache status (enabled, hits, misses), last update times per asset (AC5)
    </interface>
    <interface name="pythOracleService.getPrice(symbol)" kind="function signature" signature="async getPrice(symbol: string): Promise&lt;PythPriceData | null&gt;" path="backend/src/services/pythOracleService.ts">
      Returns `PythPriceData` with: `{ price, confidence, exponent, publishTime, timestamp, symbol }`. Cache-first with database fallback.
    </interface>
    <interface name="pythOracleService.getAllPrices()" kind="function signature" signature="async getAllPrices(): Promise&lt;Record&lt;string, number&gt;&gt;" path="backend/src/services/pythOracleService.ts">
      Returns normalized prices as `Record<string, number>`. Used by `/api/prices` endpoint.
    </interface>
    <interface name="fallbackPriceService.getLatestPrices()" kind="function signature" signature="async getLatestPrices(): Promise&lt;Record&lt;string, PriceData&gt;&gt;" path="backend/src/services/fallbackPriceService.ts">
      Returns CoinGecko fallback prices. Structure: `{ [asset]: { symbol, price, timestamp, source: 'CoinGecko' } }`
    </interface>
    <interface name="Logger" kind="class" signature="class Logger { debug(), info(), warn(), error() }" path="backend/src/utils/logger.ts">
      Winston-based structured logger. Methods: `debug(msg, meta?)`, `info(msg, meta?)`, `warn(msg, meta?)`, `error(msg, meta?)`. Logs in JSON format with timestamp.
    </interface>
  </interfaces>

  <tests>
    <standards>
      Backend tests use Vitest framework. Test files located in `backend/tests/` with subdirectories: `unit/`, `integration/`, `e2e/`. Use `supertest` for API endpoint testing. Test structure: describe blocks for endpoints, it blocks for specific cases. Mock external services (Pyth, CoinGecko) when testing error scenarios.
    </standards>
    <locations>
      - `backend/tests/unit/` - Unit tests for individual functions/services
      - `backend/tests/integration/` - Integration tests for API endpoints and service interactions
      - `backend/tests/e2e/` - End-to-end tests for full request flows
      - Example: `backend/tests/integration/oracle-integration.test.ts` - Existing oracle integration tests
    </locations>
    <ideas>
      <test ac="AC1" idea="Unit test: GET /api/oracle/price/BTC returns correct format with price, source, confidence, updatedAt within 250ms p50">
        Mock `pythOracleService.getPrice()` to return test data. Measure response time. Verify response structure matches AC1 format.
      </test>
      <test ac="AC1" idea="Integration test: GET /api/oracle/price/SOL|ETH|BTC all return HTTP 200 with valid data">
        Test all three required assets. Verify each returns correct format. Test error handling for invalid assets.
      </test>
      <test ac="AC2" idea="Integration test: GET /api/prices returns map with all fields, stale feeds fall back to CoinGecko">
        Mock Pyth to return stale data. Verify fallback to CoinGecko. Check response includes source attribution.
      </test>
      <test ac="AC3" idea="Unit test: Freshness guardrails reject prices >30s old, return 200 + source=fallback">
        Mock `pythOracleService.getPrice()` to return stale data (>30s). Verify endpoint returns 200 with source=fallback.
      </test>
      <test ac="AC3" idea="Integration test: Irrecoverable errors return 5xx, recoverable errors return 200 + source=fallback">
        Test scenarios: Pyth failure + CoinGecko failure (5xx), Pyth failure + CoinGecko success (200 + fallback).
      </test>
      <test ac="AC4" idea="Unit test: Structured logging captures all oracle operations (fetches, cache hits/misses, fallbacks)">
        Mock Logger and verify log calls for price fetches, cache operations, fallback events.
      </test>
      <test ac="AC5" idea="Integration test: GET /api/oracle/health returns freshness metrics, cache status, last update times">
        Verify health endpoint returns expected structure with freshness metrics per asset, cache stats, update times.
      </test>
    </ideas>
  </tests>
</story-context>

