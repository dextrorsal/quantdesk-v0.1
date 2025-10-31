<story-context id="bmad/bmm/workflows/4-implementation/story-context/template" v="1.0">
  <metadata>
    <epicId>0</epicId>
    <storyId>7</storyId>
    <title>MIKEY Tools Fetch Live Prices, Tweets, and News via Backend</title>
    <status>ready-for-dev</status>
    <generatedAt>2025-10-30</generatedAt>
    <generator>BMAD Story Context Workflow</generator>
    <sourceStoryPath>bmad/stories/0-7-mikey-tools-fetch-live-prices-tweets-news-backend.md</sourceStoryPath>
  </metadata>

  <story>
    <asA>AI operator</asA>
    <iWant>MIKEY tools to use backend endpoints for reliable data</iWant>
    <soThat>I avoid CORS issues and get consistent data</soThat>
    <tasks>
- [ ] Create backend proxy endpoints [AC1, AC2]
  - [ ] `/api/market/summary` - Aggregated market data endpoint
  - [ ] `/api/tweets/:query` - Twitter/X data proxy endpoint
  - [ ] `/api/news/:query` - News data proxy endpoint
  - [ ] Standardize error messages and timeouts
- [ ] Update MIKEY tools to use backend endpoints [AC1, AC2]
  - [ ] Update `get_live_price` to call `/api/oracle/price/:asset`
  - [ ] Create `get_market_summary` tool calling `/api/market/summary`
  - [ ] Create `get_tweets` tool calling `/api/tweets/:query`
  - [ ] Create `get_news` tool calling `/api/news/:query`
  - [ ] Ensure all tools return timestamps with data
  - [ ] Verify CORS-free (all calls go through backend)
- [ ] Add standardized error handling [AC2]
  - [ ] Consistent timeout values
  - [ ] Standardized error message format
  - [ ] Proper error propagation to MIKEY-AI
    </tasks>
  </story>

  <acceptanceCriteria>
1. Tools: `get_live_price(asset)`, `get_market_summary`, `get_tweets(query)`, `get_news(query)` return current data with timestamps
2. CORS-free: tools call backend proxy endpoints; timeouts and error messages standardized
  </acceptanceCriteria>

  <artifacts>
    <docs>
      <doc path="bmad/docs/tech-spec-epic-0.md" title="Epic 0 Technical Specification" section="MIKEY Tools Backend Integration">
        Defines requirements for backend-proxied MIKEY tools. Requires all external API calls routed through backend to avoid CORS.
      </doc>
      <doc path="bmad/docs/architecture.md" title="QuantDesk Architecture Documentation" section="MIKEY-AI Integration">
        Architecture outlines MIKEY-AI service integration with backend API gateway for data access.
      </doc>
    </docs>
    <code>
      <artifact path="MIKEY-AI/src/services/RealDataTools.ts" kind="service" symbol="RealDataTools" lines="1-284" reason="MIKEY AI real data tools. Currently calls some backend endpoints but needs enhancement for complete backend proxy integration.">
        Current implementation:
        - `createPythPriceTool()` already calls `/api/oracle/prices` (lines 33-80) ✅
        - `createNewsDataTool()` calls `/api/v1/market/sentiment?symbol=SOL` (line 180) - endpoint may not exist
        - `createMarketDataTool()` calls `/api/real-supabase-markets` (line 219) - exists but may need enhancement
        - **MISSING**: `get_live_price(asset)` tool for single asset
        - **MISSING**: `get_market_summary` tool
        - **MISSING**: `get_tweets(query)` tool
        - **NEEDS**: Standardized error handling and timeouts
      </artifact>
      <artifact path="backend/src/routes/oracle.ts" kind="route" symbol="router.get('/price/:asset')" lines="134-219" reason="Single-asset price endpoint. Exists but may need enhancement for AC1 (metadata, timestamps).">
        Current implementation:
        - Returns `{ success: true, price, value, data: { asset, price, confidence, timestamp, source } }`
        - Already returns timestamp ✅
        - May need to ensure consistency with `/api/prices` format
      </artifact>
      <artifact path="backend/src/routes/aiAgent.ts" kind="route" symbol="router.get('/market-summary')" lines="19-61" reason="Market summary endpoint for dev AI assistance. Exists but may need exposure as `/api/market/summary` for MIKEY tools.">
        Current implementation:
        - Endpoint: `/api/dev/market-summary` (line 19)
        - Returns aggregated market data with prices, volumes, open interest
        - Returns timestamp ✅
        - **NEEDS**: May need to be exposed as `/api/market/summary` for MIKEY tools
      </artifact>
      <artifact path="backend/src/routes/news.ts" kind="route" symbol="router.get('/')" lines="408-481" reason="News endpoint. Exists but may need query parameter support for MIKEY tools.">
        Current implementation:
        - Endpoint: `/api/news` (line 408)
        - Accepts query parameters: sources, ticker, category, keyword, limit
        - Returns news articles with timestamps ✅
        - **NEEDS**: May need `/api/news/:query` route for MIKEY tools
      </artifact>
      <artifact path="backend/src/routes/twitter.ts" kind="route" symbol="router" lines="1-160" reason="Twitter/X routes. May exist but needs verification for query support.">
        Twitter routes:
        - Need to verify if `/api/tweets/:query` exists
        - May need to create if missing
      </artifact>
    </code>
    <dependencies>
      <dependency>
        <ecosystem>Node.js</ecosystem>
        <packages>
          <package>express: ^4.18.2</package>
          <package>axios: ^1.12.2</package>
          <package>@langchain/core: ^0.x</package>
        </packages>
      </dependency>
    </dependencies>
  </artifacts>

  <constraints>
    - All external API calls must be routed through backend to avoid CORS
    - Standardized timeout: 10 seconds for all external API calls
    - Standardized error format: `{ success: false, error: string, timestamp: number }`
    - All tools must return timestamps with data
    - Tools should handle network errors gracefully
    - Error messages must be standardized across all tools
    - Backend endpoints must return consistent response format
  </constraints>
  <interfaces>
    <interface name="GET /api/oracle/price/:asset" kind="REST endpoint" signature="GET /api/oracle/price/:asset" path="backend/src/routes/oracle.ts">
      Single-asset price endpoint. Returns { success: true, price, data: { asset, price, confidence, timestamp, source } }.
    </interface>
    <interface name="GET /api/market/summary" kind="REST endpoint" signature="GET /api/market/summary" path="backend/src/routes/aiAgent.ts">
      Market summary endpoint. Currently at `/api/dev/market-summary`, may need `/api/market/summary` alias.
    </interface>
    <interface name="GET /api/tweets/:query" kind="REST endpoint" signature="GET /api/tweets/:query" path="backend/src/routes/twitter.ts">
      Twitter/X data proxy endpoint. May need to be created or enhanced.
    </interface>
    <interface name="GET /api/news/:query" kind="REST endpoint" signature="GET /api/news/:query" path="backend/src/routes/news.ts">
      News data proxy endpoint. Base endpoint exists at `/api/news`, may need query parameter route.
    </interface>
    <interface name="DynamicTool.createPythPriceTool()" kind="function signature" signature="createPythPriceTool(): DynamicTool" path="MIKEY-AI/src/services/RealDataTools.ts">
      Pyth price tool. Already calls `/api/oracle/prices` backend endpoint. Returns prices with timestamp.
    </interface>
    <interface name="DynamicTool.createNewsDataTool()" kind="function signature" signature="createNewsDataTool(): DynamicTool" path="MIKEY-AI/src/services/RealDataTools.ts">
      News data tool. Currently calls `/api/v1/market/sentiment`, may need to use `/api/news` instead.
    </interface>
  </interfaces>
  <tests>
    <standards>
      Backend tests use `vitest` in `backend/tests/`. Integration tests for API endpoints are in `backend/tests/integration/`. Unit tests for tools are in `MIKEY-AI/tests/unit/`. Mock external APIs when testing error scenarios.
    </standards>
    <locations>
      - `backend/tests/integration/` - API endpoint integration tests
      - `backend/tests/unit/` - Service unit tests
      - `MIKEY-AI/tests/unit/` - Tool unit tests
    </locations>
    <ideas>
      <test ac="AC1" idea="Integration test: get_live_price returns current data with timestamp">
        Test MIKEY tool calls `/api/oracle/price/:asset`. Verify response includes price, confidence, timestamp, source. Verify timestamp is recent.
      </test>
      <test ac="AC1" idea="Integration test: get_market_summary returns aggregated data">
        Test MIKEY tool calls `/api/market/summary`. Verify response includes markets array with prices, volumes, open interest, timestamps.
      </test>
      <test ac="AC1" idea="Integration test: get_tweets returns tweets with timestamps">
        Test MIKEY tool calls `/api/tweets/:query`. Verify response includes tweets array with content, author, timestamp, url. Verify CORS-free (called from backend).
      </test>
      <test ac="AC1" idea="Integration test: get_news returns news with timestamps">
        Test MIKEY tool calls `/api/news/:query`. Verify response includes news array with title, content, source, sentiment, timestamp, url.
      </test>
      <test ac="AC2" idea="Integration test: Standardized timeouts">
        Mock external API to timeout. Verify backend returns error within 10 seconds. Verify error format is standardized.
      </test>
      <test ac="AC2" idea="Unit test: CORS-free verification">
        Verify all MIKEY tool functions call backend endpoints (localhost:3002), not external APIs directly. Verify no CORS errors in console.
      </test>
      <test ac="AC2" idea="Integration test: Standardized error handling">
        Test error scenarios: network failure, API timeout, invalid query. Verify all errors return standardized format: { success: false, error: string, timestamp: number }.
      </test>
    </ideas>
  </tests>
</story-context>

