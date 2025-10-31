# Story 0-7: MIKEY Tools Fetch Live Prices, Tweets, and News via Backend

Status: ready-for-dev

## Story

As an AI operator, I need MIKEY tools to use backend endpoints for reliable data.

## Acceptance Criteria

1. Tools: `get_live_price(asset)`, `get_market_summary`, `get_tweets(query)`, `get_news(query)` return current data with timestamps
2. CORS-free: tools call backend proxy endpoints; timeouts and error messages standardized

## Tasks / Subtasks

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

## Dev Notes

### Architecture Alignment
- MIKEY-AI tools location: `MIKEY-AI/src/services/RealDataTools.ts`
- Existing `createPythPriceTool()` already calls `/api/oracle/prices` (lines 33-80)
- Backend routes: `backend/src/routes/` (create new routes as needed)
- Avoid CORS by routing all external API calls through backend

### References
- [Source: bmad/docs/tech-spec-epic-0.md#MIKEY Tools Backend Integration]
- [Source: MIKEY-AI/src/services/RealDataTools.ts - existing tool implementations]
- [Source: backend/src/routes/ - route patterns]

## Dev Agent Record

### Context Reference

- bmad/stories/0-7-mikey-tools-fetch-live-prices-tweets-news-backend.context.md

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

