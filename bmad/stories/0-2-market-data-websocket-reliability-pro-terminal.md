# Story 0-2: Market Data WebSocket Reliability in Pro Terminal

Status: ready-for-dev

## Story

As a user, I need price updates via WebSocket with retry and safe fallback so prices remain live.

## Acceptance Criteria

1. WebSocket connects via env URL or localhost:3002/ws; exponential backoff max 15s; errors throttled
2. On WS failure, polling fallback engages every 2s without console spam
3. A visual "Live" indicator reflects WS/polling status

## Tasks / Subtasks

- [ ] Implement WebSocket connection with retry logic [AC1]
  - [ ] Connect via env URL or localhost:3002/ws
  - [ ] Exponential backoff retry (max 15s between retries)
  - [ ] Error throttling to prevent console spam
  - [ ] Connection state management (connecting, connected, disconnected, error)
- [ ] Implement polling fallback [AC2]
  - [ ] Detect WebSocket failure after max retries
  - [ ] Start polling `/api/oracle/prices` every 2s
  - [ ] Suppress console errors/spam during polling
  - [ ] Seamlessly switch back to WebSocket when available
- [ ] Add visual "Live" indicator [AC3]
  - [ ] Show "Live" when WebSocket connected
  - [ ] Show "Polling" when in fallback mode
  - [ ] Show "Disconnected" when neither working
  - [ ] Use appropriate color coding (green/amber/red)

## Dev Notes

### Architecture Alignment
- WebSocket service location: `frontend/src/services/websocketService.ts`
- Price endpoint: `/api/oracle/prices` (already implemented)
- Price context provider: `frontend/src/contexts/PriceContext.tsx` (may need updates)
- Price store: `frontend/src/stores/PriceStore.ts` (fallback polling already implemented)
- UI component: Add indicator to Pro Terminal header or price display area

### References
- [Source: bmad/docs/tech-spec-epic-0.md#WebSocket Reliability]
- [Source: frontend/src/pro/index.tsx - Pro Terminal implementation]
- [Source: backend/src/routes/oracle.ts - `/api/oracle/prices` endpoint]

## Dev Agent Record

### Context Reference

- bmad/stories/0-2-market-data-websocket-reliability-pro-terminal.context.md

### Agent Model Used

### Debug Log References

### Completion Notes List

### File List

