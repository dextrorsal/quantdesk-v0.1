# Story 1.2: redis-enable

Status: ready-for-dev

## Story

As a platform operator,
I want Redis caching enabled and configured,
so that backend performance and throughput meet production targets.

## Acceptance Criteria

1. Redis enabled via REDIS_URL env; connection established at backend startup with health check endpoint returning OK when connected
2. Cache-aside implemented for oracle price endpoint path using Redis TTL=1s; p95 latency for GET /api/markets/price <500ms (Architecture: Oracle + Caching)
3. WebSocket pub/sub integrated with Redis for order updates; events delivered <100ms after DB update (Architecture: Real-time)
4. Configuration is toggleable (enable/disable caching) without code changes; sensible defaults in non-prod
5. Observability: basic metrics for cache hit rate and Redis availability; error logs on failures
6. Security: no sensitive data stored in Redis; only cacheable market/summary data

## Tasks / Subtasks

- [ ] Add Redis client initialization and config loader (backend/src/services/redisClient.ts) [AC1]
- [ ] Implement cache wrapper utilities and integrate in oracle route/service [AC2]
- [ ] Wire Redis pub/sub with websocket.ts for order update broadcasts [AC3]
- [ ] Add feature flags/env toggles for caching on/off and TTL [AC4]
- [ ] Add metrics counters and health check endpoint (e.g., GET /api/health/redis) [AC5]
- [ ] Ensure only non-sensitive data cached; review data classes [AC6]
- [ ] Tests: unit for cache wrapper, integration latency tests for price endpoint, WebSocket pub/sub smoke tests [AC2-AC3]

## Dev Notes

- Relevant architecture patterns and constraints
- Source tree components to touch
- Testing standards summary

### Project Structure Notes

- Alignment with unified project structure (paths, modules, naming)
- Detected conflicts or variances (with rationale)

### References

- Cite all technical details with source paths and sections, e.g. [Source: docs/<file>.md#Section]
  - [Source: bmad/docs/architecture.md#Architecture Overview]
  - [Source: bmad/docs/tech-spec-epic-1.md#Dependencies and Integrations]

## Dev Agent Record

### Context Reference

- bmad/stories/1-2-redis-enable.context.md

### Agent Model Used

GPT-5 Dev-SM

### Debug Log References

### Completion Notes List

### File List


