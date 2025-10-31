# Story 1.3: loadtest-monitor

Status: ready-for-dev

## Story

As a platform operator,
I want load test monitoring automation,
so that we can validate system performance and stability continuously.

## Acceptance Criteria

1. Generate sustained synthetic load (1000 pending conditional orders across BTC/ETH/SOL) and maintain monitor loop cadence of 1s with evaluation completing <100ms per 100 orders (p95)
2. Under load, order trigger detection latency (price cross → execution call) stays <2s p95; WebSocket execution notification emitted within <100ms after DB update
3. Oracle request rate remains within limits; no Pyth errors due to rate limiting; fallback not triggered during steady-state test
4. Circuit breaker engages on >5% oracle deviation and auto-resets on <2% without crashing the monitor; when tripped, no orders are executed
5. Metrics endpoint `/api/metrics/conditional-orders` exposes non-zero processed counts and reports p95 latency; logs contain no errors for 15-minute run
6. Test artifacts saved to `bmad/docs/reports/` with summary of throughput, latencies (p50/p95), error count, and breaker events (if any)

## Tasks / Subtasks

- [ ] Create load generator script for conditional orders (backend/scripts/loadtest-conditional-orders.ts) [AC1]
  - [ ] Seed 1000 orders split across markets and sides; randomize trigger windows
  - [ ] Cleanup routine to remove test data after run
- [ ] Add test harness to sample `/api/metrics/conditional-orders` every 5s and aggregate p50/p95 [AC2, AC5]
- [ ] Add price tick simulator (dev-only) to nudge prices across triggers safely without hitting Pyth rate limits [AC2, AC3]
- [ ] Add report writer to `bmad/docs/reports/monitor-loadtest-YYYY-MM-DD.md` [AC6]
- [ ] Document runbook (how to start/stop test, env flags) in `bmad/docs/reports/monitor-loadtest-template.md` [AC6]
- [ ] CI optional job to run reduced load (100 orders) smoke test on PRs [AC5]

## Dev Notes

- Relevant architecture patterns and constraints
- Source tree components to touch
- Testing standards summary

### Project Structure Notes

- Alignment with unified project structure (paths, modules, naming)
- Detected conflicts or variances (with rationale)

### References

- Cite all technical details with source paths and sections, e.g. [Source: docs/<file>.md#Section]
  - [Source: bmad/docs/tech-spec-epic-1.md#Workflows and Sequencing]
  - [Source: bmad/docs/architecture.md#Conventions for AI Agents (Authoritative)]

## Dev Agent Record

### Context Reference

- bmad/stories/1-3-loadtest-monitor.context.md

### Agent Model Used

GPT-5 Dev-SM

### Debug Log References

### Completion Notes List

### File List


