# Story 1.1: monitor-conditional-orders

Status: ready-for-dev

## Story

As a trader,
I want automatic monitoring and triggering of conditional orders,
so that my stop-loss/take-profit executes reliably without manual intervention.

## Acceptance Criteria

1. Monitoring loop evaluates pending conditional orders every 1s and completes each cycle <100ms per 100 orders (Spec: Monitoring Performance)
2. Stop-loss execution detected within ≤2s of price crossing trigger and executed ≤5s total; DB status updates to 'filled'; WebSocket event broadcast (Spec: AC2)
3. Take-profit execution detected within ≤2s; DB status 'filled' and event broadcast (Spec: AC3)
4. Trailing stop recalculates trigger on favorable price moves and emits trailing_stop_updated events; executes immediately on cross (Spec: AC4)
5. Circuit breaker pauses execution when oracle deviation >5% and resumes <2% (Spec: AC9)
6. Atomic evaluation prevents race conditions for same market/orders (Spec: AC8)
7. All execution paths verify on-chain via smart contract call and persist consistency to DB with retry on failure (Spec: AC10)

## Tasks / Subtasks

- [ ] Implement monitoring service (backend/src/services/orderExecutionMonitorService.ts) [AC1-AC8]
  - [ ] monitorConditionalOrders() main loop at 1s cadence with batching
  - [ ] evaluateStopLossOrder(), evaluateTakeProfitOrder(), evaluateTrailingStop()
  - [ ] executeOrder() calling smart contract execute_conditional_order()
- [ ] Integrate Pyth oracle via pythOracleService.getAllPrices() only [AC2-AC4]
- [ ] Query pending orders via databaseService (no direct Supabase) [AC1]
- [ ] Emit WebSocket events via websocket.ts (order_executed, trailing_stop_updated) [AC2-AC4]
- [ ] Add circuit breaker for >5% oracle deviation with alerts [AC5]
- [ ] Ensure atomic evaluation per market to avoid races [AC6]
- [ ] Add DB retry logic and failure alerting on persistence issues [AC7]
- [ ] Tests (backend/tests): unit for evaluators, integration for execution flow, performance for 100+ orders [AC1-AC10]

## Dev Notes

- Relevant architecture patterns and constraints
- Source tree components to touch
- Testing standards summary

### Project Structure Notes

- Alignment with unified project structure (paths, modules, naming)
- Detected conflicts or variances (with rationale)

### References

- Cite all technical details with source paths and sections, e.g. [Source: docs/<file>.md#Section]
  - [Source: bmad/docs/tech-spec-epic-1.md#Acceptance Criteria (Authoritative)]
  - [Source: bmad/docs/architecture.md#Conventions for AI Agents (Authoritative)]

## Dev Agent Record

### Context Reference

- bmad/stories/1-1-monitor-conditional-orders.context.md

### Agent Model Used

GPT-5 Dev-SM

### Debug Log References

### Completion Notes List

### File List


