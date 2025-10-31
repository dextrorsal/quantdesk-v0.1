# Story: Conditional Order Monitor Service

## Summary
Implement a backend monitoring loop that evaluates conditional orders (stop-loss, take-profit, trailing stop, bracket) against current prices and executes when triggers are met.

## Context
- Architecture specifies a 1s cadence monitor with Redis caching and WebSocket notifications.
- Database schema uses `orders` table with fields for conditional configuration.

## Scope
- Backend service `ConditionalOrderMonitor` executed on server startup.
- Trigger evaluation for all supported conditional types.
- Execution path integrates with existing order execution routes/program calls.

## Acceptance Criteria
1. Monitor cadence
   - Given the service is running, it evaluates active conditional orders every 1s (±200ms) under nominal load.
2. Trigger correctness
   - For stop-loss and take-profit orders, when Pyth normalized price crosses the trigger, the corresponding execution is initiated exactly once.
3. Trailing stop updates
   - When market moves favorably, the trailing stop trigger updates according to configured trailing percentage.
4. Bracket lifecycle
   - After entry fill, stop-loss/take-profit child conditions are armed; exactly one child closes the bracket.
5. Idempotency
   - Reprocessing the same order within 1 loop iteration does not produce duplicate executions.
6. Persistence
   - Order status transitions are written to the database with timestamps; errors are logged with correlation IDs.
7. Observability
   - Metrics: processed_orders/sec, triggers_fired/sec, execution_latency_ms (p95), errors/sec
   - Logs include order_id, market, trigger_type, trigger_price, current_price.
8. Notifications
   - WebSocket broadcast notifies clients on status change and execution completion.
9. Performance
   - Handles 10,000 active conditional orders at 1s cadence without missing cycles on a standard dev machine profile.

## Non-Goals
- UI changes beyond status updates
- New order types beyond listed conditional types

## Implementation Notes
- Service name: `backend/src/services/conditionalOrderMonitor.ts`
- Uses `pythOracleService.getAllPrices()` and `databaseService` only.
- Protect with a single instance guard to avoid duplicate loops.

## Tests
- Unit tests for trigger evaluation logic (edge cases: exact price, gap jumps)
- Integration test simulating price movements and verifying DB state and WS notifications
- Load test plan addressed in separate load-testing story
