 # Story: Load Testing for Conditional Order Monitor

## Summary
Design and execute a load test for the Conditional Order Monitor to validate performance, stability, and correctness under scale.

## Context
- Monitor processes active conditional orders every 1s, targets 10k active orders.
- Performance targets and indexing guidance defined in architecture.

## Scope
- Synthetic data generation for active conditional orders.
- Price stream simulation (Pyth-like) with controlled movements.
- Measurement of cadence adherence, trigger latency, and resource usage.

## Acceptance Criteria
1. Scale Baseline
   - Generate 10k active conditional orders with realistic distribution (stop/TP/trailing/bracket).
2. Cadence Adherence
   - p95 loop cadence ≤ 1,200ms; p99 ≤ 1,500ms during 10k load.
3. Trigger Latency
   - p95 execution latency (trigger detection → order execution initiation) ≤ 300ms.
4. Correctness Under Load
   - 0 duplicate executions per order; idempotency preserved.
5. Resource Profile
   - CPU and memory within acceptable ranges; no sustained memory leak over 30 minutes.
6. Reporting
   - Produce a report with graphs/tables: cadence histogram, latency percentiles, CPU/RAM over time, error rates.
7. Artifacts
   - Test scripts in `backend/scripts/loadtest/monitor/` with README to reproduce locally.

## Non-Goals
- Cluster/distributed load; single-node evaluation is sufficient for baseline.

## Implementation Notes
- Use a local price simulator that steps through programmed scenarios (trend, spike, gap).
- Consider using k6 or Node-based script with timers; export Prometheus metrics.

## Exit Criteria
- All ACs met; report saved to `bmad/docs/reports/monitor-loadtest-{date}.md`.
