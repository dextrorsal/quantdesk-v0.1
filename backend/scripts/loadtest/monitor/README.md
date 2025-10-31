# QuantDesk Conditional Order Monitor - Load Test

Purpose: Validate performance, stability, and correctness of the Conditional Order Monitor at scale.

## Scenarios
- Trend: steady up/down
- Spike: sudden jump/drop
- Gap: large gap then revert

## Targets
- Scale: 10,000 active conditional orders
- Cadence: p95 ≤ 1200ms, p99 ≤ 1500ms
- Trigger latency: p95 ≤ 300ms
- Idempotency: 0 duplicate executions/loop

## How to Run

```bash
# From project root
# Quick test (1000 orders, 10 seconds)
npx ts-node backend/scripts/loadtest/monitor/run.ts --orders 1000 --scenario spike --duration 10000

# Full test (10k orders, 60 seconds)
npx ts-node backend/scripts/loadtest/monitor/run.ts --orders 10000 --scenario trend --duration 60000

# Gap scenario test
npx ts-node backend/scripts/loadtest/monitor/run.ts --orders 10000 --scenario gap --duration 60000
```

**Arguments:**
- `--orders`: Number of orders to generate (default: 10000)
- `--scenario`: Price scenario - `trend`, `spike`, or `gap` (default: `trend`)
- `--duration`: Test duration in milliseconds (default: 60000 = 60s)
- `--startPrice`: Starting price for simulation (default: 100)

**Output:**
- Console: Formatted metrics and summary
- JSON file: Detailed results saved to `backend/scripts/loadtest/monitor/output/results-{timestamp}.json`

## Outputs
- JSON metrics in backend/scripts/loadtest/monitor/output/
- Markdown report in bmad/docs/reports/
