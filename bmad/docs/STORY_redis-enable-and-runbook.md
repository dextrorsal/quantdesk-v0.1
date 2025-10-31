# Story: Redis Enablement & Production Runbook

## Summary
Enable Redis for caching and pub/sub in production and provide a runbook covering configuration, health checks, and incident response.

## Context
- Architecture calls for Redis 7, cache-aside strategy, and Socket.IO pub/sub usage.
- Prior development had Redis mocked/disabled in some environments.

## Scope
- Configure Redis connection for production and staging.
- Implement health probes and dashboards.
- Document operational procedures (runbook).

## Acceptance Criteria
1. Configuration
   - Application reads Redis connection from env vars (`REDIS_URL`, timeouts, TLS options if applicable).
   - Startup logs confirm successful Redis connection and subscriber initialization.
2. Health Probes
   - `/health` reports Redis status (up/down) and latency.
   - Prometheus metrics expose redis_ping_ms and connection status.
3. Caching
   - Cache layer uses TTLs per architecture (prices 1s, portfolio 5s, order status 2s).
   - Cache hit ratio is logged/metricized; configurable thresholds for alerts.
4. Pub/Sub
   - Socket.IO (or equivalent) integrates with Redis adapter for horizontal scaling readiness.
5. Observability
   - Dashboards show Redis latency, errors, hit/miss ratio, memory and keyspace.
6. Runbook
   - Location: `docs/ops/redis-runbook.md`
   - Includes: setup, environment variables, test commands, common errors, failover steps, flush policies, alert thresholds.
7. Security
   - Credentials stored securely; no secrets in repo logs.

## Non-Goals
- Multi-region Redis deployment
- Complex sharding; use single node or managed service initially

## Tests
- Integration tests verifying cache hit/miss behavior and fallback on Redis down
- Health endpoint test verifies Redis status reporting
