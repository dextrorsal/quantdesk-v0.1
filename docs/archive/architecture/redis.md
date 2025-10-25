## Redis in QuantDesk

This document explains how Redis is used today in QuantDesk, recommended production settings, and a short implementation plan to harden reliability and security.

### Current usage
- Sessions (SIWS): `session:<walletPubkey>` → JSON session, TTL 7d
- Rate limiting: `rl:<key>:<timeBucket>` → counters with TTL equal to window
- Presence (chat): `presence:<channelId>:<walletPubkey>` → value `"1"`, TTL ~65s
- Pub/Sub: `publish(channel, message)` for chat message fanout

Code locations:
- `backend/src/services/redisClient.ts`
- `backend/src/routes/siws.ts` (sessions)
- `backend/src/routes/chat.ts` (presence, pub/sub, rate limit)
- `backend/src/middleware/auth.ts` (session check)

### Key schema (proposed namespacing)
Use an environment prefix to avoid cross‑env collisions:
- `qd:{env}:session:{wallet}`
- `qd:{env}:rl:{scope}:{bucket}`
- `qd:{env}:presence:{channel}:{wallet}`
- `qd:{env}:pubsub:{channel}` (channel name convention)

Where `{env}` ∈ `dev|staging|prod` (from `NODE_ENV` or an explicit `ENV_NAME`).

### Recommended configuration
- Persistence: enable AOF with `appendfsync everysec` for reasonable durability
- Memory control: set `maxmemory` and `maxmemory-policy allkeys-lru` for caches
- Security: require password/ACL, bind to private addresses only
- Networking: prefer bridge networking or managed Redis; avoid host networking
- Observability: export metrics (latency, used_memory, evictions, hits/misses)
- Reconnect strategy: client auto‑reconnect with jittered backoff; healthchecks

### Environment variables
```
REDIS_URL=redis://:STRONG_PASSWORD@localhost:6379
ENV_NAME=dev
```

### docker-compose (local dev example)
```yaml
version: '3.8'
services:
  redis:
    image: redis:7-alpine
    command: ["redis-server", "--appendonly", "yes", "--appendfsync", "everysec", "--requirepass", "${REDIS_PASSWORD:-devpassword}", "--maxmemory", "256mb", "--maxmemory-policy", "allkeys-lru"]
    ports:
      - "6379:6379"
    environment:
      - REDIS_PASSWORD=${REDIS_PASSWORD:-devpassword}
    volumes:
      - ./data/redis:/data
```

### Backend hardening tasks
1) Reconnect/healthcheck
- On startup, call `connectRedis()` and expose `/health/redis` that does a `PING`.
- Log and retry with backoff on disconnect events.

2) Namespacing helpers
- Centralize key building: `makeKey(...segments)` → `qd:${ENV_NAME}:${segments.join(':')}`
- Apply to sessions, rate limits, presence, pub/sub.

3) Metrics
- Add admin endpoint (or Prometheus metrics) to expose Redis stats and keyspace hits/misses.

### Operational checklist
- [ ] AOF enabled with `everysec`
- [ ] `maxmemory` and `allkeys-lru` set
- [ ] Password/ACL required; not publicly exposed
- [ ] Healthcheck endpoint green in CI and prod
- [ ] Namespaced keys per environment
- [ ] Dashboards/alerts on memory, evictions, latency

### Notes
Redis is an in‑memory system; keep Postgres as the source of truth. Use Redis for transient/real‑time concerns: sessions, presence, rate limits, pub/sub, and fast caches.


