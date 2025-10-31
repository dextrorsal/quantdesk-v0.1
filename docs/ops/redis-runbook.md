# Redis Operations Runbook - QuantDesk

**Last Updated:** 2025-01-29  
**Redis Version:** 7.x  
**Purpose:** Production operations guide for Redis caching and pub/sub

---

## Table of Contents

1. [Overview](#overview)
2. [Configuration](#configuration)
3. [Setup & Deployment](#setup--deployment)
4. [Health Checks](#health-checks)
5. [Common Operations](#common-operations)
6. [Monitoring & Metrics](#monitoring--metrics)
7. [Troubleshooting](#troubleshooting)
8. [Failover & Recovery](#failover--recovery)
9. [Security](#security)
10. [Alert Thresholds](#alert-thresholds)

---

## Overview

QuantDesk uses Redis for:
- **Caching:** Price data (1s TTL), Portfolio data (5s TTL), Order status (2s TTL)
- **Pub/Sub:** WebSocket message broadcasting for horizontal scaling
- **Sessions:** User session storage
- **Rate Limiting:** API rate limiting counters

### Architecture

- **Cache Strategy:** Cache-aside pattern
- **Pub/Sub:** Redis Pub/Sub for multi-instance WebSocket scaling
- **TTLs:** Configurable per data type (see Configuration)

---

## Configuration

### Environment Variables

```bash
# Required
REDIS_URL=redis://localhost:6379

# Optional
REDIS_TLS=true                    # Enable TLS for production
REDIS_PASSWORD=your-password       # Redis password (if required)
REDIS_CONNECT_TIMEOUT=5000         # Connection timeout (ms)
REDIS_COMMAND_TIMEOUT=5000         # Command timeout (ms)
ENV_NAME=production                # Environment name for key namespacing
```

### Redis TTL Configuration

Cache TTLs are configured in `backend/src/services/redisCache.ts`:

- **Prices:** 1 second
- **Portfolio:** 5 seconds
- **Order Status:** 2 seconds
- **Sessions:** 3600 seconds (1 hour)
- **Presence:** 65 seconds

---

## Setup & Deployment

### Local Development

```bash
# Using Docker
docker run -d --name redis-quantdesk -p 6379:6379 redis:7-alpine

# Or using Redis installed locally
redis-server
```

**Note:** Redis is optional in development - application will continue without Redis if not configured.

### Production Deployment

#### Option 1: Managed Redis (Recommended)

**Recommended providers:**
- Upstash Redis (serverless, pay-per-use)
- Redis Cloud (managed Redis)
- AWS ElastiCache (if on AWS)

#### Option 2: Self-Hosted

1. **Install Redis 7:**
   ```bash
   # Ubuntu/Debian
   sudo apt-get install redis-server
   
   # Or compile from source
   wget https://download.redis.io/redis-stable.tar.gz
   tar xzf redis-stable.tar.gz
   cd redis-stable
   make
   ```

2. **Configure Redis:**
   ```bash
   # Edit /etc/redis/redis.conf
   bind 127.0.0.1  # or your server IP
   requirepass your-secure-password
   maxmemory 512mb
   maxmemory-policy allkeys-lru
   ```

3. **Start Redis:**
   ```bash
   sudo systemctl start redis
   sudo systemctl enable redis
   ```

### Application Configuration

Set environment variables:

```bash
export REDIS_URL=redis://your-redis-host:6379
export REDIS_PASSWORD=your-password  # If required
export NODE_ENV=production
export ENV_NAME=production
```

---

## Health Checks

### Application Health Endpoints

**Main Health Check:**
```bash
curl http://localhost:3002/health
```

Response includes Redis status:
```json
{
  "status": "healthy",
  "redis": {
    "status": "healthy",
    "latency": 2,
    "available": true
  }
}
```

**Redis-Specific Health:**
```bash
curl http://localhost:3002/api/redis/health
```

**Response Codes:**
- `200`: Redis healthy
- `503`: Redis unhealthy or unavailable

### Test Commands

**Ping Redis:**
```bash
redis-cli -u $REDIS_URL PING
# Should return: PONG
```

**Check Connection:**
```bash
redis-cli -u $REDIS_URL INFO server
```

**List Keys:**
```bash
redis-cli -u $REDIS_URL KEYS "qd:production:*"
```

---

## Common Operations

### Cache Management

**View Cache Statistics:**
```bash
curl http://localhost:3002/api/redis/stats
```

**Clear Cache:**
```bash
# Clear all cache (use with caution!)
curl -X POST http://localhost:3002/api/redis/clear \
  -H "Content-Type: application/json" \
  -d '{"pattern": "*"}'

# Clear specific pattern
curl -X POST http://localhost:3002/api/redis/clear \
  -H "Content-Type: application/json" \
  -d '{"pattern": "price:*"}'
```

**Reset Statistics:**
```bash
curl -X POST http://localhost:3002/api/redis/reset-stats
```

### Redis CLI Operations

**Monitor Commands in Real-Time:**
```bash
redis-cli -u $REDIS_URL MONITOR
```

**Get Cache Hit Ratio:**
```bash
# Via API stats endpoint
curl http://localhost:3002/api/redis/stats | jq '.data.cache.hitRatio'
```

**Flush All Data (Emergency Only!):**
```bash
redis-cli -u $REDIS_URL FLUSHALL
# ⚠️ WARNING: This deletes ALL Redis data!
```

---

## Monitoring & Metrics

### Available Metrics

**Cache Metrics (via `/api/redis/stats`):**
- `hits`: Cache hits
- `misses`: Cache misses
- `errors`: Cache errors
- `sets`: Cache sets
- `hitRatio`: Hit ratio percentage
- `totalRequests`: Total requests

**Redis Metrics:**
- `status`: healthy/unhealthy/disabled
- `latency`: Ping latency (ms)
- `available`: Connection status

### Prometheus Metrics

Redis metrics exposed (via Grafana integration):
- `redis_ping_ms`: Redis ping latency
- `redis_connection_status`: 1 = connected, 0 = disconnected
- `redis_cache_hits`: Total cache hits
- `redis_cache_misses`: Total cache misses
- `redis_cache_hit_ratio`: Hit ratio (0-1)

### Dashboard Queries

**Redis Latency:**
```
redis_ping_ms
```

**Cache Hit Ratio:**
```
redis_cache_hit_ratio
```

**Cache Operations:**
```
rate(redis_cache_hits[5m])
rate(redis_cache_misses[5m])
```

---

## Troubleshooting

### Common Issues

#### Issue: Redis Connection Failed

**Symptoms:**
- Logs show: `Redis connection failed, continuing without Redis`
- Health endpoint shows Redis unavailable

**Diagnosis:**
```bash
# Check Redis is running
redis-cli -u $REDIS_URL PING

# Check network connectivity
telnet redis-host 6379

# Check Redis logs
tail -f /var/log/redis/redis-server.log
```

**Solutions:**
1. Verify `REDIS_URL` is correct
2. Check firewall rules allow Redis port (6379)
3. Verify Redis is running: `sudo systemctl status redis`
4. Check authentication if password-protected

#### Issue: High Cache Miss Rate

**Symptoms:**
- `hitRatio` below 50%
- High database query load

**Diagnosis:**
```bash
# Check cache statistics
curl http://localhost:3002/api/redis/stats | jq '.data.cache'
```

**Solutions:**
1. Increase cache TTLs if data doesn't change frequently
2. Check cache key patterns - ensure consistent key naming
3. Verify cache is being used (check application logs)
4. Review cache invalidation strategy

#### Issue: Memory Usage High

**Symptoms:**
- Redis memory usage approaching `maxmemory`
- Cache eviction errors

**Diagnosis:**
```bash
redis-cli -u $REDIS_URL INFO memory
```

**Solutions:**
1. Increase Redis `maxmemory` setting
2. Review and optimize `maxmemory-policy` (currently: `allkeys-lru`)
3. Clear old cache entries: `POST /api/redis/clear`
4. Review TTL settings - ensure keys expire

#### Issue: Slow Redis Operations

**Symptoms:**
- High latency in health checks
- Slow cache operations

**Diagnosis:**
```bash
# Check Redis performance
redis-cli -u $REDIS_URL --latency

# Check slow commands
redis-cli -u $REDIS_URL SLOWLOG GET 10
```

**Solutions:**
1. Check network latency between app and Redis
2. Monitor Redis CPU usage
3. Review slow log for problematic commands
4. Consider Redis connection pooling

---

## Failover & Recovery

### Redis Down Scenario

**Behavior:**
- Application continues operating without Redis
- Cache operations gracefully fail (returns null)
- Pub/Sub operations skip gracefully
- Health endpoint reports Redis as "unhealthy"

**Recovery Steps:**

1. **Restore Redis:**
   ```bash
   # Start Redis
   sudo systemctl start redis
   
   # Or restart Docker container
   docker restart redis-quantdesk
   ```

2. **Verify Connection:**
   ```bash
   redis-cli -u $REDIS_URL PING
   ```

3. **Monitor Application:**
   - Watch logs for "Redis connected" message
   - Check `/health` endpoint
   - Verify cache operations resume

4. **Warm Cache (Optional):**
   - Application will rebuild cache naturally
   - Or manually trigger operations that populate cache

### Data Loss Recovery

**Redis Data Persistence:**
- Configure `save` directives in `redis.conf` for persistence
- Or use Redis AOF (Append-Only File) for durability

**Recovery:**
1. Restore from backup if available
2. Application will rebuild cache from database
3. Cache is ephemeral - data can be regenerated

---

## Security

### Best Practices

1. **Use Strong Passwords:**
   ```bash
   requirepass strong-random-password-here
   ```

2. **Bind to Specific IP:**
   ```bash
   bind 127.0.0.1 your-internal-ip
   ```

3. **Enable TLS (Production):**
   ```bash
   # In redis.conf
   tls-port 6380
   tls-cert-file /path/to/cert.pem
   tls-key-file /path/to/key.pem
   ```

4. **Limit Network Access:**
   - Use firewall rules to restrict Redis port
   - Only allow access from application servers

5. **Secure Credentials:**
   - Never commit passwords to repository
   - Use environment variables or secret management
   - Rotate passwords regularly

### Environment Variable Security

```bash
# ✅ Good: Use environment variables
export REDIS_URL=redis://:password@redis-host:6379

# ❌ Bad: Hardcode in code
const redisUrl = "redis://:password@redis-host:6379";
```

---

## Alert Thresholds

### Recommended Alerts

**Redis Down:**
- Alert if `/api/redis/health` returns status != "healthy" for > 1 minute

**High Latency:**
- Alert if `redis_ping_ms` > 100ms for > 5 minutes

**High Error Rate:**
- Alert if `redis_cache_errors` rate > 10/min for > 5 minutes

**Low Cache Hit Ratio:**
- Alert if `redis_cache_hit_ratio` < 30% for > 15 minutes

**Memory Warning:**
- Alert if Redis memory usage > 80% of maxmemory

### Grafana Dashboard

Monitor Redis in Grafana dashboard:
- Redis latency graph
- Cache hit/miss rates
- Memory usage
- Connection status

---

## Additional Resources

- **Redis Documentation:** https://redis.io/documentation
- **Application Code:**
  - Cache Service: `backend/src/services/redisCache.ts`
  - Redis Client: `backend/src/services/redisClient.ts`
  - Health Endpoint: `backend/src/routes/redis.ts`
- **Configuration:** `backend/src/config/environment.ts`

---

**Last Reviewed:** 2025-01-29  
**Maintained By:** QuantDesk DevOps Team

