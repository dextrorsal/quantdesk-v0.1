# 🛠️ Troubleshooting Guide

*What is this doc?*  
This guide covers common issues and solutions for running the ML-MODEL system, including database, API, and data quality problems. It's for anyone running, debugging, or maintaining the system.

[Neon Pipeline](NEON_PIPELINE.md) | [Model Training](MODEL_TRAINING.md) | [Project README](../README.md)

## Database Issues

### PostgreSQL Permission Errors

**Symptom:** Errors like `permission denied for schema public` or `permission denied for database`

**Solution:**
1. Grant superuser privileges to your database user:
```sql
ALTER USER your_username WITH SUPERUSER;
```

2. Grant privileges to the public schema:
```sql
GRANT ALL PRIVILEGES ON SCHEMA public TO your_username;
```

3. Check database ownership and permissions:
```sql
\l  -- List databases and owners
\dn -- List schemas
```

### Connection Issues

**Symptom:** `could not connect to server` or `connection refused`

**Solution:**
1. Check if PostgreSQL service is running:
```bash
sudo systemctl status postgresql
```

2. Verify connection settings in your configuration file
3. Check host, port, username, and password
4. For Neon DB, verify your API token is valid and has appropriate permissions

## Binance API Issues

### Rate Limiting

**Symptom:** HTTP status code 429 or error messages about exceeding rate limits

**Solution:**
1. Implement exponential backoff for API calls
2. Reduce frequency of requests
3. Use WebSockets instead of REST API for real-time data when possible
4. Consider upgrading API key limits if available

### Connection Drops

**Symptom:** Frequent disconnections from the Binance API

**Solution:**
1. Check your network stability
2. If using a VPN, try a different server location
3. Implement proper error handling and reconnection logic
4. Log detailed error messages to identify patterns

## WebSocket Issues

### Connection Failures

**Symptom:** Failed to establish WebSocket connection or frequent disconnections

**Solution:**
1. Verify network allows WebSocket traffic (some corporate firewalls block it)
2. Implement proper ping/pong handling (required by Binance):
```python
def on_ping(self, wsapp, message):
    """Respond to ping with same payload in pong frame"""
    wsapp.send(message, websocket.ABNF.OPCODE_PONG)
```

3. Log connection events and errors at DEBUG level for troubleshooting:
```python
logging.getLogger('websocket').setLevel(logging.DEBUG)
```

### Data Streaming Stops

**Symptom:** Connected but not receiving new data

**Solution:**
1. Implement a heartbeat check to verify connection is alive
2. Automatically reconnect if no messages received within a timeframe
3. Ensure you're responding to ping frames correctly
4. Verify subscription to the correct streams:
```python
# Example correct subscription message
ws.send(json.dumps({
    "method": "SUBSCRIBE",
    "params": ["solusdt@kline_1m"],
    "id": 1
}))
```

## Python Library Issues

### Missing Dependencies

**Symptom:** `ModuleNotFoundError` or `ImportError`

**Solution:**
1. Install required packages:
```bash
pip install -r requirements.txt
```

2. For specific data collection tests:
```bash
pip install -r tests/data/requirements-data-tests.txt
```

3. Check for version conflicts with:
```bash
pip check
```

### Asyncio Issues

**Symptom:** Event loop errors or coroutine never finishes

**Solution:**
1. Use the correct async/await patterns
2. Avoid mixing synchronous and asynchronous code
3. Properly close resources and connections:
```python
try:
    # Your async code
finally:
    await connection.close()
```

## Data Quality Issues

### Missing Data Points

**Symptom:** Gaps in time series data or inconsistent intervals

**Solution:**
1. Implement data validation checks
2. Add logic to detect and retry for missing data
3. Consider using cache to handle temporary outages
4. Add monitoring for data quality metrics

### Invalid Prices

**Symptom:** Extreme price values or zeros in data

**Solution:**
1. Implement validation checks (e.g., price within expected range)
2. Add outlier detection and handling
3. Compare data from multiple sources when possible

## Getting Help

If you're still experiencing issues after trying these solutions:

1. Check the project's GitHub issues for similar problems
2. Enable detailed logging and share logs when asking for help
3. Include specific error messages and context

## Contributing Solutions

Found a solution to a common problem? Please contribute to this guide by:

1. Opening a pull request with your solution
2. Adding detailed step-by-step instructions
3. Including code examples where appropriate 

## See Also
- [Project README](../README.md) — Project overview and structure
- [Neon Data Pipeline](NEON_PIPELINE.md) — Database and data flow details
- [Model Training Guide](MODEL_TRAINING.md) — Training and evaluation troubleshooting
- [src/data/pipeline/neon_collector.py](../src/data/pipeline/neon_collector.py) — Data collector code
- [scripts/train_model.py](../scripts/train_model.py) — Training script
- [scripts/dashboard/model_dashboard.py](../scripts/dashboard/model_dashboard.py) — Dashboard troubleshooting 