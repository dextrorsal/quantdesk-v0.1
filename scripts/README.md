# RPC Load Balancer Testing

This directory contains comprehensive testing tools for your RPC load balancer system.

## 🧪 Testing Scripts

### 1. **test-rpc-performance.js**
Main testing script that performs comprehensive RPC performance analysis.

**Features:**
- ✅ RPC response time measurement
- ✅ Load balancer stress testing  
- ✅ Rate limit detection
- ✅ Provider health monitoring
- ✅ Performance metrics collection

### 2. **test-rpc.sh**
Shell script wrapper for easy testing.

## 🚀 Usage

### Quick Test
```bash
# Run comprehensive tests
./scripts/test-rpc.sh

# Run stress test (30 seconds, 10 req/s)
./scripts/test-rpc.sh --stress

# Custom stress test (60 seconds, 20 req/s)
./scripts/test-rpc.sh --stress --duration 60 --rps 20
```

### Direct Node.js Usage
```bash
# Basic tests
node scripts/test-rpc-performance.js

# Stress test only
node scripts/test-rpc-performance.js --stress --duration 30 --rps 10
```

### API Endpoints
```bash
# Run comprehensive tests via API
curl -X POST http://localhost:3002/api/rpc-test/test

# Run stress test via API
curl -X POST http://localhost:3002/api/rpc-test/stress-test \
  -H "Content-Type: application/json" \
  -d '{"duration": 30, "rps": 10}'

# Get current metrics
curl http://localhost:3002/api/rpc-test/metrics
```

## 📊 What Gets Tested

### 1. **RPC Speed Tests**
- `getLatestBlockhash()` - Basic RPC call
- `getCurrentSlot()` - Slot information
- `getBalance()` - Account balance query

**Metrics Collected:**
- Average response time
- Min/Max response times
- Success rate
- Error analysis

### 2. **Load Balancer Tests**
- Concurrent request handling
- Provider selection algorithm
- Load distribution across providers
- Failover mechanisms

### 3. **Rate Limit Detection**
- Rapid request testing (50 requests)
- Rate limit error detection
- Provider switching verification
- Error categorization

### 4. **Stress Testing**
- High-frequency requests
- Sustained load testing
- Performance under pressure
- Rate limit behavior

## 📈 Expected Results

### **Good Performance Indicators:**
- ✅ Average response time < 200ms
- ✅ Success rate > 95%
- ✅ No rate limit errors
- ✅ All providers healthy
- ✅ Load distributed evenly

### **Typical RPC Speeds:**
- **Excellent**: < 100ms average
- **Good**: 100-200ms average  
- **Acceptable**: 200-500ms average
- **Poor**: > 500ms average

### **Rate Limit Behavior:**
- **Free tiers**: 10-100 requests/second
- **Paid tiers**: 100-1000+ requests/second
- **Enterprise**: 1000+ requests/second

## 🔧 Configuration

Make sure your `.env` file has all RPC URLs set:

```bash
# Your providers (Set 1)
HELIUS_RPC_1_URL=https://your-helius-url.com/api-key/YOUR_KEY
QUICKNODE_1_RPC_URL=https://your-quicknode-url.com/api-key/YOUR_KEY
ALCHEMY_1_RPC_URL=https://your-alchemy-url.com/api-key/YOUR_KEY
SYNDICA_1_RPC_URL=https://your-syndica-url.com/api-key/YOUR_KEY
CHAINSTACK_1_RPC_URL=https://your-chainstack-url.com/api-key/YOUR_KEY

# Partner providers (Set 2)
HELIUS_RPC_2_URL=https://partner-helius-url.com/api-key/PARTNER_KEY
QUICKNODE_2_RPC_URL=https://partner-quicknode-url.com/api-key/PARTNER_KEY
ALCHEMY_2_RPC_URL=https://partner-alchemy-url.com/api-key/PARTNER_KEY
SYNDICA_2_RPC_URL=https://partner-syndica-url.com/api-key/PARTNER_KEY
CHAINSTACK_2_RPC_URL=https://partner-chainstack-url.com/api-key/PARTNER_KEY
```

## 🎯 Testing Scenarios

### **Scenario 1: Basic Performance**
```bash
./scripts/test-rpc.sh
```
Tests basic RPC operations and load balancer functionality.

### **Scenario 2: Rate Limit Testing**
```bash
./scripts/test-rpc.sh --stress --duration 10 --rps 50
```
Sends 50 requests/second for 10 seconds to test rate limits.

### **Scenario 3: Sustained Load**
```bash
./scripts/test-rpc.sh --stress --duration 300 --rps 5
```
5 requests/second for 5 minutes to test sustained performance.

### **Scenario 4: High Frequency**
```bash
./scripts/test-rpc.sh --stress --duration 60 --rps 100
```
100 requests/second for 1 minute to test high-frequency limits.

## 📋 Troubleshooting

### **Common Issues:**

1. **"Backend not running"**
   ```bash
   cd backend && ./start-backend.sh
   ```

2. **"No RPC URLs found"**
   - Check your `.env` file
   - Verify environment variables are loaded

3. **All providers unhealthy**
   - Check network connectivity
   - Verify RPC URLs are correct
   - Check provider status pages

4. **High response times**
   - Consider upgrading to paid RPC providers
   - Check geographic location of providers
   - Monitor network conditions

### **Debug Commands:**
```bash
# Check RPC health
curl http://localhost:3002/api/rpc/health

# Get provider stats
curl http://localhost:3002/api/rpc/stats

# Get detailed metrics
curl http://localhost:3002/api/rpc-test/metrics
```

## 🎉 Success Criteria

Your RPC load balancer is working well if you see:

- ✅ **8 providers** all healthy
- ✅ **Response times** < 200ms average
- ✅ **Success rate** > 95%
- ✅ **Load distribution** across multiple providers
- ✅ **Rate limit protection** working
- ✅ **Automatic failover** functioning

This indicates your system is ready for high-frequency trading operations! 🚀