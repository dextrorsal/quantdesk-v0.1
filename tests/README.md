# 🧪 QuantDesk Tests Directory

This directory contains all test files organized by type and scope.

## 📁 Directory Structure

```
tests/
├── integration/   # Integration tests
├── unit/         # Unit tests
├── e2e/          # End-to-end tests
├── performance/  # Performance tests
└── README.md     # This file
```

## 🔗 Integration Tests (`integration/`)

### Price Feed Tests
- `test-each-feed.js` - Test individual price feeds
- `test-hermes-client.js` - Hermes client integration
- `test-hermes-correct.js` - Hermes correctness tests
- `test-hermes-response.js` - Hermes response validation
- `test-hermes-rest-url.js` - Hermes REST URL tests
- `test-hermes-rest.js` - Hermes REST API tests
- `test-hermes-stream.js` - Hermes streaming tests
- `test-official-pyth.js` - Official Pyth network tests
- `test-oracle-service.js` - Oracle service integration
- `test-oracle.js` - Oracle functionality tests
- `test-single-feed.js` - Single feed testing
- `test-updated-service.js` - Updated service tests
- `test-working-service.js` - Working service validation

### Trading System Tests
- `test-advanced-orders.js` - Advanced order types
- `test-advanced-risk-management.js` - Risk management features
- `test-api-improvements.js` - API enhancement tests
- `test-backend-websocket.js` - WebSocket backend tests
- `test-cross-collateralization.js` - Cross-collateral features
- `test-frontend-price-system.js` - Frontend price system
- `test-jit-liquidity.js` - JIT liquidity tests
- `test-new-markets.js` - New market functionality
- `test-portfolio-analytics.js` - Portfolio analytics
- `test-pyth-fix.js` - Pyth network fixes

### Debug & Utility Tests
- `debug-pyth-connection.js` - Pyth connection debugging
- `scrape-drift-orderbook.js` - Drift orderbook scraping

## 🧩 Unit Tests (`unit/`)

*Unit tests will be added here for individual components*

## 🎯 End-to-End Tests (`e2e/`)

*E2E tests will be added here for complete user workflows*

## ⚡ Performance Tests (`performance/`)

*Performance tests will be added here for load and stress testing*

## 🚀 Running Tests

### All Integration Tests
```bash
# Run all integration tests
npm run test:integration

# Run specific test category
npm run test:hermes
npm run test:oracle
npm run test:trading
```

### Individual Tests
```bash
# Run specific test file
node tests/integration/test-hermes-client.js
node tests/integration/test-oracle-service.js
```

### Debug Tests
```bash
# Debug Pyth connection
node tests/integration/debug-pyth-connection.js

# Test single feed
node tests/integration/test-single-feed.js
```

## 📊 Test Categories

### 🔗 Hermes Service Tests
Tests for the Hermes price feed service:
- Client connectivity
- Response validation
- Streaming functionality
- REST API integration

### 🔮 Oracle Tests
Tests for price oracle functionality:
- Pyth network integration
- Price feed accuracy
- Service reliability
- Error handling

### 💹 Trading Tests
Tests for trading system features:
- Order management
- Risk management
- Portfolio analytics
- Market data integration

### 🌐 WebSocket Tests
Tests for real-time communication:
- Backend WebSocket server
- Frontend WebSocket client
- Message handling
- Connection management

## 🐛 Debugging Tests

### Common Issues
- **Connection failures**: Check network and API endpoints
- **Timeout errors**: Verify service availability
- **Data validation**: Check response format and content
- **Authentication**: Ensure proper API keys

### Debug Commands
```bash
# Debug Pyth connection
node tests/integration/debug-pyth-connection.js

# Test individual feed
node tests/integration/test-single-feed.js BTC/USDT

# Validate Hermes service
node tests/integration/test-hermes-client.js
```

## 📝 Adding New Tests

### Integration Test Template
```javascript
// tests/integration/test-new-feature.js
const assert = require('assert');

describe('New Feature Integration', () => {
  it('should test new functionality', async () => {
    // Test implementation
    const result = await testNewFeature();
    assert(result.success, 'Feature should work correctly');
  });
});
```

### Test Guidelines
1. **Use descriptive names** for test files and functions
2. **Include error handling** and edge cases
3. **Add proper assertions** with meaningful messages
4. **Document test purpose** in comments
5. **Follow naming convention**: `test-feature-name.js`

## 🔧 Test Configuration

### Environment Variables
```bash
# Required for tests
export API_URL="https://api.quantdesk.app"
export WS_URL="wss://api.quantdesk.app"
export SOLANA_RPC_URL="https://api.mainnet-beta.solana.com"
```

### Test Data
- Test data should be in `tests/data/` directory
- Use mock data for external API calls
- Include sample responses for validation

## 📈 Test Coverage Goals

- **Integration Tests**: 80% coverage of external integrations
- **Unit Tests**: 90% coverage of core business logic
- **E2E Tests**: 70% coverage of user workflows
- **Performance Tests**: All critical paths tested

## 🚨 Continuous Integration

Tests are automatically run on:
- Pull request creation
- Code push to main branch
- Scheduled nightly runs
- Before deployment

## 📚 Test Documentation

- **Test Strategy**: See main project README
- **API Documentation**: See `docs/api/`
- **Architecture**: See `docs/architecture/`
- **Deployment**: See `docs/deployment/`
