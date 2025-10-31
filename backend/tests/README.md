# Order Placement Test Suite

## Overview
This document outlines the comprehensive test suite for Story 1.2: Fix Order Placement and Execution. The test suite covers all aspects of the order placement flow from frontend to backend to smart contract execution.

## Test Structure

### 1. Unit Tests
- **File**: `backend/tests/unit/smart-contract-service.test.ts`
- **Purpose**: Test individual components in isolation
- **Coverage**:
  - Smart contract service initialization
  - Order execution logic
  - Position creation logic
  - PDA derivation
  - Error handling
  - Health checks

- **File**: `backend/tests/unit/order-authorization.test.ts`
- **Purpose**: Test order authorization and validation logic
- **Coverage**:
  - Order validation logic (1.2-UNIT-001)
  - Order parameter sanitization (1.2-UNIT-002)
  - Order execution logic validation (1.2-UNIT-003)
  - Smart contract instruction validation (1.2-UNIT-004)
  - Order status state machine (1.2-UNIT-005)
  - Position creation logic validation (1.2-UNIT-006)
  - Error message generation logic (1.2-UNIT-007)
  - Order authorization validation (1.2-UNIT-008)
  - Position creation atomicity validation (1.2-UNIT-009)

### 2. Integration Tests
- **File**: `backend/tests/integration/order-api.test.ts`
- **Purpose**: Test API endpoints with mocked dependencies
- **Coverage**:
  - Order placement API endpoint
  - Request validation
  - Error response handling
  - Authentication integration
  - Response formatting

- **File**: `backend/tests/integration/order-position-flow.test.ts`
- **Purpose**: Test complete order to position flow
- **Coverage**:
  - Order placement through matching service
  - Position creation
  - WebSocket broadcasting
  - Smart contract integration
  - Error scenarios

- **File**: `backend/tests/integration/order-authorization-integration.test.ts`
- **Purpose**: Test order authorization and execution integration
- **Coverage**:
  - Backend order placement service (1.2-INT-001)
  - Database order persistence (1.2-INT-002)
  - Backend-smart contract communication (1.2-INT-003)
  - Order execution with Oracle price feed (1.2-INT-004)
  - Atomic transaction execution (1.2-INT-005)
  - Order status synchronization across systems (1.2-INT-006)
  - WebSocket order status updates (1.2-INT-007)
  - Order-to-position creation flow (1.2-INT-008)
  - Position persistence in database (1.2-INT-009)
  - Error propagation across systems (1.2-INT-010)
  - Backend-smart contract communication failure recovery (1.2-INT-011)
  - Unauthorized order execution prevention (1.2-INT-012)
  - Position creation failure recovery (1.2-INT-013)

### 3. End-to-End Tests
- **File**: `backend/tests/e2e/order-position-flow.test.ts`
- **Purpose**: Test complete flow with real services
- **Coverage**:
  - WebSocket connection
  - Order placement
  - Real-time updates
  - Position creation
  - Error handling

- **File**: `backend/tests/e2e/order-execution-e2e.test.ts`
- **Purpose**: Test complete order execution scenarios
- **Coverage**:
  - User places market order successfully (1.2-E2E-001)
  - Order executes when price conditions met (1.2-E2E-002)
  - User sees real-time order status updates (1.2-E2E-003)
  - Position created after order fill (1.2-E2E-004)
  - User sees clear error messages (1.2-E2E-005)
  - Order execution with smart contract failure (1.2-E2E-006)

## Test Scenarios

### Happy Path Scenarios
1. **Market Order Placement**
   - Place market order
   - Verify order creation
   - Verify smart contract execution
   - Verify position creation
   - Verify WebSocket updates

2. **Limit Order Placement**
   - Place limit order with valid price
   - Verify order creation
   - Verify order remains pending
   - Verify WebSocket updates

3. **Order Filling**
   - Place order with matching counterparty
   - Verify order fills
   - Verify position updates
   - Verify WebSocket broadcasts

### Error Scenarios
1. **Validation Errors**
   - Missing required fields
   - Invalid size (negative, zero)
   - Invalid price for limit orders
   - Invalid leverage (out of range)
   - Invalid side or order type

2. **Service Errors**
   - Oracle price unavailable
   - Smart contract execution failure
   - Database connection issues
   - WebSocket connection issues

3. **Business Logic Errors**
   - Insufficient balance
   - Market not found
   - Position creation failure
   - Health factor calculation errors

## Test Data

### Mock Data
- **User ID**: `test-user-123`
- **Market ID**: `market-123`
- **Order ID**: `order-123`
- **Position ID**: `position-123`
- **Transaction Signature**: `tx-signature-123`

### Test Orders
```typescript
const testOrders = {
  marketOrder: {
    symbol: 'BTC/USD',
    side: 'buy',
    size: 0.001,
    orderType: 'market',
    leverage: 1
  },
  limitOrder: {
    symbol: 'BTC/USD',
    side: 'sell',
    size: 0.002,
    orderType: 'limit',
    price: 51000,
    leverage: 2
  }
};
```

## Test Coverage Validation

### Coverage Validation Script
- **File**: `backend/tests/validate-coverage.js`
- **Purpose**: Validate test coverage against requirements traceability matrix
- **Usage**: `node backend/tests/validate-coverage.js`

### Coverage Requirements
- **Total Tests Required**: 28 tests
- **Unit Tests**: 9 tests (P0: 5, P1: 4, P2: 0)
- **Integration Tests**: 13 tests (P0: 8, P1: 5, P2: 0)
- **E2E Tests**: 6 tests (P0: 4, P1: 2, P2: 0)

### Coverage Targets
- **Overall Coverage**: 80%+ (28/28 tests implemented)
- **P0 (Critical) Tests**: 100% (17/17 tests implemented)
- **P1 (High) Tests**: 100% (11/11 tests implemented)
- **P2 (Medium) Tests**: N/A (0 tests required)

### Test Implementation Status
✅ **All required tests are implemented**
- Unit tests: 9/9 implemented
- Integration tests: 13/13 implemented  
- E2E tests: 6/6 implemented
- Total coverage: 100%

## Running Tests

### Run All Tests
```bash
cd backend
npm test
```

### Run Specific Test Suites
```bash
# Unit tests only
npm test -- tests/unit/

# Integration tests only
npm test -- tests/integration/

# E2E tests only
npm test -- tests/e2e/
```

### Run Tests in Watch Mode
```bash
npm run test:watch
```

### Run Tests with Coverage
```bash
npm run test:coverage
```

## Test Configuration

### Vitest Configuration
- **File**: `backend/vitest.config.ts`
- **Features**:
  - TypeScript support
  - Mock support
  - Coverage reporting
  - Test environment setup

### Test Environment
- **Node.js**: 20.x
- **Test Framework**: Vitest
- **Mocking**: Vitest mocks
- **Assertions**: Vitest expect
- **Coverage**: c8

## Test Coverage Goals

### Unit Tests
- **Target**: 90%+ coverage
- **Focus**: Business logic, error handling, edge cases

### Integration Tests
- **Target**: 80%+ coverage
- **Focus**: API endpoints, service interactions

### E2E Tests
- **Target**: Critical path coverage
- **Focus**: User workflows, real-time features

## Continuous Integration

### Pre-commit Hooks
- Run unit tests
- Run linting
- Type checking

### CI Pipeline
1. Install dependencies
2. Run linting
3. Run type checking
4. Run unit tests
5. Run integration tests
6. Run E2E tests (if applicable)
7. Generate coverage report

## Test Maintenance

### Adding New Tests
1. Follow existing naming conventions
2. Use descriptive test names
3. Include both happy path and error scenarios
4. Mock external dependencies
5. Clean up after tests

### Updating Tests
1. Update tests when changing business logic
2. Maintain test data consistency
3. Update mocks when interfaces change
4. Review coverage after changes

## Troubleshooting

### Common Issues
1. **Mock not working**: Check mock setup and imports
2. **Async test failures**: Ensure proper async/await usage
3. **Timeout errors**: Increase timeout for slow operations
4. **Environment issues**: Check test environment setup

### Debug Tips
1. Use `console.log` for debugging
2. Check mock implementations
3. Verify test data
4. Review error messages carefully

## Future Enhancements

### Planned Improvements
1. **Performance Tests**: Add load testing for order placement
2. **Security Tests**: Add security-focused test scenarios
3. **Contract Tests**: Add smart contract specific tests
4. **Visual Tests**: Add frontend visual regression tests

### Test Automation
1. **Scheduled Tests**: Run tests on schedule
2. **Performance Monitoring**: Track test execution times
3. **Flaky Test Detection**: Identify and fix unstable tests
4. **Test Data Management**: Improve test data handling