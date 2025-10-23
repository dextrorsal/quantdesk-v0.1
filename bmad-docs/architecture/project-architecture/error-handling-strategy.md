# Error Handling Strategy

## General Approach

- **Error Model:** Structured error responses with error codes and user-friendly messages
- **Exception Hierarchy:** Custom error classes extending base Error with specific error types
- **Error Propagation:** Errors bubble up through service layers with context preservation

## Logging Standards

- **Library:** Winston Logger with structured JSON logging
- **Format:** JSON with correlation IDs and service context
- **Levels:** ERROR, WARN, INFO, DEBUG
- **Required Context:**
  - Correlation ID: UUID for request tracing
  - Service Context: Service name, version, environment
  - User Context: User ID, wallet address (when available)

## Error Handling Patterns

### External API Errors

- **Retry Policy:** Exponential backoff with jitter (3 attempts)
- **Circuit Breaker:** Open circuit after 5 consecutive failures
- **Timeout Configuration:** 30s for Pyth API, 10s for Solana RPC
- **Error Translation:** Map external errors to internal error codes

### Business Logic Errors

- **Custom Exceptions:** TradingError, ValidationError, AuthenticationError
- **User-Facing Errors:** Clear, actionable error messages
- **Error Codes:** Structured error codes (TRADING_001, VALIDATION_002)

### Data Consistency

- **Transaction Strategy:** Database transactions for multi-table operations
- **Compensation Logic:** Rollback mechanisms for failed operations
- **Idempotency:** Request IDs for duplicate operation prevention
