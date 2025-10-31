# Testing Reality

## Current Test Coverage

- **Unit Tests**: Minimal coverage, mostly in smart contracts
- **Integration Tests**: Basic API testing in backend
- **E2E Tests**: None implemented
- **Smart Contract Tests**: Comprehensive test suite in `contracts/tests/`
- **Manual Testing**: Primary QA method for trading functionality

## Running Tests

```bash
# Smart contract tests
cd contracts && anchor test

# Backend tests
cd backend && pnpm test

# Frontend tests
cd frontend && pnpm test
```
