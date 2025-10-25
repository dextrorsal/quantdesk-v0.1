# Testing Guide

This guide provides comprehensive instructions for testing different components of the QuantDesk system.

## Backend Testing

### Compilation Testing
```bash
# Navigate to backend directory
cd backend

# Install dependencies
pnpm install

# Compile TypeScript
pnpm run build

# Check for TypeScript errors
pnpm run type-check
```

### Development Server Testing
```bash
# Start development server
pnpm run start:dev

# Check if server is running
curl http://localhost:3002/health

# Test API endpoints
curl http://localhost:3002/api/dev/codebase-structure
curl http://localhost:3002/api/dev/market-summary
curl http://localhost:3002/api/docs/swagger
```

### Endpoint Testing
```bash
# Test markets endpoint
curl http://localhost:3002/api/markets

# Test oracle prices
curl http://localhost:3002/api/oracle/prices

# Test user endpoints
curl http://localhost:3002/api/users

# Test with authentication
curl -H "Authorization: Bearer YOUR_TOKEN" http://localhost:3002/api/users/profile
```

### Database Integration Testing
```bash
# Test database connection
curl http://localhost:3002/api/dev/codebase-structure

# Check for expected errors (non-critical for devnet)
# These errors are expected:
# - "Could not find the function public.execute_sql"
# - "Could not find the table auth_nonces"
```

### Error Handling Testing
```bash
# Test invalid endpoints
curl http://localhost:3002/api/invalid-endpoint

# Test with invalid parameters
curl http://localhost:3002/api/users/invalid-id

# Test rate limiting
for i in {1..100}; do curl http://localhost:3002/api/markets; done
```

## Smart Contract Testing

### Compilation Testing
```bash
# Navigate to contracts directory
cd contracts

# Install dependencies
pnpm install

# Build contracts
anchor build

# Check for Rust compilation errors
cargo check
```

### Local Devnet Testing
```bash
# Start local Solana validator
solana-test-validator

# Deploy contracts to local devnet
anchor deploy

# Run tests
anchor test

# Check program ID
solana program show PROGRAM_ID
```

### Contract Interaction Testing
```bash
# Test contract initialization
anchor run test-initialize

# Test position opening
anchor run test-open-position

# Test position closing
anchor run test-close-position
```

### Rust Testing
```bash
# Run Rust unit tests
cargo test

# Run integration tests
cargo test --test integration

# Run with verbose output
cargo test -- --nocapture
```

## Frontend Testing

### Development Testing
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
pnpm install

# Start development server
pnpm run dev

# Check if frontend is running
curl http://localhost:3001

# Test in browser
open http://localhost:3001
```

### Build Testing
```bash
# Build for production
pnpm run build

# Check build output
ls -la dist/

# Test production build locally
pnpm run preview
```

### Component Testing
```bash
# Run component tests
pnpm run test

# Run tests in watch mode
pnpm run test:watch

# Run tests with coverage
pnpm run test:coverage
```

## Integration Testing

### End-to-End Testing
```bash
# Start all services
npm run dev

# Test complete flow
# 1. Frontend loads
# 2. Backend API responds
# 3. Database queries work
# 4. Oracle prices fetch
# 5. Smart contract interaction
```

### API Integration Testing
```bash
# Test market data flow
curl http://localhost:3002/api/markets | jq

# Test oracle price flow
curl http://localhost:3002/api/oracle/prices | jq

# Test user authentication flow
curl -X POST http://localhost:3002/api/siws/verify \
  -H "Content-Type: application/json" \
  -d '{"message": "test", "signature": "test"}'
```

### Database Integration Testing
```bash
# Test database operations
curl http://localhost:3002/api/dev/codebase-structure | jq

# Test market data
curl http://localhost:3002/api/markets | jq

# Test user data
curl http://localhost:3002/api/users | jq
```

## Performance Testing

### Load Testing
```bash
# Test API performance
ab -n 1000 -c 10 http://localhost:3002/api/markets

# Test database performance
ab -n 1000 -c 10 http://localhost:3002/api/dev/codebase-structure

# Test oracle performance
ab -n 1000 -c 10 http://localhost:3002/api/oracle/prices
```

### Memory Testing
```bash
# Monitor memory usage
ps aux | grep node

# Check for memory leaks
node --inspect backend/dist/server.js
```

## Security Testing

### Authentication Testing
```bash
# Test without authentication
curl http://localhost:3002/api/users/profile

# Test with invalid token
curl -H "Authorization: Bearer invalid-token" http://localhost:3002/api/users/profile

# Test with valid token
curl -H "Authorization: Bearer valid-token" http://localhost:3002/api/users/profile
```

### Input Validation Testing
```bash
# Test SQL injection
curl "http://localhost:3002/api/users?name='; DROP TABLE users; --"

# Test XSS
curl "http://localhost:3002/api/users?name=<script>alert('xss')</script>"

# Test parameter validation
curl "http://localhost:3002/api/users/invalid-id"
```

## Troubleshooting Common Issues

### Port Conflicts
```bash
# Check what's using port 3002
lsof -ti:3002

# Kill process if needed
lsof -ti:3002 | xargs kill -9

# Check all ports
netstat -tulpn | grep :300
```

### Database Connection Issues
```bash
# Check Supabase connection
curl http://localhost:3002/api/dev/codebase-structure

# Check environment variables
echo $SUPABASE_URL
echo $SUPABASE_ANON_KEY

# Test database directly
psql $SUPABASE_URL
```

### Smart Contract Issues
```bash
# Check Solana CLI
solana --version

# Check Anchor CLI
anchor --version

# Check local validator
solana-test-validator --help

# Check program deployment
solana program show PROGRAM_ID
```

### Frontend Issues
```bash
# Check Node.js version
node --version

# Check pnpm version
pnpm --version

# Clear cache
pnpm store prune

# Reinstall dependencies
rm -rf node_modules
pnpm install
```

## Testing Checklist

### Backend Testing Checklist
- [ ] TypeScript compilation succeeds
- [ ] Server starts without errors
- [ ] All API endpoints respond
- [ ] Database queries work
- [ ] Oracle prices fetch successfully
- [ ] Error handling works correctly
- [ ] Rate limiting functions properly
- [ ] Authentication works
- [ ] Input validation prevents attacks

### Smart Contract Testing Checklist
- [ ] Rust compilation succeeds
- [ ] Anchor build succeeds
- [ ] Contracts deploy to devnet
- [ ] All instructions work correctly
- [ ] Events emit properly
- [ ] Error handling works
- [ ] Account validation functions
- [ ] Integration tests pass

### Frontend Testing Checklist
- [ ] React components render
- [ ] API calls work
- [ ] State management functions
- [ ] Routing works
- [ ] Styling applies correctly
- [ ] Build succeeds
- [ ] Production build works
- [ ] Performance is acceptable

### Integration Testing Checklist
- [ ] All services start together
- [ ] Frontend connects to backend
- [ ] Backend connects to database
- [ ] Backend fetches oracle prices
- [ ] Smart contracts interact correctly
- [ ] End-to-end flow works
- [ ] Error handling is consistent
- [ ] Performance meets requirements

## Continuous Integration Testing

### GitHub Actions Testing
```yaml
# .github/workflows/test.yml
name: Test Suite
on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-node@v2
        with:
          node-version: '20'
      - run: pnpm install
      - run: pnpm run test:backend
      - run: pnpm run test:frontend
      - run: pnpm run test:contracts
```

### Local CI Testing
```bash
# Run all tests locally
npm run test:all

# Run specific test suites
npm run test:backend
npm run test:frontend
npm run test:contracts
```

This testing guide should help you ensure all components of the QuantDesk system work correctly and reliably.
