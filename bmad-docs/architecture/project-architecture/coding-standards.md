# Coding Standards

## Core Standards

- **Languages & Runtimes:** TypeScript 5.3.3+, Node.js 20.11.0+
- **Style & Linting:** ESLint with TypeScript rules, Prettier for formatting
- **Test Organization:** Jest for unit tests, Supertest for API tests

## Critical Rules

- **Never use console.log in production code** - Use structured logger
- **All API responses must use ApiResponse wrapper type** - Consistent response format
- **Database queries must use repository pattern** - Never direct ORM calls
- **All external API calls must have timeout and retry logic** - Resilience patterns
- **Wallet addresses must be validated using Solana address validation** - Security requirement
- **All trading operations must be logged with correlation IDs** - Audit trail requirement
