# Environment Variables Guide for QuantDesk

## Overview

This document describes the environment variables used in the QuantDesk project and provides guidelines for managing them effectively.

## Environment Files

The project uses multiple environment files for different purposes:

- `.env` - Main environment file (not committed to git)
- `env.example` - Template file with example values
- `data-ingestion/.env` - Data ingestion service specific variables
- `data-ingestion/env.example` - Data ingestion template

## Naming Conventions

### Standard Format
- Use UPPERCASE letters
- Use underscores to separate words
- Prefix with service name when applicable

### Examples
```bash
# ✅ Good
QUANTDESK_API_URL=https://api.quantdesk.com
DATABASE_URL=postgresql://user:pass@localhost:5432/quantdesk
REDIS_URL=redis://localhost:6379

# ❌ Bad
quantdesk_api_url=https://api.quantdesk.com
DATABASE-URL=postgresql://user:pass@localhost:5432/quantdesk
```

## Required Variables

### Core Application
- `DATABASE_URL` - PostgreSQL connection string
- `REDIS_URL` - Redis connection string
- `JWT_SECRET` - Secret key for JWT tokens
- `NODE_ENV` - Environment (development, production)

### Solana Integration
- `SOLANA_NETWORK` - Network (devnet, mainnet-beta)
- `RPC_URL` - Solana RPC endpoint
- `PYTH_NETWORK_URL` - Pyth network endpoint

### External Services
- `SUPABASE_URL` - Supabase project URL
- `SUPABASE_SERVICE_ROLE_KEY` - Supabase service role key

## Security Best Practices

1. **Never commit sensitive values** to version control
2. **Use strong, unique secrets** for JWT_SECRET
3. **Rotate secrets regularly** in production
4. **Use environment-specific values** for different deployments
5. **Validate all environment variables** at startup

## Validation Tools

The project includes several tools for managing environment variables:

### 1. Environment Scanner
```bash
npm run env:scan
```
Scans for duplicates, similar names, and format issues.

### 2. Environment Validator
```bash
npm run env:validate
```
Validates environment variables and checks for common issues.

### 3. dotenv-linter
```bash
npm run env:lint
```
Lints .env files for formatting and best practices.

### 4. Combined Check
```bash
npm run env:check
```
Runs all environment variable checks.

## Common Issues

### Duplicate Variables
Having the same variable in multiple files can cause confusion:
```bash
# ❌ Problem
# .env
QUANTDESK_URL=https://app.quantdesk.com

# data-ingestion/.env
QUANTDESK_URL=https://api.quantdesk.com
```

### Similar Names
Variables with similar names can cause confusion:
```bash
# ❌ Problem
QUANTDESK_URL=https://app.quantdesk.com
QUANTDESK_API_URL=https://api.quantdesk.com
```

### Inconsistent Naming
Different naming conventions can cause issues:
```bash
# ❌ Problem
DATABASE_URL=postgresql://...
DB_CONNECTION_STRING=postgresql://...
```

## Troubleshooting

### Variable Not Found
If an environment variable is not found:
1. Check if it's defined in the correct .env file
2. Verify the variable name matches exactly
3. Ensure the .env file is in the correct location
4. Restart the application after making changes

### Duplicate Variables
If you have duplicate variables:
1. Use the environment scanner to identify them
2. Choose one canonical location for each variable
3. Remove duplicates from other files
4. Update code to reference the canonical variable

### Similar Names
If you have similar variable names:
1. Use the environment scanner to identify them
2. Choose a consistent naming convention
3. Rename variables to follow the convention
4. Update code to use the new names

## Best Practices

1. **Use consistent naming** across all environment files
2. **Document all variables** in this file
3. **Validate variables** at application startup
4. **Use type validation** for numeric and boolean values
5. **Provide default values** where appropriate
6. **Use environment-specific files** for different deployments

## Tools and Scripts

- `scripts/env-scanner.sh` - Comprehensive environment variable scanner
- `scripts/validate-env.js` - Node.js environment variable validator
- `scripts/install-env-tools.sh` - Install environment variable tools

## References

- [dotenv-linter Documentation](https://github.com/dotenv-linter/dotenv-linter)
- [Node.js Environment Variables](https://nodejs.org/en/learn/command-line/how-to-read-environment-variables-from-nodejs)
- [Environment Variables Best Practices](https://12factor.net/config)
