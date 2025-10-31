# Tech Spec — Epic 3: Mainnet Deployment Readiness

Source: PRD.md, architecture.md

## Summary
Prepare for production: security, performance, comprehensive testing, monitoring, and rollout.

## Scope
- Security audit and fixes
- Performance optimization (Redis, caching, profiling)
- Comprehensive testing (unit, integration, e2e)
- Monitoring and rollback procedures

## Architecture Notes
- Gradual rollout: devnet → limited mainnet beta → full production
- Monitoring with Grafana; alerting baseline

## Patterns & Constraints
- Backend `databaseService` only; custom error classes; tiered rate limiting
- Oracle via `pythOracleService.getAllPrices()`

## Initial Stories
- 3-1-security-audit-fixes
- 3-2-performance-optimization
- 3-3-comprehensive-testing
