# Tech Spec — Epic 5: Advanced Analytics and Reporting

Source: PRD.md, architecture.md

## Summary
Deliver professional analytics: trade history, performance attribution, funding optimization, and advanced charting.

## Scope
- Trade history analytics
- Performance attribution
- Funding rate optimization
- Advanced charting

## Architecture Notes
- Backend aggregation endpoints; caching
- Frontend charting integration

## Patterns & Constraints
- Use backend `databaseService` only; custom error classes; tiered rate limiting
- Oracle via `pythOracleService.getAllPrices()` where relevant

## Initial Stories
- 5-1-trade-history-analytics (to be drafted)
- 5-2-performance-attribution (to be drafted)
- 5-3-advanced-charting (to be drafted)
