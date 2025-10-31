# Tech Spec — Epic 2: Enhance AI Assistant Intelligence

Source: PRD.md, architecture.md

## Summary
Enhance AI analysis, portfolio risk, and trade suggestion intelligence across MIKEY-AI and frontend surfaces.

## Scope
- Advanced technical analysis
- Portfolio risk calculation and warnings
- Improved trade suggestions
- Contextual intelligence and user pattern learning

## Architecture Notes
- MIKEY-AI service (Port 3000) with LangChain pipelines
- Backend API gateway (Port 3002) contracts for analytics endpoints
- Data ingestion dependencies and caching patterns

## Patterns & Constraints
- Use backend `databaseService` only; custom error classes; tiered rate limiting
- Backend-centric oracle integration via `pythOracleService.getAllPrices()`

## Initial Stories
- 2-1-enhanced-technical-analysis
- 2-2-portfolio-risk-calculation
- 2-3-improved-trade-suggestions
