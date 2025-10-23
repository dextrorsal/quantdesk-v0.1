# Epic 2: Harden Order Flow with Pyth Oracle, Supabase RLS, and Fluent DB APIs - Brownfield Enhancement

## Epic Goal

Harden the existing QuantDesk perpetual DEX order flow architecture by replacing custom SQL patterns with secure Supabase fluent APIs, implementing reliable Pyth oracle integration, and fixing authentication mapping issues to enable production deployment.

## Epic Description

### Existing System Context

- **Current relevant functionality**: Epic 1 completed with basic order placement, position management, and P&L calculation working on devnet
- **Technology stack**: Node.js/Express backend, React frontend, Supabase PostgreSQL, Anchor smart contracts, Redis caching
- **Integration points**: Backend services (`matching`, `pythOracleService`, `supabaseService`), authentication middleware, database RLS policies, smart contract oracle integration

### Enhancement Details

- **What's being added/changed**: 
  - Replace all `execute_sql` custom RPC calls with Supabase fluent APIs
  - Implement Pyth-first oracle with in-memory caching (remove CoinGecko fallback)
  - Fix JWT to RLS mapping for proper `user_id` resolution
  - Resolve Pyth SDK dependency conflicts in Anchor smart contracts
  - Add deterministic matching and idempotency for on-chain execution

- **How it integrates**: 
  - Maintains existing API contracts and database schema
  - Enhances existing services without breaking current functionality
  - Preserves Epic 1 test script compatibility
  - Uses existing Supabase RLS framework with corrected policies

- **Success criteria**: 
  - Epic 1 test script passes completely
  - Zero `execute_sql` references in codebase
  - Market orders execute with < 60s price staleness
  - RLS policies prevent unauthorized data access
  - Anchor smart contracts compile and deploy successfully

## Stories

1. **Story 2.1: Oracle Switchboard Implementation** - Replace CoinGecko fallback with Pyth Hermes integration, add in-memory caching, implement price staleness monitoring
2. **Story 2.2: Database Security Hardening** - Remove all `execute_sql` calls, implement Supabase fluent APIs, fix RLS policies to use `auth.uid()`
3. **Story 2.3: Authentication and Smart Contract Fixes** - Fix JWT to RLS mapping, resolve Pyth SDK conflicts, add deterministic matching with idempotency

## Compatibility Requirements

- [x] Existing APIs remain unchanged
- [x] Database schema changes are backward compatible (no new columns required)
- [x] UI changes follow existing patterns (no frontend changes needed)
- [x] Performance impact is minimal (caching improves performance)

## Risk Mitigation

- **Primary Risk**: Breaking existing Epic 1 functionality during database migration
- **Mitigation**: Maintain Epic 1 test script as regression test, implement changes incrementally with feature flags
- **Rollback Plan**: Revert to `execute_sql` function if fluent API migration causes issues, maintain CoinGecko fallback during transition

## Definition of Done

- [ ] All stories completed with acceptance criteria met
- [ ] Epic 1 test script passes completely (existing functionality verified)
- [ ] Integration points working correctly (oracle, database, authentication)
- [ ] Documentation updated appropriately (conversation summary reflects changes)
- [ ] No regression in existing features (all Epic 1 functionality intact)

## Technical Implementation Notes

### Key Files to Modify
- `backend/src/services/pythOracleService.ts` - Oracle switchboard implementation
- `backend/src/services/supabaseService.ts` - Fluent API migration
- `backend/src/middleware/auth.ts` - JWT to RLS mapping fixes
- `contracts/programs/quantdesk-perp-dex/Cargo.toml` - Pyth SDK dependency resolution
- All database query locations - Replace `execute_sql` with fluent APIs

### Dependencies
- Pyth Network Hermes API access
- Supabase service role configuration
- Existing Epic 1 test script for validation

### Testing Strategy
- Epic 1 test script as primary regression test
- Unit tests for oracle caching and price staleness
- Integration tests for RLS policy enforcement
- Security tests for SQL injection prevention

---

**Epic Created**: $(date)  
**Type**: Brownfield Enhancement  
**Estimated Duration**: 2-3 weeks  
**Priority**: High (Production Blocking)  
**Status**: Ready for Story Development
