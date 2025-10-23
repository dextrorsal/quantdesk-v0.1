# Epic 2: Harden Order Flow with Pyth Oracle, Supabase RLS, and Fluent DB APIs

## Epic Overview

**Epic ID:** EP2  
**Title:** Harden Order Flow with Pyth Oracle, Supabase RLS, and Fluent DB APIs  
**Type:** Brownfield (Backend-First Architecture Hardening)  
**Priority:** High  
**Owner:** Backend Team  
**Estimated Duration:** 3 weeks  
**Dependencies:** Epic 1 completion, Supabase setup, Pyth Network access  

## Problem Statement

Following Epic 1 completion, our Solana perpetual DEX platform has critical architectural vulnerabilities that prevent reliable production deployment:

1. **Oracle Price Reliability**: Market orders fail due to price gaps and inconsistent oracle data sources
2. **Database Security**: Custom SQL RPC (`execute_sql`) creates SQL injection vulnerabilities and breaks RLS policies
3. **Authentication Mapping**: JWT to RLS mapping is inconsistent, causing NULL `user_id` inserts and security gaps
4. **Smart Contract Blocking**: Pyth SDK dependency conflicts prevent Anchor compilation
5. **Performance Issues**: Slow database queries and missing caching layers impact user experience

## Business Impact

- **Risk**: Production deployment blocked due to security vulnerabilities
- **User Experience**: Order failures and inconsistent pricing damage platform credibility
- **Technical Debt**: Custom SQL patterns create maintenance burden and security risks
- **Scalability**: Current architecture cannot handle production trading volumes

## Epic Goals

### Primary Goals
1. **Reliable Oracle Integration**: Implement Pyth-first price feeds with in-memory caching
2. **Secure Database Access**: Replace custom SQL with Supabase fluent APIs and proper RLS
3. **Robust Authentication**: Fix JWT to RLS mapping for secure user data access
4. **Smart Contract Unblocking**: Resolve Pyth SDK conflicts for Anchor compilation
5. **Performance Optimization**: Add caching layers and optimize database queries

### Success Metrics
- Market orders execute successfully 99.9% of the time
- Oracle price staleness < 60 seconds
- Database query performance P50 < 200ms
- Zero SQL injection vulnerabilities
- Anchor smart contracts compile and deploy successfully

## Scope

### In Scope
- **Backend Services**: `pythOracleService`, `matching`, `orders`, `positions`, `pnl`, `markets`, `websocket`, `auth`
- **Database Layer**: Supabase RLS policies, fluent API migration, performance optimization
- **Smart Contracts**: Pyth integration, manual deserialization, Anchor compilation fixes
- **Testing**: Comprehensive test coverage for oracle, RLS, and matching logic
- **Monitoring**: Structured logging, performance metrics, error tracking

### Out of Scope
- UI/UX changes or new features
- New market types or trading instruments
- Funding model redesign
- Cross-chain functionality
- Mobile application updates

## Work Breakdown Structure

### Story 2.1: Oracle Switchboard Implementation
**Priority:** Critical  
**Estimated Effort:** 5 days  
**Dependencies:** Pyth Network access, Hermes API keys

**Objectives:**
- Implement Pyth-first price fetching with Hermes integration
- Add in-memory caching layer with configurable TTL
- Remove CoinGecko fallback dependencies
- Add price staleness monitoring and alerts

**Key Deliverables:**
- Enhanced `pythOracleService` with cache-first architecture
- Price freshness monitoring dashboard
- Comprehensive oracle integration tests

### Story 2.2: Database Security Hardening
**Priority:** Critical  
**Estimated Effort:** 4 days  
**Dependencies:** Supabase service role configuration

**Objectives:**
- Remove all `execute_sql` custom RPC calls
- Implement Supabase fluent APIs throughout backend
- Fix RLS policies to use `auth.uid()` consistently
- Add slow query monitoring and optimization

**Key Deliverables:**
- Fluent API migration for all database operations
- RLS policy audit and standardization
- Performance monitoring and alerting

### Story 2.3: Authentication and User Mapping
**Priority:** High  
**Estimated Effort:** 3 days  
**Dependencies:** Database security hardening

**Objectives:**
- Fix JWT to RLS mapping inconsistencies
- Ensure all database writes include proper `user_id`
- Implement secure user context propagation
- Add authentication flow testing

**Key Deliverables:**
- Enhanced auth middleware with proper user resolution
- User context propagation throughout backend
- Authentication security testing suite

### Story 2.4: Smart Contract Pyth Integration
**Priority:** High  
**Estimated Effort:** 4 days  
**Dependencies:** Oracle switchboard implementation

**Objectives:**
- Resolve Pyth SDK dependency conflicts
- Implement manual Pyth deserialization
- Add price staleness and confidence validation
- Enable Anchor compilation and deployment

**Key Deliverables:**
- Working Anchor smart contracts with Pyth integration
- Manual deserialization implementation
- Smart contract testing framework

### Story 2.5: Deterministic Matching and Idempotency
**Priority:** Medium  
**Estimated Effort:** 3 days  
**Dependencies:** Database security hardening, authentication fixes

**Objectives:**
- Implement deterministic order matching logic
- Add idempotency keys for on-chain execution
- Ensure transaction finality before state updates
- Add comprehensive matching tests

**Key Deliverables:**
- Deterministic matching algorithm
- Idempotency framework for on-chain operations
- Matching logic test suite

### Story 2.6: Performance and Monitoring
**Priority:** Medium  
**Estimated Effort:** 2 days  
**Dependencies:** All previous stories

**Objectives:**
- Add structured logging throughout backend
- Implement performance monitoring
- Create observability dashboards
- Add error tracking and alerting

**Key Deliverables:**
- Structured logging implementation
- Performance monitoring dashboard
- Error tracking and alerting system

## Technical Architecture

### Oracle Architecture
```
Price Request → Cache Check → Pyth Hermes → Database Fallback
     ↓              ↓            ↓              ↓
  < 50ms        < 200ms      < 500ms        < 1000ms
```

### Database Security Model
```
JWT Token → Auth Middleware → User Resolution → RLS Policy → Database Access
     ↓              ↓              ↓              ↓              ↓
  wallet_pubkey → users.id → auth.uid() → Policy Check → Fluent API
```

### Smart Contract Integration
```
Backend → Manual Pyth Read → Staleness Check → Confidence Validation → On-Chain Execution
```

## Risk Assessment

### High Risk
- **Pyth Network Outage**: Mitigation through cache + database fallback
- **RLS Policy Breaking**: Mitigation through comprehensive testing and gradual rollout
- **Smart Contract Deployment**: Mitigation through extensive devnet testing

### Medium Risk
- **Performance Degradation**: Mitigation through monitoring and optimization
- **Data Consistency**: Mitigation through atomic transactions and idempotency

### Low Risk
- **Integration Complexity**: Mitigation through incremental implementation
- **Testing Coverage**: Mitigation through comprehensive test suite

## Definition of Done

### Technical Criteria
- [ ] All `execute_sql` calls removed from backend
- [ ] Pyth-first oracle implementation with < 60s staleness
- [ ] RLS policies use `auth.uid()` consistently
- [ ] Anchor smart contracts compile and deploy successfully
- [ ] All database writes include proper `user_id`
- [ ] Comprehensive test coverage (>90%) for all stories

### Performance Criteria
- [ ] Oracle price fetch P50 < 200ms
- [ ] Database query P50 < 200ms
- [ ] Market order execution success rate > 99.9%
- [ ] Zero SQL injection vulnerabilities

### Quality Criteria
- [ ] Code review completed for all changes
- [ ] Security audit passed
- [ ] Performance benchmarks met
- [ ] Documentation updated
- [ ] Epic 1 test script passes completely

## Milestones

### Week 1: Foundation
- **Day 1-2**: Oracle switchboard implementation
- **Day 3-4**: Database security hardening
- **Day 5**: Authentication and user mapping

### Week 2: Integration
- **Day 1-3**: Smart contract Pyth integration
- **Day 4-5**: Deterministic matching and idempotency

### Week 3: Polish and Testing
- **Day 1-2**: Performance and monitoring
- **Day 3-5**: Testing, documentation, and deployment preparation

## Dependencies

### External Dependencies
- Pyth Network Hermes API access
- Supabase service role configuration
- Solana devnet/mainnet access

### Internal Dependencies
- Epic 1 completion and stabilization
- Backend team availability
- QA team for testing support

## Success Criteria

### Immediate Success
- Epic 1 test script passes completely
- No `execute_sql` references in codebase
- Pyth oracle provides reliable price feeds
- RLS policies prevent unauthorized access

### Long-term Success
- Production-ready order flow architecture
- Scalable oracle integration
- Secure database access patterns
- Maintainable smart contract codebase

## Developer Handoff Notes

### Prerequisites
- Node.js 20+ with pnpm
- Supabase CLI and service role access
- Solana CLI and Anchor framework
- Pyth Network API keys

### Key Files to Modify
- `backend/src/services/pythOracleService.ts`
- `backend/src/services/supabaseService.ts`
- `backend/src/middleware/auth.ts`
- `contracts/programs/quantdesk-perp-dex/src/oracle.rs`
- All database query locations

### Testing Strategy
- Unit tests for each service
- Integration tests for oracle flow
- End-to-end tests with Epic 1 script
- Security tests for RLS policies
- Performance tests for query optimization

### Deployment Considerations
- Gradual rollout with feature flags
- Database migration scripts
- Environment variable updates
- Monitoring and alerting setup

---

**Epic Created:** $(date)  
**Status:** Ready for Development  
**Next Action:** Begin Story 2.1 (Oracle Switchboard Implementation)
