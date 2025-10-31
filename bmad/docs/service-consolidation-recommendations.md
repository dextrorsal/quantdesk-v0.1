# Service Consolidation Recommendations

**Date:** 2025-01-20  
**Current State:** 45 service files in `backend/src/services/`  
**Goal:** Improve cohesion and reduce complexity

## Recommended Service Groups

### 1. TradingService (consolidate 8 related files)
**Current files:**
- Order management logic
- Position management
- Liquidation handling
- Matching engine
- Advanced orders
- Cross-collateral
- Portfolio calculations
- P&L calculations

**Recommendation:** One `TradingService` with clear method grouping

### 2. PortfolioService (consolidate 5 files)
**Current files:**
- `portfolioAnalyticsService`
- `portfolioBackgroundService` 
- `portfolioCalculationService`
- `pnlCalculationService`
- `accountStateService`

**Recommendation:** One `PortfolioService` handling all portfolio operations

### 3. DatabaseService (consolidate 4 files)
**Current files:**
- `supabaseDatabase` (1,100+ lines - this is good)
- `supabaseService`
- `mcpSupabaseService`
- `optimizedDatabaseService`

**Recommendation:** Keep `supabaseDatabase.ts` as primary, remove others

### 4. InfrastructureService (consolidate 6 files)
**Current files:**
- `systemMonitor`
- `monitoringService`
- `metricsCollector`
- `grafanaMetrics`
- `performanceMonitoringService`
- `performanceTestSuite`

**Recommendation:** One `MonitoringService` handling all monitoring concerns

### 5. SecurityService (consolidate 4 files)
**Current files:**
- `securityValidationService`
- `transactionVerificationService`
- `jwtService`
- `auth.security.test`

**Recommendation:** One `SecurityService` or keep as separate auth utilities

## Keep Separate (Domain-Specific Services)
- ✅ `pythOracleService` - Keep separate (oracle concerns)
- ✅ `websocketService` - Keep separate (real-time concerns)
- ✅ `redisClient` - Keep separate (infrastructure)
- ✅ `smartContractService` - Keep separate (blockchain)
- ✅ `liquidationBot` - Keep separate (background job)
- ✅ `fundingService` - Keep separate (domain logic)
- ✅ `orderScheduler` - Keep separate (scheduling)
- ✅ `referralService` - Keep separate (business logic)

## Priority Actions

**Low Priority (Works fine as-is):**
- Current structure works, just messy
- Could stay as-is if no issues

**Medium Priority (Quality improvement):**
- Consolidate monitoring services (6 files → 1)
- Consolidate portfolio services (5 files → 1)

**High Priority (If refactoring):**
- Consolidate database services (4 files → 1)
- Consider trading service grouping for clearer boundaries

---

**Bottom Line:** 45 files is excessive but the code works. Consider consolidation ONLY if:
1. You're refactoring anyway
2. Finding related code is difficult
3. Maintenance burden is high

**Current recommendation:** Leave it as-is for now. Focus on enabling Redis and fixing documentation first.

