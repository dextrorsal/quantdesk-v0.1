# Epic Technical Specification: Restore Trading E2E and Data Parity

Date: 2025-10-30
Author: Dex
Epic ID: 0
Status: Draft

---

## Overview

This epic restores end-to-end trading functionality and ensures data parity across all QuantDesk services. The goal is to enable users to deposit, see live prices, place trades, and withdraw seamlessly while ensuring MIKEY tools operate on live data and UI theming is consistent across the platform.

The epic addresses critical gaps in the current implementation:
- Oracle price API reliability and normalization
- WebSocket price update reliability in Pro Terminal
- Smart contract deposit/withdraw instruction exposure
- Quote Monitor (QM) live price display
- MIKEY Monitor UI consistency
- MIKEY tools backend integration
- Comprehensive E2E testing for deposit/withdraw flows

All implementation aligns with the existing layered monolith architecture, leveraging current infrastructure without introducing new architectural patterns.

## Objectives and Scope

### In-Scope
- **Oracle Price API Stabilization**: Reliable `/api/oracle/price/:asset` and `/api/prices` endpoints with fresh, non-zero prices
- **WebSocket Reliability**: Robust price update mechanism with retry and polling fallback
- **Smart Contract Integration**: Expose `withdraw_native_sol` in program + IDL
- **UI Guards**: Soft-disable withdraw action when instruction unavailable
- **Quote Monitor Completion**: 100% instruments showing live, non-zero prices (Story 0-5 already complete)
- **MIKEY Monitor Theming**: Consistent UI theme tokens across all cards
- **MIKEY Tools Backend Integration**: All tools fetching live data via backend endpoints
- **E2E Testing**: CLI tests and UI e2e for deposit/withdraw flows
- **Expert Validation**: Document deposit/withdraw patterns with expert guidance

### Out-of-Scope
- Mainnet rollout (epic targets devnet and demo parity)
- New order types (handled in Epic 1)
- Major UI redesigns (theming consistency only)
- New smart contract instructions beyond withdraw exposure

## System Architecture Alignment

### Services and Modules

| Service/Module | Responsibility | Owner | Inputs | Outputs |
|----------------|---------------|-------|---------|---------|
| **Backend: `pythOracleService.ts`** | Oracle price normalization and caching | Dev | Pyth feeds, cache requests | Normalized prices `Record<string, number>` |
| **Backend: `/api/oracle/prices` route** | Price aggregation endpoint | Dev | HTTP requests | JSON response with normalized prices |
| **Backend: `/api/oracle/price/:asset` route** | Single-asset price endpoint | Dev | HTTP requests | JSON response with price metadata |
| **Frontend: WebSocket service** | Real-time price updates | Dev | WebSocket connection | Price update events |
| **Frontend: Quote Monitor (QM)** | Live price display component | Dev | Backend prices, WebSocket updates | Rendered price table |
| **Frontend: MIKEY Monitor** | AI service monitoring UI | Dev | Theme tokens | Themed monitoring cards |
| **Smart Contract: `collateral_management.rs`** | Deposit/withdraw instructions | Dev | User requests, oracle prices | On-chain state updates |
| **MIKEY-AI: `RealDataTools.ts`** | Backend-proxied data tools | Dev | Tool requests | Live price/tweet/news data |
| **Testing: CLI scripts** | Deposit/withdraw automation | Dev | Test parameters | Test results, logs |
| **Testing: E2E tests** | UI flow validation | Dev | User interactions | Screenshots, test reports |

### Architecture Constraints
- **Backend-Centric Oracle**: Pyth prices fetched by backend via `pythOracleService.getAllPrices()`, normalized and cached
- **Consolidated Database Service**: Single abstraction layer (`supabaseDatabase.ts`) prevents direct Supabase usage
- **Layered Monolith Architecture**: Backend consolidates all API routes in one process for simplicity and performance
- **WebSocket Fallback**: Polling fallback engages automatically on WebSocket failure
- **Single Solana Program**: All deposit/withdraw logic in existing `quantdesk-perp-dex` program

---

## Detailed Design

### Services and Modules

#### 1. Oracle Price API Stabilization

**File:** `backend/src/routes/oracle.ts`

**Current State:**
- `/api/oracle/prices` endpoint exists and returns normalized prices via `pythOracleService.getAllPrices()`
- Returns `Record<string, number>` with cache-first architecture (1s TTL)
- Fallback to CoinGecko if Pyth fails

**Enhancements Required:**
- Add `/api/oracle/price/:asset` single-asset endpoint with proper error handling
- Add `/api/oracle/health` route exposing freshness metrics
- Structured logging for all oracle operations
- Health and freshness guardrails (returns 5xx only on irrecoverable errors; otherwise 200 + source=fallback)

**Key Functions:**
```typescript
// Single-asset endpoint
router.get('/price/:asset', async (req, res) => {
  // Normalize asset symbol
  // Get price from pythOracleService
  // Return with metadata: { price, source, confidence, updatedAt }
  // Health checks and freshness guardrails
});

// Health endpoint
router.get('/health', async (req, res) => {
  // Return freshness metrics for all assets
  // Cache status, last update times
});
```

#### 2. WebSocket Reliability

**File:** `frontend/src/services/websocket.ts` (or equivalent)

**Current State:**
- WebSocket connection exists for price updates
- Basic connection handling

**Enhancements Required:**
- Exponential backoff retry (max 15s)
- Error throttling to prevent console spam
- Automatic polling fallback (every 2s) on WebSocket failure
- Visual "Live" indicator reflecting connection status

**Key Functions:**
```typescript
// WebSocket connection with retry
const connectWebSocket = () => {
  // Connect via env URL or localhost:3002/ws
  // Exponential backoff on failure (max 15s)
  // Throttle error messages
};

// Polling fallback
const startPollingFallback = () => {
  // Poll every 2s when WebSocket fails
  // Fetch prices from /api/oracle/prices
  // No console spam
};
```

#### 3. Smart Contract Withdraw Instruction

**File:** `contracts/programs/quantdesk-perp-dex/src/instructions/collateral_management.rs`

**Current State:**
- `withdraw_native_sol` function exists in Rust code (lines 201-243)
- Account structure `WithdrawNativeSol` defined (lines 471+)

**Required Actions:**
- Verify IDL contains `withdraw_native_sol` instruction
- Ensure IDL matches account structure exactly:
  - `user_account` (PDA)
  - `user` (signer)
  - `protocol_vault` (PDA)
  - `collateral_account` (PDA)
  - `sol_usd_price_feed` (AccountInfo)
  - `system_program` (Program)
  - `rent` (optional Sysvar)

**IDL Verification:**
- Check `contracts/deployed-idl.json` for `withdraw_native_sol` instruction
- Verify account order matches Rust struct exactly

#### 4. UI Guard for Missing Withdraw

**File:** `frontend/src/components/WithdrawButton.tsx` (or equivalent)

**Enhancement Required:**
- Detect if IDL lacks `withdraw_native_sol` instruction
- Soft-disable withdraw action (button disabled with tooltip)
- Tooltip: "Upgrade in progress"
- Prevent console errors: `program.methods.withdrawNativeSol is not a function`

**Implementation:**
```typescript
const canWithdraw = program?.idl?.instructions?.some(
  (ix: any) => ix.name === 'withdrawNativeSol'
);

if (!canWithdraw) {
  // Disable button, show tooltip
  // Never call program.methods.withdrawNativeSol
}
```

#### 5. Quote Monitor (QM) Live Prices

**File:** `frontend/src/pro/index.tsx` - `QMWindowContent` component (lines 2388-2498)

**Current State:** ✅ **COMPLETE**
- QM polls `/api/oracle/prices` every 5 seconds (line 2422)
- Backend prices used as authoritative source (line 2417)
- Prices displayed correctly for PERP pairs
- Implementation working as of today (2025-10-30)

**No Further Action Required** - Story 0-5 already complete.

#### 6. MIKEY Monitor Theming

**File:** `frontend/src/components/MIKEYMonitor.tsx` (or equivalent location)

**Enhancements Required:**
- Replace all inline gray colors with theme tokens
- Use `var(--bg-tertiary)`, `var(--text-primary)`, etc.
- Ensure light/dark theme switch preserves contrast
- Remove color blending artifacts

**Implementation:**
```typescript
// Replace
style={{ backgroundColor: '#1a1a1a', color: '#ffffff' }}

// With
style={{ 
  backgroundColor: 'var(--bg-tertiary)', 
  color: 'var(--text-primary)' 
}}
```

#### 7. MIKEY Tools Backend Integration

**File:** `MIKEY-AI/src/services/RealDataTools.ts`

**Current State:**
- `createPythPriceTool()` exists and calls `/api/oracle/prices` (lines 33-80)
- Tool returns normalized prices

**Enhancements Required:**
- Ensure all tools call backend proxy endpoints (avoid CORS)
- Add tools: `get_market_summary`, `get_tweets(query)`, `get_news(query)`
- Standardize timeouts and error messages
- Return timestamps with all data

**Backend Endpoints to Create:**
- `/api/market/summary` - Aggregated market data
- `/api/tweets/:query` - Twitter/X data (backend proxy)
- `/api/news/:query` - News data (backend proxy)

#### 8. Deposit/Withdraw CLI Tests

**File:** `scripts/test-deposit-withdraw.sh` (new)

**Test Script Requirements:**
- Test `deposit_native_sol` instruction
  - Assert PDA derivations correct
  - Verify account order matches IDL
  - Check success logs
  - Verify SOL balance increases in protocol_vault
- Test `withdraw_native_sol` instruction
  - Assert PDA derivations correct
  - Verify account order matches IDL
  - Check success logs
  - Verify SOL balance decreases in protocol_vault, increases in user wallet
- Return transaction signatures for verification

#### 9. Deposit/Withdraw UI E2E

**File:** `tests/e2e/deposit-withdraw.spec.ts` (new)

**E2E Test Flow:**
1. Connect wallet
2. Deposit SOL via UI
3. Verify USD balance updates in UI
4. Withdraw SOL via UI
5. Verify balance reduces correctly
6. Capture screenshots at each step
7. Store screenshots for regression testing

#### 10. Drift Parity Review

**File:** `docs/drift-parity-analysis.md` (new)

**Analysis Required:**
- Compare account lists (QuantDesk vs Drift)
- Evaluate WSOL vs native SOL approach
- Document remaining account requirements
- Price check integration comparison
- Recommended approach selection (native/WSOL)
- Risks and mitigation strategies (rent, signer, account order)

#### 11. Expert Confirmation of Native SOL Patterns

**File:** `docs/native-sol-patterns-expert-guidance.md` (new)

**Expert Guidance to Capture:**
- Signer first in account struct (Anchor requirement)
- SystemAccount usage for vault
- When to include rent Sysvar
- `invoke` vs `invoke_signed` for PDA vault
- Anchor 0.30+ accounts/addresses behavior
- Best practices for native SOL handling

### Data Models and Contracts

#### Oracle Price Data Structure

**Backend Service:** `pythOracleService.getAllPrices()` returns:
```typescript
Record<string, number>  // e.g., { BTC: 0.00121..., ETH: 0.0000437... }
```

**API Response Format:**
```typescript
{
  success: true,
  data: {
    BTC: 0.0012116391507573001,
    ETH: 0.000043722750125000004,
    SOL: 0.0054188075671234567
  },
  timestamp: 1234567890,
  source: 'pyth-network'
}
```

**Single-Asset Response:**
```typescript
{
  success: true,
  data: {
    price: 0.0012116391507573001,
    source: 'pyth-network',
    confidence: 0.95,
    updatedAt: 1234567890
  }
}
```

#### Smart Contract Account Structure

**Deposit Native SOL:**
```rust
pub struct DepositNativeSol<'info> {
    pub user_account: Account<'info, UserAccount>,      // PDA
    pub user: Signer<'info>,                             // User signer
    pub protocol_vault: Account<'info, ProtocolSolVault>, // PDA
    pub collateral_account: Account<'info, CollateralAccount>, // PDA
    pub sol_usd_price_feed: AccountInfo<'info>,          // Pyth feed
    pub system_program: Program<'info, System>,
    pub rent: Sysvar<'info, Rent>,                        // Optional
}
```

**Withdraw Native SOL:**
```rust
pub struct WithdrawNativeSol<'info> {
    pub user_account: Account<'info, UserAccount>,       // PDA
    pub user: Signer<'info>,                             // User signer
    pub protocol_vault: Account<'info, ProtocolSolVault>, // PDA
    pub collateral_account: Account<'info, CollateralAccount>, // PDA
    pub sol_usd_price_feed: AccountInfo<'info>,          // Pyth feed
    pub system_program: Program<'info, System>,
}
```

### APIs and Interfaces

#### Backend Oracle Endpoints

**GET `/api/oracle/prices`**
- **Request:** None (query params optional for specific assets)
- **Response:**
  ```json
  {
    "success": true,
    "data": { "BTC": 0.00121, "ETH": 0.0000437 },
    "timestamp": 1234567890,
    "source": "pyth-network"
  }
  ```
- **Error Codes:** 500 (irrecoverable), 200 with source=fallback (recoverable)

**GET `/api/oracle/price/:asset`**
- **Request:** `asset` path parameter (e.g., "BTC", "SOL")
- **Response:**
  ```json
  {
    "success": true,
    "data": {
      "price": 0.0012116391507573001,
      "source": "pyth-network",
      "confidence": 0.95,
      "updatedAt": 1234567890
    }
  }
  ```
- **Error Codes:** 400 (invalid asset), 500 (irrecoverable), 200 with source=fallback (recoverable)

**GET `/api/oracle/health`**
- **Request:** None
- **Response:**
  ```json
  {
    "status": "healthy",
    "assets": {
      "BTC": { "fresh": true, "lastUpdate": 1234567890 },
      "ETH": { "fresh": true, "lastUpdate": 1234567891 }
    },
    "cache": { "enabled": true, "hits": 100, "misses": 5 }
  }
  ```

#### Smart Contract Instructions

**`deposit_native_sol(amount: u64)`**
- **Accounts:** See DepositNativeSol struct above
- **Returns:** Transaction signature
- **Side Effects:** Increases protocol_vault balance, creates/updates collateral_account

**`withdraw_native_sol(amount: u64)`**
- **Accounts:** See WithdrawNativeSol struct above
- **Returns:** Transaction signature
- **Side Effects:** Decreases protocol_vault balance, decreases collateral_account

### Workflows and Sequencing

#### Oracle Price Flow
```
1. Frontend requests price → GET /api/oracle/prices
2. Backend checks cache (1s TTL)
3. If cache hit → return cached prices
4. If cache miss → call pythOracleService.getAllPrices()
5. Pyth service fetches from Hermes client
6. Prices normalized (exponent applied)
7. Cache updated, response returned
8. Frontend receives normalized prices
9. QM component renders prices in table
```

#### WebSocket Price Update Flow
```
1. Frontend attempts WebSocket connection (localhost:3002/ws)
2. If connection succeeds:
   a. Subscribe to price updates
   b. Receive real-time price events
   c. Update UI with "Live" indicator
3. If connection fails:
   a. Exponential backoff retry (max 15s)
   b. After max retries → start polling fallback
   c. Poll /api/oracle/prices every 2s
   d. Update UI with "Polling" indicator
```

#### Deposit/Withdraw Flow
```
1. User initiates deposit via UI
2. Frontend calls backend API (if needed for pre-validation)
3. Frontend constructs transaction with program IDL
4. User signs transaction
5. Transaction sent to Solana network
6. Smart contract executes deposit_native_sol
7. Protocol vault balance increases
8. Collateral account updated
9. Frontend receives confirmation
10. UI updates to reflect new balance

Withdraw follows similar flow with withdraw_native_sol instruction
```

## Non-Functional Requirements

### Performance

**Oracle API Performance:**
- `/api/oracle/prices` p50 latency: <250ms
- `/api/oracle/price/:asset` p50 latency: <250ms
- Cache hit rate target: >90%
- Price freshness: <3 seconds

**WebSocket Performance:**
- Connection establishment: <2s
- Price update frequency: Real-time (as available from Pyth)
- Polling fallback interval: 2s

**Smart Contract Performance:**
- Deposit transaction confirmation: <30s (Solana network)
- Withdraw transaction confirmation: <30s (Solana network)

### Security

**Oracle Security:**
- Confidence interval checks (reject prices >1% confidence)
- Staleness checks (reject prices >30s old)
- Price band validation (SOL prices within $50-$500 range)
- Source attribution (pyth-network vs fallback)

**Smart Contract Security:**
- Account ownership verification (PDA seeds)
- Signer validation
- Collateral sufficiency checks
- Oracle price validation (Pyth feed verification)

**API Security:**
- Rate limiting (existing tiered rate limits)
- Error message sanitization (no internal errors exposed)
- Input validation (asset symbol normalization)

### Reliability/Availability

**Oracle Availability:**
- Primary source: Pyth Network (Hermes client)
- Fallback source: CoinGecko (automatic on Pyth failure)
- Graceful degradation: Returns 200 with source=fallback on recoverable errors

**WebSocket Reliability:**
- Automatic reconnection with exponential backoff
- Polling fallback on persistent WebSocket failure
- Error throttling to prevent console spam
- Connection status indicators

**Smart Contract Reliability:**
- Transaction retry mechanism (client-side)
- Error handling for insufficient funds
- Collateral account validation before operations

### Observability

**Logging Requirements:**
- Structured logging for all oracle operations (price fetches, cache hits/misses)
- WebSocket connection events (connect, disconnect, retry)
- Deposit/withdraw transaction logs (user address, amount, success/failure)
- Health endpoint exposes freshness metrics

**Metrics to Track:**
- Oracle API response times (p50, p95, p99)
- Cache hit rate
- WebSocket connection uptime
- Price update frequency
- Deposit/withdraw success rate
- Transaction confirmation times

**Monitoring:**
- Alert on oracle API failures (5xx errors)
- Alert on stale prices (>30s)
- Alert on WebSocket connection failures
- Alert on high deposit/withdraw failure rate

## Dependencies and Integrations

### External Dependencies

**Pyth Network:**
- `@pythnetwork/hermes-client` - Hermes client for price feeds
- Feed IDs: BTC, ETH, SOL, ADA, DOT, LINK (verified working)
- WebSocket endpoint: `wss://hermes.pyth.network/ws`
- REST endpoint: `https://hermes.pyth.network/v2/updates/price/latest`

**Solana:**
- `@solana/web3.js` - Solana RPC client
- `@coral-xyz/anchor` - Anchor framework for smart contracts
- Program ID: `C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw` (devnet)

**CoinGecko (Fallback):**
- API endpoint: `https://api.coingecko.com/api/v3/simple/price`
- Rate limits: 50 calls/minute (free tier)

### Internal Dependencies

**Backend Services:**
- `pythOracleService.ts` - Oracle price normalization service
- `supabaseDatabase.ts` - Database abstraction layer
- `redisCacheService.ts` - Cache management (optional)

**Frontend Services:**
- WebSocket service (existing)
- Price context provider (existing)
- Wallet connection service (existing)

**Smart Contract:**
- `collateral_management.rs` - Deposit/withdraw instructions
- `oracle/mod.rs` - Oracle price validation
- `state.rs` - User account and collateral account structures

## Acceptance Criteria (Authoritative)

### Story 0-1: Stabilize Backend Oracle and Price APIs
1. ✅ GET `/api/oracle/price/SOL|BTC|ETH` returns JSON with `{ price:number, source:string, confidence:number, updatedAt }` and HTTP 200 within 250ms p50
2. ✅ GET `/api/prices` returns map of assets with same fields; stale feeds fall back to secondary provider
3. ✅ `/api/oracle/price/*` has health and freshness guardrails; returns 5xx only on irrecoverable errors; otherwise 200 + source=fallback
4. ✅ Structured logging added for all oracle operations
5. ✅ `/api/oracle/health` route exposes freshness metrics

### Story 0-2: Market Data WebSocket Reliability in Pro Terminal
1. ✅ WebSocket connects via env URL or localhost:3002/ws; exponential backoff max 15s; errors throttled
2. ✅ On WS failure, polling fallback engages every 2s without console spam
3. ✅ Visual "Live" indicator reflects WS/polling status

### Story 0-3: Expose withdraw_native_sol in Program + IDL
1. ✅ IDL contains `withdraw_native_sol` with accounts: [user_account, user signer, protocol_vault, collateral_account, sol_usd_price_feed, system_program, (rent if required)]
2. ✅ CLI test withdraw succeeds on devnet; signature returned; lamports decrease in protocol_vault and increase in user wallet

### Story 0-4: UI Guard for Missing Withdraw Instruction
1. ✅ Withdraw action is soft-disabled when IDL lacks `withdraw_native_sol`; tooltip explains "Upgrade in progress"
2. ✅ No "program.methods.withdrawNativeSol is not a function" appears in console

### Story 0-5: QM Shows Complete Live Prices (No N/A)
1. ✅ 100% instruments render price and timestamp; snapshot test baselines the list
2. ✅ Missing assets are mapped to appropriate oracle symbols or excluded with reason

**Status: ✅ COMPLETE** - Implementation verified working as of 2025-10-30

### Story 0-6: MIKEY Monitor Theming and Fonts
1. ✅ All cards use theme vars for bg/border/text; no inline grays
2. ✅ Light/Dark switch preserves contrast; no color blending artifacts

### Story 0-7: MIKEY Tools Fetch Live Prices, Tweets, and News via Backend
1. ✅ Tools: `get_live_price(asset)`, `get_market_summary`, `get_tweets(query)`, `get_news(query)` return current data with timestamps
2. ✅ CORS-free: tools call backend proxy endpoints; timeouts and error messages standardized

### Story 0-8: Deposit/Withdraw CLI Tests and UI E2E
1. ✅ CLI scripts: `deposit_native_sol`, `withdraw_native_sol`; assert PDA derivations, accounts order, success logs
2. ✅ UI e2e: connect wallet → deposit → see USD balance → withdraw → balance reduces; screenshots stored

### Story 0-9: Drift Parity Review for Deposit/Withdraw
1. ✅ Document compares account lists, WSOL vs native SOL, remaining accounts, price checks; recommended approach selected (native/WSOL)
2. ✅ Risks and mitigation listed (e.g., rent, signer, account order)

### Story 0-10: Expert Confirmation of Native SOL Patterns
1. ✅ Short note capturing expert guidance: signer first, SystemAccount for vault, when to include rent, invoke vs invoke_signed for PDA vault, Anchor 0.30 accounts/addresses behavior

## Traceability Mapping

| AC | Spec Section | Component/API | Test Idea |
|----|--------------|---------------|-----------|
| 0-1.1 | Oracle Price API | `GET /api/oracle/price/:asset` | Unit test: verify response format, latency <250ms |
| 0-1.2 | Oracle Price API | `GET /api/prices` | Integration test: verify fallback to CoinGecko on Pyth failure |
| 0-1.3 | Oracle Price API | `/api/oracle/health` | Unit test: verify health endpoint returns freshness metrics |
| 0-2.1 | WebSocket Service | `websocket.ts` | Integration test: verify exponential backoff retry behavior |
| 0-2.2 | WebSocket Service | Polling fallback | Unit test: verify polling starts on WS failure, no console spam |
| 0-2.3 | Frontend UI | Live indicator | E2E test: verify indicator reflects connection status |
| 0-3.1 | Smart Contract | IDL verification | Manual check: verify deployed-idl.json contains withdraw_native_sol |
| 0-3.2 | Smart Contract | CLI test | Integration test: verify withdraw succeeds, balances update correctly |
| 0-4.1 | Frontend UI | Withdraw button | Unit test: verify button disabled when IDL lacks instruction |
| 0-4.2 | Frontend UI | Error handling | E2E test: verify no console errors when withdraw unavailable |
| 0-5.1 | QM Component | Price display | ✅ Snapshot test: baseline price list (already complete) |
| 0-5.2 | QM Component | Asset mapping | ✅ Unit test: verify asset symbol mapping (already complete) |
| 0-6.1 | MIKEY Monitor | Theme tokens | Visual regression test: verify theme consistency |
| 0-6.2 | MIKEY Monitor | Light/Dark switch | E2E test: verify contrast preserved on theme switch |
| 0-7.1 | MIKEY Tools | Backend endpoints | Integration test: verify tools call backend, return timestamps |
| 0-7.2 | MIKEY Tools | CORS handling | Integration test: verify no CORS errors, standardized error messages |
| 0-8.1 | CLI Tests | Deposit script | Integration test: verify deposit succeeds, logs success |
| 0-8.2 | CLI Tests | Withdraw script | Integration test: verify withdraw succeeds, balances update |
| 0-8.3 | E2E Tests | Deposit flow | E2E test: connect → deposit → verify balance → screenshots |
| 0-8.4 | E2E Tests | Withdraw flow | E2E test: withdraw → verify balance reduces → screenshots |
| 0-9.1 | Documentation | Drift analysis | Manual review: verify analysis document complete |
| 0-10.1 | Documentation | Expert guidance | Manual review: verify expert guidance captured |

## Risks, Assumptions, Open Questions

### Risks

**R1: Oracle Price Staleness**
- **Risk:** Pyth feeds may become stale, causing incorrect balance calculations
- **Mitigation:** Health checks, staleness validation (30s threshold), fallback to CoinGecko

**R2: WebSocket Connection Instability**
- **Risk:** WebSocket may fail frequently, degrading user experience
- **Mitigation:** Automatic polling fallback, exponential backoff retry, error throttling

**R3: Smart Contract IDL Mismatch**
- **Risk:** IDL may not match deployed program, causing transaction failures
- **Mitigation:** IDL verification step, CLI tests validate account order, expert guidance document

**R4: Deposit/Withdraw Transaction Failures**
- **Risk:** Transactions may fail silently, leaving users confused
- **Mitigation:** Comprehensive error handling, transaction retry mechanism, clear error messages

**R5: MIKEY Tools Backend Integration Complexity**
- **Risk:** Adding new backend endpoints may introduce CORS or rate limiting issues
- **Mitigation:** Use existing backend proxy pattern, standardize error handling, comprehensive testing

### Assumptions

**A1: Pyth Network Availability**
- Oracle assumes Pyth Network is generally available; fallback to CoinGecko handles outages

**A2: Solana Network Reliability**
- Smart contract operations assume Solana devnet is stable; transaction retries handle temporary failures

**A3: Existing Backend Infrastructure**
- Assumes existing `pythOracleService` and caching infrastructure is stable and performant

**A4: Frontend WebSocket Support**
- Assumes browser environment supports WebSocket; polling fallback covers unsupported environments

### Open Questions

**Q1: Rent Sysvar Requirement**
- **Question:** Is rent Sysvar required for withdraw_native_sol instruction?
- **Status:** To be answered by expert guidance document (Story 0-10)

**Q2: Account Order Priority**
- **Question:** Does account order matter for invoke vs invoke_signed?
- **Status:** To be answered by expert guidance document (Story 0-10)

**Q3: WSOL vs Native SOL Decision**
- **Question:** Should we use WSOL wrapper or native SOL for vault?
- **Status:** To be answered by Drift parity review (Story 0-9)

## Test Strategy Summary

### Test Levels

**Unit Tests:**
- Oracle service price normalization
- WebSocket retry logic
- UI component rendering with theme tokens
- MIKEY tools backend integration

**Integration Tests:**
- Oracle API endpoints with Pyth and fallback
- WebSocket connection and polling fallback
- Deposit/withdraw CLI scripts
- MIKEY tools backend endpoint calls

**E2E Tests:**
- Deposit flow (connect → deposit → verify balance)
- Withdraw flow (withdraw → verify balance reduction)
- Quote Monitor price display
- MIKEY Monitor theme switching
- WebSocket live indicator behavior

### Test Frameworks

- **Backend:** Vitest (existing test setup)
- **Frontend:** Playwright or Cypress for E2E
- **Smart Contract:** Anchor tests (existing test setup)
- **CLI:** Shell scripts with assertion logging

### Coverage Targets

- **Oracle APIs:** 100% endpoint coverage
- **WebSocket Service:** 100% connection state coverage
- **Smart Contract:** 100% instruction coverage
- **UI Components:** Visual regression tests for all components

### Edge Cases

- Oracle price staleness (>30s)
- WebSocket connection failure on page load
- Insufficient collateral for withdrawal
- Missing IDL instruction for withdraw
- Network timeout scenarios
- Cache invalidation edge cases

---

## Implementation Notes

### Priority Stories
1. **Story 0-1** (Oracle API) - Blocking for other stories
2. **Story 0-2** (WebSocket) - Critical for live price updates
3. **Story 0-3** (Withdraw instruction) - Required for full E2E flow
4. **Story 0-5** (QM prices) - ✅ Already complete

### Dependencies
- Backend availability gates UI stories
- Program/IDL upgrade coordination required to expose withdraw
- Expert guidance needed for smart contract patterns

### Known Issues
- Story 0-5 (QM prices) already complete - no action needed
- Deposit instruction already implemented; focus on withdraw exposure

---

_This tech spec provides comprehensive context for Epic 0 implementation. All stories should reference this document for architectural guidance and constraints._

