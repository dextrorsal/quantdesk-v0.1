---
title: "Epic: Restore Trading E2E and Data Parity"
status: draft
owner: Scrum Master
created: 2025-10-30
tags: [epic, trading, oracle, websocket, ui, mikey, anchor]
---

# Epic: Restore Trading E2E and Data Parity

Goal: A user can deposit, see live prices, place a trade, and withdraw; MIKEY tools operate on live data; UI theming is consistent.

## User Story 1: Stabilize backend oracle and price APIs
As a trader, I need /api/oracle/price/:asset and /api/prices to return fresh non-zero prices so UI and tools display correct balances.

Acceptance Criteria:
- GET /api/oracle/price/SOL|BTC|ETH returns JSON with { price:number, source:string, confidence:number, updatedAt } and HTTP 200 within 250ms p50.
- GET /api/prices returns a map of assets with the same fields; stale feeds fall back to secondary provider.
- /api/oracle/price/* has health and freshness guardrails; returns 5xx only on irrecoverable errors; otherwise 200 + source=fallback.

Notes: Add structured logging and a /api/oracle/health route exposing freshness.

## User Story 2: Market data WebSocket reliability in Pro Terminal
As a user, I need price updates via WebSocket with retry and safe fallback so prices remain live.

Acceptance Criteria:
- WebSocket connects via env URL or localhost:3002/ws; exponential backoff max 15s; errors throttled.
- On WS failure, polling fallback engages every 2s without console spam.
- A visual “Live” indicator reflects WS/polling status.

## User Story 3: Expose withdraw_native_sol in program + IDL
As a trader, I need to withdraw SOL so I can exit positions and retrieve funds.

Acceptance Criteria:
- IDL contains withdraw_native_sol with accounts: [user_account, user signer, protocol_vault, collateral_account, sol_usd_price_feed, system_program, (rent if required)].
- CLI test withdraw succeeds on devnet; signature returned; lamports decrease in protocol_vault and increase in user wallet.

## User Story 4: UI guard for missing withdraw instruction
As a user, I should not hit runtime errors if withdraw is unavailable.

Acceptance Criteria:
- Withdraw action is soft-disabled when IDL lacks withdraw_native_sol; tooltip explains “Upgrade in progress”.
- No “program.methods.withdrawNativeSol is not a function” appears in console.

## User Story 5: QM shows complete live prices (no N/A)
As a user, I need every instrument in Quick Monitor to show a live, non-zero price.

Acceptance Criteria:
- 100% instruments render a price and timestamp; snapshot test baselines the list.
- Missing assets are mapped to appropriate oracle symbols or excluded with reason.

## User Story 6: MIKEY Monitor theming and fonts
As an operator, I want MIKEY Monitor cards to use theme tokens so UI is consistent.

Acceptance Criteria:
- All cards use theme vars for bg/border/text; no inline grays.
- Light/Dark switch preserves contrast; no color blending artifacts.

## User Story 7: MIKEY tools fetch live prices, tweets, and news via backend
As an AI operator, I need MIKEY tools to use backend endpoints for reliable data.

Acceptance Criteria:
- Tools: get_live_price(asset), get_market_summary, get_tweets(query), get_news(query) return current data with timestamps.
- CORS-free: tools call backend proxy endpoints; timeouts and error messages standardized.

## User Story 8: Deposit/Withdraw CLI tests and UI e2e
As a maintainer, I need automated checks to ensure deposits/withdraws function through upgrades.

Acceptance Criteria:
- CLI scripts: deposit_native_sol, withdraw_native_sol; assert PDA derivations, accounts order, success logs.
- UI e2e: connect wallet → deposit → see USD balance → withdraw → balance reduces; screenshots stored.

## User Story 9: Drift parity review for deposit/withdraw
As a protocol engineer, I need a gap analysis vs Drift to minimize bugs.

Acceptance Criteria:
- Document compares account lists, WSOL vs native SOL, remaining accounts, price checks; recommended approach selected (native/WSOL).
- Risks and mitigation listed (e.g., rent, signer, account order).

## User Story 10: Expert confirmation of native SOL patterns
As a protocol engineer, I need expert-validated patterns to avoid regressions.

Acceptance Criteria:
- Short note capturing expert guidance: signer first, SystemAccount for vault, when to include rent, invoke vs invoke_signed for PDA vault, Anchor 0.30 accounts/addresses behavior.

---
Dependencies & Risks:
- Backend availability gates UI stories.
- Program/IDL upgrade coordination required to expose withdraw.

Out-of-Scope:
- Mainnet rollout; this epic targets devnet and demo parity.


