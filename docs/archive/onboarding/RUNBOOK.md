## Early Tester Onboarding Runbook (Solana + Supabase + Redis)

Goal: Enable wallet sign-in, referral binding (25% + 10%), chat access, and claims in a local/devnet setup.

### 1) Prereqs
- Solana CLI: `solana --version`
- Anchor CLI: `anchor --version`
- Node 23.x, pnpm (or yarn)
- Supabase project: URL + anon key + service role key
- Redis instance (local or managed)

### 2) Environment
- Backend `.env` (server-only):
  - `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`
  - `SESSION_SECRET`, `REDIS_URL`
  - `FEATURE_REFERRALS=true`, `FEATURE_CHAT=true`, `REFERRAL_L1=0.25`, `TRADER_DISCOUNT=0.10`
- Frontend `.env`:
  - `VITE_SUPABASE_URL`, `VITE_SUPABASE_ANON_KEY`

### 3) Database (Supabase)
- Run `database/referrals_schema.sql` in Supabase SQL editor.
- Enable RLS and add policies as needed (see comments in SQL).
- Optional: create a DB function/webhook to flip `referrals.activated` on mock trade events.

### 4) Backend boot
- Start local validator (for E2E): `solana-test-validator`
- Backend start: `pnpm dev` (or project start script)
- Confirm routes in Postman: `/auth/nonce`, `/auth/verify`, `/referrals/*`, `/chat/*`, `/claims/*`.

### 5) Frontend boot
- Start: `pnpm dev`
- Flow: connect wallet → nonce → sign → verify → session cookie → show invite link.

### 6) Referrals (25% + 10%)
- Bind on first successful verify with `?ref=<referrer_pubkey>`.
- Activation rule: after threshold (e.g., 5 mock trades or $10 sim volume).
- Claims: preview (calc on net fees), execute weekly (SOL on devnet).

### 7) Chat
- Token endpoint issues short-lived chat token (JWT).
- WS/SSE connects with token; Redis pub/sub for presence and broadcast.
- Rate limits via Redis (e.g., 5 msgs / 10s / wallet).

### 8) Testing
- Unit/program: Anchor + Bankrun for PDA/referral logic (faster, deterministic).
- E2E/UI: solana-test-validator + Playwright + wallet-adapter mock.
- Postman: import `docs/api/QuantDesk-API-Collection.json` and `QuantDesk-Complete-Environment.json`.

### 9) Solana MCP (reference points)
- Local validator vs bankrun recommendations (faster tests, time travel).
- PDA patterns and multi-user seeding tips.
- Wallet mocking options for E2E.

### 10) Ops
- Cron for weekly claims.
- Monitoring: RPC health, Redis memory, error rates.
- Feature flags: disable/enable referrals/chat quickly if needed.


