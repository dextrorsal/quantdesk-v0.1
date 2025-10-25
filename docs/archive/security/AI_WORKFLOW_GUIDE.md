# AI Workflow Guide (Public)

This guide explains, at a high level, how QuantDesk's AI components interact without exposing proprietary code.

## Components
- MIKEY-AI (private): research assistant, signal summarization, portfolio reasoning
- Data pipeline (public docs): price feeds, analytics summaries, smart money alerts
- Backend (private): routes, auth, RPC load balancing, analytics endpoints
- Frontend (private): dashboards, charts, trading UIs

## Flow
1. Data services produce normalized events (prices, news, whale flows)
2. MIKEY-AI consumes summaries and metadata (no PII/secrets), generates human-readable insights
3. Backend exposes secure endpoints; RPC load balancer proxies Solana calls
4. Frontend renders dashboards and insights

## Safety & Keys
- Keys loaded via `.env` (never committed)
- Role-limited actions; devnet-first for testing
- RLS & rate limiting at API boundaries

For technical details, see `docs/operations/` and `docs/security/`.
