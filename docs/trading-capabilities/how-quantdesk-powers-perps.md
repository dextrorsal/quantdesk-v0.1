# How QuantDesk Powers Solana Perpetual Markets

QuantDesk delivers the full perpetual trading workflow on Solana without exposing you to infrastructure complexity. Here’s how the system runs behind the scenes and why it matters when you manage real positions.

## 1. Always-On Trading Accounts

- **Connect wallet, start trading:** As soon as you connect, QuantDesk spins up a dedicated trading profile that tracks collateral, open positions, and P&L in real time.
- **Automatic state handling:** Whether you are depositing collateral, opening a position, or unwinding exposure, the platform transitions through the right account states automatically.
- **One place for everything:** Funding, margin levels, and liquidation thresholds are surfaced inside the terminal so you never guess where you stand.

## 2. Smart Contract Stack Tuned for Perps

- **QuantDesk Perp DEX Program** (`C2T3UnvGdHwEkspXJG7JyAhwo6VKQEKjN6eCq69guYSw`) – Core perpetual trading engine
- **Additional programs:** Collateral, Oracle, Security, Trading, Core modules for specialization
- **Account program:** Keeps trading accounts, sub-accounts organized on-chain
- **Oracle layer:** Pyth Network price feeds with confidence scores for accurate mark prices
- **Rate limiting:** Per-minute limits to prevent abuse (100 pub/min, 10 trading/min, 5 auth/15min)
- **Transaction verification:** All on-chain transactions verified before processing

These pieces provide battle-tested Solana perp functionality with enterprise-grade security and monitoring.

## 3. Real-Time Data Pipeline

- **Supabase storage:** PostgreSQL database with Row Level Security for account states, historical fills, and analytics
- **Redis cache:** Optional cache for order books, funding, and sentiment metrics (disabled in development)
- **Socket.IO WebSockets:** Real-time market data, MIKEY alerts, and account changes stream to clients
- **API gateway (Port 3002):** Central entry point that validates requests, manages SIWS sessions, and applies rate limiting
- **Pyth Network integration:** Direct oracle price feeds for BTC, ETH, SOL with confidence tracking

The result is a terminal that reacts to market swings the moment they happen with sub-second updates.

## 4. Execution Built for Volatility

- **Optimized Solana transactions:** Uses address lookup tables and batching so your orders land fast even under congestion.
- **Slippage + liquidation checks:** Every ticket shows real-time impact on your collateral before you commit.
- **MEV-aware routing:** Requests are structured to reduce sandwich risk and wasted fees.
- **Cross-venue awareness:** Price comparisons across DEX venues help you decide where to route size.

## 5. Risk Controls You Can Trust

- **Continuous margin monitoring:** The backend flags unhealthy exposure before liquidation bots do.
- **Automated liquidation engine:** Unwinds positions smoothly while protecting other market participants.
- **Health alerts:** MIKEY pings you when collateral ratios tighten or funding flips unfavourably.
- **Audit-ready trails:** Every action is logged with timestamps so you can reconstruct decisions anytime.

## 6. Operational Reliability

- **Secure architecture:** SIWS authentication with client-side signing; transactions execute via PDAs so keys never leave your wallet
- **Session management:** HTTP-only cookies with 7-day expiration; Redis-backed sessions (optional in dev)
- **Observability:** Grafana dashboards track node latency, WebSocket uptime, error rates, and system metrics
- **Multi-service architecture:** Backend (3002), Frontend (3001), MIKEY-AI (3000), Data Ingestion (3003), Admin (5173)
- **Health monitoring:** Real-time system health endpoints and automated alerting for service degradation
- **Transaction verification:** All on-chain transactions verified before processing to prevent fraud

## What This Means When You Trade

| Your Need | How QuantDesk Supports It |
|-----------|---------------------------|
| Instant market awareness | Live depth, funding, and MIKEY insights feed the terminal without lag |
| Confident execution | Orders simulate risk impact, respect leverage, and route quickly |
| Capital protection | Automated risk checks and liquidation lines keep collateral safe |
| Transparency | Every transaction, event, and alert is traceable inside the workspace |

## Ready for Deeper Dives?

- [Security & Trust Practices](../security-trust/) – See the controls that protect your capital and data.
- [Core Terminal Features](../core-features/) – Explore the tools available inside the workspace.
- [Start Trading in Minutes](../getting-started/) – Spin up your environment and experience the terminal firsthand.

QuantDesk gives you the professional-grade foundation to trade Solana perpetual markets confidently, without needing to wrangle low-level infrastructure yourself.
