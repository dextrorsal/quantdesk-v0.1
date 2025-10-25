# How QuantDesk Powers Solana Perpetual Markets

QuantDesk delivers the full perpetual trading workflow on Solana without exposing you to infrastructure complexity. Here’s how the system runs behind the scenes and why it matters when you manage real positions.

## 1. Always-On Trading Accounts

- **Connect wallet, start trading:** As soon as you connect, QuantDesk spins up a dedicated trading profile that tracks collateral, open positions, and P&L in real time.
- **Automatic state handling:** Whether you are depositing collateral, opening a position, or unwinding exposure, the platform transitions through the right account states automatically.
- **One place for everything:** Funding, margin levels, and liquidation thresholds are surfaced inside the terminal so you never guess where you stand.

## 2. Smart Contract Stack Tuned for Perps

- **Account program:** Keeps your trading accounts, sub-accounts, and order history organized on-chain.
- **Collateral vault:** Manages deposits and withdrawals securely while tracking your available margin.
- **Perp engine:** Routes orders, updates positions, and enforces leverage rules.
- **Oracle layer:** Pulls price data from multiple sources to keep mark prices honest.
- **Protection pool:** Backs liquidations and extreme events so healthy traders aren’t penalized.

These pieces mirror battle-tested Solana perp protocols, giving you familiarity with the reliability you expect from venues like Drift.

## 3. Real-Time Data Pipeline

- **Supabase storage:** Persists account states, historical fills, and analytics for quick recall.
- **Redis cache:** Keeps order books, funding, and sentiment metrics hot so the terminal updates instantly.
- **WebSockets everywhere:** Market data, MIKEY alerts, and account changes stream straight to your screen.
- **API gateway:** Central entry point that signs and submits transactions, validates requests, and rate-limits abusive traffic.

The result is a terminal that reacts to market swings the moment they happen.

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

- **Secure architecture:** Wallet signatures happen client-side; transactions execute via PDAs so keys never leave your wallet.
- **Observability:** Dashboards track node latency, WebSocket uptime, and error rates so you know the system is healthy.
- **Scalability:** Horizontal scaling keeps the terminal responsive during high-volume events.
- **Failover ready:** Multi-region infrastructure means the trading experience stays online.

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
