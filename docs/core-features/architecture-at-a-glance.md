# Architecture at a Glance

QuantDesk is a layered platform: the terminal you see on screen is backed by real-time services, hardened APIs, and on-chain programs that keep perpetual trading reliable. Use this high-level view when explaining how everything fits together without diving into proprietary code.

## 1. User Experience Layer

- **Trading Terminal (Port 3001)** – React-based workspace with market boards, execution desk, risk dashboard, and MIKEY insights. Optimized for multi-monitor layouts and live data.
- **Wallet + Profile** – Non-custodial wallet connection with support for master/sub accounts so traders can segment risk and activity.
- **Admin Tools** – Separate gated interface for compliance checks, account health, and operational overrides.

## 2. API & Service Layer

- **API Gateway (Port 3002)** – Node/TypeScript service handling REST + WebSocket traffic, with authentication, rate limiting, and request validation.
- **Trading Services** – Order routing, risk calculations, and portfolio management modules that translate terminal actions into on-chain transactions.
- **Market Intelligence Services** – Data ingestion, MIKEY inference, and alert dispatchers built on Redis Streams for sub-second updates.

## 3. Data & Storage Layer

- **Supabase PostgreSQL** – Source of truth for accounts, orders, positions, and analytics with Row Level Security enforcing per-user access.
- **Timeseries Schemas** – Tables for oracle prices, funding history, and liquidations tuned with Timescale-style indexing for fast queries.
- **Redis Caching** – Hot storage for order books, whale flow, and news events feeding both the terminal and MIKEY.

## 4. Blockchain & External Integrations

- **Solana Programs** – Anchor-based smart contracts managing accounts, collateral, and liquidation logic; RPC load balancer distributes calls across providers.
- **Price Oracles** – Pyth (and optional Switchboard) deliver mark prices, funding references, and confidence scores.
- **Third-Party Data** – Birdeye, GMGN, CoinGecko, Coinglass, and others augment MIKEY’s view of liquidity and sentiment.

## 5. Hybrid Web2 ↔ Web3 Flow

- Frontend requests hit the **Web2 API gateway first** (authentication, validation, risk checks).
- The gateway decides when to call **Solana programs** (deposits, order placement, liquidation) and returns confirmed state back to the terminal.
- This split keeps off-chain services (MIKEY, analytics, compliance) fast and extensible while on-chain operations remain trustless and auditable.

## 6. Observability & Operations

- **Health Dashboards** – Monitoring for RPC latency, worker throughput, WebSocket uptime, and API response time.
- **Alerting** – Threshold-based alerts for ingestion stalls, funding anomalies, and security events.
- **CI/CD Guardrails** – Automated lint, type check, and secret scanning before any deployment.

## Why It Matters

- Traders get a professional experience because execution, risk, and data are decoupled yet synchronized through the service layer.
- Stakeholders can trust the platform knowing each layer has clear responsibilities, security controls, and failover paths.
- The modular approach lets us expand—whether that’s more perp markets, new analytics, or additional deployment regions—without rewriting the foundation.

For a deeper dive into storage specifics, check [Data & Storage Blueprint](../security-trust/data-and-storage.md). To see how this architecture powers real trading, read [How QuantDesk Powers Perps](../trading-capabilities/how-quantdesk-powers-perps.md).
