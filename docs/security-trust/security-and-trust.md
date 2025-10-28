# Security & Trust at QuantDesk

QuantDesk is built to handle live trading capital without asking you to compromise on safety. This page explains the safeguards layered through the stack so you know exactly how your data and keys stay protected.

## 1. Account & Key Protection

- **Non-custodial by default** – QuantDesk never holds your wallet keys. Orders are signed client-side and routed securely.
- **SIWS Authentication** – Wallet-based authentication (Solana In-App Web3 Signing) with signature verification
- **Environment-scoped secrets** – `.env` files stay local; production deployments expect encrypted secrets or managed vaults
- **Role-based access** – Supabase Row Level Security (RLS) policies restrict every table to the authenticated owner; no shared views, no public reads
- **Session controls** – HTTP-only cookies with JWT tokens, 7-day expiration, and automatic cleanup of inactive sessions

## 2. Data Layer Safeguards

- **Supabase policies** enforce per-user reads/writes on markets, orders, positions, and funding tables.
- **Audit trails** log inserts, updates, and deletes with user IDs and timestamps for full transparency.
- **Query rate limits** protect the database from abuse and noisy neighbours.
- **Partitioned analytics** keeps historical metrics separate from live trading data to minimize blast radius.

## 3. API & Backend Security

- **Authenticated routes** – Trading, portfolio, and alert endpoints require wallet signatures; anonymous access is limited to documentation and public status checks
- **Request validation** – Every payload passes schema checks before touching business logic; malformed requests are rejected early
- **Rate limiting** – Tiered limits: Public (100 req/min), Trading (10 req/min), Auth (5 per 15 min), Admin (50 req/min), Webhook (20 req/min)
- **Transport security** – HTTPS enforced for all external communication; WebSockets use Socket.IO with authentication
- **Transaction verification** – All on-chain transactions verified before processing to prevent fraud

## 4. Real-Time Services

- **WebSocket tokens** expire quickly and are tied to user identity; reconnects must re-authenticate.
- **Isolated ingestion workers** run without access to customer data—only to the queues they process.
- **Backpressure handling** throttles outbound events when clients fall behind, preventing buffer overflow or dropped messages.

## 5. Monitoring & Alerts

- **System health dashboards** watch RPC response times, WebSocket uptime, and queue depths so operations can react before users feel impact.
- **Security alerts** trigger when unusual API activity, rapid credential reuse, or suspicious funding flows are detected.
- **Log aggregation** keeps structured logs searchable for incident response.

## 6. Deployment Hygiene

- **Separate environments** – Local, staging, and production use distinct Supabase instances and API keys.
- **Automated CI checks** scan for committed secrets and enforce lint/tests before shipping.
- **Infrastructure as code** keeps environments reproducible and reviewable.
- **Routine key rotation** is part of the operational calendar to limit credential exposure.

## 7. Steps You Control

- Keep your `.env` file outside version control.
- Rotate exchange and Supabase keys regularly.
- Restrict CORS domains to the hosts you actually use.
- Review and adjust QuantDesk rate limits to match your team’s usage pattern.
- Enable multi-factor authentication on every connected wallet or exchange account.

---

Curious about the system internals? Dive into [How QuantDesk Powers Perps](../trading-capabilities/how-quantdesk-powers-perps.md) to see how the architecture and security layers work together.
