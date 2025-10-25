# Data & Storage Blueprint

QuantDesk’s data layer is designed to keep trading records accurate, auditable, and fast. This overview distills the Supabase/Postgres schema and surrounding controls into trader-friendly terms.

## 1. Core Databases

- **Supabase PostgreSQL 17** – Primary ledger for users, accounts, orders, positions, and trades. We use Postgres directly for speed while retaining Supabase auth and Row Level Security.
- **Redis Streams & Cache** – Event bus for high-frequency feeds (prices, whale moves, news) and low-latency state (order books, alerts).
- **Timescale Extensions** – Time-series indexes for price and funding data so historical lookups remain sub-second even at scale.

## 2. Key Tables (What They Hold)

- `users` – Wallet-linked profiles, KYC status, risk level, and volume metrics.
- `markets` – Perp market configuration (leverage limits, oracle accounts, margin ratios).
- `user_balances` & `positions` – Collateral breakdowns, entry prices, leverage, liquidation levels.
- `orders` & `trades` – Full execution history with stop/trigger parameters and fill details.
- `oracle_prices`, `funding_rates`, `liquidations` – Time-series records of the market environment.
- `market_stats`, `user_stats`, `system_events` – Daily rollups and operational audit trails.
- `admin_users`, `admin_audit_logs` – Controlled access for the admin terminal with every action recorded.

## 3. Security Controls

- **Row Level Security** ensures traders only see their own balances, orders, and analytics.
- **Role separation** keeps admin tooling isolated from everyday trading operations.
- **Audit logging** captures who performed what action and when—crucial for compliance reviews.
- **Rate limiting & request validation** protect APIs leading into the database.

## 4. Performance Tuning

- Compound indexes on `(market_id, created_at)` for price/funding tables and `(user_id, created_at)` for trade history keep dashboards responsive.
- Batched writers (analytics service) push market stats and user rollups without overwhelming Postgres.
- Redis caches the hottest data so the terminal rarely hits Postgres for live tiles or alerts.
- Redis keys are namespaced per environment (`qd:{env}:…`) with persistence (`AOF everysec`) and memory policies tuned to keep caches fresh without risking data loss.

## 5. Data Lifecycle

- **Hot Data** (seconds): Redis Streams deliver ticks, whale events, and news directly to MIKEY and the terminal.
- **Warm Data** (minutes): Supabase tables store trades, funding moves, and daily stats for dashboards and reporting.
- **Cold Data** (days+): Analytics rollups and archives remain queryable for performance reviews and investor updates.

## Why Traders Care

- Positions, P&L, and liquidation levels are always accurate because the data model mirrors how perpetual markets operate.
- Historical analytics and smart money dashboards are powered by the same authoritative tables—no conflicting data sources.
- Security-first table design keeps personal trading activity private while enabling admin oversight when necessary.

When discussing platform reliability, pair this blueprint with the broader [Architecture at a Glance](../core-features/architecture-at-a-glance.md) and the [Security & Trust](./security-and-trust.md) overview.
