# Enterprise Readiness Overview

QuantDesk is engineered for professional desks—this page highlights the advanced capabilities and operational playbooks that larger teams care about.

## Institutional Trading Features

- **Multi-Account Control** – Master/sub-account permissions, delegated access, and cross-collateral controls built for managed money.
- **SIWS Authentication** – Wallet-based signature verification with HTTP-only session management for enterprise security.
- **Rate Limiting** – Per-minute rate limits (30 req/min trading, 100 req/min admin) with tiered access controls.
- **Advanced Order Suite** – TWAP, Iceberg, bracket orders, and time-in-force options sitting alongside risk-aware execution safeguards (in development).
- **Risk & Liquidation Engine** – Continuous monitoring of health factors, configurable limits, and insurance fund hooks for future upgrades.
- **Just-in-Time Liquidity** – Architecture accommodates market maker auctions and price improvement routing for block-sized flow.
- **Multi-Program Architecture** – Separate programs for collateral, oracle, trading, security, and core operations.

## Operational Tooling

- **Admin Dashboard** – Role-based console for compliance checks, account freezes, and manual overrides when needed (Port 5173).
- **API Gateway** – Centralized API management with rate limiting, authentication, and request validation (Port 3002).
- **Real-time Monitoring** – WebSocket support via Socket.IO for live updates to trading terminals.
- **Audit Trails** – Every action (trades, transfers, permission changes) logged with timestamps and wallet signatures for easy review.
- **Reporting & Analytics** – Rollups across markets, accounts, and timeframes prepared for risk, finance, and performance reporting.
- **Grafana Integration** – Metrics collection and monitoring dashboards for production observability.

## Deployment Playbooks

- **Cloud Strategy** – Works with modern PaaS (Render, DigitalOcean) today and scales to AWS/GCP setups with GPU-backed inference for MIKEY.
- **Containerization** – Backend, ingestion workers, and AI services ship with Docker support, simplifying horizontal scaling and zero-downtime deploys.
- **Observability** – Metrics, logs, and tracing ready for Prometheus/Grafana or cloud-native monitoring stacks.

## Security & Compliance

- **Row Level Security & Rate Limiting** guard API and database access by default.
- **Secret Hygiene** – Environment guides, scanners, and automated lint checks prevent credential leaks.
- **Audit-Ready Architecture** – PDP (policy decision points) in the backend make it straightforward to enforce KYC/AML hooks if required.

## Integration Paths

- **SDK & API Roadmap** – REST/WebSocket endpoints already power the terminal; public SDKs and partner APIs are staged next.
- **White-Label Potential** – Tenant-aware database design and role separation lay the groundwork for branded experiences or institutional sub-tenants.

QuantDesk’s roadmap is intentionally enterprise-friendly: the foundation already supports professional desks today, with upcoming releases deepening automation, reporting, and governance controls. Pair this with the [Security & Trust](../security-trust/security-and-trust.md) and [Market Intelligence Pipeline](../ai-engine/market-intelligence-pipeline.md) docs when discussing due diligence with larger partners.
