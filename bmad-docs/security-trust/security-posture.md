# Security Posture Snapshot

QuantDesk treats security as a first-class feature. This snapshot consolidates the latest audits and hardening work so stakeholders know where the platform stands today and what’s being improved next.

## Recent Audit Highlights

- **External tooling coverage** – Sentry (runtime), SonarQube (code quality), Semgrep (static security), and Socket.dev (dependency supply chain) are part of the recurring review cycle.
- **Critical issues resolved** – 18 high/critical dependency vulnerabilities eliminated across backend, frontend, and admin dashboard packages (including `elliptic`, `serialize-javascript`, `@babel/*`, `cross-spawn`, and `nanoid`).
- **Secrets audit** – Confirmed no API keys or Supabase credentials remain in source; Supabase service role keys were rotated after exposure in historical examples.

## Ongoing Hardening

- **Dependency hygiene** – Remaining `bigint-buffer` and `esbuild` alerts require breaking version upgrades. They’re staged behind test plans to ensure Solana SPL token and Vite workflows remain stable.
- **Documentation CDN usage** – Legacy docs-site HTML still pulls Prism/Mermaid from public CDNs. A migration plan is in place to self-host or apply SRI hashes during the docs rewrite.
- **GitHub security tooling** – Secret scanning is active; Dependabot and CodeQL workflows are being re-enabled as the repo cleanup settles.

## Operational Safeguards

- Row Level Security policies, request validation, and rate limiting shield APIs and database access.
- Observability dashboards track RPC latency, worker throughput, and WebSocket uptime so anomalies trigger alerts quickly.
- Emergency playbooks cover credential rotation and git history scrubbing if a secret is ever exposed.

## Takeaways

- QuantDesk’s current release is backed by audited dependencies, sanitized configuration, and layered protections.
- The remaining action items are controlled upgrades or tooling automation—not undiscovered vulnerabilities in production flows.
- Security reviews are baked into each release cycle alongside the roadmap you see in the [Build Status](../success-stories/roadmap-and-status.md) page.

For more about day-to-day safeguards, see [Security & Trust](./security-and-trust.md) and the [Data & Storage Blueprint](./data-and-storage-blueprint.md).
