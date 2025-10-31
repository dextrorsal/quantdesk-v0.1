## QuantDesk Repository Structure

This document summarizes the top-level structure and where to find things quickly.

### Apps and Services
- `frontend/` — React/Vite trading UI (port 3001)
- `backend/` — API gateway, database, oracle services (port 3002)
- `MIKEY-AI/` — AI trading agent (port 3000)
- `data-ingestion/` — Real-time data pipeline (port 3003)
- `admin-dashboard/` — Admin/ops dashboard
- `docs-site/` — Static HTML docs viewer and tooling

### Smart Contracts and SDK
- `contracts/` — Anchor workspace for Solana programs, tests, and scripts
- `sdk/typescript/` — TypeScript SDK published as `@quantdesk/sdk`

### Shared, Docs, and Tooling
- `docs/` — Architecture, guides, specs, and design docs
- `docs/guides/` — How-tos and manuals (e.g., manual testing, mock data setup)
- `docs/demos/` — Demo checklists and runbooks
- `scripts/` — Dev, CI, deployment, and testing utilities
- `database/` — SQL migrations and related scripts
- `examples/` — Example integrations and scripts
- `reports/` — Generated reports and security outputs
  - `reports/fixes/` — Fix root-cause analyses and implementation summaries
  - `reports/sessions/` — Session summaries and test result reports
- `assets/` — Images, screenshots, and static assets
- `debug/` — Debug JSONs and runtime artifacts
- `misc/` — Assorted text notes and temporary references

### Workspace and Commands
- `pnpm-workspace.yaml` — Declares workspaces:
  - `backend`, `frontend`, `MIKEY-AI`, `data-ingestion`, `contracts`, `admin-dashboard`, `docs-site`, `sdk/typescript`
- Root `package.json` useful scripts:
  - `pnpm run dev` — Starts all services via script
  - `pnpm run dev:all` — Runs `dev` for all packages in parallel
  - `pnpm run start:backend|frontend|admin` — Dev entry points per app
  - `pnpm run build` — Builds core apps

### Notes
- Prefer `pnpm` over `npm` for installs and scripts
- Backend services must use `databaseService` abstraction and `pythOracleService` per project rules
- For multi-service features, use BMAD v6 workflows documented under `bmad/`


