## Port Layout (Local Dev)

Authoritative map of local service ports and how to override them. Keep this up to date when adding or changing services.

- Frontend (Vite dev server)
  - Default: 3001
  - Config: `frontend/vite.config.ts` → `server.port`
  - Start: `frontend/start-frontend.sh`
  - Health: open `http://localhost:3001/`

- Backend API
  - Default: 3002
  - Env: `PORT`
  - Code: `backend/src/server.ts` (defaults to 3002)
  - Start: `backend/start-backend.sh`
  - Health: `GET http://localhost:3002/health`

- MIKEY Bridge (integration)
  - Default: 3000
  - Env: `PORT`
  - Config: `MIKEY-AI/integration/mikey-bridge/src/config/index.ts`
  - Start: `MIKEY-AI/integration/mikey-bridge/start-bridge.sh`
  - Health: `GET http://localhost:3000/health`

- Data Ingestion Monitoring Dashboard
  - Default: 3003
  - Env: `METRICS_PORT`
  - Code: `data-ingestion/src/monitoring/dashboard.js`
  - Start: `data-ingestion/start-pipeline.sh`
  - Health: open `http://localhost:3003/`

Notes
- Frontend proxies API requests to `http://localhost:3002` (see `frontend/vite.config.ts`).
- Update this document whenever ports change to keep a single source of truth.


