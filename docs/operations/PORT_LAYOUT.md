# Port Layout (Local Dev)

Authoritative map of local service ports and how to override them. Keep this up to date when adding or changing services.

## 🚀 Running Services

| Service | Port | Purpose | Start Command |
|---------|------|---------|---------------|
| **Backend API** | 3002 | REST + WebSocket gateway | `cd backend && pnpm run dev` |
| **Frontend** | 3001 | Trading interface | `cd frontend && pnpm run dev` |
| **MIKEY-AI** | 3000 | AI trading assistant | `cd MIKEY-AI && pnpm run dev` |
| **Data Ingestion** | 3003 | Market data pipeline | `cd data-ingestion && pnpm start` |
| **Admin Dashboard** | 5173 | Admin interface | `cd admin-dashboard && pnpm run dev` |

## 📝 Detailed Configuration

### Backend API (Port 3002)
- **Default**: 3002
- **Environment**: `PORT`
- **Config**: `backend/src/server.ts` (defaults to 3002)
- **Health**: `GET http://localhost:3002/health`
- **Start**: `cd backend && pnpm run start:dev`

### Frontend (Port 3001)
- **Default**: 3001
- **Config**: `frontend/vite.config.ts` → `server.port`
- **Health**: `http://localhost:3001/`
- **Start**: `cd frontend && pnpm run dev`

### MIKEY-AI (Port 3000)
- **Default**: 3000
- **Environment**: `PORT`
- **Health**: `GET http://localhost:3000/health`
- **Start**: `cd MIKEY-AI && pnpm run dev`

### Data Ingestion (Port 3003)
- **Default**: 3003
- **Environment**: `METRICS_PORT`
- **Health**: `http://localhost:3003/`
- **Start**: `cd data-ingestion && pnpm start`

### Admin Dashboard (Port 5173)
- **Default**: 5173
- **Config**: `admin-dashboard/vite.config.ts`
- **Health**: `http://localhost:5173/`
- **Start**: `cd admin-dashboard && pnpm run dev`

## 🔄 Quick Start All Services

```bash
# From project root
pnpm run dev
```

This starts all services simultaneously via concurrent scripts.

## 📡 Service Communication

- **Frontend** → Backend API: `http://localhost:3002` (proxy in vite.config.ts)
- **Frontend** → WebSocket: `ws://localhost:3002` (Socket.IO)
- **Backend** → MIKEY-AI: `http://localhost:3000`
- **Backend** → Supabase: Database connection
- **Backend** → Redis: Session storage (optional in dev)

## ⚙️ Overriding Ports

To run on different ports:

```bash
# Backend
cd backend && PORT=3003 pnpm run start:dev

# Frontend
cd frontend && npm run dev -- --port 3002

# MIKEY-AI
cd MIKEY-AI && PORT=3001 pnpm run dev
```

## 🔍 Health Check Commands

```bash
# Backend
curl http://localhost:3002/health

# MIKEY-AI
curl http://localhost:3000/health

# Data Ingestion
curl http://localhost:3003/health
```

## 📝 Notes

- Frontend proxies API requests to backend (see `frontend/vite.config.ts`)
- WebSocket connections use Socket.IO on backend port
- All services support hot-reload in development
- Redis is optional in development (session storage disabled by default)
- Update this document whenever ports change to keep a single source of truth


