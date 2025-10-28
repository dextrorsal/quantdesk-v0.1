#!/usr/bin/env bash
set -euo pipefail

# QuantDesk - Start all dev services
# Services: backend, frontend, MIKEY-AI, data-ingestion, docs-site, admin-dashboard

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"
mkdir -p "${LOG_DIR}"

command -v pnpm >/dev/null 2>&1 || { echo "pnpm is required. Install via: corepack enable && corepack prepare pnpm@latest --activate"; exit 1; }

# Load user shell environment (nvm, PATH adjustments) for non-interactive sessions
load_shell_env() {
  if [ -f "$HOME/.bashrc" ]; then
    # shellcheck disable=SC1090
    . "$HOME/.bashrc" || true
  fi
  if [ -z "${NVM_DIR:-}" ]; then
    if [ -d "$HOME/.nvm" ]; then
      export NVM_DIR="$HOME/.nvm"
      # shellcheck disable=SC1090
      [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh" || true
      # shellcheck disable=SC1090
      [ -s "$NVM_DIR/bash_completion" ] && . "$NVM_DIR/bash_completion" || true
    fi
  fi
  if command -v corepack >/dev/null 2>&1; then
    corepack enable >/dev/null 2>&1 || true
  fi
}

port_in_use() {
  local port="$1"
  nc -z 127.0.0.1 "$port" >/dev/null 2>&1
}

kill_by_port() {
  local port="$1"
  if command -v fuser >/dev/null 2>&1; then
    fuser -k "${port}/tcp" 2>/dev/null || true
  else
    lsof -t -i :"${port}" 2>/dev/null | xargs -r kill || true
  fi
}

kill_by_pattern() {
  local pattern="$1"
  pkill -f "$pattern" 2>/dev/null || true
}

pre_clean() {
  echo "🧹 Pre-clean: stopping any existing services and freeing ports"
  ( bash "${ROOT_DIR}/scripts/stop-all.sh" ) || true
  kill_by_port 3000
  kill_by_port 3001
  kill_by_port 3002
  kill_by_port 3003
  kill_by_port 3005
  kill_by_pattern "tsx.*MIKEY-AI"
  kill_by_pattern "concurrently.*data-ingestion"
  kill_by_pattern "vite.*frontend"
  kill_by_pattern "vite.*admin-dashboard"
  kill_by_pattern "ts-node src/server.ts"
  rm -f "${LOG_DIR}"/*.pid 2>/dev/null || true
  sleep 0.3
}

# Ensure Redis is running for the backend
ensure_redis() {
  if ! (command -v docker >/dev/null 2>&1); then
    echo "⚠️  Docker not found; skipping Redis auto-start. Backend may fail without Redis."
    return 0
  fi
  if ! (nc -z 127.0.0.1 6379 >/dev/null 2>&1); then
    echo "🧱 Redis not detected on 6379. Starting via docker compose..."
    (cd "${ROOT_DIR}" && docker compose -f config/docker-compose.yml up -d redis) || true
    for i in {1..10}; do
      if nc -z 127.0.0.1 6379 >/dev/null 2>&1; then
        echo "✅ Redis is up."
        break
      fi
      sleep 0.5
    done
  else
    echo "✅ Redis already running on 6379."
  fi
}

start_service() {
  local name="$1"
  local dir="$2"
  shift 2
  local cmd="$*"
  local log_file="${LOG_DIR}/${name}.log"
  local pid_file="${LOG_DIR}/${name}.pid"

  echo "➡️  Starting ${name} in ${dir} ... (logs: ${log_file})"
  (
    cd "${dir}"
    load_shell_env
    nohup bash -lc "${cmd}" >"${log_file}" 2>&1 &
    echo $! > "${pid_file}"
  )
}

# Run pre-clean unless skipped
if [ "${SKIP_PRE_CLEAN:-0}" != "1" ]; then
  pre_clean
fi

ensure_redis

echo "🚀 QuantDesk - Launching all dev services"

# Backend (3002)
if port_in_use 3002; then
  echo "⏭️  Skipping backend start: port 3002 in use."
else
  start_service backend "${ROOT_DIR}/backend" pnpm run start:dev
fi

# Frontend (3001)
if port_in_use 3001; then
  echo "⏭️  Skipping frontend start: port 3001 in use."
else
  start_service frontend "${ROOT_DIR}/frontend" pnpm run dev
fi

# MIKEY-AI (3000)
if port_in_use 3000; then
  echo "⏭️  Skipping mikey-ai start: port 3000 in use."
else
  start_service mikey-ai "${ROOT_DIR}/MIKEY-AI" pnpm run dev
fi

# Data Ingestion (3003)
if port_in_use 3003; then
  echo "⏭️  Skipping data-ingestion start: port 3003 in use."
else
  start_service data-ingestion "${ROOT_DIR}/data-ingestion" pnpm run start:api || true
fi

# Optional: Heavy collectors (opt-in)
if [ "${START_COLLECTORS:-0}" = "1" ]; then
  echo "🧩 START_COLLECTORS=1 → Launching data collectors (price/whales/news/etc.)"
  start_service data-ingestion-collectors "${ROOT_DIR}/data-ingestion" pnpm run start:collectors || true
fi

# Docs Site (no fixed port check here)
if [ -f "${ROOT_DIR}/docs-site/start-docs.sh" ]; then
  start_service docs-site "${ROOT_DIR}/docs-site" bash start-docs.sh
fi

# Admin Dashboard (3005)
if [ -f "${ROOT_DIR}/admin-dashboard/start-admin.sh" ]; then
  if port_in_use 3005; then
    echo "⏭️  Skipping admin-dashboard start: port 3005 in use."
  else
    start_service admin-dashboard "${ROOT_DIR}/admin-dashboard" bash start-admin.sh
  fi
fi

echo "✅ All start commands issued. Tail logs with: tail -f ${LOG_DIR}/*.log"
cat << EOF
🔎 Health:
  - Frontend:          http://localhost:3001
  - Backend API:       http://localhost:3002/health (or /api/dev/codebase-structure)
  - MIKEY-AI API:      http://localhost:3000/llm/status
  - Admin Dashboard:   http://localhost:3005 (if configured)
EOF

exit 0


