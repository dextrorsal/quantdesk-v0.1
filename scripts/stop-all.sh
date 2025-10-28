#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${ROOT_DIR}/logs"

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

stop_service() {
  local name="$1"
  local pid_file="${LOG_DIR}/${name}.pid"
  if [ -f "${pid_file}" ]; then
    pid=$(cat "${pid_file}" || true)
    if [ -n "${pid:-}" ] && kill -0 "${pid}" 2>/dev/null; then
      echo "🛑 Stopping ${name} (pid ${pid})"
      kill "${pid}" 2>/dev/null || true
      # Wait briefly, then force kill if needed
      for i in {1..10}; do
        if kill -0 "${pid}" 2>/dev/null; then
          sleep 0.2
        else
          break
        fi
      done
      kill -9 "${pid}" 2>/dev/null || true
    else
      echo "ℹ️  ${name} not running"
    fi
    rm -f "${pid_file}"
  else
    echo "ℹ️  No pid file for ${name}"
  fi
}

# Stop via pid files first
for svc in backend frontend mikey-ai data-ingestion docs-site admin-dashboard; do
  stop_service "$svc"
done

# Additional cleanup: kill known ports
kill_by_port 3000  # MIKEY-AI
kill_by_port 3001  # frontend
kill_by_port 3002  # backend
kill_by_port 3003  # data-ingestion
kill_by_port 3005  # admin-dashboard

# Additional cleanup: kill common process patterns
kill_by_pattern "tsx.*MIKEY-AI"
kill_by_pattern "vite.*frontend"
kill_by_pattern "vite.*admin-dashboard"
kill_by_pattern "node .*backend/dist/server.js"
kill_by_pattern "ts-node src/server.ts"

# Final wait to ensure processes exit
sleep 0.5

echo "✅ All services stop attempted."


