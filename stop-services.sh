#!/bin/bash

# QuantDesk Service Stop Script
# Convenient script to stop all services

echo "🛑 Stopping QuantDesk Services"
echo "============================="

# Kill all Node.js processes (be careful with this)
echo "🔍 Finding QuantDesk processes..."

# Kill backend
BACKEND_PID=$(ps aux | grep "backend.*start:dev" | grep -v grep | awk '{print $2}')
if [ ! -z "$BACKEND_PID" ]; then
    echo "📡 Stopping Backend (PID: $BACKEND_PID)..."
    kill $BACKEND_PID
fi

# Kill frontend
FRONTEND_PID=$(ps aux | grep "frontend.*dev" | grep -v grep | awk '{print $2}')
if [ ! -z "$FRONTEND_PID" ]; then
    echo "🎨 Stopping Frontend (PID: $FRONTEND_PID)..."
    kill $FRONTEND_PID
fi

# Kill MIKEY-AI
MIKEY_PID=$(ps aux | grep "MIKEY-AI.*dev" | grep -v grep | awk '{print $2}')
if [ ! -z "$MIKEY_PID" ]; then
    echo "🤖 Stopping MIKEY-AI (PID: $MIKEY_PID)..."
    kill $MIKEY_PID
fi

# Kill data ingestion
DATA_PID=$(ps aux | grep "data-ingestion.*dev" | grep -v grep | awk '{print $2}')
if [ ! -z "$DATA_PID" ]; then
    echo "📊 Stopping Data Ingestion (PID: $DATA_PID)..."
    kill $DATA_PID
fi

# Kill any remaining pnpm processes
echo "🧹 Cleaning up remaining pnpm processes..."
pkill -f "pnpm.*dev" 2>/dev/null || true

echo ""
echo "✅ All services stopped!"
echo "======================="
echo "All QuantDesk services have been terminated."
