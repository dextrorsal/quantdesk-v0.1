#!/bin/bash

# QuantDesk Service Startup Script
# Convenient script to start all essential services

echo "🚀 Starting QuantDesk Services"
echo "=============================="

# Start Backend
echo "📡 Starting Backend (Port 3002)..."
cd backend && pnpm run start:dev &
BACKEND_PID=$!

# Wait a moment for backend to start
sleep 3

# Start Frontend  
echo "🎨 Starting Frontend (Port 3001)..."
cd ../frontend && pnpm run dev &
FRONTEND_PID=$!

# Start MIKEY-AI
echo "🤖 Starting MIKEY-AI (Port 3000)..."
cd ../MIKEY-AI && pnpm run dev &
MIKEY_PID=$!

# Start Data Ingestion
echo "📊 Starting Data Ingestion (Port 3003)..."
cd ../data-ingestion && pnpm run dev &
DATA_PID=$!

echo ""
echo "✅ All services started!"
echo "======================="
echo "Backend:    http://localhost:3002"
echo "Frontend:   http://localhost:3001" 
echo "MIKEY-AI:   http://localhost:3000"
echo "Data Ingestion: http://localhost:3003"
echo ""
echo "Process IDs:"
echo "Backend: $BACKEND_PID"
echo "Frontend: $FRONTEND_PID"
echo "MIKEY-AI: $MIKEY_PID"
echo "Data Ingestion: $DATA_PID"
echo ""
echo "To stop all services, run: ./stop-services.sh"
