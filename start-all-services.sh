#!/bin/bash

# 🚀 QuantDesk Complete Startup Script
# Starts all services: Backend, Frontend, Data Ingestion, MIKEY-AI

echo "🚀 Starting QuantDesk Platform..."
echo "================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to check if port is in use
check_port() {
    if lsof -Pi :$1 -sTCP:LISTEN -t >/dev/null ; then
        echo -e "${YELLOW}⚠️  Port $1 is already in use${NC}"
        return 1
    else
        echo -e "${GREEN}✅ Port $1 is available${NC}"
        return 0
    fi
}

# Function to kill processes on specific ports
kill_port() {
    echo -e "${YELLOW}🔄 Killing processes on port $1...${NC}"
    lsof -ti:$1 | xargs kill -9 2>/dev/null || true
    sleep 2
}

# Function to start service in background
start_service() {
    local service_name=$1
    local port=$2
    local start_command=$3
    local working_dir=$4
    
    echo -e "\n${BLUE}🚀 Starting $service_name...${NC}"
    
    if ! check_port $port; then
        kill_port $port
    fi
    
    # Start service in background from the correct directory
    (cd $working_dir && nohup $start_command > ../logs/${service_name}.log 2>&1 &)
    local pid=$!
    
    # Wait a moment and check if it's still running
    sleep 3
    if kill -0 $pid 2>/dev/null; then
        echo -e "${GREEN}✅ $service_name started successfully (PID: $pid)${NC}"
        echo -e "${GREEN}   Port: $port${NC}"
        echo -e "${GREEN}   Logs: logs/${service_name}.log${NC}"
    else
        echo -e "${RED}❌ $service_name failed to start${NC}"
        echo -e "${RED}   Check logs/${service_name}.log for details${NC}"
    fi
}

# Create logs directory
mkdir -p logs

echo -e "\n${YELLOW}🔍 Checking Redis connection...${NC}"
if docker ps | grep -q redis-quantdesk; then
    echo -e "${GREEN}✅ Redis is running${NC}"
else
    echo -e "${RED}❌ Redis is not running. Starting Redis...${NC}"
    docker run -d --name redis-quantdesk -p 6379:6379 redis:alpine
    sleep 3
fi

echo -e "\n${YELLOW}🔍 Checking PostgreSQL connection...${NC}"
# Test database connection
cd backend
if node -e "
const { Pool } = require('pg');
const pool = new Pool({
  connectionString: process.env.DATABASE_URL || 'postgresql://postgres:password@localhost:5432/quantdesk',
  ssl: false
});
pool.query('SELECT 1').then(() => {
  console.log('✅ Database connection successful');
  process.exit(0);
}).catch(err => {
  console.log('❌ Database connection failed:', err.message);
  process.exit(1);
});
" 2>/dev/null; then
    echo -e "${GREEN}✅ Database connection successful${NC}"
else
    echo -e "${RED}❌ Database connection failed${NC}"
    echo -e "${YELLOW}⚠️  Services may not work properly without database${NC}"
fi
cd ..

echo -e "\n${BLUE}🚀 Starting QuantDesk Services...${NC}"

# Start Backend (Port 3002)
start_service "Backend" 3002 "./start-backend.sh" "backend"

# Start Frontend (Port 3001) 
start_service "Frontend" 3001 "./start-frontend.sh" "frontend"

# Start Data Ingestion (Port 3003)
start_service "Data-Ingestion" 3003 "./start-pipeline.sh" "data-ingestion"

# Start MIKEY-AI (Port 3000)
start_service "MIKEY-AI" 3000 "npm start" "MIKEY-AI"

echo -e "\n${GREEN}🎉 All services started!${NC}"
echo -e "\n${BLUE}📊 Service Status:${NC}"
echo -e "${GREEN}✅ Backend:        http://localhost:3002${NC}"
echo -e "${GREEN}✅ Frontend:       http://localhost:3001${NC}"
echo -e "${GREEN}✅ Data Ingestion: http://localhost:3003${NC}"
echo -e "${GREEN}✅ MIKEY-AI:       http://localhost:3000${NC}"
echo -e "${GREEN}✅ Redis:          localhost:6379${NC}"

echo -e "\n${YELLOW}📋 Quick Health Checks:${NC}"
echo -e "curl http://localhost:3002/health"
echo -e "curl http://localhost:3000/health"
echo -e "curl http://localhost:3003/health"

echo -e "\n${BLUE}📝 Logs Location:${NC}"
echo -e "tail -f logs/Backend.log"
echo -e "tail -f logs/Frontend.log"
echo -e "tail -f logs/Data-Ingestion.log"
echo -e "tail -f logs/MIKEY-AI.log"

echo -e "\n${YELLOW}🛑 To stop all services:${NC}"
echo -e "pkill -f 'node.*backend'"
echo -e "pkill -f 'node.*frontend'"
echo -e "pkill -f 'node.*data-ingestion'"
echo -e "pkill -f 'node.*MIKEY-AI'"
echo -e "docker stop redis-quantdesk"

echo -e "\n${GREEN}🚀 QuantDesk Platform is ready!${NC}"
