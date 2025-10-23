#!/bin/bash

# QuantDesk Backend Kill Script
# Kills all backend processes and frees up port 3002

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Killing all QuantDesk backend processes...${NC}"

# Kill all backend-related processes
pkill -f "nodemon" 2>/dev/null || true
pkill -f "ts-node" 2>/dev/null || true
pkill -f "node.*server.ts" 2>/dev/null || true
pkill -f "node.*backend" 2>/dev/null || true

# Kill any process on port 3002
if lsof -ti:3002 >/dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port 3002...${NC}"
    lsof -ti:3002 | xargs kill -9 2>/dev/null || true
fi

# Wait a moment for processes to die
sleep 1

# Check if port is free
if lsof -ti:3002 >/dev/null 2>&1; then
    echo -e "${RED}❌ Port 3002 still in use${NC}"
    echo -e "${YELLOW}Processes still running on port 3002:${NC}"
    lsof -ti:3002 | xargs ps -p
else
    echo -e "${GREEN}✅ All backend processes killed${NC}"
    echo -e "${GREEN}✅ Port 3002 is now free${NC}"
fi
