#!/bin/bash

# QuantDesk Admin Dashboard Kill Script
# Kills all admin dashboard processes and frees up port 5173

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Killing all QuantDesk admin dashboard processes...${NC}"

# Kill all admin dashboard-related processes
pkill -f "vite" 2>/dev/null || true
pkill -f "admin-dashboard.*npm run dev" 2>/dev/null || true
pkill -f "npm run dev" 2>/dev/null || true

# Kill any process on port 5173
if lsof -ti:5173 >/dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port 5173...${NC}"
    lsof -ti:5173 | xargs kill -9 2>/dev/null || true
fi

# Wait a moment for processes to die
sleep 1

# Check if port is free
if lsof -ti:5173 >/dev/null 2>&1; then
    echo -e "${RED}❌ Port 5173 still in use${NC}"
    echo -e "${YELLOW}Processes still running on port 5173:${NC}"
    lsof -ti:5173 | xargs ps -p
else
    echo -e "${GREEN}✅ All admin dashboard processes killed${NC}"
    echo -e "${GREEN}✅ Port 5173 is now free${NC}"
fi