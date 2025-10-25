#!/bin/bash

# QuantDesk Backend + Redis Stop Script
# This script stops both backend and Redis cleanly

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Stopping QuantDesk Backend and Redis...${NC}"

# Kill backend processes
echo -e "${YELLOW}Stopping backend processes...${NC}"
pkill -f "nodemon" 2>/dev/null || true
pkill -f "ts-node" 2>/dev/null || true
pkill -f "node.*server.ts" 2>/dev/null || true

# Kill any process on port 3002
if lsof -ti:3002 >/dev/null 2>&1; then
    echo -e "${YELLOW}Killing process on port 3002...${NC}"
    lsof -ti:3002 | xargs kill -9 2>/dev/null || true
fi

# Stop Redis container
echo -e "${YELLOW}Stopping Redis container...${NC}"
if docker ps --format "table {{.Names}}" | grep -q "^quantdesk-redis$"; then
    docker stop quantdesk-redis
    echo -e "${GREEN}✅ Redis container stopped${NC}"
else
    echo -e "${YELLOW}Redis container not running${NC}"
fi

echo -e "${GREEN}✅ All services stopped cleanly${NC}"
