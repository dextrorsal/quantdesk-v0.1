#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Killing ALL QuantDesk processes (Frontend + Backend)...${NC}"

# Kill frontend processes
echo -e "${YELLOW}Killing frontend processes...${NC}"
pkill -f "vite" 2>/dev/null
pkill -f "npm run dev" 2>/dev/null

# Kill backend processes
echo -e "${YELLOW}Killing backend processes...${NC}"
pkill -f "nodemon" 2>/dev/null
pkill -f "ts-node" 2>/dev/null

# Kill any processes on common ports
echo -e "${YELLOW}Freeing up ports...${NC}"

# Frontend ports
for port in 5173 3000 8080; do
  PID=$(lsof -t -i:$port 2>/dev/null)
  if [ -n "$PID" ]; then
    echo -e "${YELLOW}Killing process on port $port (PID: $PID)${NC}"
    kill -9 "$PID" 2>/dev/null
  fi
done

# Backend ports
for port in 3002 3001; do
  PID=$(lsof -t -i:$port 2>/dev/null)
  if [ -n "$PID" ]; then
    echo -e "${YELLOW}Killing process on port $port (PID: $PID)${NC}"
    kill -9 "$PID" 2>/dev/null
  fi
done

# Kill any test processes
pkill -f "test-frontend-price-system.js" 2>/dev/null
pkill -f "test-" 2>/dev/null

echo -e "${GREEN}✅ All QuantDesk processes killed${NC}"
echo -e "${GREEN}✅ All ports are now free${NC}"
echo -e "${GREEN}✅ Ready to start fresh!${NC}"
