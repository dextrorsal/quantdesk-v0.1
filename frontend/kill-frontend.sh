#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo -e "${YELLOW}🛑 Killing all QuantDesk frontend processes...${NC}"

# Find and kill Vite processes
VITE_PIDS=$(pgrep -f "vite")
if [ -n "$VITE_PIDS" ]; then
  echo -e "${YELLOW}Found Vite processes: $VITE_PIDS. Killing...${NC}"
  kill -9 $VITE_PIDS
  echo -e "${GREEN}✅ Vite processes killed.${NC}"
else
  echo -e "${GREEN}✅ No Vite processes found.${NC}"
fi

# Find and kill npm run dev processes
NPM_PIDS=$(pgrep -f "npm run dev")
if [ -n "$NPM_PIDS" ]; then
  echo -e "${YELLOW}Found npm run dev processes: $NPM_PIDS. Killing...${NC}"
  kill -9 $NPM_PIDS
  echo -e "${GREEN}✅ npm run dev processes killed.${NC}"
else
  echo -e "${GREEN}✅ No npm run dev processes found.${NC}"
fi

# Find and kill any Node processes running on port 5173 (Vite default)
PORT_5173_PID=$(lsof -t -i:5173)
if [ -n "$PORT_5173_PID" ]; then
  echo -e "${YELLOW}Found process on port 5173 with PID: $PORT_5173_PID. Killing...${NC}"
  kill -9 "$PORT_5173_PID"
  echo -e "${GREEN}✅ Process on port 5173 killed.${NC}"
else
  echo -e "${GREEN}✅ No processes found on port 5173.${NC}"
fi

# Ensure dev port 3001 is free (Vite configured port)
PORT_3001_PID=$(lsof -t -i:3001)
if [ -n "$PORT_3001_PID" ]; then
  echo -e "${YELLOW}Found process on port 3001 with PID: $PORT_3001_PID. Killing...${NC}"
  kill -9 "$PORT_3001_PID"
  echo -e "${GREEN}✅ Process on port 3001 killed.${NC}"
else
  echo -e "${GREEN}✅ No processes found on port 3001.${NC}"
fi

echo -e "${GREEN}✅ All frontend processes killed${NC}"
echo -e "${GREEN}✅ Frontend ports are now free${NC}"
