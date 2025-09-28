#!/bin/bash

# QuantDesk Backend Startup Script with Proper Cleanup
# This script ensures processes are properly killed when stopped

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to cleanup processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down backend...${NC}"
    
    # Kill nodemon and related processes
    pkill -f "nodemon" 2>/dev/null || true
    pkill -f "ts-node" 2>/dev/null || true
    pkill -f "node.*server.ts" 2>/dev/null || true
    
    # Kill any process on port 3002
    if lsof -ti:3002 >/dev/null 2>&1; then
        echo -e "${YELLOW}Killing process on port 3002...${NC}"
        lsof -ti:3002 | xargs kill -9 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ Backend stopped cleanly${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo -e "${GREEN}🚀 Starting QuantDesk Backend...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"

# Change to backend directory
cd "$(dirname "$0")"

# Start the backend with nodemon
TS_NODE_TRANSPILE_ONLY=1 npx nodemon --exec "node -r ts-node/register" src/server.ts

# If we get here, nodemon exited
cleanup
