#!/bin/bash

# QuantDesk Admin Dashboard Startup Script with Proper Cleanup
# This script ensures processes are properly killed when stopped

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to cleanup processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Shutting down admin dashboard...${NC}"
    
    # Kill vite and related processes
    pkill -f "vite" 2>/dev/null || true
    pkill -f "admin-dashboard.*npm run dev" 2>/dev/null || true
    
    # Kill any process on port 5173
    if lsof -ti:5173 >/dev/null 2>&1; then
        echo -e "${YELLOW}Killing process on port 5173...${NC}"
        lsof -ti:5173 | xargs kill -9 2>/dev/null || true
    fi
    
    echo -e "${GREEN}✅ Admin dashboard stopped cleanly${NC}"
    exit 0
}

# Set up signal handlers
trap cleanup SIGINT SIGTERM

echo -e "${GREEN}🚀 Starting QuantDesk Admin Dashboard...${NC}"
echo -e "${YELLOW}Press Ctrl+C to stop${NC}"
echo -e "${YELLOW}Admin Dashboard will be available at: http://localhost:5173${NC}"

# Change to admin dashboard directory
cd "$(dirname "$0")"

# Start the admin dashboard with npm run dev
npm run dev

# If we get here, npm run dev exited
cleanup